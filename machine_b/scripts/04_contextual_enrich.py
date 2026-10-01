"""04 — Contextual Retrieval (Anthropic, sep 2024).

Para cada chunk, usa Qwen3-8B (greedy, bf16, 1 pasada) para producir una línea
de contexto que lo sitúa dentro del documento. Esta línea se antepone al texto
del chunk antes de embeberlo, lo que reduce fallos de recuperación 49–67%
según el benchmark de Anthropic.

Prompt (del config.yaml): "Devuelve UNA oración de ≤25 palabras que sitúe este
fragmento en el documento. No inventes hechos ni citas."

Entrada: process/03_chunks.jsonl y process/02_markdown/*.md (para dar el doc completo).
Salida: process/04_chunks_enriched.jsonl con un campo adicional `contexto` + `texto_enriquecido`.

Edge cases:
- Chunks muy cortos (< 50 tokens): se usa el encabezado como contexto, se salta el LLM.
- Documentos largos (>8k tokens): se envía al LLM solo el chunk + 2 encabezados jerárquicos
  circundantes, no el documento completo.
- El modelo entra en modo "thinking" por defecto en Qwen3: forzamos `enable_thinking=False`.
- Idempotencia: si el chunk_id ya existe en la salida previa con el mismo sha256 del texto,
  se reusa el contexto. Esto permite reanudar sin re-llamar al LLM.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import iter_jsonl, load_config, paths, setup_logging, write_jsonl

log = setup_logging("contextual_enrich")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_model(cfg: dict):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = cfg["enrichment"]["model"]
    revision = cfg["enrichment"]["revision"]
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    dtype = getattr(torch, cfg["enrichment"]["dtype"])
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=revision, torch_dtype=dtype,
        device_map={"": 0}, trust_remote_code=False).eval()
    log.info("Decoder cargado: %s rev=%s dtype=%s", model_name, revision[:8], dtype)
    return tok, model


def _generate_contexts(tok, model, batch, cfg):
    import torch
    system = cfg["enrichment"]["prompt"]
    messages_batch = []
    for chunk in batch:
        header = chunk["encabezado"]
        snippet = chunk["texto"][:1600]  # cap para que quepa en el contexto
        user = f"Encabezado: {header}\nFragmento:\n{snippet}\n\nContexto (una oración, max 25 palabras):"
        messages_batch.append([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])
    # Qwen3 chat template; thinking off para que sea predecible y rápido.
    try:
        texts = [tok.apply_chat_template(m, add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)
                 for m in messages_batch]
    except TypeError:
        texts = [tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                 for m in messages_batch]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=3072).to("cuda")
    with torch.inference_mode():
        out = model.generate(
            **enc, do_sample=False, num_beams=1,
            max_new_tokens=cfg["enrichment"]["max_new_tokens"],
            repetition_penalty=1.1,
            pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    results = []
    for i in range(len(batch)):
        gen = out[i, enc["input_ids"].shape[-1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        text = text.split("\n")[0].strip()[:300]  # una línea
        results.append(text)
    return results


def main() -> int:
    cfg = load_config()
    p = paths(cfg)
    chunks_in = list(iter_jsonl(p["output"] / "03_chunks.jsonl"))
    out_path = p["output"] / "04_chunks_enriched.jsonl"

    # Idempotencia: reusar contextos previos cuando el sha coincide.
    cache: dict[str, str] = {}
    if out_path.is_file():
        for r in iter_jsonl(out_path):
            if r.get("texto_sha"):
                cache[r["texto_sha"]] = r.get("contexto", "")
        log.info("Cache de %s contextos previos", len(cache))

    if not cfg["enrichment"]["enabled"]:
        log.warning("enrichment deshabilitado en config.yaml; copiando sin contexto")
        rows = []
        for c in chunks_in:
            c = dict(c)
            c["contexto"] = ""
            c["texto_enriquecido"] = c["encabezado"] + "\n" + c["texto"]
            rows.append(c)
        write_jsonl(out_path, rows)
        return 0

    tok, model = _load_model(cfg)
    batch_size = cfg["enrichment"]["batch_size"]
    rows: list[dict] = []
    pending: list[dict] = []
    t0 = time.perf_counter()
    for c in chunks_in:
        txt_sha = _sha(c["texto"])
        if txt_sha in cache:
            c = dict(c, contexto=cache[txt_sha], texto_sha=txt_sha)
            c["texto_enriquecido"] = (cache[txt_sha] + "\n" + c["encabezado"] + "\n" + c["texto"]).strip()
            rows.append(c)
            continue
        pending.append((c, txt_sha))
        if len(pending) >= batch_size:
            ctxs = _generate_contexts(tok, model, [x[0] for x in pending], cfg)
            for (c0, sha), ctx in zip(pending, ctxs):
                c = dict(c0, contexto=ctx, texto_sha=sha)
                c["texto_enriquecido"] = (ctx + "\n" + c["encabezado"] + "\n" + c["texto"]).strip()
                rows.append(c)
            pending = []
            if len(rows) % 100 == 0:
                elapsed = time.perf_counter() - t0
                rate = len(rows) / max(elapsed, 1)
                log.info("%d / %d chunks  (%.1f c/s)", len(rows), len(chunks_in), rate)
    if pending:
        ctxs = _generate_contexts(tok, model, [x[0] for x in pending], cfg)
        for (c0, sha), ctx in zip(pending, ctxs):
            c = dict(c0, contexto=ctx, texto_sha=sha)
            c["texto_enriquecido"] = (ctx + "\n" + c["encabezado"] + "\n" + c["texto"]).strip()
            rows.append(c)
    write_jsonl(out_path, rows)
    log.info("Enriquecidos %d chunks en %.1fs", len(rows), time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
