"""03 — Segmenta los .md en chunks de nivel artículo con ventana + solapamiento.

Reglas (alineadas con Machine A):
- Unidad primaria: artículo (`ARTÍCULO N` o `Art. N`).
- Si un artículo excede `window_tokens`, se corta en ventanas con solapamiento.
- Chunks con < `min_chunk_tokens` se fusionan con el siguiente.
- Encabezado jerárquico `[Norma — Libro — Título — Capítulo — Art. N]` se preserva.
- Las sentencias (sin "ARTÍCULO") se cortan por párrafo.

Salida: process/03_chunks.jsonl con doc_id, chunk_id, norma, numero_articulo,
encabezado, texto, parent_inicio, parent_fin, nivel.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_config, paths, setup_logging, write_jsonl

log = setup_logging("chunk")

ARTICLE_RE = re.compile(r"^\s*(?:ART[IÍ]CULO|Art\.?)\s*(\d+[A-Za-z]?)\b", re.MULTILINE | re.IGNORECASE)
HEADING_RE = re.compile(r"^(#{1,3})\s+(LIBRO|T[IÍ]TULO|CAP[IÍ]TULO)\b.*", re.MULTILINE | re.IGNORECASE)
FRONT_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def _parse_front(text: str) -> tuple[dict, str]:
    m = FRONT_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = {}
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = json.loads(v.strip())
    except Exception:
        meta = {}
    return meta, text[m.end():]


def _tokenize_count(text: str) -> int:
    # Aprox: 1 token ≈ 4 chars español; evita cargar un tokenizador aquí.
    return max(1, len(text) // 4)


def _chunk_articles(doc_id: str, meta: dict, body: str, cfg: dict) -> list[dict]:
    article_matches = list(ARTICLE_RE.finditer(body))
    chunks: list[dict] = []
    if article_matches:
        for i, m in enumerate(article_matches):
            start = m.start()
            end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(body)
            art_text = body[start:end].strip()
            if _tokenize_count(art_text) <= cfg["chunking"]["window_tokens"]:
                chunks.append(_make_chunk(doc_id, meta, m.group(1), start, end, art_text, body))
            else:
                for sub in _window(art_text, cfg):
                    chunks.append(_make_chunk(doc_id, meta, m.group(1), start, end, sub, body))
    else:
        # Documentos sin "ARTÍCULO" (sentencias): cortar por párrafo.
        for sub in _window(body, cfg):
            chunks.append(_make_chunk(doc_id, meta, None, 0, len(body), sub, body))
    # Fusionar chunks demasiado cortos
    merged: list[dict] = []
    for c in chunks:
        if merged and _tokenize_count(c["texto"]) < cfg["chunking"]["min_chunk_tokens"]:
            merged[-1]["texto"] += "\n\n" + c["texto"]
            merged[-1]["fin"] = c["fin"]
        else:
            merged.append(c)
    # Asignar chunk_id secuencial
    for i, c in enumerate(merged):
        c["chunk_id"] = f"{doc_id}:{i:04d}"
    return merged


def _window(text: str, cfg: dict) -> list[str]:
    size = cfg["chunking"]["window_tokens"] * 4
    overlap = cfg["chunking"]["overlap_tokens"] * 4
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def _make_chunk(doc_id: str, meta: dict, art: str | None, start: int, end: int,
                texto: str, body: str) -> dict:
    header_parts = []
    canonical = meta.get("norma") or meta.get("doc_id") or doc_id
    header_parts.append(canonical)
    # Buscar heading jerárquico más cercano antes del artículo
    for h in HEADING_RE.finditer(body[:start]):
        header_parts.append(re.sub(r"^#+\s*", "", h.group(0)).strip())
    if art:
        header_parts.append(f"Art. {art}")
    encabezado = "[" + " — ".join(header_parts[:4]) + "]"
    return {
        "doc_id": doc_id,
        "norma": meta.get("tipo", "") + "_" + str(meta.get("numero", "")) + "_" + str(meta.get("anio", "")) if meta.get("tipo") else doc_id,
        "numero_articulo": art,
        "parent_inicio": start,
        "parent_fin": end,
        "inicio": start,
        "fin": end,
        "encabezado": encabezado,
        "texto": texto.strip(),
        "nivel": "nucleo",
    }


def main() -> int:
    cfg = load_config()
    p = paths(cfg)
    md_dir = p["output"] / "02_markdown"
    all_chunks: list[dict] = []
    docs = 0
    for md_path in sorted(md_dir.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8")
        meta, body = _parse_front(raw)
        doc_id = meta.get("doc_id") or md_path.stem
        chunks = _chunk_articles(doc_id, meta, body, cfg)
        all_chunks.extend(chunks)
        docs += 1
    write_jsonl(p["output"] / "03_chunks.jsonl", all_chunks)
    log.info("Chunking: %s documentos -> %s chunks", docs, len(all_chunks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
