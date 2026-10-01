"""Experimentos secuenciales sobre corpus_eval_v1. Sin lectura de claves al generar.

Los resultados se guardan por etapa. El índice léxico y las matrices FAISS
viven en disco/CPU; cada modelo se carga y libera individualmente en GPU.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import closing
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / "data/releases/corpus_eval_v1"
SAMPLE = ROOT / "data/oficial/data/sample_50.jsonl"
MODEL_CATALOG = ROOT / "reports/reporte_evaluacion/modelos_verificados.json"
RUNS = ROOT / "data/experimentos/corpus_definitivo"
ENCODERS = ["BAAI/bge-m3", "intfloat/multilingual-e5-large",
            "Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B"]
DECODERS = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3.5-4B"]
RERANKER = "BAAI/bge-reranker-v2-m3"
QWEN_RERANKER = "Qwen/Qwen3-Reranker-0.6B"
WINDOW_CHARS = 1500
OVERLAP_CHARS = 200
SCHEMA_PATH = ROOT / "data/oficial/schema/submission.schema.json"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def catalog():
    return {row["repo_id"]: row for row in json.loads(MODEL_CATALOG.read_text(encoding="utf-8"))}


def hardware():
    import psutil
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere CUDA. Selecciona RTX 4090, A100 o L4 y reinstala PyTorch CUDA.")
    p = torch.cuda.get_device_properties(0)
    total = p.total_memory / 2**30
    free, _ = torch.cuda.mem_get_info()
    if total < 20:
        raise RuntimeError(f"GPU de {total:.1f} GiB: este barrido requiere al menos 20 GiB.")
    dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return {"nombre": p.name, "vram_gib": round(total, 2),
            "vram_libre_gib": round(free / 2**30, 2), "ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
            "dtype": dtype, "batch_inicial": 64 if total >= 38 else 24,
            "contexto": 8192 if total >= 38 else 6144,
            "max_nuevos_tokens": 800 if total >= 38 else 600}


def preflight(verify_hashes=True):
    import torch
    import transformers
    import faiss
    try:
        import torchvision  # requerido por AutoProcessor de Qwen3.5
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("Instala torchvision compatible con PyTorch CUDA antes de ejecutar E06") from exc
    from legalrag.evaluation.entrega import preparar_entrada
    from packaging.version import Version
    if Version(transformers.__version__) < Version("5.3"):
        raise RuntimeError("Este notebook necesita transformers >=5.3 para Qwen3.5-4B.")
    for p in [RELEASE / "snapshot.json", RELEASE / "corpus_manifest.json", SAMPLE,
              SCHEMA_PATH, ROOT / "data/oficial/scripts/evaluate.py", MODEL_CATALOG]:
        if not p.is_file():
            raise FileNotFoundError(p)
    snapshot = json.loads((RELEASE / "snapshot.json").read_text(encoding="utf-8"))
    manifest = json.loads((RELEASE / "corpus_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest) == snapshot["documentos_evaluables"] == 13962
    for record in manifest:
        path = ROOT / record["texto_archivo"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if verify_hashes and sha(path) != record["sha256_texto"]:
            raise ValueError("Cambió el texto del snapshot: " + record["doc_id"])
    items = [preparar_entrada(x) for x in read_jsonl(SAMPLE)]
    if len(items) != 50 or len({x["id"] for x in items}) != 50:
        raise ValueError("La muestra oficial no tiene 50 ids únicos")
    models = catalog()
    for name in ENCODERS + DECODERS + [RERANKER, QWEN_RERANKER]:
        m = models[name]
        if not m["cumple_limite_8000000000"] or m["parametros"] > 8_000_000_000:
            raise ValueError("Modelo fuera del límite: " + name)
    result = {"snapshot_id": snapshot["snapshot_id"], "documentos": len(manifest),
              "preguntas": len(items), "transformers": transformers.__version__,
              "torch": torch.__version__, "faiss": faiss.__version__, "hardware": hardware(),
              "textos_sha256_verificados": verify_hashes}
    dump(RUNS / "preflight.json", result)
    return result


def questions():
    from legalrag.evaluation.entrega import preparar_entrada
    return [preparar_entrada(x) for x in read_jsonl(SAMPLE)]


def query(item):
    options = item.get("opciones") or {}
    return item["pregunta"] + "\n" + "\n".join(f"{k}. {v}" for k, v in options.items())


def _db():
    path = RUNS / "chunks.sqlite"
    conn = sqlite3.connect(path, timeout=120)
    conn.row_factory = sqlite3.Row
    return conn


INSERT_CHUNK = ("INSERT INTO chunks(doc_id,titulo,tipo,articulo,seccion,unidad_id,unidad_inicio,unidad_fin,inicio,fin,"
                "texto,texto_busqueda,avisos) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)")


def chunk_rows(doc, text):
    """Filas de `chunks` de un documento (también las usa ingestion.agregar_puntuales)."""
    from legalrag.preprocessing.ingesta import segmentar_documento
    rows = []
    for part in segmentar_documento(doc, text, max_chars=WINDOW_CHARS, solapamiento=OVERLAP_CHARS):
        if not part["apta_para_busqueda"] or not part["texto"].strip():
            continue
        # La cabecera es señal de búsqueda, no se añade al pasaje literal. En Markdown (norma > sección >
        # artículo) para que BM25 y el encoder vean la jerarquía del documento; el pasaje entregado sigue siendo
        # el tramo literal del texto canónico (.txt), con sus posiciones.
        header = "\n".join(x for x in [f"# {doc['titulo']}",
                                       f"## {part['seccion']}" if part.get("seccion") else None,
                                       f"### Artículo {part['articulo']}" if part.get("articulo") else None] if x)
        rows.append((doc["doc_id"], doc["titulo"], doc["tipo"], part.get("articulo"),
                     part.get("seccion"), part["unidad_id"], part["unidad_inicio"],
                     part["unidad_fin"], part["inicio"], part["fin"], part["texto"],
                     header + "\n\n" + part["texto"], json.dumps(part["avisos"], ensure_ascii=False)))
    return rows


def build_lexical(force=False, limit_docs=None):
    """Segmentación fuente a fuente; nunca modifica corpus_eval_v1."""
    RUNS.mkdir(parents=True, exist_ok=True)
    marker = RUNS / "chunks_complete.json"
    if marker.exists() and not force:
        meta = json.loads(marker.read_text(encoding="utf-8"))
        if meta.get("piloto") != (limit_docs is not None):
            raise ValueError("Índice piloto/completo incompatible. Usa force=True para reconstruir.")
        if meta["snapshot_id"] != json.loads((RELEASE / "snapshot.json").read_text(encoding="utf-8"))["snapshot_id"]:
            raise ValueError("Índice de otro snapshot; elige otra carpeta de resultados")
        return meta
    if (force or not marker.exists()) and (RUNS / "chunks.sqlite").exists():
        (RUNS / "chunks.sqlite").unlink()
    conn = _db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS chunks (
      id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, titulo TEXT NOT NULL,
      tipo TEXT NOT NULL, articulo TEXT, seccion TEXT, unidad_id TEXT NOT NULL,
      unidad_inicio INTEGER NOT NULL, unidad_fin INTEGER NOT NULL,
      inicio INTEGER NOT NULL, fin INTEGER NOT NULL, texto TEXT NOT NULL,
      texto_busqueda TEXT NOT NULL, avisos TEXT NOT NULL);
    CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
      texto_busqueda, content='chunks', content_rowid='id',
      tokenize='unicode61 remove_diacritics 2');
    """)
    manifest = json.loads((RELEASE / "corpus_manifest.json").read_text(encoding="utf-8"))
    if limit_docs is not None:
        manifest = manifest[:limit_docs]
    counts = defaultdict(int)
    t0 = time.perf_counter()
    for i, doc in enumerate(manifest, 1):
        text = (ROOT / doc["texto_archivo"]).read_text(encoding="utf-8")
        for row in chunk_rows(doc, text):
            cur = conn.execute(INSERT_CHUNK, row)
            conn.execute("INSERT INTO fts(rowid,texto_busqueda) VALUES(?,?)", (cur.lastrowid, row[-2]))
            counts["chunks"] += 1
        counts["documentos"] += 1
        if i % 100 == 0:
            conn.commit()
            print(f"Segmentados {i}/{len(manifest)} documentos", end="\r")
    conn.commit()
    conn.execute("CREATE INDEX IF NOT EXISTS idx_doc ON chunks(doc_id)")
    conn.commit()
    conn.close()
    snapshot = json.loads((RELEASE / "snapshot.json").read_text(encoding="utf-8"))
    meta = {"snapshot_id": snapshot["snapshot_id"], **counts, "window_chars": WINDOW_CHARS,
            "overlap_chars": OVERLAP_CHARS, "segundos": round(time.perf_counter()-t0, 1),
            "piloto": limit_docs is not None}
    dump(marker, meta)
    return meta


def bm25(item, k=100):
    terms = re.findall(r"\w+", query(item).casefold())
    expression = " OR ".join('"' + t.replace('"', '') + '"' for t in dict.fromkeys(terms))
    if not expression:
        return []
    with closing(_db()) as conn:
        return [(int(r[0]), float(-r[1])) for r in conn.execute(
            "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
            (expression, k)).fetchall()]


def _release_gpu(*models):
    import torch
    for m in models:
        del m
    gc.collect()
    torch.cuda.empty_cache()


class Encoder:
    def __init__(self, name):
        import torch
        from transformers import AutoModel, AutoTokenizer
        m = catalog()[name]
        self.name = name
        self.e5 = "multilingual-e5" in name
        self.qwen = "Qwen3-Embedding" in name
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=m["revision"], trust_remote_code=False)
        dtype = getattr(torch, hardware()["dtype"])
        self.model = AutoModel.from_pretrained(name, revision=m["revision"], dtype=dtype,
                                                trust_remote_code=False).to("cuda").eval()
        self.max_tokens = 512 if self.e5 else 1024
        self.batch_size = hardware()["batch_inicial"]

    def encode(self, texts, task="passage", batch=None):
        import numpy as np
        import torch
        batch = batch or self.batch_size
        results = []
        i = 0
        while i < len(texts):
            source = texts[i:i+batch]
            if self.e5:
                source = [("query: " if task == "query" else "passage: ") + x for x in source]
            elif self.qwen and task == "query":
                source = ["Instruct: Recupera pasajes jurídicos colombianos pertinentes a la pregunta\nQuery: " + x for x in source]
            try:
                tok = self.tokenizer(source, padding=True, truncation=True,
                                     max_length=self.max_tokens, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    out = self.model(**tok)
                    hidden = out.last_hidden_state.float()
                    if self.e5:
                        mask = tok["attention_mask"].unsqueeze(-1)
                        emb = (hidden * mask).sum(1) / mask.sum(1)
                    elif self.qwen:
                        ends = tok["attention_mask"].sum(1) - 1
                        if self.tokenizer.padding_side == "left":
                            ends = torch.full_like(ends, hidden.shape[1]-1)
                        emb = hidden[torch.arange(hidden.shape[0], device="cuda"), ends]
                    else:
                        emb = hidden[:, 0]
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                    results.append(emb.cpu().numpy().astype("float32"))
                i += len(source)
                if i % 5000 < batch:
                    print(f"Embeddings {self.name}: {i}/{len(texts)}", end="\r")
            except torch.cuda.OutOfMemoryError:
                _release_gpu()
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                self.batch_size = batch
                print(f"VRAM llena; lote reducido a {batch}")
        return np.vstack(results)

    def close(self):
        model = self.model
        self.model = None
        _release_gpu(model)


def _index_dir(name):
    return RUNS / "indices" / name.replace("/", "__")


def build_dense(name, block_size=1024):
    import faiss
    import numpy as np
    if not (RUNS / "chunks_complete.json").is_file():
        raise RuntimeError("Primero construye chunks.sqlite")
    target = _index_dir(name)
    target.mkdir(parents=True, exist_ok=True)
    marker = target / "complete.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    cat = catalog()[name]
    encoder = Encoder(name)
    with closing(_db()) as conn:
        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        last = 0
        index = None
        if (target / "checkpoint.faiss").exists():
            index = faiss.read_index(str(target / "checkpoint.faiss"))
            last = index.ntotal
        started = time.perf_counter()
        while last < total:
            rows = conn.execute("SELECT id,texto_busqueda FROM chunks WHERE id>? ORDER BY id LIMIT ?", (last, block_size)).fetchall()
            if not rows or rows[0]["id"] != last+1:
                raise ValueError("Orden de chunks roto")
            vec = encoder.encode([r["texto_busqueda"] for r in rows])
            if not np.isfinite(vec).all():
                raise ValueError("Embeddings no finitos")
            if index is None:
                index = faiss.IndexFlatIP(vec.shape[1])
            index.add(vec)
            last = index.ntotal
            if last % (block_size*20) < block_size or last == total:
                checkpoint_temp = target / "checkpoint.tmp"
                faiss.write_index(index, str(checkpoint_temp))
                checkpoint_temp.replace(target / "checkpoint.faiss")
                print(f"{name}: {last}/{total} vectores")
        faiss.write_index(index, str(target / "index.faiss"))
    encoder.close()
    meta = {"modelo": name, "revision": cat["revision"], "chunks": total,
            "dimensiones": index.d, "segundos_esta_sesion": round(time.perf_counter()-started, 1),
            "snapshot_id": json.loads((RUNS / "chunks_complete.json").read_text(encoding="utf-8"))["snapshot_id"]}
    dump(marker, meta)
    return meta


def dense_results(name, items=None, k=100):
    import faiss
    items = items or questions()
    marker = _index_dir(name) / "complete.json"
    meta = json.loads(marker.read_text(encoding="utf-8"))
    if meta["snapshot_id"] != json.loads((RUNS / "chunks_complete.json").read_text(encoding="utf-8"))["snapshot_id"]:
        raise ValueError("Índice denso de otro snapshot")
    idx = faiss.read_index(str(_index_dir(name) / "index.faiss"))
    enc = Encoder(name)
    vectors = enc.encode([query(x) for x in items], task="query")
    scores, ids = idx.search(vectors, k)
    enc.close()
    del idx
    gc.collect()
    return {str(item["id"]): [(int(i)+1, float(s)) for i, s in zip(ids[j], scores[j]) if i >= 0]
            for j, item in enumerate(items)}


def rrf(*rankings, k=100):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (chunk, _) in enumerate(ranking, 1):
            fused[chunk] += 1 / (60 + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]


def _rows(ids):
    if not ids:
        return []
    with closing(_db()) as conn:
        placeholders = ",".join("?" for _ in ids)
        rows = {r["id"]: dict(r) for r in conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids)}
    return [rows[i] for i in ids]


def rerank_bge(item, ranking, top=50):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    name = RERANKER
    card = catalog()[name]
    rows = _rows([x[0] for x in ranking[:top]])
    tok = AutoTokenizer.from_pretrained(name, revision=card["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        name, revision=card["revision"], dtype=getattr(torch, hardware()["dtype"])).to("cuda").eval()
    scores = []
    batch = 12 if hardware()["vram_gib"] >= 38 else 6
    i = 0
    while i < len(rows):
        try:
            pairs = [(query(item), r["texto_busqueda"]) for r in rows[i:i+batch]]
            inp = tok(pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                scores.extend(model(**inp).logits.float().view(-1).cpu().tolist())
            i += len(pairs)
        except torch.cuda.OutOfMemoryError:
            _release_gpu()
            if batch == 1:
                raise
            batch = max(1, batch//2)
    _release_gpu(model)
    return sorted([(r["id"], float(s)) for r, s in zip(rows, scores)], key=lambda x: (-x[1], x[0]))


def rerank_many(items, rankings, top=50):
    """Carga el cross-encoder una vez, no 50 veces."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    card = catalog()[RERANKER]
    tok = AutoTokenizer.from_pretrained(RERANKER, revision=card["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER, revision=card["revision"], dtype=getattr(torch, hardware()["dtype"])).to("cuda").eval()
    batch = 12 if hardware()["vram_gib"] >= 38 else 6
    out = {}
    for n, item in enumerate(items, 1):
        rows = _rows([x[0] for x in rankings[str(item["id"])][:top]])
        scores = []
        i = 0
        while i < len(rows):
            try:
                pairs = [(query(item), r["texto_busqueda"]) for r in rows[i:i+batch]]
                inp = tok(pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    scores.extend(model(**inp).logits.float().view(-1).cpu().tolist())
                i += len(pairs)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch = max(1, batch//2)
        out[str(item["id"])] = sorted([(r["id"], float(s)) for r, s in zip(rows, scores)],
                                      key=lambda x: (-x[1], x[0]))
        print(f"Reranking {n}/{len(items)}", end="\r")
    model = model.cpu()
    _release_gpu(model)
    return out


def rerank_qwen_many(items, rankings, top=50):
    """Probabilidad yes/no, siguiendo la plantilla publicada por Qwen."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    card = catalog()[QWEN_RERANKER]
    tok = AutoTokenizer.from_pretrained(QWEN_RERANKER, revision=card["revision"], padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_RERANKER, revision=card["revision"], dtype=getattr(torch, hardware()["dtype"])).to("cuda").eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    prefix = tok.encode('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False)
    suffix = tok.encode('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n', add_special_tokens=False)
    instruction = "Dada una pregunta jurídica colombiana, identifica pasajes que permitan responderla con fundamento literal."
    batch = 6 if hardware()["vram_gib"] >= 38 else 3
    out = {}
    for n, item in enumerate(items, 1):
        rows = _rows([x[0] for x in rankings[str(item["id"])][:top]])
        ids = []
        for row in rows:
            body = f"<Instruct>: {instruction}\n<Query>: {query(item)}\n<Document>: {row['texto_busqueda']}"
            ids.append(prefix + tok.encode(body, add_special_tokens=False)[:1024] + suffix)
        scores = []
        i = 0
        while i < len(ids):
            try:
                inp = tok.pad({"input_ids": ids[i:i+batch]}, padding=True, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    logits = model(**inp).logits[:, -1, [no, yes]].float()
                    scores.extend(torch.softmax(logits, dim=-1)[:, 1].cpu().tolist())
                i += min(batch, len(ids)-i)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch == 1:
                    raise
                batch = max(1, batch//2)
        out[str(item["id"])] = sorted([(r["id"], float(s)) for r, s in zip(rows, scores)],
                                      key=lambda x: (-x[1], x[0]))
        print(f"Qwen reranker {n}/{len(items)}", end="\r")
    model = model.cpu()
    _release_gpu(model)
    return out


def save_ranking(name, rankings):
    dump(RUNS / "rankings" / f"{name}.json", rankings)


def load_ranking(name):
    return json.loads((RUNS / "rankings" / f"{name}.json").read_text(encoding="utf-8"))


def evaluate_retrieval(name, rankings, ids=None):
    """Gold usado sólo después de recuperar; jamás en la consulta o el índice."""
    sys.path.insert(0, str(ROOT / "data/oficial/scripts"))
    import citations
    gold = {x["id"]: x for x in read_jsonl(SAMPLE)}
    ids = ids or [x["id"] for x in questions()]
    hits_doc = hits_citation = evaluable = 0
    rr = []
    lat = []
    for qid in ids:
        rank = rankings[str(qid)][:10]
        rows = _rows([x[0] for x in rank])
        refs = citations.bodies(citations.extract(gold[qid].get("legal_basis") or ""))
        if refs:
            evaluable += 1
            found = [bool(refs & citations.bodies(citations.extract(r["texto"]))) for r in rows]
            if any(found):
                hits_citation += 1
                rr.append(1/(found.index(True)+1))
        # La verdad documental no existe en sample_50; no se inventa recall_doc.
    result = {"experimento": name, "n": len(ids), "items_con_cita_extraible": evaluable,
              "respaldo_literal_top10": hits_citation / evaluable if evaluable else None,
              "MRR_cita_top10": sum(rr)/evaluable if evaluable else None,
              "recall_documento": None, "nota_recall": "Sin gold de doc_id oficial; sólo proxy de citas extraíbles"}
    dump(RUNS / "metricas_retrieval" / f"{name}.json", result)
    return result


def passages(ranking, max_passages=10, max_chars=2400):
    """Pasajes literales y offsets reconstruibles del snapshot."""
    sys.path.insert(0, str(ROOT / "data/oficial/scripts"))
    import citations
    selected, seen = [], set()
    for chunk_id, score in ranking:
        row = _rows([chunk_id])[0]
        key = row["unidad_id"]
        if key in seen:
            continue
        seen.add(key)
        a, b = row["inicio"], row["fin"]
        if row["unidad_fin"]-row["unidad_inicio"] <= max_chars:
            a, b = row["unidad_inicio"], row["unidad_fin"]
        # Texto exacto; sólo se lee el documento seleccionado.
        path = ROOT / "data/processed/corpus_preparado/textos" / (row["doc_id"] + ".txt")
        text = path.read_text(encoding="utf-8")
        selected.append({"doc_id": row["doc_id"], "inicio": a, "fin": b,
                         "texto": text[a:b], "score": float(score),
                         "titulo": row["titulo"], "articulo": row["articulo"],
                         "unidad_id": key, "avisos": json.loads(row["avisos"])})
        if len(selected) >= max_passages:
            break
    # La cabecera jurídica literal a menudo está separada del artículo. Sin ella,
    # el evaluador no puede reconocer en los pasajes el cuerpo de la norma citada.
    # Se reserva como máximo dos plazas, sin fabricar títulos dentro de texto.
    supplemental = []
    seen_docs = set()
    for p in selected[:4]:
        if p["doc_id"] in seen_docs:
            continue
        seen_docs.add(p["doc_id"])
        path = ROOT / "data/processed/corpus_preparado/textos" / (p["doc_id"] + ".txt")
        beginning = path.read_text(encoding="utf-8")[:800]
        if not citations.extract(beginning):
            continue
        if citations.bodies(citations.extract(beginning)) <= citations.bodies(citations.extract(p["texto"])):
            continue
        supplemental.append({"doc_id": p["doc_id"], "inicio": 0, "fin": len(beginning),
                             "texto": beginning, "score": p["score"], "titulo": p["titulo"],
                             "articulo": None, "unidad_id": p["doc_id"] + "__cabecera_literal",
                             "avisos": ["cabecera_literal_de_fuente"]})
        if len(supplemental) == 2:
            break
    if supplemental:
        selected = selected[:1] + supplemental + selected[1:max_passages-len(supplemental)]
    return selected


def _abstain(item):
    from legalrag.generation.cliente import abstenerse
    return {"id": item["id"], "formato": item["formato"], **abstenerse(item["formato"]),
            "pasajes_recuperados": []}


def _output_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise
        return json.loads(text[start:end+1])


def generate(name, ranking_name, items=None):
    import jsonschema
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer
    from legalrag.generation.cliente import mensajes, esquema_local
    from legalrag.evaluation.entrega import auditar_citas
    items = items or questions()
    card = catalog()[name]
    if card["parametros"] > 8_000_000_000:
        raise ValueError("Decoder prohibido por parámetros")
    target = RUNS / "generaciones" / (ranking_name + "__" + name.replace("/", "__"))
    target.mkdir(parents=True, exist_ok=True)
    submission_path = target / "submission.jsonl"
    detail_path = target / "detalles.jsonl"
    ranking_path = RUNS / "rankings" / f"{ranking_name}.json"
    run_identity = {"snapshot_id": json.loads((RUNS / "chunks_complete.json").read_text(encoding="utf-8"))["snapshot_id"],
                    "ranking_sha256": sha(ranking_path), "decoder": name, "revision": card["revision"],
                    "ids": [q["id"] for q in items]}
    identity_path = target / "identidad.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != run_identity:
            raise ValueError("La generación previa pertenece a otra evidencia, snapshot o modelo")
    else:
        dump(identity_path, run_identity)
    prior = {}
    if detail_path.exists():
        prior = {x["id"]: x for x in read_jsonl(detail_path)}
    if submission_path.exists() and set(prior) == {q["id"] for q in items}:
        return submission_path
    ranking = load_ranking(ranking_name)
    multimodal = name == "Qwen/Qwen3.5-4B"
    tok = (AutoProcessor if multimodal else AutoTokenizer).from_pretrained(name, revision=card["revision"])
    dtype = getattr(torch, hardware()["dtype"])
    cls = AutoModelForMultimodalLM if multimodal else AutoModelForCausalLM
    model = cls.from_pretrained(name, revision=card["revision"], dtype=dtype,
                                device_map="cuda", trust_remote_code=False).eval()
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    details = []
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest = json.loads((RELEASE / "corpus_manifest.json").read_text(encoding="utf-8"))
    for n, item in enumerate(items, 1):
        if item["id"] in prior:
            outputs.append(prior[item["id"]]["salida"])
            details.append(prior[item["id"]])
            continue
        start = time.perf_counter()
        retrieved = passages(ranking[str(item["id"])], max_passages=10)
        selected = []
        budget = hardware()["contexto"] - hardware()["max_nuevos_tokens"] - 64
        for p in retrieved:
            candidate = selected + [p]
            msgs = mensajes(item, candidate)
            if multimodal:
                msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                        for m in msgs]
            kwargs = {"tokenize": True, "add_generation_prompt": True}
            if "Qwen3" in name:
                kwargs["enable_thinking"] = False
            if multimodal:
                kwargs.update(return_dict=True, return_tensors="pt")
            prompt_ids = tok.apply_chat_template(msgs, **kwargs)
            length = prompt_ids["input_ids"].shape[-1] if multimodal else len(prompt_ids)
            if length <= budget:
                selected = candidate
        if not selected:
            answer = _abstain(item)
            problem = "sin_evidencia_en_contexto"
        else:
            msgs = mensajes(item, selected)
            if multimodal:
                msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                        for m in msgs]
            kw = {"tokenize": True, "add_generation_prompt": True}
            if "Qwen3" in name:
                kw["enable_thinking"] = False
            if multimodal:
                kw.update(return_dict=True, return_tensors="pt")
            tokens = tok.apply_chat_template(msgs, **kw)
            inputs = tokens.to("cuda") if multimodal else torch.tensor([tokens], device="cuda")
            try:
                with torch.inference_mode():
                    gen_kwargs = {"max_new_tokens": hardware()["max_nuevos_tokens"], "do_sample": False,
                                  "pad_token_id": tok.tokenizer.eos_token_id if multimodal else tok.eos_token_id}
                    generated = model.generate(**inputs, **gen_kwargs) if multimodal else model.generate(inputs, **gen_kwargs)
                length = inputs["input_ids"].shape[-1] if multimodal else inputs.shape[-1]
                raw = tok.decode(generated[0, length:], skip_special_tokens=True)
                result = _output_json(raw)
                if not isinstance(result, dict):
                    raise ValueError("Salida no es objeto JSON")
                jsonschema.validate(result, esquema_local(item["formato"]))
                answer = {"id": item["id"], "formato": item["formato"], **result,
                          "pasajes_recuperados": [{k: p[k] for k in ("doc_id", "inicio", "fin", "texto", "score")}
                                                  for p in selected]}
                audit = auditar_citas(answer, manifest)
                if audit["sin_respaldo"] or audit["indeterminadas"]:
                    answer = _abstain(item)
                    problem = "cita_sin_respaldo_o_indeterminada"
                else:
                    jsonschema.validate(answer, schema)
                    problem = None
            except (ValueError, json.JSONDecodeError, jsonschema.ValidationError) as exc:
                answer = _abstain(item)
                problem = f"json_invalido:{type(exc).__name__}"
        generation_ms = round((time.perf_counter()-start)*1000)
        detail = {"id": item["id"], "salida": answer, "problema": problem,
                  "pasajes_candidatos": len(retrieved), "pasajes_usados": len(selected),
                  "latencia_generacion_ms": generation_ms}
        outputs.append(answer)
        details.append(detail)
        # Checkpoint atómico de los 50 ítems. Sólo la muestra pública.
        (target / "submission.tmp").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in outputs), encoding="utf-8")
        (target / "submission.tmp").replace(submission_path)
        (target / "detalles.tmp").write_text("".join(json.dumps(x, ensure_ascii=False)+"\n" for x in details), encoding="utf-8")
        (target / "detalles.tmp").replace(detail_path)
        print(f"{name} {n}/{len(items)}: {problem or 'ok'}", end="\r")
    dump(target / "recursos.json", {"gpu": hardware()["nombre"],
         "vram_pico_gib": round(torch.cuda.max_memory_allocated()/2**30, 3),
         "modelo": name, "revision": card["revision"], "ranking": ranking_name})
    model = model.cpu()
    _release_gpu(model)
    return submission_path


def official_score(submission, ragas=False):
    out = Path(submission).with_name("score_oficial.json")
    cmd = [sys.executable, str(ROOT / "data/oficial/scripts/evaluate.py"),
           "--submission", str(submission), "--split", "sample", "--out", str(out)]
    if ragas:
        cmd.append("--ragas")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.read_text(encoding="utf-8"))


def compare():
    rows = []
    for p in sorted((RUNS / "metricas_retrieval").glob("*.json")):
        rows.append(json.loads(p.read_text(encoding="utf-8")))
    return rows
