"""05 — Embebe los chunks enriquecidos con BGE-M3 y arma FAISS + SQLite.

Entrada: process/04_chunks_enriched.jsonl
Salida:
  process/05_index.faiss               (IndexFlatIP, bge-m3 dim)
  process/05_metadata.sqlite           (compatible con Machine A)
  process/manifest_v2.json             (sha256 por chunk + revisiones)

Si cfg['index']['merge_with'] apunta a index/ actual, los chunks nuevos se
APPEND (sin duplicar). Esto evita reindexar los 17k documentos existentes.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import iter_jsonl, load_config, paths, setup_logging, sha256_bytes, sha256_file, write_json

log = setup_logging("embed_index")


def _create_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE chunks (
          id INTEGER PRIMARY KEY, doc_id TEXT, norma TEXT, articulo TEXT,
          parent_id TEXT, inicio INTEGER, fin INTEGER, parent_inicio INTEGER, parent_fin INTEGER,
          encabezado TEXT, texto TEXT, metadata TEXT
        );
        CREATE INDEX by_reference ON chunks(norma, articulo);
        CREATE INDEX by_parent ON chunks(parent_id);
        CREATE VIRTUAL TABLE lexical USING fts5(texto, content='chunks', content_rowid='id',
                                               tokenize='unicode61 remove_diacritics 2');
    """)
    return db


def _load_encoder(cfg: dict):
    import torch
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(cfg["embedding"]["model"], revision=cfg["embedding"]["revision"],
                                device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = cfg["embedding"]["max_tokens"]
    return model


def main() -> int:
    cfg = load_config()
    p = paths(cfg)
    import faiss
    import numpy as np

    enc = _load_encoder(cfg)
    dim = enc.get_sentence_embedding_dimension()
    log.info("Encoder listo (dim=%s)", dim)

    index = faiss.IndexFlatIP(dim)
    db_path = p["output"] / "05_metadata.sqlite.tmp"
    db = _create_db(db_path)

    chunks = list(iter_jsonl(p["output"] / "04_chunks_enriched.jsonl"))
    log.info("A indexar: %s chunks", len(chunks))

    bs = cfg["embedding"]["batch_size"]
    norm = cfg["embedding"]["normalize"]
    t0 = time.perf_counter()
    doc_hashes: dict[str, str] = {}
    for start in range(0, len(chunks), bs):
        batch = chunks[start:start + bs]
        texts = [c["texto_enriquecido"] for c in batch]
        vecs = enc.encode(texts, normalize_embeddings=norm, show_progress_bar=False,
                          convert_to_numpy=True).astype("float32")
        index.add(vecs)
        base_id = index.ntotal - len(batch)
        for rowid, c in enumerate(batch, base_id):
            db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                rowid, c["doc_id"], c.get("norma", ""), c.get("numero_articulo"),
                c.get("parent_id", c.get("doc_id")), c.get("inicio", 0), c.get("fin", 0),
                c.get("parent_inicio", 0), c.get("parent_fin", 0),
                c["encabezado"], c["texto"],
                json.dumps({k: v for k, v in c.items() if k != "texto"}, ensure_ascii=False)))
            db.execute("INSERT INTO lexical(rowid, texto) VALUES (?,?)",
                       (rowid, c["encabezado"] + "\n" + c["texto"]))
            doc_hashes.setdefault(c["doc_id"], "")
        if (start // bs) % 50 == 0:
            elapsed = time.perf_counter() - t0
            log.info("%d / %d (%.0f chunks/s)", index.ntotal, len(chunks),
                     index.ntotal / max(elapsed, 1))
    db.commit()
    db.close()

    faiss_path = p["output"] / "05_index.faiss.tmp"
    faiss.write_index(index, str(faiss_path))
    final_faiss = p["output"] / "05_index.faiss"
    final_db = p["output"] / "05_metadata.sqlite"
    if final_faiss.exists():
        final_faiss.unlink()
    if final_db.exists():
        final_db.unlink()
    faiss_path.rename(final_faiss)
    db_path.rename(final_db)

    manifest = {
        "version": cfg["manifest"]["version"],
        "team": cfg["manifest"]["team"],
        "license": cfg["manifest"]["license"],
        "chunks": index.ntotal,
        "dim": dim,
        "encoder": cfg["embedding"]["model"],
        "encoder_revision": cfg["embedding"]["revision"],
        "enricher": cfg["enrichment"]["model"] if cfg["enrichment"]["enabled"] else None,
        "enricher_revision": cfg["enrichment"]["revision"] if cfg["enrichment"]["enabled"] else None,
        "sha256_faiss": sha256_file(final_faiss),
        "sha256_metadata": sha256_file(final_db),
        "documentos": len(doc_hashes),
    }
    write_json(p["output"] / "manifest_v2.json", manifest)
    log.info("Listo. %d chunks, %d docs. sha256_faiss=%s",
             manifest["chunks"], manifest["documentos"], manifest["sha256_faiss"][:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
