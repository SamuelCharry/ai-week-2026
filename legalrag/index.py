"""Embeddings con un encoder abierto e índice FAISS exacto.

Reutilizado y adaptado de CODEFEST Ad Astra 2026 (src/indexing/encoder.py,
build_index.py y metadata_store.py; autor en el historial git: Juanesillo).
Cambios: se indexa el texto con encabezado de norma y artículo, y se guarda
un index_info.json con el encoder y el número de pasajes para poder
verificar que el índice no cambió después de congelarlo.
"""
import json
from pathlib import Path

import numpy as np

ENCODER = "BAAI/bge-m3"
_MODEL = None


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_model(name: str = ENCODER):
    """Carga el encoder una sola vez; usa GPU si hay, si no CPU."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        device = _pick_device()
        print(f"Cargando encoder {name} en {device}...")
        _MODEL = SentenceTransformer(name, device=device)
    return _MODEL


def encode_texts(textos: list[str], batch_size: int = 16) -> np.ndarray:
    """Embeddings normalizados (coseno = producto interno, para IndexFlatIP)."""
    emb = get_model().encode(textos, batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=len(textos) > batch_size, convert_to_numpy=True)
    return emb.astype("float32")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(pasajes: list[dict], out_dir: str) -> None:
    """Guarda pasajes.jsonl (fila i = vector i de FAISS), index.faiss e index_info.json."""
    import faiss
    from legalrag.segment import indexed_text

    out = Path(out_dir)
    emb = encode_texts([indexed_text(p) for p in pasajes])
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    write_jsonl(out / "pasajes.jsonl", pasajes)
    info = {"encoder": ENCODER, "pasajes": len(pasajes), "dim": int(emb.shape[1]),
            "documentos": sorted({p["doc_id"] for p in pasajes})}
    (out / "index_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Índice FAISS con {index.ntotal} vectores en {out}")
