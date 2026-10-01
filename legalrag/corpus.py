"""Fuentes del corpus: descarga, texto procesado, pasajes y manifiesto.

corpus/sources.json declara cada norma (URL pública, tipo, número, año,
órgano emisor, áreas). La vigencia queda en null mientras no se verifique
contra una fuente oficial: el sistema no la infiere.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

import requests

from legalrag.citations import fold, norma_key
from legalrag.extract import clean_text, extract_file
from legalrag.segment import segment_document

CORPUS = Path("corpus")
USER_AGENT = "Mozilla/5.0 (hackathon-2026 corpus builder; uso academico)"


def load_sources(path: Path = CORPUS / "sources.json") -> list[dict]:
    docs = json.loads(Path(path).read_text(encoding="utf-8"))
    for d in docs:
        d["norma_key"] = norma_key(d["norma"]) or fold(d["norma"])
    return docs


def aliases(docs: list[dict]) -> dict[str, str]:
    """Nombres con que se puede citar cada norma -> clave de la norma."""
    out = {}
    for d in docs:
        for name in [d["norma"], *d.get("alias", [])]:
            out[fold(name)] = d["norma_key"]
    return out


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(doc: dict, corpus: Path = CORPUS) -> dict:
    """Descarga la norma (o usa `archivo_local`), extrae y limpia el texto.
    Devuelve el registro del manifiesto."""
    raw_dir, out_dir = corpus / "raw", corpus / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if doc.get("archivo_local"):
        raw_path = Path(doc["archivo_local"])
        fecha = doc.get("fecha_consulta")
        if not fecha:
            raise ValueError(f"{doc['doc_id']}: con archivo_local hay que declarar fecha_consulta")
    else:
        r = requests.get(doc["url"], headers={"User-Agent": USER_AGENT}, timeout=90)
        r.raise_for_status()
        ext = ".pdf" if "pdf" in r.headers.get("content-type", "") or doc["url"].lower().endswith(".pdf") else ".html"
        raw_path = raw_dir / f"{doc['doc_id']}{ext}"
        raw_path.write_bytes(r.content)
        fecha = date.today().isoformat()
    texto = clean_text(extract_file(str(raw_path), doc.get("selector")))
    out_path = out_dir / f"{doc['doc_id']}.txt"
    out_path.write_text(texto, encoding="utf-8")
    articulos = {p["articulo"] for p in passages_for(doc, texto)}
    return {
        "doc_id": doc["doc_id"],
        "titulo": doc["titulo"],
        "fuente": doc["fuente"],
        "url": doc["url"],
        "fecha_consulta": fecha,
        "areas": doc["areas"],
        "tipo": doc["tipo"],
        "numero": doc["numero"],
        "anio": doc["anio"],
        "organo_emisor": doc["organo_emisor"],
        "vigencia": doc.get("vigencia"),
        "num_articulos": len(articulos),
        "sha256_original": sha256(raw_path.read_bytes()),
        "sha256_texto": sha256(texto.encode("utf-8")),
    }


def passages_for(doc: dict, texto: str) -> list[dict]:
    """Pasajes por artículo con los metadatos de la norma (Paso 1 del enunciado)."""
    pasajes = segment_document(texto, doc)
    for p in pasajes:
        p.update({
            "norma_key": doc["norma_key"],
            "tipo": doc["tipo"],
            "numero": doc["numero"],
            "anio": doc["anio"],
            "organo_emisor": doc["organo_emisor"],
            "vigencia": doc.get("vigencia"),
        })
    return pasajes


def all_passages(docs: list[dict], corpus: Path = CORPUS) -> list[dict]:
    pasajes = []
    for d in docs:
        path = corpus / "processed" / f"{d['doc_id']}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}: ejecute primero `python cli.py fetch`")
        pasajes += passages_for(d, path.read_text(encoding="utf-8"))
    return pasajes
