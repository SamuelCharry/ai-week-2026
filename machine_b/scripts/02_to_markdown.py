"""02 — Convierte PDF/HTML/RTF descargados a Markdown limpio.

Lee process/01_downloads.jsonl y procesa cada archivo:
  .pdf  -> pymupdf4llm
  .html -> trafilatura (extrae texto principal, descarta navegación)
  .rtf  -> striprtf (fallback simple)

Escribe process/02_markdown/<doc_id>.md con un front-matter YAML mínimo:
  ---
  doc_id: ...
  norma: ...
  numero: ...
  anio: ...
  tipo: ...
  fuente: ...
  url: ...
  sha256_raw: ...
  ---
  <contenido>

Edge cases:
- PDFs escaneados (sin texto): se marca `necesita_ocr` y se salta.
- HTML de SUIN/Senado trae mucho boilerplate: trafilatura con favor_precision=True.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import iter_jsonl, load_config, paths, setup_logging, write_atomic

log = setup_logging("to_markdown")


def _pdf_to_md(path: Path) -> str | None:
    try:
        import pymupdf4llm
    except ImportError:
        log.error("pip install pymupdf4llm")
        raise
    try:
        md = pymupdf4llm.to_markdown(str(path), show_progress=False)
    except Exception as exc:
        log.warning("pymupdf4llm falló en %s: %s", path.name, exc)
        return None
    if md and len(md.strip()) > 200:
        return md
    return None


def _html_to_md(data: bytes) -> str | None:
    try:
        import trafilatura
    except ImportError:
        log.error("pip install trafilatura")
        raise
    text = trafilatura.extract(data.decode("utf-8", errors="replace"),
                               favor_precision=True, include_tables=True,
                               output_format="markdown")
    if text and len(text.strip()) > 200:
        return text
    return None


def _rtf_to_md(data: bytes) -> str | None:
    try:
        from striprtf.striprtf import rtf_to_text
    except ImportError:
        return None
    text = rtf_to_text(data.decode("utf-8", errors="replace"))
    return text if text and len(text) > 200 else None


def _derive_metadata(entry: dict) -> dict:
    canon = entry.get("canonico") or []
    kind = canon[0] if canon else ""
    num = canon[1] if len(canon) > 1 else ""
    yr = canon[2] if len(canon) > 2 else ""
    return {
        "doc_id": entry.get("doc_id"),
        "norma": entry.get("norma", ""),
        "tipo": kind,
        "numero": num,
        "anio": yr,
        "fuente": entry.get("fuente"),
        "url": entry.get("source_url"),
        "sha256_raw": entry.get("sha256"),
    }


def main() -> int:
    cfg = load_config()
    p = paths(cfg)
    downloads = list(iter_jsonl(p["output"] / "01_downloads.jsonl"))
    out_dir = p["output"] / "02_markdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = failed = 0
    for row in downloads:
        if row.get("status") != "ok":
            continue
        path = p["output"] / row["path"]
        ext = path.suffix.lower()
        if ext == ".pdf":
            md = _pdf_to_md(path)
        elif ext in (".html", ".htm"):
            md = _html_to_md(path.read_bytes())
        elif ext == ".rtf":
            md = _rtf_to_md(path.read_bytes())
        else:
            md = None
        if md is None:
            failed += 1
            log.warning("sin texto útil: %s", path.name)
            continue
        meta = _derive_metadata(row)
        front = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items()) + "\n---\n\n"
        write_atomic(out_dir / f"{row['doc_id']}.md", (front + md).encode("utf-8"))
        processed += 1
    log.info("Markdown: %s convertidos, %s fallidos", processed, failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
