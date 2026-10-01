"""01 — Descarga las normas listadas en process/00_gaps.json.

Lee process/00_gaps.json y, para cada norma faltante, llama al scraper correspondiente.
Guarda el bytestream en process/01_downloaded/<fuente>/<doc_id>.<ext>. Produce
process/01_downloads.jsonl con una línea por doc intentado (éxito o fallo).

Edge cases:
- Fuente desconocida -> se marca `skipped`.
- Scraper devuelve 0 docs -> `not_found`.
- Resolución 368 Minambiente (q748) está fuera de las fuentes del enunciado: scrapers
  dedicados no aplican; se registra como pendiente de descarga manual.
- Idempotencia: si el doc_id ya existe en disco con sha256 válido, se omite.
"""
from __future__ import annotations

import json
import mimetypes
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
from _common import load_config, paths, setup_logging, write_jsonl, sha256_bytes
from sources import REGISTRY
from sources._http import PoliteHttp

log = setup_logging("download")


def _extension(content_type: str) -> str:
    if not content_type:
        return ".bin"
    ct = content_type.split(";")[0].strip().lower()
    guess = mimetypes.guess_extension(ct) or ""
    if ct in {"text/html", "application/xhtml+xml"}:
        return ".html"
    if ct == "application/pdf":
        return ".pdf"
    if ct in {"application/rtf", "text/rtf"}:
        return ".rtf"
    return guess or ".bin"


def main() -> int:
    cfg = load_config()
    p = paths(cfg)
    gaps_path = p["output"] / "00_gaps.json"
    if not gaps_path.is_file():
        log.error("Falta %s. Corre primero scripts/00_verify_gaps.py", gaps_path)
        return 2
    gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
    missing = gaps["missing"]

    http = PoliteHttp(
        user_agent=cfg["download"]["user_agent"],
        rate_limit=cfg["download"]["rate_limit_seconds"],
        retries=cfg["download"]["retries"],
        backoff=cfg["download"]["retry_backoff"],
        timeout=cfg["download"]["timeout"],
        cache_dir=p["cache"] / "http",
    )

    down_dir = p["output"] / "01_downloaded"
    down_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    ok = fail = skipped = 0
    for entry in missing:
        fuente = entry["fuente"]
        mod = REGISTRY.get(fuente)
        if mod is None:
            rows.append({**entry, "status": "skipped", "reason": f"no scraper for {fuente}"})
            skipped += 1
            continue
        try:
            docs = mod.fetch(entry, http)
        except Exception as exc:
            log.exception("scraper %s falló en %s: %s", fuente, entry["canonico"], exc)
            rows.append({**entry, "status": "error", "reason": str(exc)})
            fail += 1
            continue
        if not docs:
            rows.append({**entry, "status": "not_found"})
            fail += 1
            continue
        for d in docs:
            ext = _extension(d.get("content_type", ""))
            target = down_dir / fuente / f"{d['doc_id']}{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(d["bytes"])
            rows.append({**entry, "status": "ok", "doc_id": d["doc_id"],
                         "path": str(target.relative_to(p["output"])),
                         "sha256": sha256_bytes(d["bytes"]),
                         "content_type": d.get("content_type", ""),
                         "source_url": d.get("url", "")})
            ok += 1
            log.info("[%s] %s -> %s", fuente, entry["canonico"], target.name)

    write_jsonl(p["output"] / "01_downloads.jsonl", rows)
    log.info("Descarga terminada. ok=%s not_found/error=%s skipped=%s", ok, fail, skipped)
    log.info("Resultados en %s/01_downloaded/", p["output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
