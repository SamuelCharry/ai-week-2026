"""Scraper de la Secretaría del Senado (basedoc). Buena para códigos y leyes antiguas."""
from __future__ import annotations

import re

BASE = "http://www.secretariasenado.gov.co/senado/basedoc/"


def fetch(entry: dict, http) -> list[dict]:
    canon = entry["canonico"]
    kind, num, yr = (canon + [None, None, None])[:3]
    kind = (kind or "").lower()
    # Convención del basedoc: "ley_1098_2006.html", "constitucion_politica_1991.html"
    candidates = []
    if "constitucion" in kind:
        candidates.append(BASE + "constitucion_politica_1991.html")
    elif kind in ("ley", "decreto") and num and yr:
        candidates.append(BASE + f"{kind}_{num}_{yr}.html")
        candidates.append(BASE + f"{kind}_{num}_{yr}_pr001.html")  # primer bloque
    for url in candidates:
        r = http.get(url, cache_key=f"sen_{_safe(url)}")
        if r:
            return [{
                "doc_id": f"{kind}_{num}_{yr}" if num else "constitucion_1991",
                "url": url,
                "content_type": r.headers.get("Content-Type", "text/html"),
                "bytes": r.content,
            }]
    return []


def _safe(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", url)[:200]
