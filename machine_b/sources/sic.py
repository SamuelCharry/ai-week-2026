"""Scraper mínimo de la SIC (Superintendencia de Industria y Comercio).

La SIC publica circulares y resoluciones en https://www.sic.gov.co/. La búsqueda
estructurada es pobre, así que esta implementación hace fallback a sic.gov.co/sites
/publicaciones y busca por número de resolución/circular en el texto plano del HTML.
"""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

BASE = "https://www.sic.gov.co/"


def fetch(entry: dict, http) -> list[dict]:
    canon = entry["canonico"]
    kind, num, yr = (canon + [None, None, None])[:3]
    kind = (kind or "").lower()
    if not (num and yr):
        return []
    # Búsqueda por el buscador interno (DuckDuckGo-style)
    url = f"{BASE}buscar?q={kind}+{num}+{yr}"
    r = http.get(url, cache_key=f"sic_{num}_{yr}")
    if not r:
        return []
    soup = BeautifulSoup(r.content, "lxml")
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if any(x in href.lower() for x in (".pdf", ".doc", "resolucion", "circular")):
            full = href if href.startswith("http") else BASE + href.lstrip("/")
            page = http.get(full, cache_key=f"sic_doc_{_safe(full)}")
            if page:
                return [{
                    "doc_id": f"{kind}_{num}_{yr}",
                    "url": full,
                    "content_type": page.headers.get("Content-Type", "text/html"),
                    "bytes": page.content,
                }]
    return []


def _safe(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", url)[:200]
