"""Scraper del Normograma DIAN (normas tributarias)."""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

BASE = "https://normograma.dian.gov.co/dian/"


def fetch(entry: dict, http) -> list[dict]:
    canon = entry["canonico"]
    kind, num, yr = (canon + [None, None, None])[:3]
    kind = (kind or "").lower()
    if not (num and yr):
        return []
    url = BASE + f"compilacion/compilacion_user_public.jsp?busqueda={kind}%20{num}%20{yr}"
    r = http.get(url, cache_key=f"dian_search_{_safe(url)}")
    if not r:
        return []
    soup = BeautifulSoup(r.content, "lxml")
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if re.search(rf"{num}.*{yr}", href):
            full = href if href.startswith("http") else f"https://normograma.dian.gov.co{href}"
            page = http.get(full, cache_key=f"dian_doc_{_safe(full)}")
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
