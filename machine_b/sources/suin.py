"""Scraper de SUIN-Juriscol (sistema oficial de información normativa)."""
from __future__ import annotations

import re

from bs4 import BeautifulSoup

BASE = "https://www.suin-juriscol.gov.co/legislacion/"


def fetch(entry: dict, http) -> list[dict]:
    canon = entry["canonico"]
    kind, num, yr = (canon + [None, None, None])[:3]
    kind = (kind or "").lower()
    if not (num and yr):
        return []
    # El buscador de SUIN acepta "Decreto 1563 de 2012" como ?q=
    q_variants = [f"{kind.capitalize()} {num} de {yr}", f"{kind.capitalize()} {num}/{yr}"]
    for q in q_variants:
        url = BASE + "?q=" + q.replace(" ", "%20")
        r = http.get(url, cache_key=f"suin_search_{_safe(url)}")
        if not r:
            continue
        soup = BeautifulSoup(r.content, "lxml")
        for a in soup.select("a[href]"):
            href = a.get("href", "")
            txt = (a.get_text() or "").strip()
            if re.search(rf"{kind}.*{num}.*{yr}", txt, re.I) or re.search(rf"{num}.*{yr}", href):
                full = href if href.startswith("http") else f"https://www.suin-juriscol.gov.co{href}"
                page = http.get(full, cache_key=f"suin_doc_{_safe(full)}")
                if page and page.content:
                    return [{
                        "doc_id": f"{kind}_{num}_{yr}",
                        "url": full,
                        "content_type": page.headers.get("Content-Type", "text/html"),
                        "bytes": page.content,
                    }]
    return []


def _safe(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", url)[:200]
