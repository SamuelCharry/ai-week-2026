"""Scraper de la relatoría de la Corte Constitucional de Colombia.

Patrón de URL del buscador:
  https://www.corteconstitucional.gov.co/relatoria/?q=Sentencia+C-355+de+2006
La página de resultados enlaza al HTML y al RTF/PDF de la sentencia. Capturamos
el HTML (más limpio para markdown) y guardamos el PDF como respaldo.

Edge cases:
- Numeración antigua con ceros (C-001 vs C-1): probamos ambos.
- Sentencias de Sala Civil, Laboral, Penal CSJ (SL, SC, SP): este scraper NO cubre
  CSJ. Para esas, usar `csj_scraper.py` (TODO) o fallback a Google Scholar manual.
- El sitio a veces responde 403 con User-Agent agresivo: pide identificable.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from bs4 import BeautifulSoup


BASE = "https://www.corteconstitucional.gov.co/relatoria/"


def _candidates(canon: list) -> list[str]:
    """Devuelve URLs candidatas para una sentencia según su canónico."""
    kind = (canon[0] if canon else "").lower()
    num = str(canon[1]) if len(canon) > 1 and canon[1] else ""
    yr = str(canon[2]) if len(canon) > 2 and canon[2] else ""
    if kind != "jurisprudencia" or not num or not yr:
        return []
    # El canonico suele venir normalizado "C-355"; variantes "C-355" y "C-0355".
    letter_num = num.replace(" ", "")
    base_queries = {letter_num}
    m = re.fullmatch(r"([A-Za-z]+)-?(\d+)", letter_num)
    if m:
        letter, digits = m.group(1), m.group(2)
        base_queries.add(f"{letter}-{digits.zfill(3)}")
        base_queries.add(f"{letter}-{int(digits)}")
    urls = []
    for q in base_queries:
        urls.append(f"{BASE}?q=Sentencia+{q}+de+{yr}")
    return urls


def fetch(entry: dict, http) -> list[dict]:
    """Devuelve lista de documentos descargados: {doc_id, bytes, content_type, url}."""
    urls = _candidates(entry["canonico"])
    docs: list[dict] = []
    for search_url in urls:
        r = http.get(search_url, cache_key=f"cc_search_{_safe(search_url)}")
        if not r:
            continue
        soup = BeautifulSoup(r.content, "lxml")
        # Buscar enlaces a /relatoria/YYYY/ARCHIVO.htm o .rtf o .pdf
        hits = soup.select("a[href]")
        wanted = []
        canon_num = str(entry["canonico"][1]).lower()
        canon_yr = str(entry["canonico"][2])
        for a in hits:
            href = a.get("href", "")
            txt = (a.get_text() or "").lower()
            if canon_num.split("-")[-1] in txt and canon_yr in txt:
                wanted.append(href)
            elif re.search(rf"/{canon_yr}/[A-Za-z]+[-_]?0*{re.escape(canon_num.split('-')[-1])}", href, re.I):
                wanted.append(href)
        for href in wanted[:2]:
            full = href if href.startswith("http") else f"https://www.corteconstitucional.gov.co{href}"
            page = http.get(full, cache_key=f"cc_doc_{_safe(full)}")
            if not page:
                continue
            doc_id = _derive_id(entry["canonico"])
            docs.append({
                "doc_id": doc_id,
                "url": full,
                "content_type": page.headers.get("Content-Type", "text/html"),
                "bytes": page.content,
            })
            break  # un hit por entrada basta
        if docs:
            break
    return docs


def _safe(url: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", url)[:200]


def _derive_id(canon: list) -> str:
    kind, num, yr = canon[0], canon[1], canon[2]
    letter_num = str(num).lower().replace("-", "_")
    return f"sentencia_cc_{letter_num}_{yr}"
