"""Catálogos oficiales de decisiones societarias y de consumo financiero."""

import hashlib
import re
from urllib.parse import urljoin, urlsplit
from bs4 import BeautifulSoup


def sociedades_candidates(config):
    from legalrag.ingestion.acquire import CLIENT
    base = "https://www.supersociedades.gov.co/web/procedimientos-mercantiles/jurisprudencia-mercantiles"
    page = CLIENT.request(base)
    categories = sorted(set(re.findall(r'findByCategory\([\s\x27\"]*(\d+)', page.text)))
    if not categories:
        categories = ["2703092"]
    seen = set()
    for category in categories:
        start, total = 0, 1
        while start < total:
            response = CLIENT.request(base + f"?id={category}&start={start}&end={start+20}")
            match = re.search(r"totalArticulos\s*=\s*'(\d+)'", response.text)
            total = int(match[1]) if match else 20
            soup = BeautifulSoup(response.content, "html.parser")
            for anchor in soup.select('a[href*="/documents/"]'):
                title = anchor.get_text(" ", strip=True)
                if not re.match(r"(?:Auto|Sentencia|Sentecia)\b", title, re.I):
                    continue
                url = urljoin(base, anchor["href"])
                key = urlsplit(url).path
                if key in seen or ".pdf" not in key.lower():
                    continue
                seen.add(key)
                year = re.findall(r"\b(?:19|20)\d{2}\b", title)
                yield {"doc_id": "supersoc_" + hashlib.sha256(key.encode()).hexdigest()[:16],
                       "titulo": title, "tipo": "auto" if title.lower().startswith("auto") else "sentencia",
                       "numero": title, "anio": int(year[-1]) if year else None,
                       "url": url, "formato_fuente": "pdf", "fuente": "Superintendencia de Sociedades",
                       "origen_ampliacion": "jurisprudencia_societaria_2026", "areas": ["comercial", "procesal"],
                       "categoria_catalogo": category}
            start += 20
        print(f"[discovery] Supersociedades categoría {category} — {len(seen)} decisiones distintas", flush=True)


def financiera_candidates(config):
    from legalrag.ingestion.acquire import CLIENT
    base = "https://www.superfinanciera.gov.co/publicaciones/10082560/consumidor-financierofunciones-jurisdiccionales-fallos-10082560/"
    soup = BeautifulSoup(CLIENT.request(base).content, "html.parser")
    months = []
    for anchor in soup.select('a[href]'):
        title = anchor.get_text(" ", strip=True)
        match = re.fullmatch(r"\w+\s+(20\d{2})", title)
        url = urljoin(base, anchor["href"])
        if match and int(match[1]) >= 2020 and urlsplit(url).hostname == "www.superfinanciera.gov.co" and "fallos" in url:
            months.append((url, int(match[1])))
    seen = set()
    for number, (url, year) in enumerate(dict.fromkeys(months), 1):
        page = BeautifulSoup(CLIENT.request(url).content, "html.parser")
        for anchor in page.select('a[href*="descargar"]'):
            reference = anchor.get_text(" ", strip=True)
            if not re.fullmatch(r"20\d{2}-\d{3,8}", reference):
                continue
            target = urljoin(url, anchor["href"])
            if target in seen:
                continue
            seen.add(target)
            # El mismo expediente puede tener varias providencias, por eso se conserva idFile.
            fileid = re.search(r"idFile=(\d+)", target)
            if not fileid:
                continue
            yield {"doc_id": f"sentencia_sfc_{reference.replace('-', '_')}_{fileid[1]}",
                   "titulo": f"Fallo SFC expediente {reference}", "tipo": "sentencia", "numero": reference,
                   "anio": year, "url": target, "pagina_catalogo": url, "formato_fuente": "pdf",
                   "fuente": "Superintendencia Financiera de Colombia",
                   "origen_ampliacion": "fallos_consumidor_financiero_2026", "areas": ["mercados", "comercial"],
                   "fecha_metodo": "Año del catálogo de publicación, expediente conservado por separado"}
        print(f"[discovery] SFC {number}/{len(months)} meses — {len(seen)} fallos", flush=True)
def penal_candidates(config):
    import re
    from pathlib import Path
    from urllib.parse import urljoin, urlsplit, unquote
    import pymupdf
    from bs4 import BeautifulSoup
    from legalrag.ingestion.acquire import CLIENT, OFFICIAL_HOSTS
    from legalrag.io import write_json, sha256
    catalog = "https://cortesuprema.gov.co/sala-de-casacion-penal-relatoria-boletines/"
    soup = BeautifulSoup(CLIENT.request(catalog).content, "html.parser")
    bulletins = sorted({urljoin(catalog, a["href"]) for a in soup.select('a[href]')
                       if ".pdf" in a["href"].lower() and re.search(r"\b202[0-6]\b", a.get_text(" ", strip=True))})
    seen = set()
    archive = config.root / "data/raw/ampliacion/catalogos_penal"
    archive.mkdir(parents=True, exist_ok=True)
    failures = []
    for i, url in enumerate(bulletins, 1):
        try:
            response = CLIENT.request(url)
            if not response.content.startswith(b"%PDF"):
                raise ValueError("El boletín no devolvió PDF")
            path = archive / f"boletin_{i:03d}.pdf"
            path.write_bytes(response.content)
            with pymupdf.open(stream=response.content, filetype="pdf") as document:
                links = sorted({x["uri"] for page in document for x in page.get_links() if x.get("uri")})
            for link in links:
                match = re.search(r"\b(SP|AP)(\d+)-(\d{4})", unquote(Path(urlsplit(link).path).name), re.I)
                if not match or urlsplit(link).hostname not in OFFICIAL_HOSTS:
                    continue
                kind, number, year = match.groups()
                kind = kind.upper()
                identity = f"{'sentencia' if kind == 'SP' else 'auto'}_csj_{kind.lower()}_{int(number)}_{year}"
                if identity in seen:
                    continue
                seen.add(identity)
                yield {"doc_id": identity, "titulo": f"{'Sentencia' if kind == 'SP' else 'Auto'} {kind}{number}-{year}",
                       "tipo": "sentencia" if kind == "SP" else "auto", "numero": kind + number,
                       "anio": int(year), "url": link, "formato_fuente": "pdf",
                       "fuente": "Corte Suprema de Justicia — Relatoría Penal", "pagina_catalogo": catalog,
                       "boletin_origen": url, "boletin_sha256": sha256(path),
                       "origen_ampliacion": "providencias_boletines_penal_2026", "areas": ["penal", "procesal"],
                       "identificador_providencia": f"{kind}{number}-{year}"}
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({"url": url, "error": str(exc)})
        print(f"[discovery] penal {i}/{len(bulletins)} boletines — {len(seen)} providencias", flush=True)
    write_json(config.reports / "descubrimiento_penal.json", {"boletines": len(bulletins), "providencias": len(seen), "fallos": failures})

