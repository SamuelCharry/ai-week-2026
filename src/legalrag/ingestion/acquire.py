"""Descarga reanudable de fuentes oficiales para ampliar el corpus."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import threading
import time
import zipfile
from xml.etree import ElementTree
from urllib.parse import urljoin, urlsplit, urldefrag, urlencode

import requests
import truststore
from bs4 import BeautifulSoup, NavigableString, Comment
from pypdf.errors import PyPdfError

from legalrag.config import CONFIG
from legalrag.citations.extract import fold, norm_identity, norm_mentions
from legalrag.io import corpus_records, records, now, dump_line, write_json
from legalrag.preprocessing.pdf import extract_pdf


AREA_TERMS = {
    "constitucional": ("derecho fundamental", "accion de tutela", "constitucionalidad", "debido proceso"),
    "administrativo": ("acto administrativo", "contratacion estatal", "funcion publica", "nulidad y restablecimiento", "responsabilidad del estado"),
    "penal": ("codigo penal", "proceso penal", "privacion de la libertad", "presuncion de inocencia", "fiscalia"),
    "procesal": ("codigo general del proceso", "recurso de casacion", "carga de la prueba", "competencia judicial", "cosa juzgada"),
    "civil": ("responsabilidad civil", "contrato de compraventa", "prescripcion adquisitiva", "codigo civil", "posesion", "sucesion"),
    "familia": ("patria potestad", "filiacion", "custodia", "divorcio", "union marital", "sociedad conyugal", "alimentos", "adopcion"),
    "comercial": ("sociedad comercial", "sociedad anonima", "codigo de comercio", "insolvencia", "accionista", "contrato de seguro"),
    "tributario": ("estatuto tributario", "impuesto sobre la renta", "obligacion tributaria", "retencion en la fuente", "sancion tributaria"),
    "laboral": ("contrato de trabajo", "pension", "despido", "codigo sustantivo del trabajo", "seguridad social", "sindical"),
    "mercados": ("proteccion al consumidor", "proteccion de datos", "competencia desleal", "propiedad industrial", "derechos de autor", "habeas data"),
}
DIAN_BASE = "https://normograma.dian.gov.co/dian/compilacion/"
CSJ_API = "https://consultaprovidenciasbk.cortesuprema.gov.co/api"
OFFICIAL_HOSTS = {"www.corteconstitucional.gov.co", "normograma.dian.gov.co",
                  "consultaprovidenciasbk.cortesuprema.gov.co", "www.secretariasenado.gov.co",
                  "www.funcionpublica.gov.co", "www.suin-juriscol.gov.co",
                  "normativa.colpensiones.gov.co", "www.cancilleria.gov.co",
                  "www.supersociedades.gov.co", "supersociedades.gov.co", "www.superfinanciera.gov.co",
                  "sidn.ramajudicial.gov.co", "cortesuprema.gov.co", "www.cortesuprema.gov.co",
                  "archivodigitalapi.cortesuprema.gov.co", "www.minambiente.gov.co",
                  "normograma.superservicios.gov.co", "cancilleria.gov.co"}


@dataclass
class Payload:
    content: bytes
    url: str
    headers: dict
    status_code: int

    @property
    def text(self):
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.content)


class Client:
    def __init__(self):
        truststore.inject_into_ssl()
        self.local = threading.local()
        self.lock = threading.Lock()
        self.last = {}

    def request(self, url, payload=None):
        host = urlsplit(url).hostname
        if host not in OFFICIAL_HOSTS:
            raise ValueError(f"Fuente no autorizada en este catálogo: {host}")
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.headers.update({"User-Agent": "LegalCorpusResearch/1.0 (academic corpus audit)",
                                               "Accept": "text/html,application/json"})
        for attempt in range(3):
            with self.lock:
                wait = max(0, self.last.get(host, 0) + 0.6 - time.monotonic())
                if wait:
                    time.sleep(wait)
                self.last[host] = time.monotonic()
            try:
                with self.local.session.request("POST" if payload else "GET", url, json=payload,
                                                headers={"Accept": "application/json"} if payload else None,
                                                timeout=(8, 35), stream=True) as response:
                    if response.status_code in (429, 500, 502, 503, 504):
                        time.sleep(min(30, 3 * 2 ** attempt))
                        continue
                    response.raise_for_status()
                    if urlsplit(response.url).hostname not in OFFICIAL_HOSTS:
                        raise ValueError(f"Redirección fuera de la fuente: {response.url}")
                    mime = response.headers.get("Content-Type", "").lower()
                    disposition = response.headers.get("Content-Disposition", "").lower()
                    if mime.startswith(("audio/", "video/")) or re.search(r'\.(?:mp3|mp4|wav|m4a)(?:[\";]|$)', disposition):
                        raise ValueError("Audio o video excluido del corpus textual")
                    maximum = 30 * 1024 * 1024
                    if int(response.headers.get("Content-Length") or 0) > maximum:
                        raise ValueError("Documento supera 30 MB")
                    content = bytearray()
                    for block in response.iter_content(64 * 1024):
                        content.extend(block)
                        if len(content) > maximum:
                            raise ValueError("Documento supera 30 MB")
                    return Payload(bytes(content), response.url, response.headers.copy(), response.status_code)
            except requests.exceptions.SSLError:
                raise
            except requests.HTTPError:
                # Un 404 confirmado no mejora repitiendo la misma URL tres veces.
                raise
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Fuente temporalmente no disponible: {url}")


CLIENT = Client()


def areas_from_text(text, hints=()):
    folded = fold(text)
    counts = {area: sum(folded.count(term) for term in terms) for area, terms in AREA_TERMS.items()}
    selected = set(hints)
    selected.update(a for a, count in counts.items() if count >= 3)
    return sorted(selected), counts


def html_text(raw):
    try:
        decoded = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        decoded = raw
    soup = BeautifulSoup(decoded, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    for tag in soup.select("script,style,nav,header,footer,form,noscript"):
        tag.decompose()
    container = soup.select_one("#aj_data, #documento, #contenido_documento, .documento, .descripcion-contenido, .panel-documento") or soup.body or soup
    # Un párrafo por línea conserva los encabezados de artículos de Word/HTML.
    for br in container.find_all("br"):
        br.replace_with("\n")
    lines, inline = [], []

    def flush():
        if inline:
            value = re.sub(r"\s+", " ", " ".join(inline)).strip()
            if value:
                lines.append(value)
            inline.clear()

    def visit(node):
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            if node.strip():
                inline.append(str(node))
            return
        block = node.name in {"p", "h1", "h2", "h3", "h4", "li"} or (
            node.name == "tr" and not node.find(["p", "li"]))
        if block:
            flush()
            value = re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()
            if value:
                lines.append(value)
        else:
            for child in node.children:
                visit(child)

    visit(container)
    flush()
    text = "\n\n".join(lines)
    return title, text


def cc_candidates(config, minimum=4):
    identities = {norm_identity(d) for d in corpus_records(config)}
    path = config.reports / "brechas_corpus.json"
    if path.exists():
        missing = json.loads(path.read_text(encoding="utf-8"))["normas_ausentes"]
    else:
        raise FileNotFoundError("Ejecutar audit antes de descubrir sentencias citadas")
    for item in missing:
        match = re.fullmatch(r"sentencia_(c|t|su)_(\d+)_(\d{4})", item["norma"])
        if not match or item["norma"] in identities or item["documentos_citantes"] < minimum:
            continue
        kind, number, year = match.groups()
        if not 1992 <= int(year) <= 2026:
            continue
        filename = f"{kind}-{int(number):03d}-{year[-2:]}.htm"
        yield {"doc_id": item["norma"], "tipo": "sentencia", "numero": f"{kind.upper()}-{number}",
               "anio": int(year), "titulo": f"Sentencia {kind.upper()}-{number} de {year}",
               "url": f"https://www.corteconstitucional.gov.co/relatoria/{year}/{filename}",
               "fuente": "Corte Constitucional", "origen_ampliacion": "grafo_normativo_2026",
               "prioridad": item["documentos_citantes"], "areas": [], "norma_objetivo": item["norma"]}


def norm_candidates(config, minimum=4):
    missing = json.loads((config.reports / "brechas_corpus.json").read_text(encoding="utf-8"))["normas_ausentes"]
    identities = {norm_identity(d) for d in corpus_records(config)}
    wanted = {d["norma"] for d in missing if d["documentos_citantes"] >= minimum}
    area_votes = {}
    for edge in records(config.reports / "grafo_normativo.jsonl"):
        if edge["destino"] in wanted:
            area_votes.setdefault(edge["destino"], Counter()).update(edge.get("areas", []))
    for item in missing:
        match = re.fullmatch(r"(ley|decreto|acto_legislativo)_(\d+)_(\d{4})", item["norma"])
        if not match or item["norma"] in identities or item["norma"] not in wanted:
            continue
        kind, number, year = match.groups()
        if not 1850 <= int(year) <= 2026:
            continue
        filename = f"{kind}_{int(number):04d}_{year}.htm"
        yield {"doc_id": item["norma"], "tipo": kind, "numero": number, "anio": int(year),
               "titulo": f"{kind.replace('_', ' ').title()} {number} de {year}",
               "url": DIAN_BASE + "docs/" + filename,
               "urls_alternativas": [f"http://www.secretariasenado.gov.co/senado/basedoc/{filename[:-4]}.html",
                                     f"https://normativa.colpensiones.gov.co/colpens/docs/{filename}"],
               "buscar_fp": True, "fuente": "Normativa oficial colombiana",
               "origen_ampliacion": "normas_ausentes_grafo_2026", "prioridad": item["documentos_citantes"],
               "areas": [], "areas_citantes": dict(area_votes.get(item["norma"], {})),
               "nivel": "complementario" if int(year) < 1991 else "nucleo", "norma_objetivo": item["norma"]}


def official_html(candidate):
    failures = []
    urls = [candidate["url"], *candidate.get("urls_alternativas", [])]
    urls = [url.replace("https://www.secretariasenado.gov.co", "http://www.secretariasenado.gov.co")
            .replace(".htm", ".html") if "secretariasenado.gov.co" in url and url.endswith(".htm") else url for url in urls]
    urls = [url.replace("/colpens/docs/", "/compilacion/docs/") if "normativa.colpensiones.gov.co" in url else url for url in urls]
    if candidate.get("buscar_fp"):
        filename = f"{candidate['tipo']}_{int(candidate['numero']):04d}_{candidate['anio']}.htm"
        extra = f"https://normativa.colpensiones.gov.co/compilacion/docs/{filename}"
        if extra not in urls:
            urls.append(extra)
    if candidate.get("buscar_fp"):
        query = urlencode({"t": "ejecuta_busqueda_avanzada2", "nrodoc": candidate["numero"],
                           "ano": candidate["anio"], "pagina": 1})
        base = "https://www.funcionpublica.gov.co/eva/gestornormativo/"
        try:
            page = CLIENT.request(base + "gestion/funphp/funajax.php?" + query)
            soup = BeautifulSoup(page.content, "html.parser")
            urls = [urljoin(base, a["href"]) for a in soup.select('a[href*="norma.php"]')
                    if candidate["norma_objetivo"] in set(norm_mentions(a.get_text(" ", strip=True)))] + urls
        except requests.RequestException as exc:
            failures.append(f"Consulta de Función Pública: {exc}")
    for url in urls:
        try:
            response = CLIENT.request(url)
            if "html" not in response.headers.get("Content-Type", "").lower():
                raise ValueError("La fuente no devolvió HTML")
            title, text = html_text(response.content)
            if any(term in fold(title) for term in ("not found", "access denied", "just a moment", "buscador")):
                raise ValueError("Página de error o búsqueda")
            normative = candidate["tipo"] in {"ley", "decreto", "acto_legislativo", "resolucion"}
            if len(text) < (300 if normative else 1500):
                raise ValueError("Texto insuficiente")
            expected = candidate.get("norma_objetivo")
            if expected and expected not in set(norm_mentions(text[:15000])):
                raise ValueError("Encabezado sin identidad esperada")
            candidate["url"] = response.url
            candidate["fuente"] = {"www.funcionpublica.gov.co": "Función Pública - Gestor Normativo",
                                    "www.secretariasenado.gov.co": "Secretaría del Senado - Normograma",
                                    "normograma.dian.gov.co": "DIAN - Normograma",
                                    "normativa.colpensiones.gov.co": "Colpensiones - Normograma",
                                    "normograma.superservicios.gov.co": "Superservicios - Normograma"}.get(
                                        urlsplit(response.url).hostname, candidate["fuente"])
            return response, text
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            failures.append(f"{url}: {exc}")
    raise ValueError(" | ".join(failures))


def dian_candidates(config):
    existing = {norm_identity(d) for d in corpus_records(config)}
    seen = set()
    for tree in ("t_1_normativa_tributaria", "t_2_doctrina_tributaria", "t_3_jurisprudencia_tributaria"):
        page = CLIENT.request(DIAN_BASE + tree + ".html")
        soup = BeautifulSoup(page.content, "html.parser")
        count = len(soup.select(".opcion-nueva"))
        for part in range(1, count + 1):
            url = DIAN_BASE + f"{tree}_parte_{part:02d}.html"
            response = CLIENT.request(url)
            section = BeautifulSoup(response.content, "html.parser")
            for anchor in section.select('a[href*="docs/"]'):
                link = urldefrag(urljoin(DIAN_BASE, anchor["href"]))[0]
                if link in seen:
                    continue
                seen.add(link)
                title = anchor.get_text(" ", strip=True)
                filename = Path(urlsplit(link).path).stem
                match = re.fullmatch(r"(ley|decreto)_(\d+)_(\d{4})", filename)
                if match:
                    kind, number, year = match.groups()
                    identity = f"{kind}_{int(number)}_{year}"
                    if identity in existing:
                        continue
                else:
                    year_match = re.search(r"_(\d{4})$", filename)
                    year = year_match[1] if year_match else "0"
                    if int(year) < 2018:
                        continue
                    kind = "concepto" if "doctrina" in tree else "sentencia" if "jurisprudencia" in tree else "resolucion"
                    number = filename
                    identity = "dian_" + filename
                yield {"doc_id": identity, "titulo": title or filename, "tipo": kind, "numero": number,
                       "anio": int(year), "url": link, "fuente": "DIAN - Normograma",
                       "origen_ampliacion": "catalogo_tributario_2026", "areas": ["tributario"],
                       "nivel": "complementario" if kind == "concepto" else "nucleo",
                       "norma_objetivo": identity if match else None}
            print(f"[discovery] DIAN {tree} {part}/{count} — {len(seen)} enlaces revisados", flush=True)


def colpens_candidates(config):
    folder = config.root / "data/raw/ampliacion"
    failed = {d["doc_id"] for d in records(folder / "incidencias_normas.jsonl")}
    for original in records(folder / "candidatos_normas.jsonl"):
        if original["doc_id"] not in failed:
            continue
        doc = dict(original)
        filename = f"{doc['tipo']}_{int(doc['numero']):04d}_{doc['anio']}.htm"
        doc.update(url=f"https://normativa.colpensiones.gov.co/compilacion/docs/{filename}",
                   urls_alternativas=[], buscar_fp=False, fuente="Colpensiones — Normograma",
                   origen_ampliacion="rescate_normas_colpensiones_2026")
        yield doc


def csj_candidates(config, per_query=60):
    topics = {
        "Civil": {"civil": ["responsabilidad civil", "prescripción adquisitiva", "contrato", "sucesión"],
                  "familia": ["filiación", "unión marital", "sociedad conyugal", "alimentos", "custodia"],
                  "comercial": ["sociedad", "insolvencia", "contrato de seguro", "título valor"],
                  "mercados": ["consumidor", "competencia desleal", "datos personales", "propiedad intelectual"],
                  "procesal": ["carga de la prueba", "recurso de revisión", "nulidad procesal"]},
        "Laboral": {"laboral": ["pensión", "despido", "contrato de trabajo", "sindicato", "estabilidad laboral", "riesgos laborales"]},
        "Penal": {"penal": ["presunción de inocencia", "prueba ilícita", "tipicidad", "preacuerdo"],
                  "procesal": ["cadena de custodia", "doble conformidad"]},
    }
    seen = set()
    for room, mapping in topics.items():
        for area, queries in mapping.items():
            for term in queries:
                for start in range(0, per_query, 10):
                    fields = {"query": term, "typeOfQuery": room, "start": start, "isExact": True,
                              "magistrate": "", "year": "", "autoSentencia": "SENTENCIA", "order": "NEW_FIRST",
                              "roomTutelas": "", "addedQueries": []}
                    args = " ".join(k + ":" + json.dumps(v, ensure_ascii=False) for k, v in fields.items())
                    query = "{getSearchResult(searchQuery:{" + args + "}){searchResults{title id onlinePath ano autoSentencia} numOfResults}}"
                    result = CLIENT.request(CSJ_API, {"query": query}).json()
                    if result.get("errors"):
                        raise ValueError(str(result["errors"]))
                    result = result["data"]["getSearchResult"]
                    for row in result["searchResults"]:
                        match = re.search(r"\b(SC|SL|SP|STC|STL|STP)\s*(\d+)[- ](\d{4})", row["title"], re.I)
                        if not match:
                            continue
                        kind, number, year = match.groups()
                        identity = f"sentencia_csj_{kind.lower()}_{int(number)}_{year}"
                        if identity in seen:
                            continue
                        seen.add(identity)
                        yield {"doc_id": identity, "titulo": f"Sentencia {kind.upper()}{number}-{year}",
                               "tipo": "sentencia", "numero": kind.upper() + number, "anio": int(year),
                               "url": "https://consultaprovidencias.cortesuprema.gov.co/", "api_url": CSJ_API,
                               "source_id": row["id"], "sala": room, "fuente": "Corte Suprema de Justicia",
                               "origen_ampliacion": "consulta_oficial_csj_2026", "areas": [area], "consulta": term}
                    if start + 10 >= result["numOfResults"]:
                        break
                print(f"[discovery] CSJ {room} — {term} — {len(seen)} sentencias distintas", flush=True)


def download(candidate, root):
    candidate = dict(candidate)
    try:
        if candidate.get("api_url"):
            source_id = candidate["source_id"]
            suffix = Path(source_id).suffix.lower()
            if suffix == ".doc":
                source_id = source_id[:-4] + ".pdf"
                suffix = ".pdf"
            payload = {"path": source_id}
            response = CLIENT.request(candidate["api_url"].replace("/api", "/downloadFile"), payload)
            if suffix == ".docx" and response.content.startswith(b"PK"):
                with zipfile.ZipFile(BytesIO(response.content)) as archive:
                    xml = ElementTree.fromstring(archive.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                text = "\n\n".join("".join(p.itertext()) for p in xml.findall(".//w:p", ns))
            elif response.content.startswith(b"%PDF"):
                text = extract_pdf(response.content)
                suffix = ".pdf"
            else:
                raise ValueError("La descarga no contiene un PDF o DOCX válido")
            candidate["solicitud_api"] = payload
        elif candidate.get("formato_fuente") == "pdf":
            response = CLIENT.request(candidate["url"])
            if not response.content.startswith(b"%PDF"):
                raise ValueError("La fuente no devolvió PDF")
            text = extract_pdf(response.content)
            suffix = ".pdf"
        else:
            response, text = official_html(candidate)
            suffix = ".html"
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        normative = candidate["tipo"] in {"ley", "decreto", "acto_legislativo", "resolucion"}
        if len(text) < (300 if normative else 1500) or text.count("\ufffd") / max(1, len(text)) > 0.002:
            raise ValueError("Texto insuficiente o codificación defectuosa")
        if normative and not re.search(r"\bart[ií]culo\s+(?:\d+|[uú]nico|primero)", text, re.I):
            raise ValueError("Norma sin articulado reconocible")
        expected = candidate.get("norma_objetivo")
        if expected and expected not in set(norm_mentions(text[:15000])):
            raise ValueError("El encabezado no identifica la norma esperada")
        if candidate["tipo"] == "sentencia" and not re.search(r"resuelve|decisi[oó]n|fall[ao]|decide", text, re.I):
            raise ValueError("Providencia sin sección de decisión reconocible")
        if candidate.get("identificador_providencia"):
            compact = re.sub(r"\s+", "", text[:15000]).upper()
            if candidate["identificador_providencia"].upper() not in compact:
                raise ValueError("La providencia descargada no coincide con su identificador")
        hints = list(candidate.get("areas", []))
        if "corteconstitucional.gov.co" in candidate["url"]:
            hints.append("constitucional")
        areas, area_scores = areas_from_text(text, hints)
        if not areas and candidate.get("areas_citantes"):
            areas = [max(candidate["areas_citantes"], key=candidate["areas_citantes"].get)]
        if not areas:
            raise ValueError("No se reconoce un área del banco")
        relative = Path("data/raw/ampliacion") / candidate["origen_ampliacion"] / candidate["doc_id"]
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        raw_path = directory / ("original" + suffix)
        raw_path.write_bytes(response.content)
        encoded = text.encode("utf-8")
        (directory / "texto.txt").write_bytes(encoded)
        candidate.update(areas=areas, areas_metodo="heuristica_de_terminos_y_catalogo", areas_puntajes=area_scores,
                         fecha_descarga=now(), fecha_consulta=now()[:10], url_final=response.url,
                         archivo_raw=(relative / raw_path.name).as_posix(), sha256_raw=hashlib.sha256(response.content).hexdigest(),
                         bytes_raw=len(response.content), texto_archivo=(relative / "texto.txt").as_posix(),
                         sha256_texto=hashlib.sha256(encoded).hexdigest(), caracteres=len(text),
                         http_status=response.status_code, content_type=response.headers.get("Content-Type"),
                         nivel=candidate.get("nivel", "nucleo"), apta_para_busqueda=True,
                         vigencia="por_verificar", revision_juridica="pendiente",
                         avisos=["vigencia_por_verificar", "extraccion_automatica"] + (["edicion_con_anotaciones"] if "DIAN" in candidate["fuente"] or "Normograma" in candidate["fuente"] else []),
                         edicion_con_anotaciones="DIAN" in candidate["fuente"] or "Normograma" in candidate["fuente"],
                         redistribuir_raw=not ("DIAN" in candidate["fuente"] or "Normograma" in candidate["fuente"]),
                         licencia_fuente="Texto oficial. Se conserva la licencia de origen y las excepciones de data/raw/LICENSE. Notas editoriales de terceros no se relicencian.")
        write_json(directory / "procedencia.json", candidate)
        return candidate, None
    except (requests.RequestException, ValueError, KeyError, TypeError, RuntimeError, zipfile.BadZipFile,
            ElementTree.ParseError, PyPdfError) as exc:
        return None, {"doc_id": candidate["doc_id"], "url": candidate["url"], "error": str(exc), "fecha": now()}


def acquire(config, source, per_query=60, minimum=4, workers=3, retry_failures=False):
    folder = config.root / "data/raw/ampliacion"
    folder.mkdir(parents=True, exist_ok=True)
    result_path = folder / f"resultado_{source}.json"
    result_path.unlink(missing_ok=True)
    write_json(folder / f"estado_{source}.json", {"estado": "en_curso", "pid": os.getpid(), "inicio": now()})
    inventory = folder / f"documentos_{source}.jsonl"
    existing = list(corpus_records(config))
    done = {d["doc_id"] for d in existing}
    identities = {norm_identity(d) for d in existing}
    hashes = {d.get("sha256_texto") for d in existing}
    if inventory.exists():
        for doc in records(inventory):
            done.add(doc["doc_id"])
            hashes.add(doc["sha256_texto"])
    discovery_path = folder / f"candidatos_{source}.jsonl"
    if not discovery_path.exists():
        from legalrag.ingestion.institutions import sociedades_candidates, financiera_candidates, penal_candidates
        candidates = {"cc": lambda: cc_candidates(config, minimum), "normas": lambda: norm_candidates(config, minimum), "dian": lambda: dian_candidates(config),
                      "csj": lambda: csj_candidates(config, per_query),
                      "sociedades": lambda: sociedades_candidates(config),
                      "financiera": lambda: financiera_candidates(config), "penal": lambda: penal_candidates(config),
                      "colpens": lambda: colpens_candidates(config)}[source]()
        from legalrag.io import atomic_text
        with atomic_text(discovery_path) as stream:
            for candidate in candidates:
                if candidate["doc_id"] not in done and candidate.get("norma_objetivo") not in identities:
                    dump_line(stream, candidate)
    errors_path = folder / f"incidencias_{source}.jsonl"
    failed = {d["doc_id"] for d in records(errors_path)} if errors_path.exists() and not retry_failures else set()
    candidates = [c for c in records(discovery_path) if c["doc_id"] not in done and c["doc_id"] not in failed]
    print(f"[ingestion] {source} — {len(candidates)} fuentes pendientes", flush=True)
    stats = Counter()
    with inventory.open("a", encoding="utf-8") as output, errors_path.open("a", encoding="utf-8") as errors:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map limita los trabajadores activos. Cada respuesta se libera al guardarse.
            for i, (doc, error) in enumerate(pool.map(lambda c: download(c, config.root), candidates), 1):
                if error:
                    dump_line(errors, error)
                    errors.flush()
                    stats["incidencias"] += 1
                elif doc["sha256_texto"] in hashes or doc["doc_id"] in done:
                    stats["duplicados"] += 1
                else:
                    dump_line(output, doc)
                    output.flush()
                    hashes.add(doc["sha256_texto"])
                    done.add(doc["doc_id"])
                    stats["incorporados"] += 1
                if i % 10 == 0 or i == len(candidates):
                    print(f"[ingestion] {source} {i}/{len(candidates)} — {dict(stats)}", flush=True)
    write_json(result_path, {"fecha": now(), "candidatos": len(candidates),
                             "fallos_previos_no_reintentados": len(failed), **stats})
    write_json(folder / f"estado_{source}.json", {"estado": "terminado", "fin": now(), "pid": os.getpid()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=["cc", "dian", "csj", "normas", "sociedades", "financiera", "penal", "colpens"])
    parser.add_argument("--per-query", type=int, default=60)
    parser.add_argument("--minimum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    acquire(CONFIG, args.source, args.per_query, args.minimum, args.workers, args.retry_failures)
