"""Ronda 5 del corpus: barrido exhaustivo de las fuentes primarias colombianas.

Las rondas anteriores eligen documentos por señal (seed, muestra, grafo normativo). El
banco tiene ~1.000 preguntas que no conocemos, así que esta ronda deja de muestrear y
toma el universo completo de cada fuente primaria, leído de su propio índice oficial:

    cc_datos_abiertos  todas las sentencias C y SU de la Corte Constitucional (1992-2026),
                       del conjunto oficial "Sentencias proferidas por la Corte
                       Constitucional" (datos.gov.co, v2k4-2t8s). Fuera: control de leyes
                       aprobatorias de tratados (derecho internacional, excluido del banco).
                       Las T no entran en bloque (21.000 casos concretos): siguen llegando
                       por el grafo cuando el corpus las cita.
    senado_arbol       todas las leyes desde 1992, los decretos con fuerza de ley desde 1991,
                       los decretos que modifican leyes, los decretos por año, los actos
                       legislativos y las sentencias de Sala Plena de la Corte Suprema
                       (control constitucional antes de 1991), del árbol de la base
                       documental del Senado. Fuera: la rama de decretos relacionados con
                       actos internacionales.
    sic_circular_unica los títulos vigentes de la Circular Única de la SIC (consumidor,
                       datos personales, competencia, propiedad industrial).
    supersociedades    los conceptos jurídicos (oficios 220) y los libros de jurisprudencia
                       societaria de la Superintendencia de Sociedades, por la API pública de
                       su repositorio, y su Circular Básica Jurídica.
    superfinanciera    la Circular Básica Jurídica de la Superintendencia Financiera
                       (C.E. 029 de 2014), publicada como ZIP de capítulos en Word.
    gestor             Gestor Normativo de Función Pública por tipo de documento: leyes
                       (incluye las anteriores a 1992), decretos de nivel nacional, actos
                       legislativos, sentencias del Consejo de Estado y de la Corte Suprema,
                       conceptos de la Sala de Consulta y Servicio Civil y conceptos de
                       Función Pública. Senado sigue siendo la fuente preferida de leyes y
                       decretos (legalrag.ingestion.reconstruir); el Gestor queda de respaldo.

Escribe configs/corpus_ronda5.json, que legalrag.ingestion.objetivos incorpora al inventario.
Las leyes aprobatorias de tratados que no se pueden reconocer por el índice se reconocen
por su epígrafe al descargarlas (legalrag.ingestion.auditoria las marca).

Uso: python -m legalrag.ingestion.fuentes_ronda5 [--arbol data/cache/arbol_senado.json] [--refrescar]
"""
import argparse
import html as html_lib
import json
import re
import sys
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from legalrag.ingestion.fuentes_ronda4 import A  # noqa: E402
from legalrag.ingestion.reconstruir import pedir  # noqa: E402

CC_DATOS = "https://www.datos.gov.co/resource/v2k4-2t8s.json?$limit=50000&$order=fecha_sentencia"
CC_FUERA = {"Ley Aprobatoria de Tratado", "Tratado Internacional"}
CC_AREAS = {"Proyecto de Ley Estatutaria": ["const"], "Decreto Legislativo": ["const", "adm"],
            "Objeción Gubernamental": ["const"], "Objeción de Inconstitucionalidad": ["const"]}

SENADO_ARBOL = "http://www.secretariasenado.gov.co/senado/basedoc/arbol/"
SENADO = "http://www.secretariasenado.gov.co/senado/basedoc/"
# Ramas del árbol que se recorren (las demás repiten documentos o no son normas).
SENADO_RAMAS = {"LEYES : POR ORDEN CRONOLÓGICO", "LEYES : POR TIPO", "OTROS DECRETOS QUE MODIFICAN LEYES DESDE 1992",
                "ACTOS LEGISLATIVOS", "CÓDIGOS Y ESTATUTOS NACIONALES", "DECRETOS", "LEYES", "CORTE SUPREMA DE JUSTICIA"}
SENADO_FUERA = re.compile(r"(?i)ACTOS INTERNACIONALES")

SIC = "https://sedeelectronica.sic.gov.co/transparencia/normativa/"
# Página de la versión vigente de cada título (el buscador de la sede las lista de la más nueva
# a la más vieja). Los títulos III y IV no publican archivo.
SIC_TITULOS = {
    "I": ("titulo-i-10", "Disposiciones generales", ["merc"]),
    "II": ("titulo-ii-actualizado-el-9-de-enero-de-2026", "Protección al consumidor", ["merc"]),
    "V": ("titulo-v-actualizado-el-9-de-enero-de-2026", "Protección de datos personales / reglamentos", ["merc"]),
    "VI": ("titulo-vi-13", "Metrología legal y reglamentos técnicos", ["merc"]),
    "VII": ("titulo-vii-actualizado-por-la-resolucion-56579-del-12-de-agosto-de-2025",
            "Protección de la competencia", ["merc"]),
    "X": ("titulo-x-actualizado-el-9-de-diciembre-de-2025", "Propiedad industrial", ["merc"]),
}


SUPERSOCIEDADES = "https://www.supersociedades.gov.co"
SS_API = SUPERSOCIEDADES + "/api/jsonws/dlapp/get-file-entries{}/repository-id/{}/folder-id/{}{}"
# (grupo, carpeta, origen): conceptos jurídicos (oficios 220) y libros de jurisprudencia societaria.
SS_CARPETAS = [(107391, 159040, "supersociedades_concepto"), (20122, 2703311, "supersociedades_jurisprudencia")]
SS_CBJ = ("https://www.supersociedades.gov.co/documents/107391/897146/Circular+Basica+Jur%C3%ADdica.pdf/"
          "e78cdcc8-4d0c-40a1-177c-de4d365da1ee")
SFC_CBJ = ("https://www.superfinanciera.gov.co/loader.php?lServicio=Tools2&lTipo=descargas&lFuncion=descargar"
           "&idFile=1010065")
INSOLVENCIA = re.compile(r"(?i)insolven|reorganizaci|liquidaci|concordato|acreedor")


GESTOR = ("https://www.funcionpublica.gov.co/dafpIndexerBGN/norma/index?find=FindNext&filtroTipoDocumento={}"
          "&max=100&offset={}")
# Tipo del Gestor -> (prefijo del doc_id, tipo, origen). Las sentencias de la Corte Constitucional
# del Gestor vienen sin la sala ("Sentencia 402 de 2013") y ya están todas las C/SU: fuera.
GESTOR_TIPOS = {"Ley": "ley", "Decreto": "decreto", "Decreto Ley": "decreto", "Acto Legislativo": "acto_legislativo",
                "Sentencia": "sentencia", "Concepto Sala de Consulta C.E.": "concepto_sala_consulta",
                "Concepto": "concepto_fp"}
GESTOR_TITULO = re.compile(r"^(.+?)\s+([\w.-]+)\s+de\s+(\d{4})\s*-\s*(.+)$")


def html(url):
    estado, _, _, contenido = pedir(url)
    if estado != 200:
        raise RuntimeError(f"{url} respondió HTTP {estado}")
    return BeautifulSoup(contenido, "html.parser")


def corte_constitucional():
    estado, _, _, contenido = pedir(CC_DATOS)
    if estado != 200:
        raise RuntimeError(f"datos.gov.co respondió HTTP {estado}")
    filas = json.loads(contenido)
    documentos, fuera = {}, 0
    for f in filas:
        m = re.fullmatch(r"(C|SU)\s*-?\s*(\d+)([A-Z]?)\s*/\s*(\d{2})", f.get("sentencia", "").strip().upper())
        if not m:
            continue
        if f.get("proceso") in CC_FUERA:
            fuera += 1
            continue
        sala, numero, letra, yy = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        # El año de la sentencia sale del sufijo; el siglo, de la fecha.
        anio = int(f["fecha_sentencia"][:2] + yy) if f.get("fecha_sentencia") else (1900 if yy > "90" else 2000) + int(yy)
        radical = f"{sala.lower()}{'' if sala == 'SU' else '-'}{numero:03d}{letra.lower()}-{yy}"
        doc_id = f"sentencia_cc_{sala.lower()}{numero:03d}{letra.lower()}_{anio}"
        documentos[doc_id] = {
            "doc_id": doc_id, "titulo": f"Sentencia {sala}-{numero:03d}{letra} de {anio}",
            "tipo": "sentencia", "numero": f"{sala}-{numero:03d}{letra}", "anio": anio,
            "organo_emisor": "Corte Constitucional", "fuente": "Corte Constitucional - Relatoría",
            "url": f"https://www.corteconstitucional.gov.co/relatoria/{anio}/{radical}.htm",
            "areas": [A[a] for a in CC_AREAS.get(f.get("proceso"), ["const"])],
            "origen": "cc_datos_abiertos", "proceso": f.get("proceso"),
            "magistrado": f.get("magistrado_a"), "fecha": (f.get("fecha_sentencia") or "")[:10],
            "min_caracteres": 3000}
    print(f"  Corte Constitucional: {len(documentos)} C/SU; {fuera} de tratados fuera", flush=True)
    return list(documentos.values())


def recorrer_arbol():
    """Recorre el árbol del Senado por niveles y devuelve {archivo: {titulo, rutas}}."""
    def leer(item):
        pagina, ruta = item
        for _ in range(3):
            try:
                estado, _, _, contenido = pedir(SENADO_ARBOL + pagina)
                if estado == 200:
                    return pagina, ruta, contenido
            except Exception:  # noqa: BLE001 - reintento de red
                pass
        return pagina, ruta, None

    vistos, documentos, nivel = {"1000.html"}, {}, [("1000.html", [])]
    while nivel:
        siguiente = []
        with ThreadPoolExecutor(3) as ex:
            for pagina, ruta, contenido in ex.map(leer, nivel):
                if contenido is None:
                    print(f"  árbol: no respondió {pagina}", flush=True)
                    continue
                for a in BeautifulSoup(contenido, "html.parser").find_all("a", href=True):
                    destino = a["href"].split("#")[0]
                    texto = re.sub(r"\s+", " ", a.get_text(" ", strip=True))
                    if re.fullmatch(r"\d+\.html", destino):
                        if destino not in vistos and texto != "Inicio" and (ruta or texto in SENADO_RAMAS):
                            vistos.add(destino)
                            siguiente.append((destino, ruta + [texto]))
                    elif destino.startswith("../") and destino.endswith(".html") and "_pr" not in destino:
                        archivo = destino.rsplit("/", 1)[-1][:-5]
                        documentos.setdefault(archivo, {"titulo": texto, "rutas": []})["rutas"].append(" > ".join(ruta))
        print(f"  árbol: {len(vistos)} páginas, {len(documentos)} documentos", flush=True)
        nivel = siguiente
    return documentos


def senado(arbol):
    documentos = []
    for archivo, info in sorted(arbol.items()):
        if all(SENADO_FUERA.search(r) for r in info["rutas"]):
            continue
        sala_plena = re.fullmatch(r"csj_sp_.*_(\d{4})", archivo)
        if sala_plena:
            # Control de constitucionalidad de la Corte Suprema (Sala Plena) antes de 1991.
            documentos.append({
                "doc_id": f"sentencia_csj_sala_plena_{archivo[7:]}", "titulo": info["titulo"], "tipo": "sentencia",
                "numero": None, "anio": int(sala_plena.group(1)), "organo_emisor": "Corte Suprema de Justicia",
                "fuente": "Secretaría General del Senado - Base documental", "url": f"{SENADO}{archivo}.html",
                "areas": [A["const"]], "origen": "senado_arbol", "ramas_senado": sorted(set(info["rutas"]))[:3],
                "min_caracteres": 1500})
            continue
        m = re.fullmatch(r"(ley|decreto|acto_legislativo)_(\d+)_(\d{4})", archivo)
        if not m:
            continue
        tipo, numero, anio = m.group(1), int(m.group(2)), int(m.group(3))
        doc_id = f"{tipo}_{numero}_{anio}"
        nombre = {"ley": "Ley", "decreto": "Decreto", "acto_legislativo": "Acto Legislativo"}[tipo]
        # El índice trae a veces el epígrafe ("CÓDIGO GENERAL DEL PROCESO - Ley 1564 de 2012 - ...").
        titulo = info["titulo"] if len(info["titulo"]) > 25 else f"{nombre} {numero} de {anio}"
        documentos.append({
            "doc_id": doc_id, "titulo": titulo, "tipo": tipo, "numero": str(numero), "anio": anio,
            "organo_emisor": "Congreso de la República" if tipo != "decreto" else "Presidencia de la República",
            "fuente": "Secretaría General del Senado - Base documental", "url": f"{SENADO}{archivo}.html",
            # Hay decretos completos de un solo artículo (~1.100 caracteres): la identidad ya se
            # valida con el número y el año, así que el mínimo de longitud es bajo.
            "areas": [], "origen": "senado_arbol", "ramas_senado": sorted(set(info["rutas"]))[:3],
            "min_caracteres": 500})
    print(f"  Senado: {len(documentos)} normas", flush=True)
    return documentos


def sic():
    documentos = []
    for romano, (pagina, tema, areas) in SIC_TITULOS.items():
        pdfs = [urllib.parse.urljoin(SIC, a["href"]) for a in html(SIC + pagina).find_all("a", href=True)
                if re.search(r"(?i)/normativa/[^\"]+\.pdf$", a["href"]) and "rminos" not in a["href"]]
        if not pdfs:
            print(f"  SIC: el título {romano} no publica archivo", flush=True)
            continue
        documentos.append({
            "doc_id": f"sic_circular_unica_titulo_{romano.lower()}",
            "titulo": f"Circular Única de la Superintendencia de Industria y Comercio - Título {romano}: {tema}",
            "tipo": "circular", "numero": f"Título {romano}", "anio": None,
            "organo_emisor": "Superintendencia de Industria y Comercio",
            "fuente": "Superintendencia de Industria y Comercio - Sede electrónica", "url": pdfs[0],
            "areas": [A[a] for a in areas], "origen": "sic_circular_unica", "min_caracteres": 1500})
    print(f"  SIC: {len(documentos)} títulos de la Circular Única", flush=True)
    return documentos


def supersociedades():
    """Doctrina y jurisprudencia de la Superintendencia de Sociedades, por su API pública (Liferay).

    La descripción que acompaña a cada oficio en el repositorio no siempre corresponde a su
    contenido; por eso el título es el número del oficio y la descripción va aparte.
    """
    documentos = []
    for grupo, carpeta, origen in SS_CARPETAS:
        estado, _, _, total = pedir(SS_API.format("-count", grupo, carpeta, ""))
        total = int(total)
        for inicio in range(0, total, 500):
            estado, _, _, contenido = pedir(SS_API.format("", grupo, carpeta, f"/start/{inicio}/end/{inicio + 500}"))
            if estado != 200:
                raise RuntimeError(f"Supersociedades respondió HTTP {estado} en {carpeta}/{inicio}")
            print(f"  Supersociedades {carpeta}: {inicio + 500}/{total}", flush=True)
            for e in json.loads(contenido):
                if e.get("extension", "").lower() not in ("pdf", "doc", "docx"):
                    continue
                nombre = re.sub(r"(?i)\.(pdf|docx?)$", "", e["title"]).strip()
                oficio = re.match(r"(?i)(?:oficio\s+)?(\d{3}-\d{4,7})", nombre)
                anio = re.search(r"\b(19|20)\d{2}\b", nombre)
                concepto = origen == "supersociedades_concepto"
                titulo = (f"Superintendencia de Sociedades, Oficio {oficio.group(1)}" if concepto and oficio else
                          f"Superintendencia de Sociedades - {nombre}")
                resumen = (e.get("description") or "").strip()
                documentos.append({
                    "doc_id": f"{'doctrina' if concepto else 'jurisprudencia'}_supersociedades_{e['fileEntryId']}",
                    "titulo": titulo, "tipo": "concepto" if concepto else "compendio",
                    "numero": oficio.group(1) if oficio else None, "anio": int(anio.group()) if anio else None,
                    "organo_emisor": "Superintendencia de Sociedades",
                    "fuente": "Superintendencia de Sociedades - Repositorio documental",
                    "url": f"{SUPERSOCIEDADES}/documents/{e['groupId']}/{e['folderId']}/"
                           f"{urllib.parse.quote(e['fileName'])}/{e['uuid']}",
                    "areas": [A["com"]] + ([A["proc"]] if INSOLVENCIA.search(resumen + nombre) else []),
                    "origen": origen, "resumen_listado": resumen[:200], "min_caracteres": 800})
    documentos.append({
        "doc_id": "circular_supersociedades_basica_juridica", "tipo": "circular", "numero": "100-000008", "anio": 2022,
        "titulo": "Circular Básica Jurídica de la Superintendencia de Sociedades (Circular Externa 100-000008 de 2022)",
        "organo_emisor": "Superintendencia de Sociedades", "fuente": "Superintendencia de Sociedades - Repositorio documental",
        "url": SS_CBJ, "areas": [A["com"]], "origen": "supersociedades_circular", "min_caracteres": 20000})
    print(f"  Supersociedades: {len(documentos)} documentos", flush=True)
    return documentos


def superfinanciera():
    """Circular Básica Jurídica (C.E. 029 de 2014): la SFC la publica como ZIP de capítulos en Word."""
    return [{"doc_id": "circular_superfinanciera_basica_juridica", "tipo": "circular", "numero": "029", "anio": 2014,
             "titulo": "Circular Básica Jurídica de la Superintendencia Financiera (Circular Externa 029 de 2014)",
             "organo_emisor": "Superintendencia Financiera de Colombia",
             "fuente": "Superintendencia Financiera de Colombia", "url": SFC_CBJ,
             "areas": [A["com"], A["merc"]], "origen": "superfinanciera_circular", "min_caracteres": 100000}]


def gestor():
    """Colecciones del Gestor Normativo de Función Pública, recorridas por tipo de documento.

    El buscador no pagina más allá de 10.000 resultados: un tipo que los supera (los conceptos
    de Función Pública, ~24.000) se recorre año por año. Cada página trae además la lista
    completa de filtros (hasta 11 MB), así que los resultados se leen con expresiones
    regulares y no con un analizador HTML.
    """
    documentos = {}
    for tipo_gestor, clase in GESTOR_TIPOS.items():
        total = gestor_total(tipo_gestor)
        anios = [None] if total <= GESTOR_VENTANA else range(1886, date.today().year + 1)
        for anio_filtro in anios:
            for titulo, url, resumen in gestor_resultados(tipo_gestor, anio_filtro):
                doc = gestor_ficha(tipo_gestor, clase, titulo, url, resumen)
                if doc:
                    documentos.setdefault(doc["doc_id"], doc)
        print(f"  Gestor {tipo_gestor}: {total} en el buscador; acumulado {len(documentos)}", flush=True)
    return list(documentos.values())


GESTOR_VENTANA = 9900
GESTOR_RESULTADO = re.compile(r'<p class="norma-titulo"><a href="([^"]+norma\.php\?i=\d+)"[^>]*>(.*?)</a></p>\s*'
                              r'<p class="norma-resumen">(.*?)</p>', re.S)


def gestor_pagina(tipo_gestor, anio, offset):
    url = GESTOR.format(urllib.parse.quote_plus(tipo_gestor), offset) + (f"&filtroAnio={anio}" if anio else "")
    estado, _, _, contenido = pedir(url)
    if estado != 200:
        raise RuntimeError(f"{url} respondió HTTP {estado}")
    return contenido.decode("utf-8", errors="replace")


def gestor_total(tipo_gestor, anio=None):
    m = re.search(r"devuelve\s*(?:<[^>]+>\s*)*([\d.]+)\s*(?:<[^>]+>\s*)*resultados", gestor_pagina(tipo_gestor, anio, 0))
    return int(m.group(1).replace(".", "")) if m else 0


def gestor_resultados(tipo_gestor, anio):
    offset = 0
    while True:
        pagina = gestor_pagina(tipo_gestor, anio, offset)
        encontrados = GESTOR_RESULTADO.findall(pagina)
        for url, titulo, resumen in encontrados:
            limpio = lambda t: html_lib.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))).strip()
            yield limpio(titulo), url, limpio(resumen)[:300]
        if len(encontrados) < 100 or offset + 100 >= GESTOR_VENTANA + 100:
            return
        offset += 100


def gestor_ficha(tipo_gestor, clase, titulo, url, resumen):
    m = GESTOR_TITULO.match(titulo)
    if not m or m.group(1) != tipo_gestor:
        return None
    numero, anio, entidad = m.group(2), int(m.group(3)), m.group(4).strip()
    gid = re.search(r"i=(\d+)", url).group(1)
    base = {"titulo": f"{m.group(1)} {numero} de {anio} - {entidad}", "anio": anio, "url": url,
            "fuente": "Departamento Administrativo de la Función Pública - Gestor Normativo", "resumen_listado": resumen}
    if clase in ("ley", "decreto", "acto_legislativo"):
        if not numero.isdigit() or (clase == "decreto" and entidad != "Nivel Nacional"):
            return None
        return {**base, "doc_id": f"{clase}_{int(numero)}_{anio}", "tipo": clase, "numero": str(int(numero)),
                "organo_emisor": "Presidencia de la República" if clase == "decreto" else "Congreso de la República",
                "areas": [], "origen": "gestor_normas", "min_caracteres": 500}
    if clase == "sentencia":
        if entidad not in ("Consejo de Estado", "Corte Suprema de Justicia"):
            return None
        corte = "ce" if entidad == "Consejo de Estado" else "csj"
        # El número del Gestor no siempre es el radicado: la identidad no se valida por él.
        return {**base, "doc_id": f"sentencia_{corte}_gestor_{gid}", "tipo": "sentencia", "numero": None,
                "organo_emisor": entidad, "areas": [A["adm"], A["lab"]] if corte == "ce" else [A["lab"]],
                "origen": "gestor_jurisprudencia", "min_caracteres": 2000}
    if clase == "concepto_sala_consulta":
        return {**base, "doc_id": f"concepto_ce_sala_consulta_{numero}_{anio}_{gid}", "tipo": "concepto",
                "numero": None, "organo_emisor": "Consejo de Estado - Sala de Consulta y Servicio Civil",
                "areas": [A["adm"]], "origen": "gestor_sala_consulta", "min_caracteres": 1500}
    return {**base, "doc_id": f"doctrina_fp_concepto_{gid}", "tipo": "concepto", "numero": None,
            "organo_emisor": entidad, "areas": [A["adm"], A["lab"]], "origen": "gestor_conceptos_fp",
            "min_caracteres": 800}


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--arbol", type=Path, help="JSON del árbol del Senado ya recorrido (se crea si no existe)")
    ap.add_argument("--refrescar", action="store_true", help="ignora la caché de data/cache/ronda5_*.json")
    args = ap.parse_args()
    if args.arbol and args.arbol.is_file():
        arbol = json.loads(args.arbol.read_text(encoding="utf-8"))
    else:
        arbol = recorrer_arbol()
        if args.arbol:
            args.arbol.write_text(json.dumps(arbol, ensure_ascii=False), encoding="utf-8")
    # Cada fuente se guarda en data/cache/ronda5_<fuente>.json: si algo falla al final no hay
    # que repetir horas de consultas. --refrescar las vuelve a pedir.
    fuentes = {"cc_datos_abiertos": corte_constitucional, "senado_arbol": lambda: senado(arbol), "sic_circular_unica": sic,
               "supersociedades": supersociedades, "superfinanciera": superfinanciera, "gestor": gestor}
    grupos = {}
    for nombre, funcion in fuentes.items():
        cache = RAIZ / f"data/cache/ronda5_{nombre}.json"
        if cache.is_file() and not args.refrescar:
            grupos[nombre] = json.loads(cache.read_text(encoding="utf-8"))
            print(f"  {nombre}: {len(grupos[nombre])} (caché)", flush=True)
        else:
            grupos[nombre] = funcion()
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(grupos[nombre], ensure_ascii=False), encoding="utf-8")
    # Una ley o un decreto puede venir del Senado y del Gestor: queda la primera ficha (Senado,
    # fuente preferida) y la URL de la otra pasa a urls_alternas como respaldo.
    documentos, por_id, repetidos = [], {}, 0
    for doc in (d for g in grupos.values() for d in g):
        previo = por_id.get(doc["doc_id"])
        if previo is None:
            por_id[doc["doc_id"]] = doc
            documentos.append(doc)
            continue
        repetidos += 1
        if doc.get("url") and doc["url"] != previo.get("url") and doc["url"] not in previo.get("urls_alternas", []):
            previo.setdefault("urls_alternas", []).append(doc["url"])
        if not previo.get("resumen_listado") and doc.get("resumen_listado"):
            previo["resumen_listado"] = doc["resumen_listado"]
    print(f"  fichas repetidas entre fuentes (unidas): {repetidos}")
    ruta = RAIZ / "configs/corpus_ronda5.json"
    ruta.write_text(json.dumps({"descripcion": __doc__.strip().splitlines()[0], "documentos": documentos},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{ruta.relative_to(RAIZ)}: {len(documentos)} documentos; " + ", ".join(f"{k}: {len(v)}" for k, v in grupos.items()))


if __name__ == "__main__":
    main()
