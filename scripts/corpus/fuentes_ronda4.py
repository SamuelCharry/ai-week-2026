"""Ronda 4 del corpus: jurisprudencia de cierre y doctrina oficial que el grafo no alcanza.

El grafo normativo encuentra lo que el corpus cita con un identificador reconocible. El
Consejo de Estado cita por radicado y la Corte Suprema casi no aparece citada en el
corpus, así que sus decisiones de cierre no salen del grafo. La doctrina tampoco: la
muestra oficial tiene 2 de 50 ítems con fundamento "Doctrina".

Fuentes, todas oficiales y leídas de sus propias páginas índice:

    ce_unificacion     sentencias de unificación del Consejo de Estado 2012-2025
                       (servicios.consejodeestado.gov.co/testmaster/nue_unifi.asp; en .doc)
    csj_compendio      compendios temáticos de la Sala Civil y extractos del esquema
                       jurisprudencial del sistema penal acusatorio (cortesuprema.gov.co)
    csj_spa_providencia  las providencias completas que ese esquema enlaza por tema (.doc/.docx)
    doctrina_dian      Conceptos Generales Unificados de la DIAN (normograma DIAN)
    doctrina_fp        Conceptos Marco de Función Pública (Gestor Normativo)

Escribe configs/corpus_ronda4.json, que scripts.corpus.objetivos incorpora al inventario.

Uso: python -m scripts.corpus.fuentes_ronda4
"""
import hashlib
import json
import re
import sys
import unicodedata
import urllib.parse
from pathlib import Path

from bs4 import BeautifulSoup

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.corpus.reconstruir import pedir  # noqa: E402

A = {"const": "Derecho constitucional", "adm": "Derecho administrativo", "penal": "Derecho penal",
     "proc": "Derecho procesal", "com": "Derecho comercial y sociedades", "civil": "Derecho civil",
     "fam": "Derecho de familia", "trib": "Derecho tributario", "lab": "Derecho laboral",
     "merc": "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]"}

CE_LISTADO = "https://servicios.consejodeestado.gov.co/testmaster/nue_unifi.asp"
CE_ARCHIVO = "https://servicios.consejodeestado.gov.co/WebRelatoria/FileReferenceServlet?corp=ce&ext=doc&file={}"
CSJ_INDICES = [
    "https://cortesuprema.gov.co/sala-de-casacion-civil-y-agraria-relatoria-publicaciones-tematicas/",
    "https://cortesuprema.gov.co/sala-de-casacion-penal-relatoria-esquema-jurisprudencial-sistema-penal-acusatorio1-2/",
    "https://cortesuprema.gov.co/sala-de-casacion-laboral-relatoria-publicaciones-compendio-providencias-relevantes/",
    "https://cortesuprema.gov.co/corte/index.php/tag/compendio/",
    "https://cortesuprema.gov.co/corte/index.php/compendio-jurisprudencial-del-cgp/",
]
# Publicaciones periódicas que repiten providencias completas del mes: fuera.
CSJ_EXCLUIR = re.compile(r"(?i)bolet[ií]n|gaceta")
DIAN = "https://normograma.dian.gov.co/dian/compilacion/docs/"
DOCTRINA_DIAN = [
    ("concepto_tributario_dian_0000912_2018", "Concepto General Unificado 912 de 2018 - Renta de personas naturales"),
    ("oficio_dian_3966_2023", "Concepto General 3966 de 2023 - Renta de personas naturales (Ley 2277 de 2022)"),
    ("concepto_tributario_dian_0000001_2003", "Concepto Unificado 1 de 2003 - Impuesto sobre las ventas"),
    ("concepto_tributario_dian_0000106_2022", "Concepto General Unificado 106 de 2022 - Facturación electrónica"),
    ("concepto_tributario_dian_0000481_2018", "Concepto General Unificado 481 de 2018 - Régimen Tributario Especial (ESAL)"),
    ("concepto_tributario_dian_0000577_2020", "Concepto 577 de 2020 DIAN"),
    ("concepto_tributario_dian_0001364_2018", "Concepto 1364 de 2018 DIAN"),
    ("concepto_dian_c000003_2002", "Concepto 3 de 2002 DIAN"),
]
DIAN_PDF = [("https://www.dian.gov.co/impuestos/Reforma%20Tributaria%20Estructural/"
             "Concepto%20Unificado%20Sobre%20Procedimiento%20Tributario%20y%20R%C3%A9gimen%20Tributario%20Sancionatorio.pdf",
             "concepto_dian_unificado_procedimiento_sancionatorio",
             "Concepto Unificado sobre Procedimiento Tributario y Régimen Tributario Sancionatorio")]
FP_BUSQUEDA = ("https://www.funcionpublica.gov.co/dafpIndexerBGN/norma/index?find=FindNext"
               "&filtroTipoDocumento=Concepto%20Marco&max=50")


def slug(texto):
    texto = unicodedata.normalize("NFD", urllib.parse.unquote(texto))
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn").lower()
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")[:60]


def html(url):
    estado, _, _, contenido = pedir(url)
    if estado != 200:
        raise RuntimeError(f"{url} respondió HTTP {estado}")
    return BeautifulSoup(contenido, "html.parser")


def consejo_de_estado():
    documentos, radicados = [], set()
    for enlace in html(CE_LISTADO).find_all("a", href=True):
        archivo = re.search(r"FileReferenceServlet\?corp=ce&ext=doc&file=(\d+)", enlace["href"])
        fila = enlace.find_parent("tr")
        if not archivo or fila is None:
            continue
        celdas = [re.sub(r"\s+", " ", c.get_text(" ", strip=True)) for c in fila.find_all("td")]
        texto = " ".join(celdas)
        radicado = re.search(r"\d{5}-\d{2}-\d{2}-\d{3}-\d{4}-\d{5}-\d{2}", texto)
        # El listado repite algunas sentencias con dos archivos idénticos: una por radicado.
        if radicado and radicado.group() in radicados:
            continue
        if radicado:
            radicados.add(radicado.group())
        fecha = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", texto)
        # El listado trae "1/01/1900" cuando no registra la fecha: se trata como sin fecha.
        anio = fecha.group(3) if fecha and fecha.group(3) != "1900" else "s_f"
        tema = next((c for c in celdas if "UNIFICACI" in c.upper() or len(c) > 80), texto)[:180]
        area = ["trib"] if re.search(r"(?i)impuest|tribut|iva|renta|ica\b", tema) else \
            ["lab", "adm"] if re.search(r"(?i)pensi|laboral|salari|prestaci|empleado", tema) else ["adm"]
        documentos.append({
            "doc_id": f"sentencia_ce_suj_{archivo.group(1)}_{anio}",
            "titulo": f"Consejo de Estado, sentencia de unificación {radicado.group() if radicado else archivo.group(1)} "
                      f"({fecha.group() if anio != 's_f' else 's. f.'})",
            "tipo": "sentencia", "numero": radicado.group() if radicado else archivo.group(1),
            "anio": int(anio) if anio.isdigit() else None, "organo_emisor": "Consejo de Estado",
            "fuente": "Consejo de Estado - Relatoría", "url": CE_ARCHIVO.format(archivo.group(1)),
            "areas": [A[a] for a in area], "origen": "ce_unificacion", "resumen_listado": tema,
            "min_caracteres": 3000})
    return documentos


def corte_suprema():
    vistos, documentos = set(), []
    for indice in CSJ_INDICES:
        for enlace in html(indice).find_all("a", href=True):
            url = enlace["href"].strip()
            if not re.search(r"(?i)\.pdf$", url) or CSJ_EXCLUIR.search(url):
                continue
            url = urllib.parse.urljoin(indice, url)
            nombre = slug(url.rsplit("/", 1)[-1][:-4])
            if nombre in vistos:
                continue
            vistos.add(nombre)
            penal = "penal" in indice
            laboral = "laboral" in indice
            titulo = urllib.parse.unquote(url.rsplit("/", 1)[-1][:-4]).replace("-", " ").strip()
            areas = ["penal", "proc"] if penal else ["lab"] if laboral else \
                ["fam", "civil"] if re.search(r"(?i)filiaci|marital|famil|redes sociales por los padres", titulo) else \
                ["proc", "civil"] if re.search(r"(?i)CGP|competencia|revisi|demanda|ejecutivo", titulo) else \
                ["com", "civil"] if re.search(r"(?i)seguro|fiduci|comercial|compraventa|arbitraje|competencia|consumo", titulo) \
                else ["civil"]
            documentos.append({
                "doc_id": f"csj_compendio_{'spa_' if penal else ''}{nombre}_{hashlib.sha256(url.encode()).hexdigest()[:6]}",
                "titulo": ("Corte Suprema, Sala Penal, esquema jurisprudencial SPA: " if penal else
                           "Corte Suprema, compendio de jurisprudencia: ") + titulo,
                "tipo": "compendio", "numero": None, "anio": None,
                "organo_emisor": "Corte Suprema de Justicia", "fuente": "Corte Suprema de Justicia - Relatoría",
                "url": url, "areas": [A[a] for a in areas], "origen": "csj_compendio",
                "min_caracteres": 300 if penal else 1500})
    return documentos


SPA_INDICE = CSJ_INDICES[1]


def providencias_spa():
    """Providencias que el esquema del sistema penal acusatorio enlaza por tema (.doc/.docx).

    El nombre del archivo trae la providencia ("AP2491-2016(47221).doc") o el radicado y
    la fecha ("21347(14-12-05).doc"); la ruta trae el tema y el subtema del esquema.
    """
    por_id = {}
    for enlace in html(SPA_INDICE).find_all("a", href=True):
        url = urllib.parse.urljoin(SPA_INDICE, enlace["href"].strip()).replace(" ", "%20")
        if not re.search(r"(?i)\.docx?$", url):
            continue
        partes = [urllib.parse.unquote(p) for p in urllib.parse.urlsplit(url).path.split("/")]
        archivo, temas = partes[-1], [p for p in partes[partes.index("spa") + 1:-1]] if "spa" in partes else []
        nombre = re.sub(r"(?i)\.docx?$", "", archivo)
        moderna = re.match(r"([A-Z]{2,3})\s*(\d+)\s*-\s*(\d{4})", nombre)
        antigua = re.match(r"(\d{4,6})\s*\((\d{1,2})-(\d{1,2})-(\d{2,4})\)", nombre)
        if moderna:
            sala, numero, anio = moderna.group(1).upper(), int(moderna.group(2)), int(moderna.group(3))
            doc_id, titulo = f"sentencia_csj_{sala.lower()}{numero}_{anio}", f"Sentencia {sala}{numero}-{anio}"
        elif antigua:
            anio = int(antigua.group(4))
            anio = anio if anio > 100 else (1900 + anio if anio > 90 else 2000 + anio)
            numero = int(antigua.group(1))
            doc_id, titulo = f"sentencia_csj_rad{numero}_{anio}", f"Corte Suprema, Sala Penal, radicado {numero} ({anio})"
        else:
            continue
        tema = " / ".join(t.title() for t in temas)
        if doc_id in por_id:
            por_id[doc_id]["temas_esquema"].append(tema)
            continue
        por_id[doc_id] = {
            "doc_id": doc_id, "titulo": titulo, "tipo": "sentencia", "numero": nombre, "anio": anio,
            "organo_emisor": "Corte Suprema de Justicia", "fuente": "Corte Suprema de Justicia - Relatoría",
            "url": url, "areas": [A["penal"], A["proc"]], "origen": "csj_spa_providencia",
            "temas_esquema": [tema], "min_caracteres": 1500}
    return list(por_id.values())


def doctrina():
    documentos = [{"doc_id": f"doctrina_{clave}", "titulo": titulo, "tipo": "concepto", "numero": None, "anio":
                   int(clave[-4:]), "organo_emisor": "DIAN", "fuente": "DIAN - Compilación jurídica",
                   "url": f"{DIAN}{clave}.htm", "areas": [A["trib"]], "origen": "doctrina_dian", "min_caracteres": 1500}
                  for clave, titulo in DOCTRINA_DIAN]
    documentos += [{"doc_id": f"doctrina_{clave}", "titulo": titulo, "tipo": "concepto", "numero": None, "anio": 2016,
                    "organo_emisor": "DIAN", "fuente": "DIAN", "url": url, "areas": [A["trib"]],
                    "origen": "doctrina_dian", "min_caracteres": 1500} for url, clave, titulo in DIAN_PDF]
    for enlace in html(FP_BUSQUEDA).find_all("a", href=True):
        if "norma.php?i=" not in enlace["href"]:
            continue
        titulo = re.sub(r"\s+", " ", enlace.parent.get_text(" ", strip=True))
        marco = re.match(r"Concepto Marco (\d+) de (\d{4})", titulo)
        if not marco:
            continue
        documentos.append({
            "doc_id": f"doctrina_fp_concepto_marco_{int(marco.group(1))}_{marco.group(2)}",
            "titulo": f"Concepto Marco {marco.group(1)} de {marco.group(2)} - Función Pública",
            "tipo": "concepto", "numero": marco.group(1), "anio": int(marco.group(2)),
            "organo_emisor": "Departamento Administrativo de la Función Pública",
            "fuente": "Departamento Administrativo de la Función Pública - Gestor Normativo",
            "url": enlace["href"], "areas": [A["adm"], A["lab"]], "origen": "doctrina_fp", "min_caracteres": 1500})
    return documentos


def main():
    grupos = {"ce_unificacion": consejo_de_estado(), "csj_compendio": corte_suprema(),
              "csj_spa_providencia": providencias_spa(), "doctrina": doctrina()}
    documentos = [d for g in grupos.values() for d in g]
    ids = [d["doc_id"] for d in documentos]
    if len(ids) != len(set(ids)):
        raise ValueError("doc_id repetidos en la ronda 4")
    ruta = RAIZ / "configs/corpus_ronda4.json"
    ruta.write_text(json.dumps({"descripcion": __doc__.strip().splitlines()[0], "documentos": documentos},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{ruta.relative_to(RAIZ)}: " + ", ".join(f"{k}: {len(v)}" for k, v in grupos.items()))


if __name__ == "__main__":
    main()
