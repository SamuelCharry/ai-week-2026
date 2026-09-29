"""Ingesta y segmentación reproducibles de originales jurídicos.

Los offsets se refieren al texto normalizado. La atribución estructural no
certifica vigencia. Las ventanas permiten localizar unidades completas.
"""
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
import csv
import hashlib
from html.entities import name2codepoint
import io
import json
import re
import unicodedata
from xml.etree import ElementTree
import zipfile

from bs4 import BeautifulSoup, Comment, NavigableString, Tag, UnicodeDammit
from ftfy import fix_encoding
import pdfplumber

VERSION_INGESTA = "0.5.0-local"
MAX_CARACTERES = 1800
SOLAPAMIENTO = 200
MAX_BLOQUE_JUDICIAL = 6000

# Defectos comprobados en reports/a_muestra.csv, revisión del 2026-09-26.
# La restricción solo aplica al original auditado, identificado por su SHA-256.
FUENTES_RESTRINGIDAS = {
    "f3708f6b39baf84a413956d444a62858a16f5f99ea295568bf5a582f128b5808": "OCR_erroneo_confirmado_SC5191",
    "ba6b8388c4690612026f9953b5236da8ace97f16a069b0c04d78c1963508cfaf": "mojibake_residual_confirmado_C264",
    "0235d1f005c68b43c400c1d72ee473135e11ec74b9e5e2bbc02c33a9621d8621": "mojibake_irrecuperable_confirmado_Ley1551",
}


def evaluar_fuente(documento):
    rechazos = [{"archivo": a["archivo"], "sha256": a["sha256"],
                 "motivo": FUENTES_RESTRINGIDAS[a["sha256"]],
                 "evidencia": "reports/a_muestra.csv", "fecha_revision": "2026-09-26"}
                for a in documento.get("archivos_raw", []) if a.get("sha256") in FUENTES_RESTRINGIDAS]
    return {"estado_fuente": "defecto_confirmado" if rechazos else "sin_defecto_confirmado",
            "apta_para_busqueda": not rechazos, "rechazos_fuente": rechazos}

def sha256(contenido):
    return hashlib.sha256(contenido).hexdigest()


def normalizar(texto):
    texto = unicodedata.normalize("NFC", fix_encoding(texto))
    texto = texto.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\xad\u200b\ufeff]", "", texto)
    lineas = [re.sub(r"[^\S\n]+", " ", linea).strip() for linea in texto.split("\n")]
    return "\n".join(linea for linea in lineas if linea)


def leer_original(archivo, raw_dir):
    RAW = Path(raw_dir)
    ruta = (RAW / archivo["archivo"]).resolve()
    if not ruta.is_relative_to(RAW.resolve()):
        raise ValueError("La ruta sale del directorio raw autorizado.")
    contenido = ruta.read_bytes()
    if len(contenido) != archivo["bytes"] or sha256(contenido) != archivo["sha256"]:
        raise ValueError(f"El original no coincide con el manifiesto: {archivo['archivo']}")
    return contenido


def extraer_html(contenido, archivo):
    html, encoding, encoding_origen = None, None, None
    errores_decodificacion = 0
    declarado = re.search(br"<meta\b[^>]{0,1000}\bcharset\s*=\s*['\"]?\s*([a-zA-Z0-9._:-]+)", contenido, re.I)
    charset_html = declarado.group(1).decode("ascii") if declarado else None
    for candidato, origen in [("utf-8-sig", "utf8_comprobado"), (charset_html, "charset_html"),
                              (archivo.get("encoding"), "manifiesto")]:
        if not candidato:
            continue
        try:
            html, encoding = contenido.decode(candidato), candidato
            encoding_origen = origen
            break
        except UnicodeDecodeError:
            if origen == "charset_html":
                # La fuente c226_2004 declara windows-1252 pero incluye un byte
                # indefinido 0x81. Cambiar todo a otro alfabeto por ese byte
                # transforma las ñ válidas. Mantener charset y señalar el daño.
                html, encoding = contenido.decode(candidato, errors="replace"), candidato
                errores_decodificacion = html.count("\ufffd")
                encoding_origen = "charset_html_con_reemplazos"
                break
        except LookupError:
            continue
    if html is None:
        detectado = UnicodeDammit(contenido, is_html=True)
        html, encoding = detectado.unicode_markup, detectado.original_encoding
        encoding_origen = "autodetectado"
    if html is None:
        raise ValueError("No fue posible decodificar el HTML.")
    # Entidades de acentos sin punto y coma, presentes en Ley 2191.
    # Solo se completa una entidad HTML conocida. No se reconstruyen palabras.
    html, entidades_reparadas = re.subn(
        r"&([AEIOUaeiou](?:acute|uml)|[Nn]tilde)(?=[A-Za-zÁÉÍÓÚáéíóúÑñ<\s])",
        lambda m: chr(name2codepoint[m.group(1)]), html,
    )
    soup = BeautifulSoup(html, "html.parser")
    principal, selector = soup.body or soup, "body"
    for candidato in ["#aj_data", ".descripcion-contenido", ".panel-documento",
                      "#documento", "#contenidoNorma", "article"]:
        elementos = soup.select(candidato)
        if elementos and len(elementos[-1].get_text()) > 200:
            principal, selector = elementos[-1], candidato
            break
    editorial = bool(soup.select("#aj_data, .panel-documento"))
    retiradas_html = []
    # Selectores comprobados en ley_100_1993/000.html del Senado. Las cajas de
    # vigencia/concordancias y el título de la norma no forman parte de esta lista.
    if selector == "#aj_data":
        for etiqueta in list(principal.select("#selector_aj, #imprimir, #logo_aj, a.hlk_inicio, a.antsig")):
            if etiqueta.parent is not None:
                retiradas_html.append({"selector": etiqueta.get("id") or ".".join(etiqueta.get("class", [])),
                                       "texto": normalizar(etiqueta.get_text(" ")),
                                       "motivo": "navegacion_o_pie_editorial_Senado"})
                etiqueta.decompose()
    for etiqueta in list(principal.select("head, script, style, noscript, nav, footer, form, button, iframe, xml")):
        if etiqueta.parent is not None:
            etiqueta.decompose()
    # En ley_27_1977 el servidor ya aplanó los metadatos Word/CSS como texto
    # anterior al primer párrafo. Retirar sólo ese prefijo con ambas firmas.
    if selector == ".descripcion-contenido":
        prefijo = []
        for nodo in principal.contents:
            if not isinstance(nodo, NavigableString):
                break
            prefijo.append(nodo)
        ruido = "".join(str(nodo) for nodo in prefijo)
        if ("/* Style Definitions */" in ruido and "table.MsoNormalTable" in ruido
                and not re.search(r"\b(?:LEY|DECRETO|ART[ÍI]CULO)\s+\d", ruido, re.I)):
            retiradas_html.append({"selector": ".descripcion-contenido > prefijo_textual",
                                   "texto": normalizar(ruido), "motivo": "metadatos_Word_y_CSS_aplanados"})
            for nodo in prefijo:
                nodo.extract()
    bloques = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table", "br"}
    partes = []
    tablas = {"tablas_estructuradas": 0, "tablas_en_texto": 0}

    def esta_tachado(nodo):
        return nodo.name in {"s", "strike", "del"} or "line-through" in nodo.get("style", "").lower()

    def texto_celda(nodo):
        if isinstance(nodo, Comment):
            return ""
        if isinstance(nodo, NavigableString):
            return str(nodo)
        if not isinstance(nodo, Tag):
            return ""
        texto = "".join(texto_celda(hijo) for hijo in nodo.children)
        if esta_tachado(nodo):
            texto = " [TEXTO TACHADO EN LA FUENTE: " + texto + "] "
        if nodo.name in bloques:
            texto = " " + texto + " "
        return texto

    def tabla_markdown(tabla):
        # Las tablas de maquetación o con celdas combinadas conservan el recorrido original.
        if (tabla.get("role") in {"presentation", "none"}
                or tabla.find(["table", "caption", "h1", "h2", "h3", "h4", "h5", "h6"])):
            return None
        if any(isinstance(nodo, NavigableString) and not isinstance(nodo, Comment)
               and str(nodo).strip() and nodo.find_parent(["td", "th"]) is None
               for nodo in tabla.descendants):
            return None
        filas = [fila.find_all(["td", "th"], recursive=False) for fila in tabla.find_all("tr")]
        if len(filas) < 2 or len(filas[0]) < 2 or any(len(f) != len(filas[0]) for f in filas):
            return None
        if sum(len(fila) for fila in filas) != len(tabla.find_all(["td", "th"])):
            return None
        if any(str(celda.get(atributo, "1")) != "1"
               for fila in filas for celda in fila for atributo in ("rowspan", "colspan")):
            return None
        if re.search(r"\b(?:art[ií]culo|art\.|cap[ií]tulo|t[ií]tulo)\s+",
                     tabla.get_text(" "), re.I):
            return None
        lineas = []
        for i, fila in enumerate(filas):
            valores = []
            for celda in fila:
                valor = re.sub(r"\s+", " ", texto_celda(celda)).strip().replace("|", "\\|")
                padre = celda.parent
                while padre is not tabla:
                    if esta_tachado(padre):
                        valor = "[TEXTO TACHADO EN LA FUENTE: " + valor + "]"
                    padre = padre.parent
                valores.append(valor)
            lineas.append("| " + " | ".join(valores) + " |")
            if i == 0 and all(celda.name == "th" for celda in fila):
                lineas.append("| " + " | ".join("---" for _ in fila) + " |")
        return "\n".join(lineas)

    def recorrer(nodo):
        if isinstance(nodo, Comment):
            return
        if isinstance(nodo, NavigableString):
            partes.append(str(nodo))
        elif isinstance(nodo, Tag):
            tachado = esta_tachado(nodo)
            if nodo.name in bloques:
                partes.append("\n")
            if tachado:
                partes.append(" [TEXTO TACHADO EN LA FUENTE: ")
            tabla = tabla_markdown(nodo) if nodo.name == "table" else None
            if nodo.name == "table":
                tablas["tablas_estructuradas" if tabla is not None else "tablas_en_texto"] += 1
            if tabla is not None:
                partes.append(tabla)
            else:
                for hijo in nodo.children:
                    recorrer(hijo)
            if tachado:
                partes.append("] ")
            if nodo.name in {"td", "th"}:
                partes.append(" | ")
            if nodo.name in bloques:
                partes.append("\n")

    recorrer(principal)
    texto = normalizar("".join(partes))
    if len(texto) < 1500 and re.search(r"captcha|access denied|just a moment|acceso denegado|page not found", texto, re.I):
        raise ValueError("El HTML contiene una página de bloqueo o error.")
    return [{"pagina": None, "texto": texto}], {
        "selector": selector, "encoding": encoding, "encoding_origen": encoding_origen,
        "charset_html": charset_html, "n_errores_decodificacion": errores_decodificacion,
        "anotaciones_detectadas": editorial,
        "paginas_pdf": None, "paginas_con_poco_texto": [], "entidades_reparadas": entidades_reparadas,
        "elementos_html_retirados": retiradas_html,
        **tablas,
    }


def extraer_pdf(contenido):
    inicio = contenido[:1024].find(b"%PDF-")
    if inicio < 0:
        raise ValueError("El archivo no tiene cabecera PDF.")
    with pdfplumber.open(io.BytesIO(contenido[inicio:])) as lector:
        paginas = [{"pagina": i, "texto": normalizar(pagina.extract_text(use_text_flow=True) or "")}
                   for i, pagina in enumerate(lector.pages, start=1)]
    paginas, retiradas = retirar_cabeceras_editoriales(paginas)
    poco_texto = [p["pagina"] for p in paginas if len(p["texto"].strip()) < 40]
    return paginas, {
        "selector": "pdfplumber_text_flow", "encoding": "PDF", "anotaciones_detectadas": False,
        "paginas_pdf": len(paginas), "paginas_con_poco_texto": poco_texto,
        "cabeceras_retiradas": retiradas, "offset_cabecera_pdf": inicio, "cierre_pdf_completo": contenido.rstrip().endswith(b"%%EOF"),
    }


def extraer_docx(contenido):
    """Lee XML de Word sin ejecutar macros, relaciones externas ni descomprimir a disco."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    mc = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
    alternativas = {"choice": 0, "fallback": 0, "sin_rama_textual": 0}

    def elegir_alternativa(nodo):
        # Word guarda representaciones equivalentes para distintos lectores.
        # El texto WordML/txbxContent es soportado; no concatenar sus copias.
        for opcion in nodo.findall(mc + "Choice"):
            if any(t.text and t.text.strip() for t in opcion.iter(ns + "t")):
                alternativas["choice"] += 1
                return opcion
        respaldo = nodo.find(mc + "Fallback")
        if respaldo is not None:
            alternativas["fallback"] += 1
            return respaldo
        alternativas["sin_rama_textual"] += 1
        return None

    def leer_xml(archivo, nombre):
        info = archivo.getinfo(nombre)
        if info.file_size > 64 * 1024 * 1024:
            raise ValueError("El XML DOCX supera el límite de extracción de 64 MiB.")
        datos = archivo.read(nombre)
        if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", datos.replace(b"\x00", b""), re.I):
            raise ValueError("Declaraciones DTD o ENTITY no admitidas en DOCX.")
        return ElementTree.fromstring(datos)

    def texto_parrafo(nodo):
        def inline(elemento):
            if elemento.tag == mc + "AlternateContent":
                elegido = elegir_alternativa(elemento)
                return inline(elegido) if elegido is not None else ""
            if elemento.tag == ns + "txbxContent":
                # Las cajas pueden contener tablas y párrafos completos.
                return "\n" + "\n".join(bloques(elemento)) + "\n"
            if elemento.tag == ns + "t":
                return elemento.text or ""
            if elemento.tag == ns + "tab":
                return "\t"
            if elemento.tag in {ns + "br", ns + "cr"}:
                return "\n"
            if elemento.tag in {ns + "footnoteReference", ns + "endnoteReference"}:
                return " [NOTA " + elemento.get(ns + "id", "?") + "] "
            return "".join(inline(hijo) for hijo in elemento)
        return inline(nodo)

    def bloques(nodo):
        if nodo.tag == mc + "AlternateContent":
            elegido = elegir_alternativa(nodo)
            if elegido is not None:
                yield from bloques(elegido)
            return
        for hijo in nodo:
            if hijo.tag == ns + "p":
                yield texto_parrafo(hijo)
            elif hijo.tag == ns + "tbl":
                for fila in hijo.findall(ns + "tr"):
                    yield " | ".join(" / ".join(bloques(celda)) for celda in fila.findall(ns + "tc"))
            else:
                yield from bloques(hijo)

    with zipfile.ZipFile(io.BytesIO(contenido)) as archivo:
        arbol = leer_xml(archivo, "word/document.xml")
        partes = list(bloques(arbol))
        anexos = []
        for nombre, etiqueta in [("word/footnotes.xml", "footnote"), ("word/endnotes.xml", "endnote")]:
            if nombre not in archivo.namelist():
                continue
            for nota in leer_xml(archivo, nombre).findall(ns + etiqueta):
                if nota.get(ns + "type") in {"separator", "continuationSeparator"}:
                    continue
                partes.append("[NOTA " + nota.get(ns + "id", "?") + "]\n" + "\n".join(bloques(nota)))
            anexos.append(nombre)
    return [{"pagina": None, "texto": normalizar("\n".join(partes))}], {
        "selector": "docx_xml_stdlib", "encoding": "XML", "anotaciones_detectadas": False,
        "paginas_pdf": None, "paginas_con_poco_texto": [], "anexos_docx": anexos,
        "alternativas_docx": alternativas,
        "limitaciones": ["sin_paginacion_original", "cabeceras_pies_y_objetos_incrustados_no_extraidos"],
    }


def extraer_texto_derivado(archivo, contenido_original, raw_dir):
    """Usa el derivado declarado sólo después de verificar sus bytes y hash."""
    derivado = archivo["texto_derivado"]
    if not isinstance(derivado, dict) or Path(derivado.get("archivo", "")).suffix.lower() != ".txt":
        raise ValueError("texto_derivado debe describir un archivo .txt verificable.")
    contenido = leer_original(derivado, raw_dir)
    texto = contenido.decode("utf-8-sig", errors="strict")
    trozos = texto.split("\f")
    if len(trozos) > 1 and not trozos[-1].strip():
        trozos.pop()
    paginas_pdf, mapeo = None, False
    error_paginacion = None
    es_pdf = Path(archivo["archivo"]).suffix.lower() == ".pdf"
    if es_pdf:
        inicio = contenido_original[:1024].find(b"%PDF-")
        if inicio < 0:
            raise ValueError("El original del derivado no tiene cabecera PDF.")
        try:
            with pdfplumber.open(io.BytesIO(contenido_original[inicio:])) as lector:
                paginas_pdf = len(lector.pages)
            mapeo = len(trozos) == paginas_pdf
        except Exception as error:
            error_paginacion = f"{type(error).__name__}: {error}"
    paginas = [{"pagina": i if mapeo else None, "texto": normalizar(t)}
               for i, t in enumerate(trozos, 1)]
    return paginas, {
        "selector": "texto_derivado_verificado", "encoding": "utf-8-sig", "anotaciones_detectadas": False,
        "paginas_pdf": paginas_pdf, "paginas_derivado": len(trozos), "mapeo_paginas_verificado": mapeo,
        "paginas_con_poco_texto": [p["pagina"] for p in paginas if p["pagina"] is not None and len(p["texto"]) < 40],
        "archivo_texto_derivado": derivado["archivo"], "sha256_texto_derivado": derivado["sha256"],
        "bytes_texto_derivado": len(contenido), "metodo_derivado": derivado.get("metodo"),
        "motivo_derivado": derivado.get("motivo"), "error_verificacion_paginas": error_paginacion,
    }


def procesar_documento(documento, raw_dir):
    texto, partes, extraccion, avisos = "", [], [], []
    if not documento.get("archivos_raw"):
        raise ValueError("El documento no tiene originales.")
    for archivo in documento["archivos_raw"]:
        contenido = leer_original(archivo, raw_dir)
        extension = Path(archivo["archivo"]).suffix.lower()
        if extension == ".docx" and archivo.get("texto_derivado"):
            # Los 316 derivados DOCX recibidos se generaron sólo desde
            # word/document.xml. 255 originales contienen además notas XML.
            # Verificar el sidecar, pero recuperar cuerpo, tablas y notas nativos.
            derivado = archivo["texto_derivado"]
            contenido_derivado = leer_original(derivado, raw_dir)
            bloques, info = extraer_docx(contenido)
            info["derivado_conservado_no_utilizado"] = {
                "archivo": derivado["archivo"], "sha256": derivado["sha256"],
                "bytes": len(contenido_derivado), "metodo": derivado.get("metodo"),
                "motivo": "derivado_solo_document_xml_incompleto_para_notas",
            }
            avisos.extend(["docx_nativo_preferido_para_preservar_notas",
                           "revisar_estructura_word_sin_paginacion"])
        elif archivo.get("texto_derivado"):
            bloques, info = extraer_texto_derivado(archivo, contenido, raw_dir)
            avisos.append("texto_derivado_verificado_utilizado")
            if "ocr" in archivo["texto_derivado"]["archivo"].lower() or "tesseract" in str(info.get("metodo_derivado", "")).lower():
                avisos.append("revisar_calidad_OCR")
            if extension == ".pdf" and not info["mapeo_paginas_verificado"]:
                avisos.append("paginacion_derivado_no_verificada")
        elif extension in {".html", ".htm"}:
            bloques, info = extraer_html(contenido, archivo)
        elif extension == ".pdf":
            bloques, info = extraer_pdf(contenido)
            avisos.append("revisar_orden_y_texto_pdf")
        elif extension == ".docx":
            bloques, info = extraer_docx(contenido)
            avisos.append("revisar_estructura_word_sin_paginacion")
        elif extension == ".doc":
            raise ValueError("Word binario .doc requiere texto_derivado con bytes y SHA-256; no se decodifica como texto.")
        else:
            raise ValueError(f"Formato no admitido: {extension}")
        extraccion.append({"archivo": archivo["archivo"], **info})
        if info.get("cierre_pdf_completo") is False:
            avisos.append("original_pdf_sin_cierre_completo")
        if info.get("n_errores_decodificacion", 0):
            avisos.append("bytes_no_decodificables_en_charset_declarado")
        if info["anotaciones_detectadas"]:
            avisos.append("anotaciones_editoriales")
        if info["paginas_con_poco_texto"]:
            avisos.append("paginas_con_poco_texto_revisar_OCR")
        for bloque in bloques:
            if texto:
                texto += "\n\n"
            cabecera = f"[PÁGINA {bloque['pagina']}]\n" if bloque["pagina"] is not None else ""
            texto += cabecera
            inicio = len(texto)
            texto += bloque["texto"]
            partes.append({"archivo": archivo["archivo"], "url": archivo.get("url_final") or archivo.get("url") or documento.get("url"),
                           "archivo_texto_derivado": info.get("archivo_texto_derivado"),
                           "pagina": bloque["pagina"], "inicio": inicio, "fin": len(texto)})
    caracteres_extraidos = sum(p["fin"] - p["inicio"] for p in partes)
    if caracteres_extraidos == 0:
        raise ValueError("No se extrajo texto; revisar original u OCR.")
    if caracteres_extraidos < 200:
        avisos.append("texto_corto")
    if "\ufffd" in texto:
        avisos.append("caracteres_de_reemplazo")
    coincidencias_mojibake = len(re.findall(r"\ufffd|Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|â[€\u0080-\u00bf]", texto))
    if coincidencias_mojibake:
        avisos.append("posible_mojibake_revisar_fuente")
    if documento.get("edicion_con_anotaciones") or documento.get("redistribuir_raw") is not True:
        avisos.append("revisar_anotaciones_y_publicacion")
    avisos = sorted(set(avisos))
    registro = {**documento, **evaluar_fuente(documento), "version_ingesta": VERSION_INGESTA,
                "estado_extraccion": "revisar" if avisos else "extraido",
                "revision_juridica": "pendiente", "avisos": avisos,
                "texto_archivo": f"textos/{documento['doc_id']}.txt",
                "sha256_texto": sha256(texto.encode("utf-8")), "caracteres": len(texto),
                "coincidencias_mojibake": coincidencias_mojibake,
                "partes": partes, "extraccion": extraccion, "error": None}
    return registro, texto


def retirar_cabeceras_editoriales(paginas):
    """Retira solo la cabecera repetida identificada de EVA, con rastro por página."""
    patron = re.compile(
        r"^Departamento Administrativo de la Función Pública\n"
        r"[\s\S]{0,180}?\n\d+ EVA - Gestor Normativo\n"
    )
    coincidencias = [patron.match(p["texto"]) for p in paginas]
    if sum(m is not None for m in coincidencias) < 2:
        return paginas, []
    resultado, retiradas = [], []
    for pagina, m in zip(paginas, coincidencias):
        if m:
            retiradas.append({"pagina": pagina["pagina"], "inicio_texto_pagina": 0,
                              "fin_texto_pagina": m.end(), "texto": m.group(),
                              "motivo": "cabecera_editorial_EVA_repetida"})
        resultado.append({**pagina, "texto": pagina["texto"][m.end():] if m else pagina["texto"]})
    return resultado, retiradas


_ORDINALES = dict(zip(
    "primero segundo tercero cuarto quinto sexto septimo octavo noveno decimo undecimo duodecimo".split(),
    range(1, 13),
))
_ORDINALES.update({"primer": 1, "tercer": 3, "decimoprimero": 11, "decimosegundo": 12,
                  "decimotercero": 13, "decimocuarto": 14, "decimoquinto": 15,
                  "decimosexto": 16, "decimoseptimo": 17, "decimoctavo": 18,
                  "decimonoveno": 19, "vigesimo": 20, "trigesimo": 30,
                  "cuadragesimo": 40, "quincuagesimo": 50, "sexagesimo": 60,
                  "septuagesimo": 70, "octogesimo": 80, "nonagesimo": 90})
_NUMERO = r"\d+(?:\.\d+)*(?:[ \t]?[A-Za-z](?=[º°ª.\s:;–—-]|$))?[º°ª]?"
_ARTICULO = re.compile(
    r'(?<!\w)(?P<comilla>["“«][ \t]*)?(?P<etiqueta>ART[ÍI]CULO\.?|ART\.)\s+'
    r'(?P<transitorio>TRANSITORIO\s+)?(?P<numero>' + _NUMERO +
    r'|[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:[ \t]+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)?)', re.I,
)
_REFERENCIA = re.compile(r"art[íi]culos?\s*:?[ \t\n]*(\d+(?:\.\d+)*(?:[A-Z](?![A-Za-z]))?)", re.I)
_REFORMA = re.compile(r"modif[íi]qu|modificar|adici[óo]nese|adici[óo]nense|adicionar|sustit[úu]y", re.I)
_INTRO_CITA = re.compile(
    r"quedar[áa](?:n)?[\s\S]{0,70}?as[íi]\s*:|"
    r"(?:siguiente(?:s)?\s+(?:texto|tenor|t[ée]rminos)|texto\s+siguiente)[^:]{0,70}:|"
    r"(?:transcribe|dispone|establece|señala|consagra|dice)[^:]{0,120}:\s*$", re.I,
)
_SECCION_NORMA = re.compile(
    r"^(?:(?:T[ÍI]TULO|CAP[ÍI]TULO|LIBRO|SECCI[ÓO]N|PARTE)\s+"
    r"(?:[IVXLCDM]+|\d+|PRIMER[OA]|SEGUND[OA]|TERCER[OA]|GENERAL|ESPECIAL)\b.*|"
    r"DISPOSICIONES (?:TRANSITORIAS|GENERALES|FINALES)|ART[ÍI]CULOS TRANSITORIOS)$", re.I,
)
_MARCADOR_PAGINA = re.compile(r"(?m)^\[P[ÁA]GINA \d+\][ \t]*$")


def _sin_tildes(texto):
    return texto.lower().translate(str.maketrans("áéíóúü", "aeiouu"))


def _numero_articulo(texto):
    numero = _sin_tildes(texto.strip())
    if numero in {"unico", "transitorio"}:
        return numero
    if numero[:1].isdigit():
        numero = re.sub(r"[º°ª]$", "", numero)
        numero = re.sub(r"(?<=\d)o$", "", numero)
        return re.sub(r"\s+", "", numero).upper()
    palabras = numero.split()
    if all(p in _ORDINALES for p in palabras):
        numeros = [_ORDINALES[p] for p in palabras]
        if len(numeros) == 1 or (len(numeros) == 2 and numeros[0] >= 10 and numeros[1] < 10):
            return str(sum(numeros))
    return None


def candidatos_articulo(texto):
    """Localiza cabeceras, sin decidir que los artículos sean propios de la norma."""
    candidatos = []
    for m in _ARTICULO.finditer(texto):
        numero = _numero_articulo(m.group("numero"))
        if numero is None:
            continue
        inicio_linea = texto.rfind("\n", 0, m.start()) + 1
        prefijo = texto[inicio_linea:m.start()].strip()
        if prefijo.count("(") > prefijo.count(")"):
            # Remisión dentro de una nota editorial de la fuente, no una cabecera:
            # "(Ver Ley 388 de 1997; Art. 1.; Art. 6.)".
            continue
        alineado = not prefijo or prefijo in {'"', '“', '«'}
        pegado = bool(prefijo and prefijo[-1] in ".;:!?" and m.group("etiqueta")[0].isupper())
        if not alineado and not pegado and not m.group("comilla"):
            continue
        siguiente = texto[m.end():m.end() + 160]
        puntuado = bool(re.match(r"^\s*[.°ºª:;–—-]", siguiente)) or bool(re.search(r"[º°ª]$", m.group("numero")))
        primera = siguiente.lstrip(" \t")
        titulo = primera.lstrip("\n \t")
        titulo_sin_punto = bool(titulo[:1].isupper()) and not re.match(r"^(?:DE|DEL|EN|QUE|SE|Y|O|A|POR)\b", titulo)
        if not puntuado and not (alineado and titulo_sin_punto):
            continue
        transitorio = bool(m.group("transitorio")) or numero == "transitorio" or bool(re.match(r"^[ .:-]*transitorio\b", siguiente, re.I))
        candidatos.append({"inicio": m.start(), "fin_encabezado": m.end(), "numero": numero,
                           "transitorio": transitorio, "comilla": bool(m.group("comilla")),
                           "multilinea": "\n" in m.group(), "pegado": pegado,
                           "sin_punto": not puntuado, "encabezado": m.group()})
    return candidatos


def _lineas(texto):
    return [(m.start(), m.end(), m.group().strip()) for m in re.finditer(r"[^\n]+(?:\n|$)", texto)]


def _unidad(inicio, tipo, articulo=None, seccion=None, avisos=None, candidato=None,
            version_fuente="no_verificada", apta=True):
    return {"inicio": inicio, "tipo": tipo, "articulo": articulo, "seccion": seccion,
            "avisos": list(avisos or []), "articulo_candidato": candidato,
            "articulos_referidos": [], "version_fuente": version_fuente,
            "apta_para_busqueda": apta}


def _siguiente(numero, anterior):
    if anterior is None:
        return numero in {"1", "unico", "transitorio"} or bool(re.fullmatch(r"(?:\d+\.)+1", numero))
    if numero.isdigit() and anterior.isdigit():
        return int(numero) == int(anterior) + 1
    if re.fullmatch(re.escape(anterior) + "A", numero):
        return True
    if re.fullmatch(r"\d+[A-Z]", anterior) and numero.isdigit():
        return int(numero) == int(anterior[:-1]) + 1
    if re.fullmatch(r"\d+[A-Z]", numero) and re.fullmatch(r"\d+[A-Z]", anterior):
        return numero[:-1] == anterior[:-1] and ord(numero[-1]) == ord(anterior[-1]) + 1
    if "." in numero and "." in anterior:
        a, b = numero.split("."), anterior.split(".")
        if all(n.isdigit() for n in a + b):
            return tuple(map(int, a)) > tuple(map(int, b))
    return False


def _unidades_norma(texto):
    candidatos = candidatos_articulo(texto)
    por_inicio = {c["inicio"]: c for c in candidatos}
    secciones = {a: linea for a, b, linea in _lineas(texto)
                 if len(linea) <= 160 and _SECCION_NORMA.fullmatch(linea)}
    unidades = [_unidad(0, "preambulo")]
    previos = {False: None, True: None}
    vistos = Counter()
    seccion, transitorios, previo_inicio = None, False, 0
    reforma_activa = False
    for inicio in sorted(set(por_inicio) | set(secciones)):
        contexto = texto[previo_inicio:inicio][-1800:]
        if inicio in secciones and inicio not in por_inicio:
            if reforma_activa and _INTRO_CITA.search(contexto):
                continue
            seccion = secciones[inicio]
            transitorios = transitorios or "transitori" in _sin_tildes(seccion)
            unidades.append(_unidad(inicio, "seccion", seccion=seccion))
            reforma_activa = False
            continue
        candidato = por_inicio[inicio]
        numero = candidato["numero"]
        transitorio = candidato["transitorio"] or transitorios
        previo = previos[transitorio]
        rotulo = ("transitorio " + numero) if transitorio and numero != "transitorio" else numero
        referencias = {_numero_articulo(n) for n in _REFERENCIA.findall(contexto)}
        introduccion = bool(_INTRO_CITA.search(contexto))
        historica = bool(re.search(r"(?:^|\n)Vigencia anterior\s*$", contexto, re.I))
        esperado = _siguiente(numero, previo)
        citado = (not historica and (candidato["comilla"] or
                  (numero in referencias and (introduccion or (reforma_activa and _REFORMA.search(contexto)))) or
                  (introduccion and reforma_activa and not esperado)))
        if citado and unidades[-1]["tipo"] == "articulo":
            unidades[-1]["articulos_referidos"].append(numero)
            unidades[-1]["avisos"].append("incluye_articulo_citado_o_reformado")
            previo_inicio = inicio
            continue
        avisos, articulo = [], rotulo
        version_fuente = "anterior_segun_fuente" if historica else "no_verificada"
        if historica:
            avisos.append("version_anterior_conservada")
        elif citado:
            articulo = None
            avisos.append("articulo_citado_sin_unidad_reformadora")
        elif not esperado:
            articulo = None
            avisos.append("atribucion_articulo_ambigua")
            if previo and previo.isdigit() and numero.isdigit() and 1 < int(numero) - int(previo) <= 5:
                previos[transitorio] = numero
        if articulo is not None:
            if not historica:
                previos[transitorio] = numero
            vistos[rotulo] += 1
            if vistos[rotulo] > 1:
                avisos.append("articulo_con_varias_ocurrencias")
        elif unidades[-1]["tipo"] == "articulo":
            unidades[-1]["avisos"].append("limite_siguiente_con_atribucion_ambigua")
        unidades.append(_unidad(inicio, "articulo" if articulo else "encabezado_ambiguo",
                                articulo, seccion, avisos, None if articulo else rotulo, version_fuente))
        fin_linea = texto.find("\n", candidato["fin_encabezado"])
        if fin_linea < 0:
            fin_linea = len(texto)
        apertura = texto[candidato["fin_encabezado"]:max(fin_linea, candidato["fin_encabezado"] + 250)]
        reforma_activa = bool(_REFORMA.search(apertura[:350]) or _INTRO_CITA.search(apertura[:350]))
        previo_inicio = inicio
    if not candidatos:
        unidades[0]["avisos"].append("sin_encabezados_de_articulo_reconocidos")
    return unidades


_NUMERAL_JUDICIAL = re.compile(r"^(?:\d{1,4}(?:\.\d+)*[.)-]|[IVXLCDM]+[.)])(?:\s|$)")
_ORDEN_FALLO = re.compile(r"^(?:PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO)[.:-](?:\s|$)", re.I)
_TITULO_FALLO = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)-]?\s+)?(?:EL |LOS |LA |LAS )?"
    r"(?:ANTECEDENTES|CONSIDERACIONES(?: DE LA (?:CORTE|SALA))?|COMPETENCIA|"
    r"FUNDAMENTOS JUR[ÍI]DICOS|PROBLEMAS? JUR[ÍI]DICOS?|RESUELVE|DECISI[ÓO]N|FALLA|"
    r"S[ÍI]NTESIS|ANEXOS?(?:\s+\d+)?|(?:SALVAMENTO|ACLARACI[ÓO]N)(?: PARCIAL)? DE VOTO(?: DEL? .*)?)$", re.I,
)


def _titulo_judicial(lineas, indice):
    """Combina solo rótulos partidos, no un párrafo completo para fabricar un título."""
    a, b, linea = lineas[indice]
    candidatos = [(linea, b, 1)]
    if len(linea) <= 45 and indice + 1 < len(lineas):
        otra = lineas[indice + 1][2]
        if len(otra) <= 100:
            candidatos.insert(0, (linea + " " + otra, lineas[indice + 1][1], 2))
    for titulo, fin, consumidas in candidatos:
        if not _TITULO_FALLO.fullmatch(titulo):
            continue
        # «Consideraciones\nsimilares deben...» es una oración, no una sección.
        siguiente = lineas[indice + consumidas][2] if indice + consumidas < len(lineas) else ""
        if _sin_tildes(titulo) == "consideraciones" and siguiente[:1].islower():
            continue
        if re.search(r"\ben rad\b|\ben rid\b|\ben red\b", titulo, re.I):
            continue
        return titulo, fin, consumidas
    return None


def _limites_indice(texto, lineas):
    for indice, (a, b, linea) in enumerate(lineas):
        if a > 25000:
            break
        if _sin_tildes(linea).strip(" .:") not in {"contenido", "contenido de la providencia", "tabla de contenido", "indice"}:
            continue
        siguientes = lineas[indice + 1:indice + 80]
        finales = [(x, y, t) for x, y, t in siguientes if re.search(r"\s\d{1,4}$", t)]
        if len(finales) >= 3:
            return a, finales[-1][1]
    return None


def _unidades_jurisprudencia(texto):
    lineas = _lineas(texto)
    unidades = [_unidad(0, "preambulo")]
    indice_rango = _limites_indice(texto, lineas)
    seccion, saltar_hasta = None, -1
    if indice_rango:
        unidades.append(_unidad(indice_rango[0], "indice", seccion="Índice", apta=False))
        unidades.append(_unidad(indice_rango[1], "bloque_jurisprudencia"))
    for indice, (a, b, linea) in enumerate(lineas):
        if a < saltar_hasta or _MARCADOR_PAGINA.fullmatch(linea):
            continue
        if indice_rango and indice_rango[0] <= a < indice_rango[1]:
            continue
        titulo = _titulo_judicial(lineas, indice)
        if titulo:
            seccion, saltar_hasta, consumidas = titulo
            unidades.append(_unidad(a, "seccion", seccion=seccion))
        elif _NUMERAL_JUDICIAL.match(linea) or _ORDEN_FALLO.match(linea):
            unidades.append(_unidad(a, "parrafo", seccion=seccion))
    unidades = sorted({u["inicio"]: u for u in unidades}.values(), key=lambda u: u["inicio"])
    # Un título sin cuerpo y una enumeración breve se conservan con su continuación.
    fusionadas = []
    for indice, u in enumerate(unidades):
        fin = unidades[indice + 1]["inicio"] if indice + 1 < len(unidades) else len(texto)
        siguiente = unidades[indice + 1] if indice + 1 < len(unidades) else None
        corto = len(texto[u["inicio"]:fin].strip()) < 80
        if corto and siguiente and siguiente["tipo"] in {"parrafo", "bloque_jurisprudencia"} and u["tipo"] in {"parrafo", "seccion"}:
            siguiente["inicio"] = u["inicio"]
            if u["tipo"] == "seccion":
                siguiente["tipo"], siguiente["seccion"] = "seccion", u["seccion"]
            continue
        fusionadas.append(u)
    # Un bloque técnico no afirma ser un párrafo original. El artículo permanece nulo.
    limitadas = []
    for indice, u in enumerate(fusionadas):
        fin = fusionadas[indice + 1]["inicio"] if indice + 1 < len(fusionadas) else len(texto)
        if fin - u["inicio"] <= MAX_BLOQUE_JUDICIAL or u["tipo"] == "indice":
            limitadas.append(u)
            continue
        puntos = list(_cortes_sin_solapar(texto, u["inicio"], fin, MAX_BLOQUE_JUDICIAL))
        for numero, (a, b) in enumerate(puntos):
            nueva = {**u, "inicio": a, "avisos": list(u["avisos"])}
            nueva["tipo"] = "bloque_jurisprudencia"
            nueva["division_tecnica"] = True
            limitadas.append(nueva)
    return limitadas


def _cortes_sin_solapar(texto, inicio, fin, limite):
    while inicio < fin:
        corte = min(inicio + limite, fin)
        if corte < fin:
            tramo = texto[inicio + limite // 2:corte]
            finales = list(re.finditer(r"[.!?]\s*\n|\n", tramo))
            if finales:
                corte = inicio + limite // 2 + finales[-1].end()
        yield inicio, corte
        inicio = corte


def _ventanas(texto, inicio, fin, max_chars, solapamiento):
    while inicio < fin and texto[inicio].isspace():
        inicio += 1
    while fin > inicio and texto[fin - 1].isspace():
        fin -= 1
    while inicio < fin:
        corte = min(inicio + max_chars, fin)
        if corte < fin:
            minimo = inicio + max(max_chars // 2, solapamiento + 1)
            alternativas = [texto.rfind(s, minimo, corte + 1) for s in ("\n", ". ", "; ", " ")]
            if max(alternativas) > inicio:
                corte = max(alternativas)
        yield inicio, corte
        if corte >= fin:
            break
        siguiente = max(inicio + 1, corte - solapamiento)
        if siguiente > 0 and not texto[siguiente - 1].isspace():
            espacio = re.search(r"\s", texto[siguiente:corte])
            if espacio:
                siguiente += espacio.end()
        inicio = siguiente


def _origenes(documento, inicio, fin):
    return [{"archivo": p["archivo"], "url": p["url"], "pagina": p["pagina"],
             "archivo_texto_derivado": p.get("archivo_texto_derivado"),
             "inicio": max(inicio, p["inicio"]), "fin": min(fin, p["fin"])}
            for p in documento.get("partes", []) if max(inicio, p["inicio"]) < min(fin, p["fin"])]


def familia_documental(documento):
    tipo = _sin_tildes(str(documento.get("tipo", ""))).strip()
    if tipo in {"sentencia", "auto", "providencia", "fallo"}:
        return "jurisprudencia"
    if tipo in {"ley", "decreto", "acto_legislativo", "constitucion", "acuerdo", "resolucion", "decision"}:
        return "norma"
    return "documental"


def _unidades_documentales(texto):
    # Conceptos, circulares, compendios y tipos desconocidos pueden transcribir
    # artículos de terceros. Se conserva la referencia sin atribuirles autoría.
    unidades = _unidades_jurisprudencia(texto)
    for indice, unidad in enumerate(unidades):
        if unidad["tipo"] in {"preambulo", "bloque_jurisprudencia"}:
            unidad["tipo"] = "bloque_documental"
        fin = unidades[indice + 1]["inicio"] if indice + 1 < len(unidades) else len(texto)
        unidad["articulos_referidos"] = list(dict.fromkeys(_REFERENCIA.findall(texto[unidad["inicio"]:fin])))
    return unidades


def segmentar_documento(documento, texto, max_chars=MAX_CARACTERES, solapamiento=SOLAPAMIENTO):
    """Fragmentos exactos del texto, con atribución conservadora y unidades recuperables."""
    if not isinstance(max_chars, int) or not isinstance(solapamiento, int) or not 0 <= solapamiento < max_chars:
        raise ValueError("Se requiere 0 <= solapamiento < max_chars, ambos enteros")
    if not documento.get("doc_id"):
        raise ValueError("Falta doc_id")
    if not texto.strip():
        return []
    familia = familia_documental(documento)
    funciones = {"jurisprudencia": _unidades_jurisprudencia, "norma": _unidades_norma,
                 "documental": _unidades_documentales}
    unidades = funciones[familia](texto)
    unidades = sorted({u["inicio"]: u for u in unidades}.values(), key=lambda u: u["inicio"])
    campos = ("doc_id", "titulo", "tipo", "numero", "anio", "organo_emisor", "vigencia", "fuente", "url",
              "areas", "estado_extraccion", "revision_juridica", "texto_archivo", "redistribuir_raw",
              "edicion_con_anotaciones", "licencia_fuente", "fecha_consulta", "version_ingesta",
              "estado_fuente", "rechazos_fuente", "nivel", "vigencia_fuente", "origen_ampliacion",
              "alcance_vigencia", "temas", "advertencias_preliminares_fuente", "actualizacion_declarada_fuente")
    fragmentos = []
    for indice, u in enumerate(unidades):
        fin_unidad = unidades[indice + 1]["inicio"] if indice + 1 < len(unidades) else len(texto)
        for parte, (a, b) in enumerate(_ventanas(texto, u["inicio"], fin_unidad, max_chars, solapamiento), 1):
            trozo = texto[a:b]
            if not _MARCADOR_PAGINA.sub("", trozo).strip():
                continue
            origenes = _origenes(documento, a, b)
            avisos = sorted(set(u["avisos"]))
            if not origenes:
                avisos.append("sin_origen_solapado")
            huella = sha256(trozo.encode("utf-8"))
            orden = len(fragmentos) + 1
            fragmentos.append({**{k: documento.get(k) for k in campos},
                "fragmento_id": f"{documento['doc_id']}__{orden:06d}__{huella[:12]}", "orden": orden,
                "texto": trozo, "inicio": a, "fin": b, "sha256_texto": huella,
                "sha256_documento": documento.get("sha256_texto"), "articulo": u["articulo"],
                "articulo_candidato": u["articulo_candidato"], "articulos_referidos": sorted(set(u["articulos_referidos"])),
                "unidad_id": f"{documento['doc_id']}__u{indice + 1:05d}", "unidad_tipo": u["tipo"],
                "unidad_inicio": u["inicio"], "unidad_fin": fin_unidad, "parte": parte,
                "seccion": u["seccion"], "version_fuente": u["version_fuente"],
                "familia_documental": familia,
                "division_tecnica": u.get("division_tecnica", False),
                "apta_para_busqueda": u["apta_para_busqueda"] and documento.get("apta_para_busqueda", True),
                "origenes": origenes, "archivos": list(dict.fromkeys(o["archivo"] for o in origenes)),
                "paginas": sorted({o["pagina"] for o in origenes if o["pagina"] is not None}),
                "estado_segmentacion": "revisar" if any(a != "incluye_articulo_citado_o_reformado" for a in avisos) else "segmentado",
                "avisos": sorted(set(list(documento.get("avisos", [])) + avisos))})
    return fragmentos


def unidades_de_fragmentos(documento, texto, fragmentos):
    resultado, vistos = [], set()
    for f in fragmentos:
        if f["unidad_id"] in vistos:
            continue
        vistos.add(f["unidad_id"])
        a, b = f["unidad_inicio"], f["unidad_fin"]
        unidad = {k: v for k, v in f.items() if k not in {
            "fragmento_id", "orden", "parte", "sha256_texto", "inicio", "fin", "texto",
            "unidad_inicio", "unidad_fin", "unidad_tipo", "archivos", "paginas", "origenes"}}
        origenes = _origenes(documento, a, b)
        unidad.update(tipo_unidad=f["unidad_tipo"], inicio=a, fin=b, texto=texto[a:b],
                      origenes=origenes, paginas=sorted({p["pagina"] for p in origenes if p["pagina"] is not None}),
                      sha256_texto=sha256(texto[a:b].encode("utf-8")))
        resultado.append(unidad)
    return resultado


def diagnosticar_continuidad(documento, texto, unidades):
    """Cuenta candidatos y discontinuidades. No estima artículos jurídicos esperados."""
    candidatos = candidatos_articulo(texto) if familia_documental(documento) == "norma" else []
    propios = [u for u in unidades if u["articulo"] is not None]
    limites = {u["inicio"] for u in unidades}
    sin_explicar = []
    indice = 0
    for c in candidatos:
        while indice + 1 < len(unidades) and unidades[indice]["fin"] <= c["inicio"]:
            indice += 1
        u = unidades[indice] if unidades else None
        if c["inicio"] not in limites and (u is None or c["numero"] not in u["articulos_referidos"]):
            sin_explicar.append(c["inicio"])
    simples = sorted({int(u["articulo"]) for u in propios if u["articulo"].isdigit()})
    saltos = [[a, b] for a, b in zip(simples, simples[1:]) if b > a + 1]
    return {"doc_id": documento["doc_id"], "candidatos": len(candidatos),
            "candidatos_multilinea": sum(c["multilinea"] for c in candidatos),
            "candidatos_pegados": sum(c["pegado"] for c in candidatos),
            "candidatos_sin_punto": sum(c["sin_punto"] for c in candidatos),
            "unidades_con_articulo": len(propios),
            "articulos_distintos": len({u["articulo"] for u in propios}),
            "versiones_anteriores": sum(u["version_fuente"] == "anterior_segun_fuente" for u in unidades),
            "encabezados_ambiguos": sum(u["tipo_unidad"] == "encabezado_ambiguo" for u in unidades),
            "candidatos_sin_limite_o_cita": sin_explicar, "saltos_simples_por_revisar": saltos}


def comprobar_resultados(registros, textos, unidades, fragmentos, max_chars=MAX_CARACTERES):
    filas = []
    def comprobar(nombre, valor):
        filas.append({"comprobacion": nombre, "resultado": "ok" if valor else "fallo"})
    por_doc, por_unidad = defaultdict(list), {u["unidad_id"]: u for u in unidades}
    for f in fragmentos:
        por_doc[f["doc_id"]].append(f)
    comprobar("Extracción de todos los documentos", all(r["estado_extraccion"] != "error" for r in registros) and len(registros) == len(textos))
    comprobar("Todos los documentos tienen fragmentos", set(por_doc) == set(textos))
    comprobar("Identificadores únicos", len({f["fragmento_id"] for f in fragmentos}) == len(fragmentos) and len(por_unidad) == len(unidades))
    comprobar("Substrings exactos y límites de ventana", all(f["texto"] == textos[f["doc_id"]][f["inicio"]:f["fin"]] and 0 < len(f["texto"]) <= max_chars for f in fragmentos))
    comprobar("Ventanas dentro de su unidad", all(f["unidad_inicio"] <= f["inicio"] < f["fin"] <= f["unidad_fin"] for f in fragmentos))
    comprobar("Unidades exactas", all(u["texto"] == textos[u["doc_id"]][u["inicio"]:u["fin"]] for u in unidades))
    unidades_por_doc = defaultdict(list)
    for u in unidades:
        unidades_por_doc[u["doc_id"]].append(u)
    comprobar("Unidades sin solapamiento", all(a["fin"] <= b["inicio"] for us in unidades_por_doc.values()
               for a, b in zip(sorted(us, key=lambda u: u["inicio"]), sorted(us, key=lambda u: u["inicio"])[1:])))
    comprobar("Sin artículos propios en jurisprudencia", all(f["articulo"] is None for f in fragmentos if f["tipo"] in {"sentencia", "auto", "providencia", "fallo"}))
    comprobar("Bloques judiciales acotados", all(len(u["texto"]) <= MAX_BLOQUE_JUDICIAL for u in unidades if u["tipo"] in {"sentencia", "auto"} and u["tipo_unidad"] != "indice"))
    comprobar("Procedencia de todos los fragmentos", all(f["origenes"] for f in fragmentos))
    huecos = []
    for doc_id, grupos in por_doc.items():
        cursor, texto = 0, textos[doc_id]
        for f in sorted(grupos, key=lambda f: (f["inicio"], f["fin"])):
            if f["inicio"] > cursor and _MARCADOR_PAGINA.sub("", texto[cursor:f["inicio"]]).strip():
                huecos.append((doc_id, cursor, f["inicio"]))
            cursor = max(cursor, f["fin"])
        if _MARCADOR_PAGINA.sub("", texto[cursor:]).strip():
            huecos.append((doc_id, cursor, len(texto)))
    comprobar("Cobertura salvo blancos y marcadores de página", not huecos)
    for doc_id, n in [("co_decreto_306_1992", 10), ("ley_1032_2006", 5), ("ley_1581_2012", 30), ("ley_2445_2025", 45)]:
        if doc_id in textos:
            arts = [u["articulo"] for u in unidades if u["doc_id"] == doc_id and u["articulo"] is not None]
            comprobar(f"{doc_id}: {n} artículos revisados en el original", len(arts) == n and set(arts) == {str(i) for i in range(1, n + 1)})
    if "ley_1032_2006" in textos:
        art3 = [u for u in unidades if u["doc_id"] == "ley_1032_2006" and u["articulo"] == "3"]
        comprobar("Artículo 3 de Ley 1032 conserva páginas 2 y 3", len(art3) == 1 and {2, 3}.issubset(art3[0]["paginas"]))
    if "co_decreto_1083_2015" in textos:
        comprobar("Decreto 1083 conserva jerarquías sin punto", any(u["doc_id"] == "co_decreto_1083_2015" and u["articulo"] == "2.1.1.1" for u in unidades))
        comprobar("Decreto 1083 conserva versión anterior explícita", any(u["doc_id"] == "co_decreto_1083_2015" and u["articulo"] == "2.2.1.3.6" and u["version_fuente"] == "anterior_segun_fuente" for u in unidades))
    if "sentencia_cc_c029_2009" in textos:
        comprobar("C029 no separa el título 2. El", not any(u["doc_id"] == "sentencia_cc_c029_2009" and u["texto"].strip() == "2. El" for u in unidades))
    if "sentencia_cc_c055_2022" in textos:
        comprobar("C055 separa anexos", any(u["doc_id"] == "sentencia_cc_c055_2022" and (u["seccion"] or "").upper() == "ANEXO 1" for u in unidades))
    if "ley_599_2000" in textos:
        comprobar("Marcas de tachado conservadas", "[TEXTO TACHADO EN LA FUENTE:" in textos["ley_599_2000"])
    return filas


def ejecutar_ingesta(raiz, modo="corpus", ids_muestra=None, max_chars=MAX_CARACTERES, solapamiento=SOLAPAMIENTO,
                     raw_dir=None):
    raiz = Path(raiz)
    raw = Path(raw_dir) if raw_dir is not None else raiz / "data/raw"
    if raw_dir is not None and not raw.is_absolute():
        raw = raiz / raw
    manifest_bytes = (raw / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    ids = [d["doc_id"] for d in manifest]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
        raise ValueError("Identificadores de manifiesto vacíos, repetidos o no válidos")
    if modo not in {"corpus", "muestra"}:
        raise ValueError("Modo no admitido")
    if modo == "muestra" and (not ids_muestra or set(ids_muestra) - set(ids)):
        raise ValueError("Muestra vacía o con identificadores desconocidos")
    seleccion = sorted([d for d in manifest if modo == "corpus" or d["doc_id"] in ids_muestra], key=lambda d: d["doc_id"])
    registros, textos, unidades, fragmentos, continuidad = [], {}, [], [], []
    for posicion, documento in enumerate(seleccion, 1):
        try:
            registro, texto = procesar_documento(documento, raw)
            textos[documento["doc_id"]] = texto
            segmentos = segmentar_documento(registro, texto, max_chars, solapamiento)
            propias = unidades_de_fragmentos(registro, texto, segmentos)
            diagnostico = diagnosticar_continuidad(registro, texto, propias)
            registro["avisos_segmentacion"] = sorted({a for u in propias for a in u["avisos"] if a not in registro["avisos"]})
            continuidad.append(diagnostico)
            unidades.extend(propias)
            fragmentos.extend(segmentos)
        except Exception as error:
            registro = {**documento, "version_ingesta": VERSION_INGESTA, "estado_extraccion": "error",
                        "revision_juridica": "pendiente", "avisos": ["error_extraccion_o_segmentacion"],
                        "texto_archivo": None, "sha256_texto": None, "caracteres": 0,
                        "partes": [], "extraccion": [], "error": f"{type(error).__name__}: {error}"}
        registros.append(registro)
        if posicion % 25 == 0 or posicion == len(seleccion):
            print(f"Procesados: {posicion}/{len(seleccion)}", flush=True)
    comprobaciones = comprobar_resultados(registros, textos, unidades, fragmentos, max_chars)
    comprobaciones.append({"comprobacion": "Candidatos normativos con límite o cita trazable",
                           "resultado": "ok" if all(not d["candidatos_sin_limite_o_cita"] for d in continuidad) else "fallo"})
    avisos = Counter(a for u in unidades for a in u["avisos"])
    resumen = {"version_ingesta": VERSION_INGESTA, "modo": modo, "sha256_manifest": sha256(manifest_bytes),
               "sha256_parser": sha256(Path(__file__).read_bytes()), "documentos": len(registros),
               "textos_guardados": len(textos), "estados_extraccion": dict(Counter(r["estado_extraccion"] for r in registros)),
               "unidades": len(unidades), "unidades_por_tipo": dict(Counter(u["tipo_unidad"] for u in unidades)),
               "fragmentos": len(fragmentos), "evidencia": "unidad completa; ventanas para localización",
               "fragmentos_por_estado": dict(Counter(f["estado_segmentacion"] for f in fragmentos)),
               "fragmentos_con_avisos": sum(bool(f["avisos"]) for f in fragmentos),
               "documentos_con_avisos_unidades": len({u["doc_id"] for u in unidades if u["avisos"]}),
               "avisos_unidades": dict(avisos),
               "documentos_fuente_restringida": [r["doc_id"] for r in registros if r.get("apta_para_busqueda") is False],
               "entidades_html_reparadas": sum(e.get("entidades_reparadas", 0) for r in registros for e in r["extraccion"]),
               "cabeceras_editoriales_retiradas": sum(len(e.get("cabeceras_retiradas", [])) for r in registros for e in r["extraccion"]),
               "parametros_segmentacion": {"max_caracteres": max_chars, "solapamiento": solapamiento, "max_bloque_judicial": MAX_BLOQUE_JUDICIAL},
               "comprobaciones_correctas": sum(c["resultado"] == "ok" for c in comprobaciones),
               "comprobaciones_totales": len(comprobaciones),
               "dependencias": {p: version(p) for p in ["beautifulsoup4", "ftfy", "pdfplumber"]}}
    return {"registros": registros, "textos": textos, "unidades": unidades, "fragmentos": fragmentos,
            "continuidad": continuidad, "comprobaciones": comprobaciones, "resumen": resumen}


def _escribir_jsonl(ruta, filas):
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    with temporal.open("w", encoding="utf-8", newline="\n") as salida:
        for fila in filas:
            salida.write(json.dumps(fila, ensure_ascii=False, sort_keys=True) + "\n")
    temporal.replace(ruta)


def guardar_resultados(destino, resultado):
    if any(c["resultado"] != "ok" for c in resultado["comprobaciones"]):
        raise ValueError("Las comprobaciones fallaron. No se exportan derivados.")
    destino = Path(destino)
    (destino / "textos").mkdir(parents=True, exist_ok=True)
    for doc_id, texto in sorted(resultado["textos"].items()):
        ruta = destino / "textos" / f"{doc_id}.txt"
        temporal = ruta.with_suffix(".txt.tmp")
        temporal.write_bytes(texto.encode("utf-8"))
        temporal.replace(ruta)
    for nombre, filas in [("documentos", resultado["registros"]), ("unidades", resultado["unidades"]),
                          ("fragmentos", ({**f, "recuperar_unidad_completa": True} for f in resultado["fragmentos"])),
                          ("continuidad", resultado["continuidad"])]:
        _escribir_jsonl(destino / f"{nombre}.jsonl", filas)
    conteo_u = Counter(u["doc_id"] for u in resultado["unidades"])
    conteo_f = Counter(f["doc_id"] for f in resultado["fragmentos"])
    reporte = [{"doc_id": r["doc_id"], "estado_extraccion": r["estado_extraccion"], "caracteres": r["caracteres"],
                "unidades": conteo_u[r["doc_id"]], "fragmentos": conteo_f[r["doc_id"]],
                "avisos": ", ".join(r["avisos"]), "avisos_segmentacion": ", ".join(r.get("avisos_segmentacion", [])),
                "error": r["error"]} for r in resultado["registros"]]
    for nombre, filas in [("reporte", reporte), ("comprobaciones", resultado["comprobaciones"])]:
        with (destino / f"{nombre}.csv").open("w", encoding="utf-8-sig", newline="") as salida:
            writer = csv.DictWriter(salida, fieldnames=list(filas[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(filas)
    ruta = destino / "resumen.json"
    temporal = ruta.with_suffix(".json.tmp")
    temporal.write_text(json.dumps(resultado["resumen"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporal.replace(ruta)
