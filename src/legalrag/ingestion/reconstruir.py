"""Reconstruye data/raw desde cero, eligiendo por documento la fuente más completa.

El corpus anterior tomó casi todas las normas del Gestor Normativo de la Función
Pública, que las publica en una sola página con formato irregular, y en varias
quedaron artículos por fuera (Ley 600 de 2000: 421 unidades de 536 artículos).
La Secretaría del Senado publica las normas largas completas, partidas en varias
páginas enlazadas ("ley_0600_2000_pr012.html"), y es fuente admitida por el reto.

Por cada documento del inventario (corpus_manifest.json + configs/corpus_objetivos.json):

1. Arma las fuentes candidatas: Senado para leyes, decretos y códigos, más la URL
   declarada en el inventario. Jurisprudencia: la URL declarada (relatorías).
2. Descarga cada candidata completa (sigue la cadena _prNNN de Senado) y extrae el
   texto con el mismo extractor de la ingesta.
3. Mide completitud: artículos propios distintos, huecos en la numeración y, en
   jurisprudencia, que el texto llegue a la parte resolutiva y las firmas.
4. Se queda con la candidata más completa y valida que el texto nombre la norma.

Escribe data/raw/<doc_id>/NNN.<ext>, data/raw/manifest.json (formato de la ingesta)
y data/raw/reconstruccion.csv con las métricas de todas las candidatas. Se puede
interrumpir y reanudar: un documento ya guardado con sus hashes no se descarga otra vez.

Solo fuentes oficiales colombianas. Los certificados de Función Pública y SUIN no
envían el intermedio de Sectigo; se agrega desde configs/certificados sin desactivar
la verificación TLS.

Uso: python -m legalrag.ingestion.reconstruir [--solo ID ...] [--forzar] [--hilos 4]
"""
import argparse
import csv
import hashlib
import html as html_lib
import json
import re
import socket
import ssl
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from legalrag.ingestion.ampliar import AREA_SLUG, LICENCIA  # noqa: E402
from legalrag.preprocessing.ingesta import candidatos_articulo, extraer_html, extraer_pdf  # noqa: E402
from legalrag.preprocessing.ocr import SEPARADOR_PAGINA, es_legible, ocr_pdf  # noqa: E402
from legalrag.preprocessing.ocr import disponible as ocr_disponible, metodo as metodo_ocr  # noqa: E402
from legalrag.preprocessing.word import es_docx, es_paquete, es_word, metodo as metodo_word, texto_paquete, texto_word  # noqa: E402

AGENTE = "Mozilla/5.0 (X11; Linux x86_64) ai-week-2026 corpus (uso académico)"
SENADO = "http://www.secretariasenado.gov.co/senado/basedoc"
PAUSA_HOST = 1.0
# Senado responde hoy en ~0,05 s: una pausa menor no le pesa y acelera el barrido completo.
PAUSA_POR_SERVIDOR = {"www.secretariasenado.gov.co": 0.3}
MAX_TRAMOS = 300  # el Código Civil en Senado pasa de 80 tramos
TZ = timezone(timedelta(hours=-5))

# Códigos que Senado publica con nombre propio y no como ley_NNNN_AAAA.
SENADO_CODIGOS = {
    "co_constitucion_1_1991": "constitucion_politica_1991.html",
    "ley_84_1873": "codigo_civil.html",
    "decreto_410_1971": "codigo_comercio.html",
    "decreto_624_1989": "estatuto_tributario.html",
    "decreto_2663_1950": "codigo_sustantivo_trabajo.html",
}
NOMBRE_CODIGO = {
    "ley_84_1873": "codigo civil", "decreto_410_1971": "codigo de comercio",
    "decreto_624_1989": "estatuto tributario", "decreto_2663_1950": "codigo sustantivo del trabajo",
}
# Códigos que el extractor oficial (citations.CODES) reconoce solo por su nombre, no por
# "Decreto 2663 de 1950": el título debe nombrarlos para que la identidad del documento
# incluya el cuerpo del código y las citas al código cuenten como respaldadas.
TITULO_CANONICO = {
    "decreto_2663_1950": "Código Sustantivo del Trabajo",
    "decreto_2158_1948": "Código Procesal del Trabajo y de la Seguridad Social",
    "decreto_410_1971": "Código de Comercio",
    "ley_84_1873": "Código Civil",
}
ORGANO = {"ley": "Congreso de la República", "decreto": "Presidencia de la República",
          "acto_legislativo": "Congreso de la República",
          "constitucion": "Asamblea Nacional Constituyente"}
# Marcas de cierre de una providencia: parte resolutiva y firmas.
CIERRE_JUDICIAL = re.compile(r"(?i)(notif[ií]quese|c[oó]piese|c[uú]mplase|comun[ií]quese|magistrad[oa]|"
                             r"presidente|secretari[oa] general|aclaraci[oó]n de voto|salvamento de voto)")

RESUELVE = re.compile(r"R\s?E\s?S\s?U\s?E\s?L\s?V\s?E|\bresuelve\b|\bFALLA\b|\bDECIDE\b")

_locks, _ultimo, _cupos = {}, {}, {}
# Concurrencia solo donde sirve: Senado tarda 15-40 s por respuesta. A la relatoría de la
# Corte Constitucional, con 3 simultáneas, le bastaron unos minutos para cortarnos el TLS.
MAX_SIMULTANEAS = 1
# ver candidatas(): sentencias C impares primero al Senado; "todas" = todas las C (p. ej. cuando la
# relatoría deja de atender). --cc-senado lo activa desde la línea de comandos.
REPARTIR_CC = True
SIMULTANEAS_POR_SERVIDOR = {"www.secretariasenado.gov.co": 3}
_lock_global = threading.Lock()


def contexto_tls():
    contexto = ssl.create_default_context()
    for pem in sorted((RAIZ / "configs/certificados").glob("*.pem")):
        contexto.load_verify_locations(cafile=str(pem))
    return contexto


TLS = contexto_tls()

# Varios servidores oficiales (Supersociedades, Corte Suprema) anuncian IPv6 pero no lo
# atienden: urllib prueba las direcciones en orden y espera el timeout completo en cada una.
# IPv4 primero; IPv6 queda de respaldo.
_getaddrinfo = socket.getaddrinfo


def _ipv4_primero(*args, **kwargs):
    return sorted(_getaddrinfo(*args, **kwargs), key=lambda d: d[0] != socket.AF_INET)


socket.getaddrinfo = _ipv4_primero


def pedir(url, reintentos=2):
    """GET cortés por servidor. Devuelve (estado, url_final, content_type, bytes).

    A lo sumo MAX_SIMULTANEAS peticiones en curso por servidor y al menos PAUSA_HOST
    segundos entre inicios. Serializar del todo hacía que un servidor lento (Senado,
    17 s por respuesta) atendiera una petición cada 17 s sin aliviarle la carga.
    """
    host = urllib.parse.urlsplit(url).netloc
    with _lock_global:
        cupo = _cupos.setdefault(host, threading.BoundedSemaphore(SIMULTANEAS_POR_SERVIDOR.get(host, MAX_SIMULTANEAS)))
        ritmo = _locks.setdefault(host, threading.Lock())
    with cupo:
        with ritmo:
            espera = PAUSA_POR_SERVIDOR.get(host, PAUSA_HOST) - (time.monotonic() - _ultimo.get(host, 0))
            if espera > 0:
                time.sleep(espera)
            _ultimo[host] = time.monotonic()
        peticion = urllib.request.Request(url, headers={"User-Agent": AGENTE})
        for intento in range(reintentos + 1):
            try:
                with urllib.request.urlopen(peticion, timeout=120, context=TLS) as r:
                    return r.status, r.geturl(), r.headers.get("Content-Type", ""), r.read()
            except urllib.error.HTTPError as error:
                return error.code, url, "", b""
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if intento == reintentos:
                    return 0, url, str(error), b""
                time.sleep(4 * (intento + 1))


def _decodificar(contenido):
    for codificacion in ("utf-8", "windows-1252", "latin-1"):
        try:
            return contenido.decode(codificacion)
        except UnicodeDecodeError:
            continue
    return contenido.decode("latin-1", errors="replace")


def descargar_con_tramos(url):
    """Página base y, en Senado, las continuaciones _prNNN en orden de lectura."""
    estado, final, tipo, contenido = pedir(url)
    if estado != 200 or not contenido:
        return None, f"HTTP {estado}" + (f" ({tipo})" if estado == 0 else "")
    partes = [{"url": url, "url_final": final, "tipo_contenido": tipo, "contenido": contenido}]
    if "secretariasenado.gov.co" not in url:
        return partes, "ok"
    base, archivo = url.rsplit("/", 1)
    raiz_nombre = re.sub(r"(_pr\d+)?\.html?$", "", archivo, flags=re.I)
    patron = re.compile(re.escape(raiz_nombre) + r"_pr\d+\.html?", re.I)
    vistos, actual = {archivo.lower()}, contenido
    while len(partes) < MAX_TRAMOS:
        siguiente = next((n for n in patron.findall(_decodificar(actual)) if n.lower() not in vistos), None)
        if siguiente is None:
            return partes, "ok"
        vistos.add(siguiente.lower())
        estado, final, tipo, actual = pedir(f"{base}/{siguiente}")
        if estado != 200 or len(actual) < 1000:
            return partes, f"cadena cortada en {siguiente} (HTTP {estado})"
        partes.append({"url": f"{base}/{siguiente}", "url_final": final, "tipo_contenido": tipo, "contenido": actual})
    return partes, f"cadena de más de {MAX_TRAMOS} tramos"


def texto_de(partes):
    """Texto con el mismo extractor de la ingesta."""
    trozos = []
    for parte in partes:
        contenido = parte["contenido"]
        if parte.get("derivado") is not None:
            # Escaneo o Word: el texto sale del derivado guardado junto al original.
            trozos.append(parte["derivado"]["texto"].replace(SEPARADOR_PAGINA, "\n"))
            continue
        if contenido[:1024].find(b"%PDF-") >= 0:
            bloques, _ = extraer_pdf(contenido)
        else:
            bloques, _ = extraer_html(contenido, {"archivo": "x.html"})
        trozos.extend(b["texto"] for b in bloques)
    return "\n".join(trozos)


def metricas(texto, judicial):
    fila = {"caracteres": len(texto), "articulos": 0, "max_articulo": 0, "huecos": 0, "cierre": None}
    if judicial:
        # La parte resolutiva suele quedar antes de las notas al pie, que en la relatoría
        # ocupan el final del documento: basta con "RESUELVE" en cualquier parte.
        cola = texto[int(len(texto) * 0.8):]
        fila["cierre"] = bool(CIERRE_JUDICIAL.search(cola) or RESUELVE.search(texto))
        return fila
    numeros = set()
    for c in candidatos_articulo(texto):
        if c["comilla"] or c["transitorio"]:
            continue
        m = re.match(r"\d+", str(c["numero"]))
        if m and int(m.group()) < 5000:
            numeros.add(int(m.group()))
    if numeros:
        fila.update(articulos=len(numeros), max_articulo=max(numeros),
                    huecos=len(set(range(1, max(numeros) + 1)) - numeros))
    return fila


DIAS_EN_LETRAS = [None, "uno|primero", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
                  "once", "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho",
                  "diecinueve", "veinte", "veintiuno|veintiun", "veintidos", "veintitres", "veinticuatro",
                  "veinticinco", "veintiseis", "veintisiete", "veintiocho", "veintinueve", "treinta",
                  "treinta y uno|treinta y un"]
DIAS_EN_LETRAS = [None] + [f"(?:{d})" for d in DIAS_EN_LETRAS[1:]]


def anio_en_letras(anio):
    """1998 -> "mil novecientos noventa y ocho"; 2012 -> "dos mil doce" (sin tildes)."""
    unidades = ["", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once",
                "doce", "trece", "catorce", "quince", "dieciseis", "diecisiete", "dieciocho", "diecinueve",
                "veinte", "veintiuno", "veintidos", "veintitres", "veinticuatro", "veinticinco", "veintiseis",
                "veintisiete", "veintiocho", "veintinueve"]
    decenas = {3: "treinta", 4: "cuarenta", 5: "cincuenta", 6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"}
    resto = anio % 100
    cola = unidades[resto] if resto < 30 else decenas[resto // 10] + (f" y {unidades[resto % 10]}" if resto % 10 else "")
    base = "dos mil" if anio >= 2000 else "mil novecientos"
    return f"{base} {cola}".strip()


def identidad_valida(texto, ficha):
    """El texto nombra la norma que se pidió (evita portadas y páginas de error con 200)."""
    plano = re.sub(r"\s+", " ", texto[:80000]).lower()
    plano = plano.translate(str.maketrans("áéíóúü", "aeiouu"))
    tipo, numero, anio = ficha.get("tipo"), str(ficha.get("numero") or ""), str(ficha.get("anio") or "")
    if tipo == "acto_legislativo" and numero and anio:
        return bool(re.search(rf"\bacto legislativo\b[^\d]{{0,25}}0*{int(numero)}\s+del?\s+(\d+\s+de\s+\w+\s+de\s+)?{anio}", plano))
    if tipo in ("ley", "decreto") and numero and anio:
        # Senado intercala anotaciones en el encabezado: "DECRETO <LEY> 4334 DE 2008",
        # y a veces la fecha completa: "DECRETO LEY 2663 DEL 5 DE AGOSTO DE 1950".
        if re.search(rf"\b{tipo}\b[^\d]{{0,25}}0*{int(numero)}\s+del?\s+(\d+\s+de\s+\w+\s+de\s+)?{anio}", plano):
            return True
        nombre = NOMBRE_CODIGO.get(ficha["doc_id"])
        return bool(nombre and nombre in plano and str(int(numero)) in plano)
    if tipo == "constitucion":
        return "constitucion politica" in plano
    if tipo in ("sentencia", "auto") and numero:
        sala = re.match(r"[a-z]+", numero.lower())
        digitos = re.search(r"\d+", numero)
        if ficha.get("organo_emisor") not in ("Corte Constitucional", "Corte Suprema de Justicia"):
            # Consejo de Estado y superintendencias numeran por radicado ("2020CE-SUJ-4-005"):
            # basta con que aparezcan el año y los dígitos del número, con o sin separadores.
            compacto = re.sub(r"[^0-9]", "", plano)
            if ficha.get("organo_emisor") == "Consejo de Estado" and re.fullmatch(r"\d{5,8}", numero):
                # El listado de unificación no siempre trae radicado: el "número" es el id del
                # archivo en la relatoría, que no aparece en el cuerpo. Basta el año del fallo.
                return not anio or anio in plano
            return anio in plano and bool(digitos) and digitos.group().lstrip("0") in compacto
        antigua = re.fullmatch(r"(\d+)\s*\((\d{1,2})-(\d{1,2})-(\d{2,4})\)\s*[a-z]*", numero.strip().lower())
        if antigua and ficha.get("organo_emisor") == "Corte Suprema de Justicia":
            # Sala Penal anterior a 2014 ("21347(14-12-05).doc", "38311(13-08-12)imp.doc"): el
            # radicado suele ir solo en el nombre del archivo. Se valida la sala y la fecha
            # exacta, que a veces va solo en letras ("diecisiete de octubre de dos mil doce").
            dia = int(antigua.group(2))
            encabezado = plano[:6000]
            fecha_ok = (anio in encabezado or anio_en_letras(int(anio)) in encabezado) and bool(
                re.search(rf"\(0?{dia}\)|\b0?{dia}\s+de\s+[a-z]+\s+de|\b{DIAS_EN_LETRAS[dia]}\s+(\(0?{dia}\)\s+)?de\s+[a-z]+\s+de",
                          encabezado))
            return "casacion penal" in plano and fecha_ok
        if sala and digitos:
            if ficha.get("organo_emisor") == "Corte Suprema de Justicia" and \
                    re.search(rf"{int(digitos.group())}\s*-\s*{anio}\b", plano):
                # "SC8453-2016" aunque el OCR pegue ruido delante del sello ("SsC8453-2016").
                return True
            # "SU-917/10", "SU917/10" y "SU.917/10": la relatoría usa las tres.
            letra = re.fullmatch(r"[a-z]+\W*\d+([a-z]?)", numero.lower())
            letra = letra.group(1) if letra else ""
            return bool(re.search(rf"\b{sala.group()}\s*[-.]?\s*0*{int(digitos.group())}{letra}\b", plano))
        return bool(digitos and digitos.group().lstrip("0") in plano)
    return True


def ficha_base(doc):
    """Metadatos de la ingesta a partir del doc_id y del inventario."""
    doc_id = doc["doc_id"]
    ficha = {"doc_id": doc_id, "titulo": TITULO_CANONICO.get(doc_id) or doc.get("titulo") or doc_id,
             "fuente": doc.get("fuente"),
             "vigencia": "por_verificar",
             "alcance_vigencia": "Se preservan señales de la fuente. No se certifica vigencia ni consolidación completa.",
             "redistribuir_raw": True, "edicion_con_anotaciones": False, "licencia_fuente": LICENCIA,
             "advertencias_preliminares_fuente": [], "actualizacion_declarada_fuente": [], "temas": [],
             "areas": sorted({AREA_SLUG.get(a, a) for a in doc.get("areas") or []}),
             "origen_ampliacion": doc.get("origen", "reconstruccion_v06")}
    for campo in ("tipo", "numero", "anio", "organo_emisor"):
        if doc.get(campo) is not None:
            ficha[campo] = doc[campo]
    if "tipo" in ficha:
        return ficha
    m = re.fullmatch(r"(?:co_)?(ley|decreto|acto_legislativo)_(\d+)_(\d{4})", doc_id)
    if m:
        ficha.update(tipo=m.group(1), numero=m.group(2), anio=int(m.group(3)), organo_emisor=ORGANO[m.group(1)])
    elif doc_id == "co_constitucion_1_1991":
        ficha.update(tipo="constitucion", numero="1991", anio=1991, organo_emisor=ORGANO["constitucion"])
    elif (m := re.fullmatch(r"sentencia_cc_(c|t|su)(\d+)([a-z]?)_(\d{4})", doc_id)):
        ficha.update(tipo="sentencia", numero=f"{m.group(1).upper()}-{int(m.group(2)):03d}{m.group(3).upper()}",
                     anio=int(m.group(4)),
                     organo_emisor="Corte Constitucional")
    elif (m := re.fullmatch(r"auto_cc_a(\d+)_(\d{4})", doc_id)):
        ficha.update(tipo="auto", numero=f"A-{m.group(1)}", anio=int(m.group(2)), organo_emisor="Corte Constitucional")
    elif (m := re.fullmatch(r"sentencia_csj_([a-z]*)(\d+)_(\d{4})", doc_id)):
        ficha.update(tipo="sentencia", numero=f"{m.group(1).upper()}{m.group(2)}", anio=int(m.group(3)),
                     organo_emisor="Corte Suprema de Justicia")
    elif (m := re.fullmatch(r"sentencia_ce_(.+)_(\d{4})", doc_id)):
        ficha.update(tipo="sentencia", numero=m.group(1).upper(), anio=int(m.group(2)), organo_emisor="Consejo de Estado")
    elif (m := re.fullmatch(r"sentencia_sic_(\d+)_(\d{4})", doc_id)):
        ficha.update(tipo="sentencia", numero=m.group(1), anio=int(m.group(2)),
                     organo_emisor="Superintendencia de Industria y Comercio")
    elif (m := re.fullmatch(r"decision_(\d+)_(\d{4})", doc_id)):
        ficha.update(tipo="decision", numero=m.group(1), anio=int(m.group(2)),
                     organo_emisor="Comisión de la Comunidad Andina")
    else:
        ficha.update(tipo="documento", numero=None, anio=None, organo_emisor=None)
    return ficha


GESTOR_BUSQUEDA = ("https://www.funcionpublica.gov.co/dafpIndexerBGN/norma/index?find=FindNext"
                   "&filtroNumero={numero}&filtroAnio={anio}")
TIPO_GESTOR = {"ley": "Ley", "decreto": "Decreto", "acto_legislativo": "Acto Legislativo"}


def buscar_gestor(tipo, numero, anio):
    """URL, título y epígrafe de la norma en el buscador del Gestor Normativo, o None.

    Se consulta solo por número y año: con el filtro de tipo el buscador no devuelve
    los "Decreto Ley" ni los "Decreto Legislativo". Se acepta el resultado cuyo título
    es "<Tipo> [Ley|Legislativo] <número> de <año>"; el buscador devuelve también
    normas de otros tipos con el mismo número.
    """
    if tipo not in TIPO_GESTOR:
        return None
    url = GESTOR_BUSQUEDA.format(numero=int(numero), anio=anio)
    estado, _, _, contenido = pedir(url)
    if estado != 200:
        return None
    html = _decodificar(contenido)
    tipo_titulo = {"ley": r"Ley(?:\s+Estatutaria|\s+Org[aá]nica)?", "decreto": r"Decreto(?:\s+Ley|\s+Legislativo)?",
                   "acto_legislativo": r"Acto\s+Legislativo"}[tipo]
    for m in re.finditer(r'href="(https://www\.funcionpublica\.gov\.co/eva/gestornormativo/norma\.php\?i=\d+)"', html):
        bloque = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html[m.end():m.end() + 1500].split(">", 1)[-1])).strip()
        titulo = re.match(rf"(?i)({tipo_titulo}\s+0*{int(numero)}\s+de\s+{anio})\b(?:\s*-\s*[^A-ZÁÉÍÓÚ]*[\w ]*?Nacional)?\s*(.*)", bloque)
        if titulo:
            epigrafe = html_lib.unescape(titulo.group(2).split("$(")[0])
            epigrafe = re.sub(r"^-?\s*(Congreso de la República|Nivel Nacional)\s*", "", epigrafe).strip()
            return {"url": m.group(1), "titulo": titulo.group(1), "epigrafe": epigrafe[:400]}
    return None


def candidatas(doc, ficha):
    urls = []
    if doc["doc_id"] in SENADO_CODIGOS:
        urls.append(f"{SENADO}/{SENADO_CODIGOS[doc['doc_id']]}")
    elif ficha.get("tipo") in ("ley", "decreto") and ficha.get("numero"):
        urls.append(f"{SENADO}/{ficha['tipo']}_{int(ficha['numero']):04d}_{ficha['anio']}.html")
    elif ficha.get("organo_emisor") == "Corte Constitucional" and ficha.get("tipo") == "sentencia":
        # La relatoría es la fuente primaria; "C-055" -> "c-055-22", "SU-016" -> "su016-20".
        # Hay números con letra ("C-155A/93" -> "c-155a-93").
        sala, numero, letra = re.fullmatch(r"([A-Z]+)-(\d+)([A-Z]?)", ficha["numero"].upper()).groups()
        radical = f"{sala.lower()}{'' if sala == 'SU' else '-'}{int(numero):03d}{letra.lower()}-{str(ficha['anio'])[2:]}"
        urls.append(f"https://www.corteconstitucional.gov.co/relatoria/{ficha['anio']}/{radical}.htm")
        if REPARTIR_CC and sala == "C" and (int(numero) % 2 or REPARTIR_CC == "todas"):
            # La relatoría solo admite una petición a la vez; el Senado publica las mismas
            # sentencias C. Las impares van primero al Senado y la relatoría queda de respaldo.
            urls.insert(0, respaldos(ficha)[0])
    elif ficha.get("tipo") == "acto_legislativo" and ficha.get("numero"):
        urls.append(f"{SENADO}/acto_legislativo_{int(ficha['numero']):02d}_{ficha['anio']}.html")
    for url in [doc.get("url"), *(doc.get("urls_alternas") or [])]:
        if url and url not in urls:
            urls.append(url)
    if not any("funcionpublica" in u for u in urls) and ficha.get("tipo") in TIPO_GESTOR and ficha.get("numero"):
        # Senado no publica todos los decretos; el Gestor Normativo queda de respaldo.
        encontrada = buscar_gestor(ficha["tipo"], ficha["numero"], ficha["anio"])
        if encontrada:
            urls.append(encontrada["url"])
    return urls


# Compilaciones jurídicas oficiales con la misma estructura de URL (Avance Jurídico):
# docs/ley_0074_1968.htm. Se consultan solo si ninguna fuente principal fue válida.
COMPILACIONES = ("https://www.cancilleria.gov.co/normograma/compilacion/docs",
                 "https://normograma.dian.gov.co/dian/compilacion/docs",
                 # Cobertura histórica (leyes y decretos anteriores a 1992): Colpensiones y
                 # el normograma de la Cancillería publican normas que Senado y el Gestor no.
                 "https://normativa.colpensiones.gov.co/colpens/docs",
                 "https://www.cancilleria.gov.co/sites/default/files/Normograma/docs")


def respaldos(ficha):
    tipo, numero, anio = ficha.get("tipo"), ficha.get("numero"), ficha.get("anio")
    if tipo in ("ley", "decreto", "acto_legislativo") and numero:
        return [f"{base}/{tipo}_{int(numero):04d}_{anio}.htm" for base in COMPILACIONES]
    if tipo == "sentencia" and ficha.get("organo_emisor") == "Corte Constitucional" and numero:
        # Senado aloja parte de la jurisprudencia constitucional: basedoc/c-411_1993.html
        sala, n, letra = re.fullmatch(r"([A-Z]+)-(\d+)([A-Z]?)", numero.upper()).groups()
        return [f"{SENADO}/{sala.lower()}-{int(n):03d}{letra.lower()}_{anio}.html",
                f"{SENADO}/{sala.lower()}-{int(n)}{letra.lower()}_{anio}.html"]
    return []


def fuente_de(url, declarada):
    if "secretariasenado.gov.co" in url:
        return "Secretaría General del Senado - Base documental"
    if "suin-juriscol.gov.co" in url:
        return "SUIN-Juriscol - Ministerio de Justicia y del Derecho"
    if "cancilleria.gov.co" in url:
        return "Ministerio de Relaciones Exteriores - Compilación jurídica"
    if "colpensiones.gov.co" in url:
        return "Colpensiones - Compilación normativa"
    if "normograma.dian.gov.co" in url:
        return "DIAN - Compilación jurídica"
    if "funcionpublica.gov.co" in url:
        return "Departamento Administrativo de la Función Pública - Gestor Normativo"
    return declarada


def elegir(evaluadas, judicial):
    validas = [e for e in evaluadas if e["valida"]]
    if not validas:
        return None
    if judicial:
        return max(validas, key=lambda e: (bool(e["metricas"]["cierre"]), e["metricas"]["caracteres"]))
    # Más artículos propios; a igualdad, menos huecos y Senado primero (va primero en la lista).
    # Una cadena de Senado cortada pierde frente a cualquier fuente completa.
    return max(validas, key=lambda e: (e["estado"] == "ok", e["metricas"]["articulos"], -e["metricas"]["huecos"], -e["orden"]))


def procesar(doc):
    ficha = ficha_base(doc)
    judicial = ficha.get("tipo") in ("sentencia", "auto")
    evaluadas = []
    urls, con_respaldos, orden = candidatas(doc, ficha), False, -1
    while True:
        orden += 1
        if orden >= len(urls):
            if con_respaldos or any(e["valida"] for e in evaluadas):
                break
            # Ninguna fuente principal sirvió: se prueban las compilaciones oficiales.
            con_respaldos = True
            # La página del Gestor a veces responde "en mantenimiento": su versión PDF sirve.
            pdf_gestor = [u.replace("/norma.php?", "/norma_pdf.php?") for u in urls if "gestornormativo/norma.php?" in u]
            urls += [r for r in pdf_gestor + respaldos(ficha) if r not in urls]
            orden -= 1
            continue
        url = urls[orden]
        partes, estado = descargar_con_tramos(url)
        fila = {"doc_id": doc["doc_id"], "url": url, "orden": orden, "estado": estado,
                "tramos": len(partes or []), "valida": False, "metricas": {}, "partes": partes}
        if partes:
            try:
                for parte in partes:
                    if es_paquete(parte["contenido"]):
                        parte["derivado"] = {"texto": texto_paquete(parte["contenido"]), "sufijo": ".txt",
                                             "metodo": metodo_word(parte["contenido"]),
                                             "motivo": "original publicado como ZIP de capítulos en Word"}
                    elif es_word(parte["contenido"]):
                        # Word binario (.doc): la ingesta no lo lee; se guarda el texto al lado.
                        parte["derivado"] = {"texto": texto_word(parte["contenido"]), "sufijo": ".txt",
                                             "metodo": metodo_word(parte["contenido"]), "motivo": "original en Word (.doc/.docx)"}
                texto = texto_de(partes)
                if not es_legible(texto) and ocr_disponible() and \
                        any(p["contenido"][:1024].find(b"%PDF-") >= 0 for p in partes):
                    for parte in partes:
                        if parte["contenido"][:1024].find(b"%PDF-") >= 0:
                            parte["derivado"] = {"texto": ocr_pdf(parte["contenido"]), "sufijo": ".ocr.txt",
                                                 "metodo": metodo_ocr(),
                                                 "motivo": "PDF escaneado: la capa de texto del original es ilegible"}
                    texto = texto_de(partes)
                    fila["estado"] = f"{fila['estado']} (OCR)"
                fila["metricas"] = metricas(texto, judicial)
                # La legibilidad solo descarta PDF (escaneos); un HTML con muchas tablas puede
                # tener pocas palabras funcionales y aun así ser el texto correcto.
                es_pdf = any(p["contenido"][:1024].find(b"%PDF-") >= 0 for p in partes)
                fila["valida"] = len(texto) >= doc.get("min_caracteres", 1500) and identidad_valida(texto, ficha) and \
                    (es_legible(texto) or not es_pdf)
                if not fila["valida"]:
                    fila["estado"] = "no nombra la norma o texto corto"
            except Exception as error:  # un original ilegible descarta la candidata, no el documento
                fila["estado"] = f"extracción: {type(error).__name__}: {error}"
        evaluadas.append(fila)
        if con_respaldos and fila["valida"]:
            break  # entre las compilaciones de respaldo basta la primera que sirva
        # Senado completo y sin huecos basta: no hace falta bajar la declarada.
        m = fila["metricas"]
        if fila["valida"] and fila["estado"] == "ok" and orden == 0 and "secretariasenado" in url and judicial \
                and m.get("cierre"):
            break  # sentencia repartida al Senado, completa hasta la parte resolutiva
        if fila["valida"] and fila["estado"] == "ok" and orden == 0 and "secretariasenado" in url and not judicial \
                and m.get("articulos", 0) >= 3 and m.get("huecos", 1) == 0:
            break
    return ficha, evaluadas, elegir(evaluadas, judicial)


def guardar(raiz, ficha, elegida, declarada):
    archivos = []
    carpeta = raiz / "data/raw" / ficha["doc_id"]
    carpeta.mkdir(parents=True, exist_ok=True)
    for viejo in carpeta.glob("*"):
        viejo.unlink()
    for orden, parte in enumerate(elegida["partes"]):
        contenido = parte["contenido"]
        ext = ".pdf" if contenido[:1024].find(b"%PDF-") >= 0 else ".docx" if es_docx(contenido) \
            else ".zip" if es_paquete(contenido) else ".doc" if es_word(contenido) else ".html"
        archivo = f"{ficha['doc_id']}/{orden:03d}{ext}"
        (raiz / "data/raw" / archivo).write_bytes(contenido)
        archivos.append({"url": parte["url"], "url_final": parte["url_final"],
                         "sha256": hashlib.sha256(contenido).hexdigest(), "bytes": len(contenido),
                         "http_status": 200, "content_type": parte["tipo_contenido"], "last_modified": None,
                         "encoding": "PDF" if ext == ".pdf" else "auto", "archivo": archivo})
        if parte.get("derivado") is not None:
            derivado = parte["derivado"]
            texto_derivado = derivado["texto"].encode("utf-8")
            archivo_derivado = f"{ficha['doc_id']}/{orden:03d}{derivado['sufijo']}"
            (raiz / "data/raw" / archivo_derivado).write_bytes(texto_derivado)
            archivos[-1]["texto_derivado"] = {
                "archivo": archivo_derivado, "sha256": hashlib.sha256(texto_derivado).hexdigest(),
                "bytes": len(texto_derivado), "metodo": derivado["metodo"], "motivo": derivado["motivo"]}
    ahora = datetime.now(TZ).replace(microsecond=0)
    return {**ficha, "fuente": fuente_de(elegida["url"], declarada), "url": elegida["url"],
            "fecha_consulta": ahora.date().isoformat(), "fecha_descarga": ahora.isoformat(),
            "archivos_raw": archivos}


def completo_en_disco(raiz, entrada):
    for archivo in [a for x in entrada.get("archivos_raw") or [] for a in (x, x.get("texto_derivado")) if a]:
        ruta = raiz / "data/raw" / archivo["archivo"]
        if not ruta.is_file() or hashlib.sha256(ruta.read_bytes()).hexdigest() != archivo["sha256"]:
            return False
    return bool(entrada.get("archivos_raw"))


def servidor(doc):
    """Servidor principal de un documento, para repartir el trabajo por servidor."""
    if doc["doc_id"].startswith(("sentencia_cc", "auto_cc")):
        return "relatoria_cc"
    if doc["doc_id"].startswith("sentencia_csj"):
        return "corte_suprema"
    return urllib.parse.urlsplit(doc.get("url") or SENADO).netloc


def intercalar(docs):
    """Reparte los pendientes en turnos por servidor principal.

    La pausa es por servidor: si los hilos toman documentos en orden alfabético, todos
    esperan al mismo (Senado para las leyes) y los demás quedan ociosos.
    """
    grupos = {}
    for doc in docs:
        grupos.setdefault(servidor(doc), []).append(doc)
    colas = list(grupos.values())
    salida = []
    while any(colas):
        for cola in colas:
            if cola:
                salida.append(cola.pop(0))
    return salida


def excluidas(raiz):
    ruta = Path(raiz) / "configs/corpus_exclusiones.json"
    return {e["clave"] for e in json.loads(ruta.read_text(encoding="utf-8"))["exclusiones"]} if ruta.is_file() else set()


def inventario(raiz):
    docs = {d["doc_id"]: d for d in json.loads((raiz / "corpus_manifest.json").read_text(encoding="utf-8"))["documentos"]}
    extra = raiz / "configs/corpus_objetivos.json"
    if extra.is_file():
        for d in json.loads(extra.read_text(encoding="utf-8"))["documentos"]:
            docs[d["doc_id"]] = {**docs.get(d["doc_id"], {}), **d}
    # Lo excluido con evidencia (configs/corpus_exclusiones.json) no se vuelve a intentar.
    fuera = excluidas(raiz)
    return [docs[k] for k in sorted(docs) if k not in fuera]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", nargs="*", help="doc_id a procesar (por defecto todo el inventario)")
    ap.add_argument("--forzar", action="store_true", help="vuelve a descargar aunque ya esté en disco")
    ap.add_argument("--cc-senado", action="store_true",
                    help="todas las sentencias C primero al Senado (la relatoría queda de respaldo)")
    ap.add_argument("--hilos", type=int, default=4, help="documentos en paralelo (la pausa es por host)")
    ap.add_argument("--omitir", nargs="*", default=[],
                    help="servidores a no tocar en esta ejecución (relatoria_cc, corte_suprema o un dominio), "
                         "p. ej. mientras uno nos tiene bloqueados")
    args = ap.parse_args()
    global REPARTIR_CC
    if args.cc_senado:
        REPARTIR_CC = "todas"

    raw = RAIZ / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    # Cada ejecución reescribe manifest.json desde su copia en memoria: dos a la vez se
    # pisarían y perderían entradas sin error. El candado lo impide.
    import fcntl
    candado = (raw / ".reconstruir.lock").open("w")
    try:
        fcntl.flock(candado, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("Otra reconstrucción está corriendo sobre data/raw. Esperar a que termine.")
    ruta_manifiesto = raw / "manifest.json"
    manifiesto = {d["doc_id"]: d for d in json.loads(ruta_manifiesto.read_text(encoding="utf-8"))} \
        if ruta_manifiesto.is_file() else {}
    for doc_id, titulo in TITULO_CANONICO.items():
        if doc_id in manifiesto and manifiesto[doc_id].get("titulo") != titulo:
            manifiesto[doc_id]["titulo"] = titulo
    docs = inventario(RAIZ)
    if args.solo:
        docs = [d for d in docs if d["doc_id"] in set(args.solo)]
    pendientes = [d for d in docs if args.forzar or d["doc_id"] not in manifiesto
                  or not completo_en_disco(RAIZ, manifiesto[d["doc_id"]])]
    omitidos = [d for d in pendientes if servidor(d) in set(args.omitir)]
    pendientes = intercalar([d for d in pendientes if servidor(d) not in set(args.omitir)])
    if omitidos:
        print(f"Omitidos por servidor ({', '.join(args.omitir)}): {len(omitidos)}", flush=True)
    print(f"Inventario: {len(docs)} | ya en disco: {len(docs) - len(pendientes)} | por descargar: {len(pendientes)}", flush=True)

    ruta_informe = raw / "reconstruccion.csv"
    informe = {}
    if ruta_informe.is_file():
        with ruta_informe.open(encoding="utf-8-sig") as f:
            for fila in csv.DictReader(f):
                informe.setdefault(fila["doc_id"], []).append(fila)
    campos = ["doc_id", "url", "estado", "tramos", "valida", "elegida", "caracteres", "articulos",
              "max_articulo", "huecos", "cierre"]
    bloqueo = threading.Lock()

    def escribir():
        temporal = ruta_manifiesto.with_suffix(".json.tmp")
        temporal.write_text(json.dumps([manifiesto[k] for k in sorted(manifiesto)], ensure_ascii=False, indent=1),
                            encoding="utf-8")
        temporal.replace(ruta_manifiesto)
        with ruta_informe.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
            w.writeheader()
            for k in sorted(informe):
                w.writerows(informe[k])

    hechos = 0
    # Un grupo de hilos por servidor: con uno solo, los hilos se acumulan esperando al
    # servidor lento (Senado llegó a 17 s por respuesta) y los rápidos quedan ociosos.
    por_servidor = {}
    for d in pendientes:
        por_servidor.setdefault(servidor(d), []).append(d)
    pools = {k: ThreadPoolExecutor(max_workers=max(1, args.hilos // 2)) for k in por_servidor}
    try:
        futuros = {pools[k].submit(procesar, d): d for k, docs_k in por_servidor.items() for d in docs_k}
        for futuro in as_completed(futuros):
            doc = futuros[futuro]
            hechos += 1
            try:
                ficha, evaluadas, elegida = futuro.result()
            except Exception as error:
                print(f"[{hechos}/{len(pendientes)}] {doc['doc_id']}: ERROR {type(error).__name__}: {error}", flush=True)
                continue
            with bloqueo:
                informe[doc["doc_id"]] = [{**{k: e[k] for k in ("doc_id", "url", "estado", "tramos", "valida")},
                                           **e["metricas"], "elegida": e is elegida} for e in evaluadas]
                if elegida:
                    manifiesto[doc["doc_id"]] = guardar(RAIZ, ficha, elegida, doc.get("fuente"))
                escribir()
            m = elegida["metricas"] if elegida else {}
            resumen = (f"{elegida['url'][:70]} tramos={elegida['tramos']} arts={m.get('articulos')} "
                       f"huecos={m.get('huecos')} cierre={m.get('cierre')}") if elegida else \
                "SIN FUENTE VÁLIDA: " + "; ".join(f"{e['url'][:60]} -> {e['estado']}" for e in evaluadas)
            print(f"[{hechos}/{len(pendientes)}] {doc['doc_id']}: {resumen}", flush=True)
    finally:
        for pool in pools.values():
            pool.shutdown(wait=True)
    faltan = [d["doc_id"] for d in docs if d["doc_id"] not in manifiesto]
    print(f"\nEn el manifiesto: {len(manifiesto)} | sin fuente válida: {len(faltan)} {faltan[:40]}")


if __name__ == "__main__":
    main()
