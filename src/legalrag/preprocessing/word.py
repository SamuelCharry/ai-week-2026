"""Texto de documentos Word binarios (.doc), como los de la relatoría del Consejo de Estado.

El Consejo de Estado entrega sus sentencias de unificación en .doc (Word 97-2003) y la
Sala Penal de la Corte Suprema enlaza sus providencias en .doc y .docx. La ingesta no lee
esos formatos, así que el texto se extrae una vez y se guarda al lado del original como
<NNN>.txt, con su hash y el método en el manifiesto. El paquete que se comparte no
necesita las herramientas de conversión.

También lee paquetes ZIP de capítulos en Word, como la Circular Básica Jurídica de la
Superintendencia Financiera (texto_paquete).

Requiere: antiword para .doc; .docx se lee con la biblioteca estándar.
"""
import html
import io
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

FIRMA_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def es_docx(contenido):
    """Word 2007+ (.docx): un ZIP con word/document.xml."""
    if contenido[:2] != b"PK":
        return False
    try:
        return "word/document.xml" in zipfile.ZipFile(io.BytesIO(contenido)).namelist()
    except zipfile.BadZipFile:
        return False


def es_word(contenido):
    return contenido[:8] == FIRMA_OLE or es_docx(contenido)


def texto_docx(contenido):
    """Texto de un .docx sin dependencias: un párrafo (w:p) por línea."""
    xml = zipfile.ZipFile(io.BytesIO(contenido)).read("word/document.xml").decode("utf-8")
    parrafos = []
    for parrafo in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        parrafo = re.sub(r"<w:tab/>", "\t", parrafo)
        parrafo = re.sub(r"<w:br[^>]*/>", "\n", parrafo)
        texto = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", parrafo))
        parrafos.append(texto)
    return html.unescape("\n".join(parrafos))


# Dentro de un paquete: anexos, formatos e instructivos son plantillas de reporte, no norma.
PAQUETE_FUERA = re.compile(r"(?i)anexo|formato|instructivo|\.tmp$|~\$")


def es_paquete(contenido):
    """ZIP de capítulos en Word (p. ej. la Circular Básica Jurídica de la Superfinanciera)."""
    if contenido[:2] != b"PK" or es_docx(contenido):
        return False
    try:
        nombres = zipfile.ZipFile(io.BytesIO(contenido)).namelist()
    except zipfile.BadZipFile:
        return False
    return any(re.search(r"(?i)\.docx?$", n) for n in nombres)


def texto_paquete(contenido):
    """Capítulos del paquete en orden de ruta, cada uno bajo "=== <ruta> ===".

    Un capítulo publicado en .doc/.docx y también en .pdf se toma una sola vez (del Word).
    """
    paquete = zipfile.ZipFile(io.BytesIO(contenido))
    nombres = sorted(n for n in paquete.namelist() if re.search(r"(?i)\.docx?$", n) and not PAQUETE_FUERA.search(n))
    trozos = []
    for nombre in nombres:
        try:
            texto = texto_word(paquete.read(nombre))
        except (subprocess.CalledProcessError, KeyError, zipfile.BadZipFile) as error:
            texto = f"[capítulo ilegible: {type(error).__name__}]"
        trozos.append(f"=== {nombre} ===\n{texto.strip()}")
    return "\n\n".join(trozos)


def disponible():
    return bool(shutil.which("antiword"))


def metodo(contenido=b""):
    if es_paquete(contenido):
        return "ZIP de capítulos Word: cada .doc/.docx (sin anexos ni formatos) con antiword o XML, en orden de ruta"
    if es_docx(contenido):
        return "docx: párrafos de word/document.xml (biblioteca estándar de Python)"
    return "antiword -w 0 (texto sin cortes de línea artificiales), codificación UTF-8"


def texto_word(contenido):
    """Texto del .doc (antiword; -w 0 evita cortes a 80 columnas) o del .docx (XML)."""
    if es_docx(contenido):
        return texto_docx(contenido)
    if not disponible():
        raise RuntimeError("Falta antiword para leer documentos Word (.doc)")
    with tempfile.TemporaryDirectory() as carpeta:
        ruta = Path(carpeta) / "doc.doc"
        ruta.write_bytes(contenido)
        salida = subprocess.run(["antiword", "-w", "0", "-m", "UTF-8.txt", str(ruta)],
                                capture_output=True, check=True)
    return salida.stdout.decode("utf-8", errors="replace")
