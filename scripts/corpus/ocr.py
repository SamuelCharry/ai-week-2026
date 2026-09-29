"""OCR de PDF escaneados y medida de legibilidad del texto extraído.

Varias providencias de la Corte Suprema son escaneos: no traen capa de texto o
traen una basura ("Repdbli鍛deColombia") que no sirve para recuperar. Se pasan por
Tesseract en español a 300 ppp, un hilo, para que el resultado sea determinista.

El texto se guarda al lado del original como <NNN>.ocr.txt y queda referenciado en
el manifiesto con su hash y el método. La ingesta lo lee en vez de la capa de texto
del PDF, así que el paquete de Colab no necesita Tesseract.

Requiere: tesseract con el idioma spa y pdftoppm (poppler).
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

DPI = 300
IDIOMA = "spa"
FUNCIONALES = {"de", "la", "el", "en", "y", "a", "los", "las", "que", "del", "se", "por", "con", "no", "una",
               "un", "para", "al", "es", "lo", "como", "su", "o", "sus", "le", "sobre", "este", "esta"}
SEPARADOR_PAGINA = "\f"


def legibilidad(texto):
    """(proporción de letras no latinas, proporción de palabras funcionales del español)."""
    muestra = texto[:200000]
    letras = [c for c in muestra if c.isalpha()]
    no_latinas = sum(1 for c in letras if not ("a" <= c.lower() <= "z" or c in "áéíóúüñÁÉÍÓÚÜÑ"))
    palabras = re.findall(r"[a-záéíóúñü]+", muestra.lower())
    funcionales = sum(1 for p in palabras if p in FUNCIONALES)
    return (no_latinas / len(letras) if letras else 1.0), (funcionales / len(palabras) if palabras else 0.0)


def es_legible(texto):
    """Texto en español utilizable: casi sin letras no latinas y con palabras funcionales normales.

    Un texto jurídico en español ronda el 35-45 % de palabras funcionales; una capa
    de OCR dañada o sin espacios queda por debajo del 25 %.
    """
    if len(texto.strip()) < 500:
        return False
    no_latinas, funcionales = legibilidad(texto)
    return no_latinas <= 0.01 and funcionales >= 0.25


def disponible():
    return bool(shutil.which("tesseract") and shutil.which("pdftoppm"))


def version():
    salida = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout
    return salida.splitlines()[0].strip() if salida else "tesseract"


def ocr_pdf(contenido):
    """Texto del PDF por OCR, páginas separadas por \\f. Determinista: un hilo, ppp fijos."""
    if not disponible():
        raise RuntimeError("Falta tesseract (idioma spa) o pdftoppm para el OCR")
    inicio = contenido[:1024].find(b"%PDF-")
    entorno = {**os.environ, "OMP_THREAD_LIMIT": "1"}
    with tempfile.TemporaryDirectory() as carpeta:
        carpeta = Path(carpeta)
        (carpeta / "doc.pdf").write_bytes(contenido[max(inicio, 0):])
        subprocess.run(["pdftoppm", "-r", str(DPI), "-gray", "-png", "doc.pdf", "p"], cwd=carpeta, check=True,
                       capture_output=True)
        paginas = []
        for imagen in sorted(carpeta.glob("p-*.png"), key=lambda p: int(re.search(r"-(\d+)\.png$", p.name).group(1))):
            salida = subprocess.run(["tesseract", str(imagen), "stdout", "-l", IDIOMA, "--psm", "3"],
                                    capture_output=True, text=True, env=entorno, check=True).stdout
            paginas.append(salida.strip())
    return SEPARADOR_PAGINA.join(paginas)


def metodo():
    return f"{version()} -l {IDIOMA} --psm 3, pdftoppm {DPI} ppp en gris, OMP_THREAD_LIMIT=1"
