"""Extracción de texto (HTML, PDF, TXT) y limpieza.

Reutilizado y adaptado de CODEFEST Ad Astra 2026 (src/extraction/extract_html.py,
extract_pdf.py, extract_txt.py y clean_text.py; autores en el historial git:
Juanesillo, Angie Gutierrez, SamuelCharry). Cambios para normas:
- HTML: se admite un selector CSS para tomar solo el cuerpo de la norma y se
  conserva un párrafo por bloque (<p>, <li>, <h1-4>), que es lo que usa la
  segmentación por artículo.
- PDF: se conserva el recorte de encabezado/pie; el OCR de imágenes es opcional.
- Limpieza: se quitaron las reglas del reto aeroespacial que borraban fechas,
  URL y números sueltos, porque en un texto normativo pueden ser contenido.
"""
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup


def extract_html(raw: bytes, selector: str | None = None) -> str:
    """Texto visible de un HTML, un párrafo por bloque. Con `selector` se toma
    solo ese contenedor (p. ej. 'div.descripcion-contenido' en Función Pública)."""
    soup = BeautifulSoup(raw, "html.parser")  # bytes: respeta el charset declarado
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    root = soup
    if selector:
        root = soup.select_one(selector)
        if root is None:
            raise ValueError(f"El selector {selector!r} no aparece en el HTML")
    bloques = root.find_all(["h1", "h2", "h3", "h4", "p", "li"])
    if not bloques:
        return root.get_text(separator="\n")
    textos = (re.sub(r"\s+", " ", b.get_text()).strip() for b in bloques)
    return "\n\n".join(t for t in textos if t)


def extract_pdf(path: str, ocr: bool = False, idioma_ocr: str = "spa") -> str:
    """Texto de un PDF por página, recortando 8% arriba y abajo (encabezado y
    pie). Con ocr=True también lee el texto de las imágenes (requiere pytesseract)."""
    import pdfplumber

    paginas = []
    with pdfplumber.open(path) as pdf:
        for pagina in pdf.pages:
            x0, y0, x1, y1 = pagina.bbox
            alto = y1 - y0
            area = (x0, y0 + alto * 0.08, x1, y1 - alto * 0.08)
            texto = (pagina.crop(area).extract_text() or "").strip()
            if ocr:
                import pytesseract
                for img in pagina.images:
                    try:
                        bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
                        pil = pagina.within_bbox(bbox).to_image(resolution=300).original
                        extra = pytesseract.image_to_string(pil, lang=idioma_ocr).strip()
                        if extra:
                            texto += "\n\n" + extra
                    except Exception:
                        continue  # una imagen rota no tumba la página
            if texto:
                paginas.append(texto)
    return "\n\n".join(paginas)


def extract_file(path: str, selector: str | None = None) -> str:
    ext = Path(path).suffix.lower()
    if ext in (".html", ".htm"):
        return extract_html(Path(path).read_bytes(), selector)
    if ext == ".pdf":
        return extract_pdf(path)
    if ext == ".txt":
        return Path(path).read_text(encoding="utf-8", errors="ignore")
    raise ValueError(f"Formato no soportado: {ext}")


def clean_text(texto: str) -> str:
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    texto = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", texto)
    texto = re.sub(r"(?i)\b(página|pág|page)\s+\d+(\s+(de|of)\s+\d+)?\b", "", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip() + "\n"
