from io import BytesIO
import logging
import re
import threading

_fallback_lock = threading.Lock()


def fallback_pdf(raw):
    import pymupdf
    # MuPDF se usa en serie porque su API no admite extracción multihilo.
    with _fallback_lock, pymupdf.open(stream=raw, filetype="pdf") as document:
        pages = []
        for page in document:
            blocks = page.get_text("blocks", sort=True)
            pages.append("\n\n".join(block[4].strip() for block in blocks if block[6] == 0 and block[4].strip()))
    text = "\n\f\n".join(pages)
    if (not text.strip() or text.count("\ufffd") / max(1, len(text)) > 0.002
            or len(re.findall(r"[\u3400-\u9fff]", text)) / max(1, len(text)) > 0.001):
        raise ValueError("PDF sin texto fiable. Requiere OCR u otra fuente oficial.")
    return text


def extract_pdf(raw):
    try:
        return fallback_pdf(raw)
    except (RuntimeError, ValueError):
        return extract_pypdf(raw)


def extract_pypdf(raw):
    from pypdf import PdfReader
    messages = []
    current_thread = threading.get_ident()

    class Capture(logging.Handler):
        def emit(self, record):
            if record.thread == current_thread:
                messages.append(record.getMessage())

    logger = logging.getLogger("pypdf")
    capture = Capture()
    logger.addHandler(capture)
    try:
        reader = PdfReader(BytesIO(raw))
        text = "\n\f\n".join(page.extract_text() or "" for page in reader.pages)
    finally:
        logger.removeHandler(capture)
    if (any("not implemented" in message.lower() for message in messages) or not text.strip()
            or text.count("\ufffd") / max(1, len(text)) > 0.002
            or len(re.findall(r"[\u3400-\u9fff]", text)) / max(1, len(text)) > 0.001):
        raise ValueError("PDF sin extracción fiable. Requiere OCR u original DOCX.")
    return text
