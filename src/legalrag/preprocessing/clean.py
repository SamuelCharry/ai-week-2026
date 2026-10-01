from collections import Counter
import re
import unicodedata


def clean_text(text):
    """Limpieza para buscar. Los offsets siempre apuntan al original."""
    text = unicodedata.normalize("NFC", text).replace("\u00ad", "")
    text = text.replace("\ufeff", "").replace("\u00a0", " ")
    text = re.sub(r"(?<=\w)-\r?\n(?=[a-záéíóúñ])", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0e-\x1f]", "", text)
    pages = text.split("\f")
    if len(pages) >= 3:
        edges = Counter()
        for page in pages:
            lines = [x.strip() for x in page.splitlines() if x.strip()]
            edges.update(set(lines[:2] + lines[-2:]))
        repeated = {s for s, n in edges.items() if n >= max(3, len(pages) * .7)
                    and len(s) < 160 and not re.search(r"art[ií]culo|par[aá]grafo", s, re.I)}
        text = "\n".join("\n".join(line for line in page.splitlines()
                                  if line.strip() not in repeated) for page in pages)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()
