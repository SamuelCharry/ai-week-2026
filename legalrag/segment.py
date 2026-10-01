"""Segmentación de normas colombianas por artículo.

Cada artículo es un pasaje. Sus incisos, literales, numerales y parágrafos
se quedan dentro del mismo pasaje. Un artículo más largo que `max_chars`
se divide en partes consecutivas por límites de párrafo (o de oración si un
párrafo solo ya es demasiado largo); cada parte conserva el número de artículo.

Cada pasaje guarda `inicio` y `fin`: posiciones de caracteres en el texto
procesado del documento, de modo que `texto_doc[inicio:fin] == pasaje["texto"]`.
"""
import re

ORDINALES = (
    "PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO|[ÚU]NICO"
)

# Encabezado de artículo al inicio de un párrafo. Ejemplos admitidos:
#   "Artículo 1°.", "ARTICULO 1o.", "ARTÍCULO 42. DEBERES DEL JUEZ.",
#   "ARTÍCULO 2.2.1.1.1.", "ARTÍCULO 20-A.", "ARTÍCULO 20 BIS.",
#   "ARTÍCULO TRANSITORIO 1o.", "ARTÍCULO PRIMERO."
# No coincide con referencias como "Artículo 15 de la Constitución".
ARTICLE_RE = re.compile(
    r"(?:ART[ÍI]CULO|Art[íi]culo|ART\.|Art\.)\s+"
    r"(?P<num>"
    r"TRANSITORIO(?:\s+\d+(?:\s*[°º]|o(?=\s*[\.\-]))?)?"
    rf"|(?:{ORDINALES})"
    r"|\d+(?:\.\d+)*(?:\s*[°º]|o(?=\s*[\.\-]))?(?:\s*-?\s*(?:[A-Z]\b|[Bb][Ii][Ss]\b))?"
    r")"
    r"\s*[\.\-:–]"
)

# Encabezados de estructura: terminan el artículo en curso.
HEADING_RE = re.compile(r"(?:LIBRO|T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|PARTE)\b")

LEVELS = {"LIBRO": 0, "PARTE": 1, "TÍTULO": 2, "TITULO": 2, "CAPÍTULO": 3, "CAPITULO": 3,
          "SECCIÓN": 4, "SECCION": 4}


def _is_closing(para: str) -> bool:
    """Bloque de firmas y publicación al final de la norma: no es artículo."""
    p = para.strip()
    if re.match(r"(?:PUBL[ÍI]QUESE|COMUN[ÍI]QUESE|Publ[íi]quese|Comun[íi]quese)", p):
        return True
    if re.match(r"NOTA:\s*Publicad", p):
        return True
    if re.match(r"Dad[ao] en\s", p) and len(p) < 300:
        return True
    return bool(re.match(r"(?:El|La)\s+(?:Presidente|Presidenta|Secretario|Secretaria)\b", p)
                and len(p) < 150 and p.endswith(","))


PARAGRAFO_RE = re.compile(r"(?:PAR[ÁA]GRAFO|Par[áa]grafo)(?:\s+(?:TRANSITORIO|transitorio))?(?:\s+\d+\s*[°ºo]?)?")


def normalize_article_number(raw: str) -> str:
    """'1°' -> '1', '5o' -> '5', '20 - A' -> '20A', '20 BIS' -> '20bis',
    'TRANSITORIO 1o' -> 'transitorio 1', 'PRIMERO' -> 'primero'."""
    s = raw.strip()
    s = re.sub(r"(\d)\s*[°º]", r"\1", s)
    s = re.sub(r"(\d)o$", r"\1", s)
    s = re.sub(r"(\d)\s*-?\s*([A-Z])$", r"\1\2", s)
    s = re.sub(r"(\d)\s*-?\s*[Bb][Ii][Ss]$", r"\1bis", s)
    s = re.sub(r"\s+", " ", s)
    if not s[:1].isdigit():
        s = s.lower()
    return s


def _paragraph_spans(text: str, start: int = 0, end: int | None = None):
    """(inicio, fin) de cada párrafo no vacío, separados por líneas en blanco."""
    end = len(text) if end is None else end
    for m in re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", text[start:end]):
        s, e = start + m.start(), start + m.end()
        # recorta espacios finales para que el pasaje termine en texto
        while e > s and text[e - 1].isspace():
            e -= 1
        yield s, e


def _split_long_span(text: str, s: int, e: int, max_chars: int):
    """Divide un párrafo demasiado largo cortando en fin de oración ('. ' o '; ');
    si no hay, en un espacio; si tampoco, a la fuerza."""
    pieces = []
    while e - s > max_chars:
        window = text[s:s + max_chars]
        cut = max(window.rfind(". "), window.rfind("; "))
        if cut > max_chars // 3:
            cut += 1  # conserva el signo
        else:
            cut = window.rfind(" ")
            if cut <= 0:
                cut = max_chars
        pieces.append((s, s + cut))
        s += cut
        while s < e and text[s].isspace():
            s += 1
    if s < e:
        pieces.append((s, e))
    return pieces


def _pack(text: str, s: int, e: int, max_chars: int):
    """Agrupa los párrafos de un artículo en partes de hasta max_chars sin
    cortar ningún párrafo (salvo que uno solo exceda el límite)."""
    units = []
    for ps, pe in _paragraph_spans(text, s, e):
        units.extend(_split_long_span(text, ps, pe, max_chars) if pe - ps > max_chars else [(ps, pe)])
    parts, cur_s, cur_e = [], None, None
    for us, ue in units:
        if cur_s is None:
            cur_s, cur_e = us, ue
        elif ue - cur_s <= max_chars:
            cur_e = ue
        else:
            parts.append((cur_s, cur_e))
            cur_s, cur_e = us, ue
    if cur_s is not None:
        parts.append((cur_s, cur_e))
    return parts


def segment_document(text: str, doc: dict, max_chars: int = 4000) -> list[dict]:
    """Devuelve los pasajes (uno o más por artículo) de una norma.

    `doc` es la entrada de corpus/sources.json: se usan doc_id, norma, fuente, url.
    """
    # 1) recorrer párrafos y marcar dónde empieza/termina cada artículo
    articles = []  # dicts con num, inicio, fin, seccion
    current = None
    headings: dict[int, str] = {}  # nivel -> encabezado vigente (TÍTULO, CAPÍTULO...)
    last_level = None  # para pegarle al encabezado la línea con su nombre

    def close():
        nonlocal current
        if current is not None:
            articles.append(current)
            current = None

    for ps, pe in _paragraph_spans(text):
        para = text[ps:pe]
        m = ARTICLE_RE.match(para)
        if m:
            close()
            current = {"num": normalize_article_number(m.group("num")), "inicio": ps, "fin": pe,
                       "seccion": " / ".join(headings[k] for k in sorted(headings))}
            last_level = None
            continue
        kind = para.split()[0].upper() if para.split() else ""
        if HEADING_RE.match(para) and kind in LEVELS and len(para) < 200:
            close()
            last_level = LEVELS[kind]
            headings = {k: v for k, v in headings.items() if k < last_level}
            headings[last_level] = para.strip()
            continue
        if last_level is not None and current is None and len(para) < 200:
            # línea con el nombre del título o capítulo ("PRINCIPIOS RECTORES")
            headings[last_level] += " " + para.strip()
            last_level = None
            continue
        last_level = None
        if _is_closing(para):
            close()
            continue
        if current is not None:
            current["fin"] = pe
    close()

    # 2) partir artículos largos y armar los pasajes
    seen: dict[str, int] = {}
    passages = []
    for art in articles:
        spans = _pack(text, art["inicio"], art["fin"], max_chars)
        seen[art["num"]] = seen.get(art["num"], 0) + 1
        dup = seen[art["num"]]
        base_id = f"{doc['doc_id']}:art{art['num'].replace(' ', '_')}" + (f"#{dup}" if dup > 1 else "")
        for i, (s, e) in enumerate(spans, start=1):
            texto = text[s:e]
            passages.append({
                "chunk_id": base_id + (f":p{i}" if len(spans) > 1 else ""),
                "doc_id": doc["doc_id"],
                "norma": doc["norma"],
                "articulo": art["num"],
                "parte": i,
                "partes": len(spans),
                "repeticion": dup,
                "seccion": art["seccion"],
                "paragrafos": [p.group(0).strip() for p in PARAGRAFO_RE.finditer(texto)
                               if p.start() == 0 or texto[p.start() - 1] == "\n"],
                "inicio": s,
                "fin": e,
                "texto": texto,
                "fuente": doc.get("fuente", ""),
                "url": doc.get("url", ""),
            })
    return passages


def indexed_text(p: dict) -> str:
    """Texto que se representa con el encoder y con BM25: encabezado con la
    norma y el artículo (también en las partes 2..n) + el texto del pasaje."""
    head = f"{p['norma']}, artículo {p['articulo']}"
    if p["partes"] > 1:
        head += f" (parte {p['parte']} de {p['partes']})"
    return f"{head}\n{p['texto']}"
