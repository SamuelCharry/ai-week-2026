from dataclasses import dataclass, asdict
import re
import unicodedata


def fold(value):
    text = str(value).lower().replace("–", "-").replace("—", "-").replace("\u00a0", " ")
    return re.sub(r"[\u0300-\u036f]", "", unicodedata.normalize("NFD", text))


ALIASES = {
    "constitucion politica": "constitucion_1991",
    "constitucion": "constitucion_1991",
    "codigo penal": "ley_599_2000",
    "codigo general del proceso": "ley_1564_2012",
    "codigo de procedimiento administrativo y de lo contencioso administrativo": "ley_1437_2011",
    "cpaca": "ley_1437_2011",
    "codigo de procedimiento penal": "ley_906_2004",
    "codigo sustantivo del trabajo": "decreto_2663_1950",
    "codigo civil": "ley_84_1873",
    "codigo de comercio": "decreto_410_1971",
    "estatuto tributario": "decreto_624_1989",
    "codigo de la infancia y la adolescencia": "ley_1098_2006",
}
NORM = re.compile(
    r"\b(ley|decreto(?:[ -]+ley)?|acto\s+legislativo|resolucion)\s+"
    r"(?:n(?:umero|ro|o)?\.?\s*)?(\d[\d.]*)\s*(?:de(?:l)?|/)\s*(\d{4})\b"
)
JUDGMENT = re.compile(r"\b(?:sentencia\s+)?(su|c|t)[\s-]*(\d+)\s*(?:de\s*|/|-)(\d{2,4})\b")
CSJ_JUDGMENT = re.compile(r"\b(?:sentencia\s+)?(stc|stl|stp|sc|sl|sp)[\s-]*(\d+)\s*(?:de\s*|/|-)(\d{4})\b")
ARTICLES = re.compile(r"\bart(?:iculo)?s?\.?\s+(\d+[a-z]?(?:\s*(?:,|y|e)\s*\d+[a-z]?)*)(?!\d)")
ALIAS_PATTERN = re.compile(r"\b(?:" + "|".join(re.escape(a) for a in sorted(ALIASES, key=len, reverse=True)) + r")\b")


def norm_mentions(text):
    """Menciones de normas para el grafo, sin asociación de artículos."""
    text = fold(text)
    for match in NORM.finditer(text):
        kind = "decreto" if match[1].startswith("decreto") else re.sub(r"\s+", "_", match[1])
        yield f"{kind}_{int(match[2].replace('.', ''))}_{match[3]}"
    for match in JUDGMENT.finditer(text):
        year = int(match[3])
        if year < 100:
            year += 2000 if year <= 30 else 1900
        yield f"sentencia_{match[1]}_{int(match[2])}_{year}"
    for match in CSJ_JUDGMENT.finditer(text):
        yield f"sentencia_csj_{match[1]}_{int(match[2])}_{match[3]}"
    for match in ALIAS_PATTERN.finditer(text):
        yield ALIASES[match[0]]


@dataclass(frozen=True)
class Citation:
    norma: str
    articulo: str | None = None

    def dict(self):
        return asdict(self)


def article_number(value):
    value = re.sub(r"[\s°.º]", "", fold(value))
    ordinals = {"primero": "1", "segundo": "2", "tercero": "3", "cuarto": "4",
                "quinto": "5", "sexto": "6", "septimo": "7", "octavo": "8",
                "noveno": "9", "decimo": "10", "unico": "unico"}
    return ordinals.get(value, re.sub(r"^(\d+)o$", r"\1", value))


def norm_identity(doc):
    tipo = fold(doc.get("tipo", "")).replace(" ", "_")
    number = str(doc.get("numero") or "").replace(".", "").lstrip("0") or "0"
    year = str(doc.get("anio") or "")
    if "constitucion" in tipo or "constitucion" in fold(doc.get("doc_id", "")):
        return "constitucion_1991"
    if tipo in {"ley", "decreto", "decreto_ley", "acto_legislativo", "resolucion"} and year:
        return f"{'decreto' if tipo == 'decreto_ley' else tipo}_{number}_{year}"
    if tipo in {"auto", "concepto", "compendio", "circular", "decision", "acuerdo"}:
        return doc["doc_id"]
    csj = re.fullmatch(r"(stc|stl|stp|sc|sl|sp)[\s-]*(\d+)", fold(number))
    if tipo == "sentencia" and csj and year:
        return f"sentencia_csj_{csj[1]}_{int(csj[2])}_{year}"
    parsed = extract_references(doc.get("titulo", ""))
    return parsed[0].norma if parsed else doc["doc_id"]


def extract_references(text, deduplicate=True):
    text = fold(text)
    # Las cláusulas separadas por punto y coma no comparten norma.
    result = []
    for clause in re.split(r"[;\n]", text):
        mentions = []
        for m in NORM.finditer(clause):
            kind = "decreto" if m[1].startswith("decreto") else m[1].replace(" ", "_")
            mentions.append((m.start(), m.end(), f"{kind}_{int(m[2].replace('.', ''))}_{m[3]}"))
        for m in JUDGMENT.finditer(clause):
            year = int(m[3])
            if year < 100:
                year += 2000 if year <= 30 else 1900
            mentions.append((m.start(), m.end(), f"sentencia_{m[1]}_{int(m[2])}_{year}"))
        for m in CSJ_JUDGMENT.finditer(clause):
            mentions.append((m.start(), m.end(), f"sentencia_csj_{m[1]}_{int(m[2])}_{m[3]}"))
        occupied = [(a, b) for a, b, _ in mentions]
        for alias, identity in sorted(ALIASES.items(), key=lambda x: -len(x[0])):
            for m in re.finditer(r"\b" + re.escape(alias) + r"\b", clause):
                if not any(a < m.end() and m.start() < b for a, b in occupied):
                    mentions.append((m.start(), m.end(), identity))
                    occupied.append((m.start(), m.end()))
        mentions.sort()
        associated = set()
        for article in ARTICLES.finditer(clause):
            if not mentions:
                continue
            candidate = min(mentions, key=lambda n: min(abs(article.end() - n[0]), abs(article.start() - n[1])))
            distance = min(abs(article.end() - candidate[0]), abs(article.start() - candidate[1]))
            if distance > 150:
                continue
            associated.add(candidate)
            for number in re.findall(r"\d+[a-z]?", article[1]):
                result.append(Citation(candidate[2], article_number(number)))
        result.extend(Citation(identity) for a, b, identity in mentions if (a, b, identity) not in associated)
    return list(dict.fromkeys(result)) if deduplicate else result


def response_text(response):
    fields = ("justificacion", "respuesta", "referencia_legal", "marco_normativo",
              "analisis", "jurisprudencia", "conclusion")
    return "\n".join(str(response.get(f, "")) for f in fields)


def trace_citations(response, passages):
    references = extract_references(response_text(response))
    traces = []
    for ref in references:
        supporting = []
        for p in passages:
            own = not p.get("atribucion_ambigua") and p.get("norma") == ref.norma and (
                ref.articulo is None or str(p.get("numero_articulo", "")).lower() == ref.articulo)
            explicit = extract_references(p["texto"])
            cited = any(c.norma == ref.norma and (ref.articulo is None or c.articulo == ref.articulo)
                        for c in explicit)
            if own or cited:
                supporting.append({"doc_id": p["doc_id"], "inicio": p["inicio"], "fin": p["fin"]})
        traces.append({**ref.dict(), "verificada": bool(supporting), "pasajes": supporting})
    return traces
