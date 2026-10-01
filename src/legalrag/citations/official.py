"""Extractor de citas en paridad con el evaluador oficial.

El cuerpo (normalizacion y regex) se copia de data/oficial/scripts/citations.py
para que el saneo y el aumento de citas del generador computen exactamente las
mismas tuplas que el jurado. Toda logica agregada queda en este archivo marcada
con el comentario `# extension local`.
"""
from __future__ import annotations

import re
import unicodedata


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s or "")
                   if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(s or "").lower()).strip()


CODES: dict[str, tuple[str, ...]] = {
    "constitucion": ("constitucion politica", "constitucion nacional", "constitucion",
                     "c.p.", "cp", "c.n.", "carta politica"),
    "codigo_civil": ("codigo civil", "c.c.", "cc"),
    "codigo_penal": ("codigo penal", "ley 599 de 2000", "c.p.p", "codigo penal colombiano"),
    "codigo_procedimiento_penal": ("codigo de procedimiento penal", "ley 906 de 2004", "cpp"),
    "codigo_comercio": ("codigo de comercio", "codigo del comercio", "c.co.", "cco"),
    "codigo_sustantivo_trabajo": ("codigo sustantivo del trabajo", "c.s.t.", "cst"),
    "codigo_procesal_trabajo": ("codigo procesal del trabajo", "cpts", "cpt"),
    "codigo_general_proceso": ("codigo general del proceso", "ley 1564 de 2012", "cgp"),
    "cpaca": ("codigo de procedimiento administrativo y de lo contencioso administrativo",
              "ley 1437 de 2011", "cpaca"),
    "estatuto_tributario": ("estatuto tributario", "decreto 624 de 1989", "e.t.", "et"),
    "codigo_infancia": ("codigo de la infancia y la adolescencia", "ley 1098 de 2006"),
    "codigo_nacional_policia": ("codigo nacional de seguridad y convivencia ciudadana",
                                "codigo nacional de policia", "ley 1801 de 2016"),
    "codigo_disciplinario": ("codigo general disciplinario", "codigo disciplinario unico",
                             "ley 1952 de 2019"),
    "estatuto_consumidor": ("estatuto del consumidor", "estatuto de proteccion al consumidor"),
    "decision_andina_486": ("decision 486 de la comision de la comunidad andina",
                            "decision andina 486", "decision 486"),
}

CODES["estatuto_consumidor"] += ("ley 1480 de 2011",)
CODES["codigo_civil"] += ("c.c",)

_ALIAS_NUM: dict[tuple[str, str], str] = {
    ("ley", "599"): "codigo_penal",
    ("ley", "906"): "codigo_procedimiento_penal",
    ("ley", "1564"): "codigo_general_proceso",
    ("ley", "1437"): "cpaca",
    ("ley", "1098"): "codigo_infancia",
    ("ley", "1801"): "codigo_nacional_policia",
    ("ley", "1952"): "codigo_disciplinario",
    ("decreto", "624"): "estatuto_tributario",
    ("ley", "1480"): "estatuto_consumidor",
}

_BARE_LAW_RE = re.compile(r"\b(ley|decreto)\s*(?:n[°ºo]?\.?\s*)?(\d{2,5})\b(?!\s*(?:de|del|/|-)\s*\d{4})")

NORM_TYPES = {
    "ley": ("ley", "leyes"),
    "decreto": ("decreto ley", "decreto-ley", "decreto legislativo", "decreto unico reglamentario",
                "decreto reglamentario", "decreto"),
    "acto_legislativo": ("acto legislativo",),
    "resolucion": ("resolucion",),
    "circular": ("circular externa", "circular"),
    "acuerdo": ("acuerdo",),
}

_ART = re.compile(r"\barts?\b\.?|\bart[ií]culos?\b")
_NUMLIST = re.compile(r"^[\s:.,;\-]*(\d+[a-z]?(?:\s*(?:,|y|e)\s*\d+[a-z]?)*)")

_NORM_RE = re.compile(
    r"\b(" + "|".join(sorted((v for vs in NORM_TYPES.values() for v in vs),
                             key=len, reverse=True)).replace(" ", r"\s+") + r")\s*"
    r"(?:n[°ºo]?\.?\s*)?(\d{1,5})\s*(?:de|del|/|-)\s*(\d{4})\b")

_SENT_RE = re.compile(
    r"\b(c|t|su|sl|sc|sp|stc|stl|ac|au)\s*[-\s]?\s*(\d{1,5})\s*(?:de|del|/|-)\s*(\d{2,4})\b",
    re.IGNORECASE)


def _expand_numbers(chunk: str) -> list[str]:
    m = _NUMLIST.match(chunk)
    if not m:
        return []
    parts = re.split(r"\s*(?:,|\by\b|\be\b)\s*", m.group(1))
    return [p.strip() for p in parts if p.strip()]


def _articles_near(text: str, end: int, start: int) -> list[str]:
    before = text[max(0, start - 90):start]
    after = text[end:end + 60]
    arts: list[str] = []
    m = list(_ART.finditer(before))
    if m and not re.search(r"[.;]", before[m[-1].end():]):
        arts += _expand_numbers(before[m[-1].end():])
    m2 = _ART.search(after)
    if m2 and not re.search(r"\w", after[:m2.start()].replace(".", " ").replace(",", " ")):
        arts += _expand_numbers(after[m2.end():m2.end() + 40])
    return arts


def extract(text: str) -> set[tuple]:
    t = norm(text)
    found: set[tuple] = set()

    for m in _NORM_RE.finditer(t):
        raw_type, number, year = m.group(1), m.group(2), m.group(3)
        kind = next(k for k, vs in NORM_TYPES.items()
                    if any(re.fullmatch(v.replace(" ", r"\s+"), raw_type) for v in vs))
        alias = _ALIAS_NUM.get((kind, number))
        arts = _articles_near(t, m.end(), m.start())
        if alias:
            body, number, year = alias, None, None
        else:
            body = kind
        if arts:
            found.update((body, number, year, a) for a in arts)
        else:
            found.add((body, number, year, None))

    for m in _BARE_LAW_RE.finditer(t):
        alias = _ALIAS_NUM.get((m.group(1), m.group(2)))
        if alias:
            arts = _articles_near(t, m.end(), m.start())
            if arts:
                found.update((alias, None, None, a) for a in arts)
            else:
                found.add((alias, None, None, None))

    for code, variants in CODES.items():
        for v in sorted(variants, key=len, reverse=True):
            pat = r"(?<![\w.])" + re.escape(v).replace(r"\ ", r"\s+") + r"(?![\w])"
            for m in re.finditer(pat, t):
                arts = _articles_near(t, m.end(), m.start())
                if arts:
                    found.update((code, None, None, a) for a in arts)
                else:
                    found.add((code, None, None, None))
            if re.search(pat, t):
                break

    for m in _SENT_RE.finditer(t):
        sala, number, year = m.group(1).upper(), m.group(2), m.group(3)
        if len(year) == 2:
            year = ("20" if int(year) < 50 else "19") + year
        found.add(("jurisprudencia", f"{sala}-{int(number)}", year, None))

    return found


def bodies(cites: set[tuple]) -> set[tuple]:
    return {(c[0], c[1], c[2]) for c in cites}


# ──────────────────────────── extension local ────────────────────────────

# Normas que aparecen en el corpus con doc_id/encabezado distinto al que el
# extractor oficial reconoce. Expandimos el texto del pasaje con la mencion
# canonica equivalente antes de extraer, para que el soporte se calcule igual.
HEADER_ALIASES: dict[str, str] = {
    "ley 84 de 1873": "Codigo Civil",
    "decreto 410 de 1971": "Codigo de Comercio",
    "decreto 2663 de 1950": "Codigo Sustantivo del Trabajo",
    "decreto 1400 de 1970": "Codigo de Procedimiento Civil",
    "ley 57 de 1887": "Codigo Civil",
    "ley 153 de 1887": "Codigo Civil",
    "ley 100 de 1993": "Sistema de Seguridad Social Integral",
    "decreto 2158 de 1948": "Codigo Procesal del Trabajo",
    "constitucion politica de colombia": "Constitucion Politica",
    "constitucion politica": "Constitucion Politica",
}


def enrich_header(encabezado: str) -> str:
    """Agrega al encabezado el nombre canonico del codigo si procede.

    El extractor oficial reconoce 'codigo civil' pero no 'Ley 84 de 1873'.
    Este helper agrega el alias canonico cuando el encabezado solo trae la
    referencia primitiva.
    """
    low = norm(encabezado)
    for key, canonical in HEADER_ALIASES.items():
        if key in low and norm(canonical) not in low:
            encabezado = f"{encabezado} [{canonical}]"
            low = norm(encabezado)
    return encabezado


# Rendering canonico de un cuerpo normativo para incluirlo en una respuesta.
_CANONICAL = {
    "constitucion": "Constitucion Politica",
    "codigo_civil": "Codigo Civil",
    "codigo_penal": "Codigo Penal",
    "codigo_procedimiento_penal": "Codigo de Procedimiento Penal",
    "codigo_comercio": "Codigo de Comercio",
    "codigo_sustantivo_trabajo": "Codigo Sustantivo del Trabajo",
    "codigo_procesal_trabajo": "Codigo Procesal del Trabajo",
    "codigo_general_proceso": "Codigo General del Proceso",
    "cpaca": "Codigo de Procedimiento Administrativo y de lo Contencioso Administrativo (CPACA)",
    "estatuto_tributario": "Estatuto Tributario",
    "codigo_infancia": "Codigo de la Infancia y la Adolescencia",
    "codigo_nacional_policia": "Codigo Nacional de Seguridad y Convivencia Ciudadana",
    "codigo_disciplinario": "Codigo General Disciplinario",
    "estatuto_consumidor": "Estatuto del Consumidor",
    "decision_andina_486": "Decision Andina 486",
}


def render(body: tuple) -> str:
    """Convierte un cuerpo (tipo, numero, anio) en una mencion canonica."""
    kind, number, year = body
    if kind == "jurisprudencia":
        return f"Sentencia {number}"
    if kind in _CANONICAL:
        return _CANONICAL[kind]
    if number and year:
        return f"{kind.capitalize()} {number} de {year}"
    if number:
        return f"{kind.capitalize()} {number}"
    return kind.capitalize()
