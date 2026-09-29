"""Extraccion y normalizacion determinista de citas normativas colombianas.

Una cita se normaliza a una tupla canonica, por ejemplo:
    "Art. 41 de la Ley 1563 de 2012"  -> ("ley", "1563", "2012", "41")
    "articulo 404 del Codigo Penal"   -> ("codigo_penal", None, None, "404")
    "Sentencia C-355 de 2006"         -> ("jurisprudencia", "C-355", "2006", None)
"""
from __future__ import annotations

import re
import unicodedata


def strip_accents(s: str) -> str:
    """Elimina los diacriticos de una cadena."""
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    """Normaliza a minusculas, sin acentos y con espacios colapsados."""
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
    """Extrae los numeros de articulo de una enumeracion como '13, 15 y 42'."""
    m = _NUMLIST.match(chunk)
    if not m:
        return []
    parts = re.split(r"\s*(?:,|\by\b|\be\b)\s*", m.group(1))
    return [p.strip() for p in parts if p.strip()]


def _articles_near(text: str, end: int, start: int) -> list[str]:
    """Articulos citados inmediatamente antes o despues de una norma.

    Cubre las dos formas del corpus: "Art. 41 de la Ley 1563 de 2012" y
    "Ley 1480 de 2011. Art. 23."
    """
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
    """Conjunto de citas canonicas presentes en un texto libre."""
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


def article_level(cites: set[tuple]) -> set[tuple]:
    """Solo las citas que llegan a nivel de articulo (las exigibles)."""
    return {c for c in cites if c[3] is not None}


def bodies(cites: set[tuple]) -> set[tuple]:
    """Cuerpo normativo sin el articulo, para comparar a grano grueso."""
    return {(c[0], c[1], c[2]) for c in cites}


def score(answer: str, reference_basis: str, respaldadas: set[tuple]) -> dict:
    """Compara las citas de una respuesta contra el fundamento de referencia.

    answer: texto de la respuesta del que se extraen las citas.
    reference_basis: fundamento normativo de referencia del item.
    respaldadas: citas extraidas de los pasajes que el sistema recupero para este
    mismo item. Una cita ausente de ese conjunto se cuenta como afirmada sin
    respaldo en la evidencia presentada.

    El universo de verificacion es el corpus del equipo, sin limite superior: toda
    norma que el equipo incorpore y recupere queda disponible como respaldo.

    Devuelve conteos crudos; la agregacion y el puntaje viven en evaluate.py.
    """
    got, ref = extract(answer), extract(reference_basis)
    got_b, ref_b, resp_b = bodies(got), bodies(ref), bodies(respaldadas)

    aciertos = got_b & ref_b
    fuera = got_b - ref_b
    citas_sin_respaldo = {c for c in fuera if c not in resp_b}
    incorrectas = fuera - citas_sin_respaldo
    aciertos_respaldados = aciertos & resp_b
    aciertos_sin_respaldo = aciertos - resp_b

    return {
        "n_ref": len(ref_b),
        "n_citadas": len(got_b),
        "aciertos": len(aciertos),
        "aciertos_respaldados": len(aciertos_respaldados),
        "aciertos_sin_respaldo": len(aciertos_sin_respaldo),
        "incorrectas": len(incorrectas),
        "citas_sin_respaldo": len(citas_sin_respaldo),
        "recall_citas": len(aciertos) / len(ref_b) if ref_b else None,
        "precision_citas": len(aciertos) / len(got_b) if got_b else None,
        "detalle": {
            "aciertos": sorted(map(list, aciertos)),
            "incorrectas": sorted(map(list, incorrectas)),
            "citas_sin_respaldo": sorted(map(list, citas_sin_respaldo)),
        },
    }
