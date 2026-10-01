"""Detección de normas citadas y verificación contra los pasajes recuperados.

Una cita es (norma, artículo). Está respaldada si alguno de los pasajes
recuperados para esa misma respuesta pertenece a esa norma y a ese artículo.
"""
import re
import unicodedata

TIPOS = r"(?:Ley\s+Estatutaria|Ley|Decreto[\s-]+Ley|Decreto\s+Legislativo|Decreto|Acto\s+Legislativo|Resoluci[óo]n)"
NORMA_RE = re.compile(rf"(?P<tipo>{TIPOS})\s+(?:No\.?\s*)?(?P<num>\d[\d\.]*)\s+de\s+(?P<anio>\d{{4}})", re.I)
# "artículo 10", "art. 42", "artículos 10, 11 y 12", "artículo 2.2.1.1.1"
ART_RE = re.compile(
    r"\bart(?:[íi]culos?|s?\.)\s+"
    r"(?P<lista>\d+(?:\.\d+)*(?:\s*[°º])?(?:\s*(?:,|y|e)\s*\d+(?:\.\d+)*(?:\s*[°º])?)*)",
    re.I,
)
VENTANA = 120  # caracteres para asociar un artículo con la norma más cercana


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.lower()).strip()


def norma_key(texto: str) -> str | None:
    """'Ley Estatutaria 1581 de 2012' -> 'ley 1581 de 2012'."""
    m = NORMA_RE.search(texto)
    if not m:
        return None
    tipo = fold(m.group("tipo")).replace(" estatutaria", "").replace("-", " ")
    return f"{tipo} {m.group('num').replace('.', '')} de {m.group('anio')}"


def _norm_mentions(text: str, aliases: dict[str, str]):
    """Posiciones de menciones de normas: por número (Ley X de AAAA) o por
    alias declarados en sources.json (p. ej. 'Código General del Proceso')."""
    out = [(m.start(), m.end(), norma_key(m.group(0))) for m in NORMA_RE.finditer(text)]
    low = fold(text)  # fold conserva la longitud en textos en español
    for alias, key in aliases.items():
        for m in re.finditer(re.escape(alias), low):
            out.append((m.start(), m.end(), key))
    return sorted(out)


def extract_citations(text: str, aliases: dict[str, str] | None = None) -> list[dict]:
    """Lista de {'norma': key|None, 'articulo': '10'} en orden de aparición."""
    aliases = aliases or {}
    mentions = _norm_mentions(text, aliases)
    cites = []
    for m in ART_RE.finditer(text):
        nums = re.findall(r"\d+(?:\.\d+)*", m.group("lista"))
        after = [k for s, e, k in mentions if 0 <= s - m.end() <= VENTANA]
        before = [k for s, e, k in mentions if 0 <= m.start() - e <= VENTANA]
        key = after[0] if after else (before[-1] if before else None)
        for n in nums:
            c = {"norma": key, "articulo": n}
            if c not in cites:
                cites.append(c)
    return cites


def is_supported(cita: dict, pasajes: list[dict]) -> bool:
    for p in pasajes:
        if p["articulo"] != cita["articulo"]:
            continue
        if cita["norma"] is None or cita["norma"] == p.get("norma_key"):
            return True
    return False


def format_citation(cita: dict, nombres: dict[str, str]) -> str:
    """'Ley 1581 de 2012, artículo 10' (o solo 'Artículo 10' si no se sabe la norma)."""
    norma = nombres.get(cita["norma"], cita["norma"] or "")
    return f"{norma}, artículo {cita['articulo']}" if norma else f"Artículo {cita['articulo']}"
