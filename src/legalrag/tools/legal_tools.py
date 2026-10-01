"""Herramientas numericas deterministas para preguntas juridicas.

Las funciones toman datos de la pregunta y devuelven resultados calculados
que se inyectan como un bloque de texto `[DATOS CALCULADOS]` en el user prompt
del generador. Son datos publicos (DANE / Mintrabajo); no leen el banco de
preguntas ni respuestas esperadas.
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Optional


# ------------------------------------------------------------------
# Tabla SMLMV historica (COP). Fuente: decretos anuales de Mintrabajo /
# compendios DANE. Dato publico.
# ------------------------------------------------------------------
SMLMV: dict[int, int] = {
    2015: 644350, 2016: 689455, 2017: 737717, 2018: 781242,
    2019: 828116, 2020: 877803, 2021: 908526, 2022: 1000000,
    2023: 1160000, 2024: 1300000, 2025: 1423500, 2026: 1500000,
}

# Umbrales de cuantia del Codigo General del Proceso (Art. 25 CGP).
# Mínima: hasta 40 SMLMV
# Menor:  más de 40 y hasta 150 SMLMV
# Mayor:  más de 150 SMLMV
CUANTIA_UMBRAL_MINIMA = 40
CUANTIA_UMBRAL_MENOR = 150


# ------------------------------------------------------------------
# Festivos nacionales de Colombia (fijos + Ley Emiliani aproximada por anio).
# Para fines del reto solo necesitamos restar fines de semana + festivos de
# una ventana de dias habiles pequena (<60), asi que incluimos los principales
# fijos. No pretende ser calendario completo.
# ------------------------------------------------------------------
_FESTIVOS_FIJOS = [(1, 1), (5, 1), (7, 20), (8, 7), (12, 8), (12, 25)]


def _es_festivo(d: _dt.date) -> bool:
    if (d.month, d.day) in _FESTIVOS_FIJOS:
        return True
    return False


# ---------- Tool 1: SMLMV a COP ----------
def smlmv_valor(anio: int) -> Optional[int]:
    """Devuelve el SMLMV del anio solicitado en COP (None si no esta tabulado)."""
    return SMLMV.get(anio)


def cop_a_smlmv(valor_cop: float, anio: int) -> Optional[float]:
    """Convierte COP a numero de SMLMV del anio dado."""
    base = SMLMV.get(anio)
    if not base:
        return None
    return round(valor_cop / base, 2)


# ---------- Tool 2: Cuantia CGP ----------
def cuantia_cgp(valor_cop: float, anio: int) -> Optional[dict]:
    """Clasifica la cuantia (minima / menor / mayor) segun Art. 25 CGP y el SMLMV del anio."""
    smlmv = cop_a_smlmv(valor_cop, anio)
    if smlmv is None:
        return None
    if smlmv <= CUANTIA_UMBRAL_MINIMA:
        clase = "minima"
    elif smlmv <= CUANTIA_UMBRAL_MENOR:
        clase = "menor"
    else:
        clase = "mayor"
    return {"smlmv_equivalente": smlmv, "clase": clase, "umbral_minima_smlmv": CUANTIA_UMBRAL_MINIMA,
            "umbral_menor_smlmv": CUANTIA_UMBRAL_MENOR, "smlmv_del_anio": SMLMV[anio], "anio": anio}


# ---------- Tool 3: Dias habiles ----------
def sumar_dias_habiles(fecha: _dt.date, dias: int) -> _dt.date:
    """Suma dias habiles (excluye sabados, domingos y festivos fijos)."""
    sign = 1 if dias >= 0 else -1
    pendientes = abs(dias)
    actual = fecha
    while pendientes > 0:
        actual = actual + _dt.timedelta(days=sign)
        if actual.weekday() < 5 and not _es_festivo(actual):
            pendientes -= 1
    return actual


# ---------- Deteccion y rendering ----------
_SMLMV_RE = re.compile(r"\b(?:smlmv|salario(?:s)?\s+m[ií]nimo(?:s)?(?:\s+legal(?:es)?)?(?:\s+mensual(?:es)?)?(?:\s+vigente(?:s)?)?)\b", re.I)
_CUANTIA_RE = re.compile(r"\bcuant[ií]a(?:s)?\b|\bprocesos?\s+(?:de\s+)?m[ií]nima\s+cuant[ií]a\b", re.I)
_PLAZO_RE = re.compile(r"\bd[ií]as?\s+h[áa]biles\b|\bplazo\b|\bliquidaci[óo]n\s+(?:unilateral|del\s+contrato)\b", re.I)
_COP_RE = re.compile(r"(\d{1,3}(?:[.,]\d{3}){1,3}|\d{7,10})\s*(?:cop|pesos?|\$)?", re.I)
_YEAR_RE = re.compile(r"\b(20[0-2]\d|201[0-9])\b")


def _extract_amount_cop(text: str) -> Optional[float]:
    """Primer valor monetario grande (>= 1 millon) en el texto."""
    for m in _COP_RE.finditer(text):
        raw = m.group(1).replace(".", "").replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        if v >= 1_000_000:
            return v
    return None


def _extract_year(text: str, default: int = 2024) -> int:
    m = _YEAR_RE.search(text)
    if m:
        return int(m.group(1))
    return default


def tool_block(question_text: str, opciones: Optional[dict] = None) -> str:
    """Devuelve un bloque [DATOS CALCULADOS] cuando la pregunta activa un trigger.

    Vacio si ningun trigger aplica.
    """
    text = question_text
    if opciones:
        text = text + " " + " ".join(opciones.values() if isinstance(opciones, dict) else opciones)
    hits: list[str] = []

    year = _extract_year(text)
    amount = _extract_amount_cop(text)

    # Trigger 1: cuantia (requiere monto)
    if _CUANTIA_RE.search(text) and amount:
        c = cuantia_cgp(amount, year)
        if c:
            hits.append(
                f"- Monto {amount:,.0f} COP equivale a {c['smlmv_equivalente']} SMLMV "
                f"del ano {c['anio']} ({c['smlmv_del_anio']:,} COP por SMLMV)."
                .replace(",", ".")
            )
            hits.append(
                f"- Segun Art. 25 CGP: hasta {CUANTIA_UMBRAL_MINIMA} SMLMV = minima cuantia; "
                f"hasta {CUANTIA_UMBRAL_MENOR} SMLMV = menor cuantia; mas = mayor cuantia. "
                f"Por tanto esta cuantia es de tipo **{c['clase']}**."
            )

    # Trigger 2: SMLMV (solo numero del ano)
    elif _SMLMV_RE.search(text):
        v = SMLMV.get(year)
        if v:
            hits.append(f"- SMLMV del ano {year} = {v:,} COP (dato oficial Mintrabajo).".replace(",", "."))

    # Trigger 3: plazo / liquidacion unilateral
    if _PLAZO_RE.search(text):
        # Documentar los plazos tipicos citados en la hackathon:
        hits.append(
            "- Art. 11 Ley 1150 de 2007: liquidacion bilateral del contrato estatal = "
            "dentro de los 4 meses siguientes a la terminacion."
        )
        hits.append(
            "- Art. 11 Ley 1150 de 2007: liquidacion unilateral = "
            "dentro de los 2 meses siguientes al vencimiento del plazo para la bilateral."
        )
        hits.append(
            "- Art. 11 Ley 1150 de 2007: liquidacion por mutuo acuerdo despues de la unilateral = "
            "hasta 2 anos despues del vencimiento del plazo de liquidacion unilateral."
        )

    if not hits:
        return ""
    return "[DATOS CALCULADOS]\n" + "\n".join(hits)


# ==================================================================
# Herramientas v2 (Cerberus Mark 43, --herramientas-v2). tool_block queda igual para reproducir Mark 42.
#   - SMLMV 2026 = $1.750.905 (Decreto 0159 de 2026; la tabla v1 tenía 1.500.000).
#   - UVT 2015-2026 (resoluciones anuales de la DIAN).
#   - Sin año en la pregunta: se calcula con los dos años más recientes en vez de suponer 2024; si la
#     clasificación de cuantía coincide en ambos, se da una sola respuesta.
#   - Los plazos de liquidación (art. 11 Ley 1150 de 2007) solo se agregan si la pregunta habla de liquidar
#     un contrato: con cualquier «plazo» desorientaban preguntas de tutela, procesales o laborales.
# ==================================================================
SMLMV_V2: dict[int, int] = {**SMLMV, 2026: 1750905}
UVT: dict[int, int] = {
    2015: 28279, 2016: 29753, 2017: 31859, 2018: 33156, 2019: 34270, 2020: 35607,
    2021: 36308, 2022: 38004, 2023: 42412, 2024: 47065, 2025: 49799, 2026: 52374,
}
_UVT_RE = re.compile(r"\buvt\b|unidad(?:es)?\s+de\s+valor\s+tributario", re.I)
_CANTIDAD_UVT_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+)\s*(?:uvt\b|unidades\s+de\s+valor\s+tributario)", re.I)
_LIQUIDACION_RE = re.compile(r"liquidaci[óo]n\s+(?:unilateral|bilateral|del\s+contrato|de\s+(?:los|el)\s+contratos?)"
                             r"|liquidar\s+(?:el|los)\s+contratos?", re.I)


def _anios_v2(text: str) -> list[int]:
    m = _YEAR_RE.search(text)
    if m and int(m.group(1)) in SMLMV_V2:
        return [int(m.group(1))]
    ultimo = max(SMLMV_V2)
    return [ultimo, ultimo - 1]


def _cop(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def _clase(smlmv: float) -> str:
    return "minima" if smlmv <= CUANTIA_UMBRAL_MINIMA else "menor" if smlmv <= CUANTIA_UMBRAL_MENOR else "mayor"


def tool_block_v2(question_text: str, opciones: Optional[dict] = None) -> str:
    text = question_text
    if opciones:
        text = text + " " + " ".join(opciones.values() if isinstance(opciones, dict) else opciones)
    hits: list[str] = []
    anios = _anios_v2(text)
    amount = _extract_amount_cop(text)

    if _CUANTIA_RE.search(text) and amount:
        clases = []
        for anio in anios:
            smlmv = round(amount / SMLMV_V2[anio], 2)
            clases.append(_clase(smlmv))
            hits.append(f"- Monto {_cop(amount)} COP equivale a {smlmv} SMLMV del ano {anio} "
                        f"({_cop(SMLMV_V2[anio])} COP por SMLMV).")
        hits.append(f"- Segun Art. 25 CGP: hasta {CUANTIA_UMBRAL_MINIMA} SMLMV = minima cuantia; hasta "
                    f"{CUANTIA_UMBRAL_MENOR} SMLMV = menor cuantia; mas = mayor cuantia.")
        if len(set(clases)) == 1:
            hits.append(f"- Por tanto esta cuantia es de tipo **{clases[0]}**.")
        else:
            hits.append("- La clase depende del ano: " + "; ".join(f"{a}: {c}" for a, c in zip(anios, clases)) + ".")
    elif _SMLMV_RE.search(text):
        for anio in anios:
            linea = f"- SMLMV del ano {anio} = {_cop(SMLMV_V2[anio])} COP"
            hits.append(linea + (f"; {_cop(amount)} COP = {round(amount / SMLMV_V2[anio], 2)} SMLMV." if amount else "."))

    if _UVT_RE.search(text):
        cantidad = _CANTIDAD_UVT_RE.search(text)
        for anio in anios:
            linea = f"- UVT del ano {anio} = {_cop(UVT[anio])} COP"
            if amount:
                linea += f"; {_cop(amount)} COP = {round(amount / UVT[anio], 2)} UVT"
            if cantidad:
                n = float(cantidad.group(1).replace(".", "").replace(",", "."))
                linea += f"; {cantidad.group(1)} UVT = {_cop(n * UVT[anio])} COP"
            hits.append(linea + ".")

    if _LIQUIDACION_RE.search(text):
        hits.append("- Art. 11 Ley 1150 de 2007: liquidacion bilateral del contrato estatal = "
                    "dentro de los 4 meses siguientes a la terminacion.")
        hits.append("- Art. 11 Ley 1150 de 2007: liquidacion unilateral = "
                    "dentro de los 2 meses siguientes al vencimiento del plazo para la bilateral.")
        hits.append("- Art. 11 Ley 1150 de 2007: liquidacion por mutuo acuerdo despues de la unilateral = "
                    "hasta 2 anos despues del vencimiento del plazo de liquidacion unilateral.")

    if not hits:
        return ""
    return "[DATOS CALCULADOS]\n" + "\n".join(hits)
