from legalrag.citations.extract import fold

SCHEMAS = {
    "multiple_choice": {"respuesta_correcta": "A|B|C|D", "justificacion": "texto con fundamento",
                        "descarte_opciones": {"letra incorrecta": "razón breve basada en evidencia"}},
    "semi_open": {"respuesta": "3 a 5 oraciones, máximo 150 palabras",
                  "palabras_clave": ["concepto"], "referencia_legal": "norma y artículo"},
    "open_ended": {"marco_normativo": "normas aplicables", "analisis": "5 a 8 oraciones",
                   "jurisprudencia": "precedente presente en el contexto o indicación de su ausencia",
                   "conclusion": "respuesta al caso"},
}


def classify(question):
    kind = question.get("formato")
    if kind is None:
        kind = "multiple_choice" if question.get("opciones") else "semi_open"
    if kind not in SCHEMAS:
        raise ValueError(f"Formato desconocido: {kind}")
    complexity = fold(question.get("complejidad") or "")
    complex_question = complexity in {"alta", "high", "3"} or kind == "open_ended"
    return {"formato": kind, "compleja": complex_question, "esquema": SCHEMAS[kind]}


def blind_question(question):
    allowed = {"id", "pregunta", "opciones", "formato", "area", "tema", "complejidad", "sub_tarea"}
    return {key: value for key, value in question.items() if key in allowed}
