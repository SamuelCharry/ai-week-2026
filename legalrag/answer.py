"""Generación de la respuesta en los tres formatos del Anexo A del enunciado,
con verificación de citas y abstención.

Formatos y claves (obligatorias, no se renombran):
  multiple_choice: respuesta_correcta, justificacion, descarte_opciones
  semi_open:       respuesta, palabras_clave, referencia_legal
  open_ended:      marco_normativo, analisis, jurisprudencia, conclusion
Todos llevan id, formato, abstencion y pasajes_recuperados.
"""
import re
import time

from legalrag.citations import extract_citations, format_citation, is_supported

CAMPOS = {
    "multiple_choice": ["respuesta_correcta", "justificacion", "descarte_opciones"],
    "semi_open": ["respuesta", "palabras_clave", "referencia_legal"],
    "open_ended": ["marco_normativo", "analisis", "jurisprudencia", "conclusion"],
}
VACIO = {"respuesta_correcta": "", "justificacion": "", "descarte_opciones": {}, "respuesta": "",
         "palabras_clave": [], "referencia_legal": "", "marco_normativo": "", "analisis": "",
         "jurisprudencia": "", "conclusion": ""}
MAX_PALABRAS_SEMI = 150

SYSTEM = (
    "Eres un asistente jurídico para derecho colombiano. Respondes únicamente con base en los "
    "pasajes normativos numerados que se te entregan. Cuando cites una norma, usa solo artículos "
    "que aparezcan en esos pasajes y escribe la cita completa, por ejemplo 'artículo 10 de la "
    "Ley 1581 de 2012'. No cites normas ni sentencias de memoria. Si los pasajes no permiten "
    "responder, devuelve abstencion = true y deja los demás campos vacíos. Responde en español."
)

INSTRUCCIONES = {
    "multiple_choice": (
        "Pregunta de selección múltiple. En respuesta_correcta pon solo la letra de la opción "
        "correcta. En justificacion explica en 2 a 4 oraciones por qué, citando el artículo. En "
        "descarte_opciones pon, para cada opción incorrecta (clave = letra), una frase de por qué "
        "se descarta."
    ),
    "semi_open": (
        "Pregunta semiabierta. En respuesta escribe de 3 a 5 oraciones y máximo 150 palabras. "
        "En palabras_clave pon de 3 a 6 términos jurídicos. En referencia_legal pon la norma y "
        "el artículo que fundamentan la respuesta."
    ),
    "open_ended": (
        "Pregunta abierta sobre un caso. En marco_normativo enumera las normas aplicables. En "
        "analisis escribe de 5 a 8 oraciones aplicando esas normas al caso. En jurisprudencia "
        "menciona solo sentencias que aparezcan en los pasajes; si no hay, deja cadena vacía. "
        "En conclusion responde el problema jurídico."
    ),
}


def _schema(formato: str) -> dict:
    s = {"type": "string"}
    props = {
        "multiple_choice": {"respuesta_correcta": s, "justificacion": s,
                            "descarte_opciones": {"type": "object", "additionalProperties": s}},
        "semi_open": {"respuesta": s, "palabras_clave": {"type": "array", "items": s}, "referencia_legal": s},
        "open_ended": {"marco_normativo": s, "analisis": s, "jurisprudencia": s, "conclusion": s},
    }[formato]
    props = {"abstencion": {"type": "boolean"}, **props}
    return {"type": "object", "properties": props, "required": list(props)}


def build_user_prompt(item: dict, pasajes: list[dict]) -> str:
    bloques = []
    for n, p in enumerate(pasajes, start=1):
        head = f"[{n}] {p['norma']}, artículo {p['articulo']}"
        if p["partes"] > 1:
            head += f" (parte {p['parte']} de {p['partes']})"
        bloques.append(f"{head}\n{p['texto']}")
    partes = ["PASAJES:\n\n" + "\n\n".join(bloques), f"PREGUNTA:\n{item['pregunta']}"]
    if item.get("opciones"):
        partes.append("OPCIONES:\n" + "\n".join(f"{k}) {v}" for k, v in item["opciones"].items()))
    partes.append(INSTRUCCIONES[item["formato"]])
    return "\n\n".join(partes)


def abstention(item: dict) -> dict:
    out = {"id": item["id"], "formato": item["formato"], "abstencion": True}
    out.update({k: VACIO[k] for k in CAMPOS[item["formato"]]})
    out["pasajes_recuperados"] = []
    return out


def _limit_words(texto: str, maximo: int) -> str:
    palabras = texto.split()
    if len(palabras) <= maximo:
        return texto.strip()
    corto = " ".join(palabras[:maximo])
    fin = corto.rfind(".")  # termina en la última oración completa, si no queda muy corta
    return corto[:fin + 1] if fin > len(corto) // 2 else corto


def _text_fields(out: dict) -> str:
    partes = []
    for k in ("justificacion", "respuesta", "referencia_legal", "marco_normativo", "analisis",
              "jurisprudencia", "conclusion"):
        partes.append(str(out.get(k, "")))
    partes += [str(v) for v in out.get("descarte_opciones", {}).values()]
    return "\n".join(partes)


def postprocess(item: dict, raw: dict, pasajes: list[dict], aliases: dict, nombres: dict) -> tuple[dict, dict]:
    formato = item["formato"]
    if raw.get("abstencion") is True:
        return abstention(item), {"motivo_abstencion": "el modelo indicó evidencia insuficiente"}

    out = {"id": item["id"], "formato": formato, "abstencion": False}
    for k in CAMPOS[formato]:
        out[k] = raw.get(k, VACIO[k])

    if formato == "multiple_choice":
        letra = str(out["respuesta_correcta"]).strip().upper()[:1]
        validas = [k.upper() for k in (item.get("opciones") or {})]
        if validas and letra not in validas:
            return abstention(item), {"motivo_abstencion": f"letra inválida {out['respuesta_correcta']!r}"}
        out["respuesta_correcta"] = letra
        out["descarte_opciones"] = {k.upper(): str(v) for k, v in dict(out["descarte_opciones"]).items()
                                    if k.upper() != letra and (not validas or k.upper() in validas)}
    if formato == "semi_open":
        out["respuesta"] = _limit_words(str(out["respuesta"]), MAX_PALABRAS_SEMI)
        out["palabras_clave"] = [str(p) for p in out["palabras_clave"]][:6]

    citas = extract_citations(_text_fields(out), aliases)
    respaldadas = [c for c in citas if is_supported(c, pasajes)]
    sin_respaldo = [c for c in citas if not is_supported(c, pasajes)]
    traza = {"citas_respaldadas": respaldadas, "citas_sin_respaldo": sin_respaldo}

    if formato == "semi_open":
        # la referencia legal se arma solo con citas que están en los pasajes recuperados
        con_norma = [c for c in respaldadas if c["norma"]]
        out["referencia_legal"] = "; ".join(format_citation(c, nombres) for c in con_norma)
    if formato != "multiple_choice" and citas and not respaldadas:
        traza["motivo_abstencion"] = "ninguna norma citada figura en los pasajes recuperados"
        return abstention(item), traza

    out["pasajes_recuperados"] = [
        {"doc_id": p["doc_id"], "inicio": p["inicio"], "fin": p["fin"], "texto": p["texto"], "score": p["score"]}
        for p in pasajes
    ]
    return out, traza


def answer(item: dict, retriever, llm, aliases: dict, nombres: dict, k: int = 10) -> tuple[dict, dict]:
    """Recupera, genera y verifica. Devuelve (objeto JSON de entrega, traza para depurar)."""
    t0 = time.time()
    pasajes = retriever.search(item["pregunta"], k=k)
    if not pasajes:
        out, traza = abstention(item), {"motivo_abstencion": "sin pasajes recuperados"}
    else:
        raw = llm(SYSTEM, build_user_prompt(item, pasajes), _schema(item["formato"]))
        out, traza = postprocess(item, raw, pasajes, aliases, nombres)
    traza["segundos"] = round(time.time() - t0, 2)
    traza["pasajes"] = [{"rank": n, "chunk_id": p["chunk_id"], "norma": p["norma"], "articulo": p["articulo"],
                         "parte": f"{p['parte']}/{p['partes']}", "url": p["url"], "score": p["score"]}
                        for n, p in enumerate(pasajes, start=1)]
    return out, traza


def validate(obj: dict) -> list[str]:
    """Chequeo provisional de forma mientras llega schema/submission.schema.json."""
    errores = []
    formato = obj.get("formato")
    if formato not in CAMPOS:
        return [f"formato inválido: {formato!r}"]
    for k in ["id", "abstencion", *CAMPOS[formato], "pasajes_recuperados"]:
        if k not in obj:
            errores.append(f"falta la clave {k}")
    if not isinstance(obj.get("abstencion"), bool):
        errores.append("abstencion debe ser booleano")
    tipos = {"descarte_opciones": dict, "palabras_clave": list}
    for k in CAMPOS[formato]:
        if k in obj and not isinstance(obj[k], tipos.get(k, str)):
            errores.append(f"{k} tiene tipo {type(obj[k]).__name__}")
    for p in obj.get("pasajes_recuperados", []):
        for k in ("doc_id", "texto", "score"):
            if k not in p:
                errores.append(f"pasaje sin {k}")
    if formato == "semi_open" and len(str(obj.get("respuesta", "")).split()) > MAX_PALABRAS_SEMI:
        errores.append("respuesta supera 150 palabras")
    return errores
