"""Pasos extra del agente alrededor de la respuesta, cada uno activable en configs/sistema.json.

Expansión de la consulta (Gao et al., ACL 2023, HyDE; Wang et al., EMNLP 2023, Query2doc): el decoder
escribe, sin ver documentos, una respuesta breve que nombra las normas que cree aplicables. Ese
texto se parece más a los pasajes que la pregunta y se usa como una consulta más (BM25, denso y
búsqueda dentro de las normas que nombra). El reranker sigue puntuando contra la pregunta, así que
una norma inventada en la hipótesis solo agrega candidatos y nunca llega a la respuesta sin pasaje.

Cuándo expandir (Yan et al., 2024, CRAG): la búsqueda extra solo se paga cuando la evidencia es
débil: el mejor puntaje del reranker queda bajo el umbral o entre los primeros pasajes no hay
ninguna ley, decreto, código o Constitución. Sin búsqueda web: el índice queda congelado.

Verificación en cadena (Dhuliawala et al., ACL Findings 2024, CoVe), versión "factorizada": del
borrador salen preguntas de verificación, se responden solo con los pasajes y sin ver el borrador,
y la respuesta se vuelve a generar con esas verificaciones a la vista.
"""
import re

SISTEMA_HIPOTESIS = "Eres un abogado colombiano experto. Respondes en español, en texto corrido y sin JSON."

INSTRUCCION_HIPOTESIS = (
    "Sin consultar documentos, responde en máximo tres oraciones la pregunta siguiente y nombra las "
    "normas colombianas que la regulan (tipo, número y año, y el artículo si lo conoces).")

SISTEMA_VERIFICACION = ("Eres un abogado colombiano que verifica respuestas jurídicas. Respondes en español, "
                        "en texto corrido y sin JSON.")


def mensajes_hipotesis(entrada):
    opciones = "\n".join(f"{letra}) {texto}" for letra, texto in (entrada.get("opciones") or {}).items())
    pregunta = entrada["pregunta"].strip() + ("\n" + opciones if opciones else "")
    return [{"role": "system", "content": SISTEMA_HIPOTESIS},
            {"role": "user", "content": f"{INSTRUCCION_HIPOTESIS}\n\nPREGUNTA ({entrada.get('area', '')})\n{pregunta}"}]


SISTEMA_REFORMULADOR = ("Eres un abogado colombiano experto en identificar qué normas regulan un caso. Respondes "
                        "solo con una lista de normas, sin explicaciones.")


def mensajes_reformulador(entrada):
    """Agente reformulador: lista las normas y artículos aplicables, para buscarlos por nombre (como hace el
    enrutador cuando la pregunta los nombra). La pregunta de un caso («ampliar la avenida 68») rara vez se
    parece al texto de la norma («acciones populares»)."""
    opciones = "\n".join(f"{letra}) {texto}" for letra, texto in (entrada.get("opciones") or {}).items())
    pregunta = entrada["pregunta"].strip() + ("\n" + opciones if opciones else "")
    return [{"role": "system", "content": SISTEMA_REFORMULADOR},
            {"role": "user", "content": (
                f"PREGUNTA ({entrada.get('area', '')})\n{pregunta}\n\nLista las normas colombianas que regulan este "
                "caso, una por línea, con tipo, número y año, y el artículo cuando lo sepas (por ejemplo: «Ley 472 de "
                "1998, artículo 2» o «Código General del Proceso, artículo 25»). Máximo 5 líneas, sin explicaciones.")}]


def evidencia_debil(pasajes, umbral, normativos=3, criterio="ambos"):
    """(débil, motivo) según CRAG. criterio: "puntaje" (mejor puntaje del reranker bajo el umbral), "normas"
    (ninguna ley, decreto o código entre los primeros) o "ambos" (cualquiera de los dos)."""
    from legalrag.retrieval.hibrido import es_normativo

    if not pasajes:
        return True, "sin_pasajes"
    mejor = max(p["score"] for p in pasajes)
    if criterio in ("puntaje", "ambos") and umbral is not None and mejor < umbral:
        return True, f"puntaje_maximo_{mejor:.2f}"
    if criterio in ("normas", "ambos") and not any(es_normativo(p.get("tipo")) for p in pasajes[:normativos]):
        return True, "sin_normas_entre_los_primeros"
    return False, None


def mensajes_preguntas(entrada, borrador, maximo=3):
    return [{"role": "system", "content": SISTEMA_VERIFICACION},
            {"role": "user", "content": (
                f"PREGUNTA\n{entrada['pregunta'].strip()}\n\nBORRADOR DE RESPUESTA\n{borrador.strip()}\n\n"
                f"Escribe hasta {maximo} preguntas cortas, una por línea y numeradas, que permitan comprobar las "
                "afirmaciones jurídicas centrales del borrador: qué norma y artículo aplica, plazos, requisitos, "
                "cuantías o autoridades competentes. Solo las preguntas.")}]


def preguntas_de(texto, maximo=3):
    lineas = [re.sub(r"^\s*(\d+[.)-]|[-*•])\s*", "", l).strip() for l in (texto or "").splitlines()]
    return [l for l in lineas if len(l) > 8 and l.endswith("?")][:maximo] or \
        [l for l in lineas if len(l) > 8][:maximo]


def mensajes_respuestas(entrada, preguntas, bloque_pasajes):
    """Las preguntas se responden solo con los pasajes, sin ver el borrador (CoVe factorizado)."""
    lista = "\n".join(f"{i}. {p}" for i, p in enumerate(preguntas, 1))
    return [{"role": "system", "content": SISTEMA_VERIFICACION},
            {"role": "user", "content": (
                f"PASAJES\n{bloque_pasajes}\n\nResponde cada pregunta en una oración usando solo los pasajes y "
                "citando la norma y el artículo del pasaje. Si los pasajes no lo dicen, escribe «Los pasajes no lo "
                f"indican».\n\n{lista}")}]


def bloque_verificacion(preguntas, respuestas):
    return ("VERIFICACIONES HECHAS SOBRE LOS PASAJES (si contradicen lo que ibas a responder, corrige; no "
            "agregues normas que no estén en los pasajes)\n" + "\n".join(f"- {p}" for p in preguntas) +
            "\n" + (respuestas or "").strip())


# --------------------------------------------------------------------------- multiagente
#
# Un segundo modelo de otra familia (≤ 8.000 M) actúa antes que el decoder principal:
#   juez de evidencia (MAIN-RAG, ACL 2025; L-MARS, 2025): ¿cada pasaje sirve para responder? Se juzga con
#       P(«Sí») del siguiente token, sin generar texto; los pasajes se reordenan (lo útil primero, Lost in the
#       Middle) y los que el juez descarta salen del prompt, no de la evidencia entregada.
#   segunda opinión en cerradas (ReConcile, ACL 2024): sus probabilidades de A-D se combinan con las del
#       principal, ponderadas por la confianza de cada uno. Elegir por probabilidades, no fusionar textos
#       (The Selection Bottleneck, 2026).

SISTEMA_JUEZ = ("Eres un abogado colombiano que evalúa si un pasaje sirve como fundamento para responder una "
                "pregunta jurídica. Respondes solo «Sí» o «No».")


def mensajes_juez(entrada, texto_pasaje, max_caracteres=1800):
    opciones = "\n".join(f"{letra}) {texto}" for letra, texto in (entrada.get("opciones") or {}).items())
    texto = texto_pasaje.strip()
    if len(texto) > max_caracteres:
        texto = texto[:max_caracteres].rsplit(" ", 1)[0] + " […]"
    return [{"role": "system", "content": SISTEMA_JUEZ},
            {"role": "user", "content": (
                f"PREGUNTA ({entrada.get('area', '')})\n{entrada['pregunta'].strip()}" + (f"\n{opciones}" if opciones else "")
                + f"\n\nPASAJE\n{texto}\n\n¿El pasaje contiene la norma o la información jurídica necesaria para "
                  "responder la pregunta? Responde solo «Sí» o «No».")}]


def ordenar_por_juez(pasajes, juicios, minimo=3, umbral=0.5):
    """(ordenados, para_el_prompt): todos los pasajes ordenados por P(«Sí») (empates: orden del reranker), y
    los que el juez acepta (P ≥ 0,5, la frontera natural entre «Sí» y «No»), nunca menos de `minimo`."""
    orden = sorted(range(len(pasajes)), key=lambda i: (-juicios[i], i))
    ordenados = [pasajes[i] for i in orden]
    aceptados = [pasajes[i] for i in orden if juicios[i] >= umbral]
    return ordenados, aceptados if len(aceptados) >= minimo else ordenados[:minimo]


def combinar_probabilidades(*distribuciones):
    """Promedio ponderado por la confianza de cada modelo (su probabilidad máxima), como ReConcile."""
    pesos = [max(d.values()) for d in distribuciones]
    letras = list(distribuciones[0])
    return {l: sum(w * d.get(l, 0.0) for w, d in zip(pesos, distribuciones)) / sum(pesos) for l in letras}
