"""Prompts del agente reformulador de Cerberus: con qué figura y normas se resuelve el caso, como nueva consulta.

En las preguntas de caso («ampliar la avenida 68») el texto no se parece al de la norma («acciones populares»):
la auditoría de fallas (src/auditar_fallas.py) mostró que la norma correcta no llegaba a los candidatos, aunque el
reranker la habría aceptado. El enunciado (anexo B) propone justamente reformular «con el nombre del cuerpo
normativo probable» y repetir la recuperación. Tres formas, en `recuperacion.agente_expansion`:

    "iterativo"   respuesta breve escrita con los primeros pasajes a la vista (ITER-RETGEN, Shao et al., 2023)
    "normas"      lista de normas y artículos aplicables, de memoria (Rewrite-Retrieve-Read, Ma et al., 2023);
                  Qwen3-8B nombró leyes que no aplican o no existen, por eso se pasó a "iterativo"
    "hipotesis"   respuesta breve sin pasajes (HyDE, Gao et al., 2023; Query2doc, Wang et al., 2023)

Las normas que nombra el texto entran como candidatos al reranker y con lugares reservados (retrieval.hibrido);
el reranker sigue puntuando contra la pregunta y una norma equivocada no resta en citas.
"""

SISTEMA_HIPOTESIS = "Eres un abogado colombiano experto. Respondes en español, en texto corrido y sin JSON."

INSTRUCCION_HIPOTESIS = (
    "Sin consultar documentos, responde en máximo tres oraciones la pregunta siguiente y nombra las "
    "normas colombianas que la regulan (tipo, número y año, y el artículo si lo conoces).")

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


def mensajes_iterativo(entrada, bloque_pasajes):
    """Reformulador anclado en la evidencia (ITER-RETGEN, Shao et al., 2023): una respuesta breve escrita con los
    primeros pasajes a la vista, que se usa como nueva consulta. Pedir las normas de memoria falló: Qwen3-8B nombró
    leyes que no aplican o no existen (pregunta 247: «Ley 1295 de 2009» en vez de la Ley 472 de 1998)."""
    opciones = "\n".join(f"{letra}) {texto}" for letra, texto in (entrada.get("opciones") or {}).items())
    pregunta = entrada["pregunta"].strip() + ("\n" + opciones if opciones else "")
    return [{"role": "system", "content": SISTEMA_HIPOTESIS},
            {"role": "user", "content": (
                f"PASAJES DE UNA PRIMERA BÚSQUEDA\n{bloque_pasajes}\n\nPREGUNTA ({entrada.get('area', '')})\n{pregunta}"
                "\n\nCon base en esos pasajes, responde en máximo tres oraciones y nombra la figura jurídica y las normas "
                "colombianas que regulan el caso (tipo, número y año, y el artículo si aparece). Si los pasajes no "
                "bastan, nombra la figura jurídica que habría que buscar.")}]


SISTEMA_SUFICIENCIA = ("Eres un abogado colombiano que decide si unos pasajes bastan para responder una pregunta "
                       "jurídica con fundamento. Respondes solo «Sí» o «No».")


def mensajes_suficiencia(entrada, bloque_pasajes):
    """Juicio de contexto suficiente (Joren et al., ICLR 2025, Google): ¿los pasajes contienen lo necesario para
    responder? Decide si el reformulador actúa, como pide el enunciado: la iteración extra solo cuando la primera
    recuperación resulte insuficiente. Se mide P(«Sí») del siguiente token, sin generar texto."""
    opciones = "\n".join(f"{letra}) {texto}" for letra, texto in (entrada.get("opciones") or {}).items())
    pregunta = entrada["pregunta"].strip() + ("\n" + opciones if opciones else "")
    return [{"role": "system", "content": SISTEMA_SUFICIENCIA},
            {"role": "user", "content": (
                f"PASAJES\n{bloque_pasajes}\n\nPREGUNTA ({entrada.get('area', '')})\n{pregunta}\n\n¿Los pasajes "
                "contienen la norma o la información jurídica necesaria para responder la pregunta con fundamento? "
                "Responde solo «Sí» o «No».")}]
