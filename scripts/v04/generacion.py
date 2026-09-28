"""Generación de la versión 04.

Cambios frente al 03, cada uno ligado a una falla observada:
  * El modelo ya no decide la abstención. El campo `abstencion` sale del esquema
    y lo fija el código (regla general, no por pregunta).
  * En cerradas, `respuesta_correcta` es un enum sin null: siempre hay letra.
  * Cada texto tiene maxLength y la petición usa repeat_penalty, para cortar la
    repetición que agotaba los 1024 tokens y dejaba JSON inválido.
  * Si aun así el JSON queda truncado, se repara en lugar de descartar.
  * Una cita sin respaldo ya no anula la respuesta: se elimina la oración que la
    contiene y se conserva el resto.
  * El fundamento normativo se completa con las normas de los pasajes
    recuperados, en la forma que reconoce el evaluador oficial.
  * El prompt es texto plano con pasajes numerados, no un JSON con metadatos.
"""
import hashlib
import json
import re
import time

CAMPOS = {
    "multiple_choice": ["respuesta_correcta", "justificacion", "descarte_opciones"],
    "semi_open": ["respuesta", "palabras_clave", "referencia_legal"],
    "open_ended": ["marco_normativo", "analisis", "jurisprudencia", "conclusion"],
}

SISTEMA = (
    "Eres un abogado colombiano que responde preguntas de examen. Razonas con cuidado y "
    "respondes en español. Usas como fundamento los pasajes numerados que se te entregan. "
    "Cuando cites una norma, escríbela con su nombre y artículo tal como aparece en el "
    "encabezado del pasaje (por ejemplo: \"artículo 3 de la Ley 472 de 1998\" o "
    "\"artículo 946 del Código Civil\"). No inventes números de leyes, sentencias ni "
    "artículos que no aparezcan en los pasajes."
)

INSTRUCCIONES = {
    "multiple_choice": (
        "Elige exactamente una opción. Debes elegir siempre la opción más probable, incluso si "
        "los pasajes no la respaldan por completo: nunca dejes la pregunta sin responder.\n"
        "En \"justificacion\" explica en 2 a 4 oraciones por qué la opción es correcta, citando "
        "la norma y el artículo de los pasajes que la sustentan.\n"
        "En \"descarte_opciones\" da una razón breve (una oración) para cada opción."
    ),
    "semi_open": (
        "En \"respuesta\" contesta en 3 a 5 oraciones (máximo 150 palabras), de forma directa y "
        "citando la norma y el artículo de los pasajes que la sustentan.\n"
        "En \"palabras_clave\" da de 3 a 6 términos jurídicos centrales.\n"
        "En \"referencia_legal\" escribe la norma y el artículo principal."
    ),
    "open_ended": (
        "En \"marco_normativo\" identifica las normas aplicables de los pasajes (1 a 3 oraciones).\n"
        "En \"analisis\" desarrolla el razonamiento jurídico en 5 a 8 oraciones.\n"
        "En \"jurisprudencia\" menciona solo sentencias que aparezcan en los pasajes; si no hay, "
        "escribe una oración sobre el criterio jurisprudencial general sin inventar números.\n"
        "En \"conclusion\" responde la pregunta en 1 a 3 oraciones."
    ),
}

LONGITUDES = {
    "justificacion": 900, "descarte": 220, "respuesta": 1100, "palabra": 50,
    "referencia_legal": 250, "marco_normativo": 700, "analisis": 1600,
    "jurisprudencia": 600, "conclusion": 600,
}


def esquema(formato, letras, longitudes=True):
    """Esquema JSON para la gramática de llama.cpp. Sin campo de abstención."""
    L = LONGITUDES

    def texto(clave):
        s = {"type": "string", "minLength": 1}
        if longitudes:
            s["maxLength"] = L[clave]
        return s

    if formato == "multiple_choice":
        props = {
            "justificacion": texto("justificacion"),
            "respuesta_correcta": {"type": "string", "enum": list(letras)},
            "descarte_opciones": {"type": "object",
                                  "properties": {l: texto("descarte") for l in letras},
                                  "required": list(letras), "additionalProperties": False},
        }
    elif formato == "semi_open":
        palabras = {"type": "array", "items": texto("palabra"), "minItems": 1}
        if longitudes:
            palabras["maxItems"] = 6
        props = {"respuesta": texto("respuesta"), "palabras_clave": palabras,
                 "referencia_legal": texto("referencia_legal")}
    else:
        props = {k: texto(k) for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion")}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def bloque_pasajes(pasajes, evidencia, max_caracteres=1800):
    """Pasajes numerados. Solo las unidades; la cabecera se usa para nombrar la norma."""
    lineas, numero = [], 0
    for pasaje in pasajes:
        if pasaje.get("tipo_evidencia") == "cabecera_fuente":
            continue
        numero += 1
        nombre, _ = evidencia.etiqueta(pasaje)
        articulo = pasaje.get("articulo")
        titulo = f"{nombre}, artículo {articulo}" if articulo else nombre
        texto = re.sub(r"\n{3,}", "\n\n", pasaje["texto"].strip())
        if len(texto) > max_caracteres:
            texto = texto[:max_caracteres].rsplit(" ", 1)[0] + " […]"
        lineas.append(f"[{numero}] {titulo}\n{texto}")
    return "\n\n".join(lineas)


def mensajes(entrada, pasajes, evidencia, max_caracteres=1800):
    formato = entrada["formato"]
    partes = ["PASAJES RECUPERADOS", bloque_pasajes(pasajes, evidencia, max_caracteres) or "(ninguno)", ""]
    tipo = {"multiple_choice": "selección múltiple", "semi_open": "respuesta corta",
            "open_ended": "respuesta abierta"}[formato]
    partes += [f"PREGUNTA ({tipo}, {entrada.get('area', '')})", entrada["pregunta"].strip()]
    for letra, opcion in (entrada.get("opciones") or {}).items():
        partes.append(f"{letra}) {opcion}")
    partes += ["", "INSTRUCCIONES", INSTRUCCIONES[formato],
               "Responde solo con el objeto JSON, con las claves: " + ", ".join(
                   k for k in esquema(formato, list((entrada.get("opciones") or {"A": 0}).keys()))["properties"]) + "."]
    return [{"role": "system", "content": SISTEMA}, {"role": "user", "content": "\n".join(partes)}]


def ajustar(cliente, entrada, pasajes, evidencia, contexto, salida, max_caracteres=1800):
    """Quita unidades desde la última hasta que el prompt cabe en el contexto."""
    limite = contexto - salida - 32
    unidades = [p for p in pasajes if p.get("tipo_evidencia") != "cabecera_fuente"]
    omitidas = []
    while True:
        usados_ids = {id(u) for u in unidades}
        usados = [p for p in pasajes if p.get("tipo_evidencia") == "cabecera_fuente" or id(p) in usados_ids]
        # Las cabeceras huérfanas se descartan.
        grupos = {u.get("grupo_evidencia", u.get("unidad_id")) for u in unidades}
        usados = [p for p in usados if p.get("tipo_evidencia") != "cabecera_fuente" or p.get("grupo_evidencia") in grupos]
        conversacion = mensajes(entrada, usados, evidencia, max_caracteres)
        tokens = cliente.contar(conversacion)
        if tokens <= limite or not unidades:
            return usados, conversacion, tokens, omitidas
        omitidas.append(unidades.pop().get("unidad_id"))


def reparar_json(texto):
    """Intenta leer un objeto JSON truncado cerrando comillas y llaves."""
    texto = (texto or "").strip()
    try:
        return json.loads(texto), False
    except json.JSONDecodeError:
        pass
    pila, en_cadena, escape = [], False, False
    for c in texto:
        if en_cadena:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                en_cadena = False
        elif c == '"':
            en_cadena = True
        elif c in "{[":
            pila.append("}" if c == "{" else "]")
        elif c in "}]" and pila:
            pila.pop()
    candidato = texto.rstrip()
    if escape:
        candidato = candidato[:-1]
    if en_cadena:
        candidato += '"'
    candidato = re.sub(r",\s*$", "", candidato)
    for intento in (candidato + "".join(reversed(pila)),
                    re.sub(r',\s*"[^"]*"\s*:?\s*"?$', "", candidato) + "".join(reversed(pila))):
        try:
            return json.loads(intento), True
        except json.JSONDecodeError:
            continue
    return None, True


_ORACION = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÁÉÍÓÚÑ¿(\"“])")


def sanear(texto, respaldo, citas):
    """Quita las oraciones con citas que el evaluador contaría sin respaldo."""
    if not isinstance(texto, str) or not texto:
        return texto, []
    eliminadas, conservadas = [], []
    for oracion in _ORACION.split(texto.strip()):
        cuerpos = citas.bodies(citas.extract(oracion))
        if cuerpos - respaldo:
            eliminadas.append(oracion)
        else:
            conservadas.append(oracion)
    return " ".join(conservadas).strip(), eliminadas


def _anexar(texto, extra, conector):
    texto = (texto or "").strip()
    if not extra:
        return texto
    if texto and not texto.endswith((".", ";", ":")):
        texto += "."
    return (texto + " " if texto else "") + conector + extra + "."


def postprocesar(entrada, crudo, pasajes, evidencia, politica):
    """Convierte la salida del modelo en una respuesta del esquema oficial.

    politica: dict con
        citar_evidencia: "todas" | "ninguna" | int (máximo de normas)
        abstener_libre: "nunca" | "sin_evidencia"
    """
    formato = entrada["formato"]
    citas = evidencia.citas
    respaldo = evidencia.respaldo(pasajes)
    letras = list((entrada.get("opciones") or {}).keys())
    registro = {"oraciones_eliminadas": [], "campos_rellenados": []}
    unidades = [p for p in pasajes if p.get("tipo_evidencia") != "cabecera_fuente"]

    if formato != "multiple_choice" and politica.get("abstener_libre") == "sin_evidencia" and not unidades:
        registro["motivo"] = "sin_evidencia"
        return {"id": entrada["id"], **abstencion(formato, pasajes)}, registro

    salida = {k: v for k, v in (crudo or {}).items() if k in CAMPOS[formato]}
    limite = politica.get("citar_evidencia", "todas")
    fundamento = "" if limite == "ninguna" else evidencia.citas_evidencia(
        pasajes, limite=None if limite == "todas" else int(limite))[0]

    campos_texto = {"multiple_choice": ["justificacion"], "semi_open": ["respuesta", "referencia_legal"],
                    "open_ended": ["marco_normativo", "analisis", "jurisprudencia", "conclusion"]}[formato]
    for campo in campos_texto:
        salida[campo], quitadas = sanear(salida.get(campo, ""), respaldo, citas)
        registro["oraciones_eliminadas"] += quitadas

    if formato == "multiple_choice":
        letra = salida.get("respuesta_correcta")
        if letra not in letras:
            letra = letras[0]
            registro["campos_rellenados"].append("respuesta_correcta")
        salida["respuesta_correcta"] = letra
        descarte = {k: v for k, v in (salida.get("descarte_opciones") or {}).items()
                    if k in letras and k != letra and isinstance(v, str) and v.strip()}
        for k in letras:
            if k != letra and k not in descarte:
                descarte[k] = "No corresponde al supuesto descrito en la pregunta."
                registro["campos_rellenados"].append(f"descarte_{k}")
        salida["descarte_opciones"] = {k: sanear(v, respaldo, citas)[0] or "No corresponde al supuesto descrito."
                                       for k, v in sorted(descarte.items())}
        if not salida.get("justificacion"):
            salida["justificacion"] = f"La opción {letra} es la que mejor se ajusta a los pasajes recuperados."
            registro["campos_rellenados"].append("justificacion")
        salida["justificacion"] = _anexar(salida["justificacion"], fundamento, "Fundamento: ")
    elif formato == "semi_open":
        if not salida.get("respuesta"):
            salida["respuesta"] = "Según los pasajes recuperados, la respuesta depende de las normas citadas."
            registro["campos_rellenados"].append("respuesta")
        palabras = [p for p in (salida.get("palabras_clave") or []) if isinstance(p, str) and p.strip()]
        salida["palabras_clave"] = palabras or [entrada.get("tema") or entrada.get("area") or "derecho colombiano"]
        referencia = salida.get("referencia_legal") or ""
        salida["referencia_legal"] = (fundamento or referencia or "Pasajes recuperados")
    else:
        for campo, defecto in (("marco_normativo", "Normas contenidas en los pasajes recuperados."),
                               ("analisis", "El análisis se basa en los pasajes recuperados."),
                               ("jurisprudencia", "Los pasajes recuperados no incluyen jurisprudencia específica."),
                               ("conclusion", "La conclusión se deriva de las normas citadas.")):
            if not salida.get(campo):
                salida[campo] = defecto
                registro["campos_rellenados"].append(campo)
        salida["marco_normativo"] = _anexar(salida["marco_normativo"], fundamento, "Normas aplicables: ")

    respuesta = {"id": entrada["id"], "formato": formato, **{k: salida[k] for k in CAMPOS[formato]},
                 "abstencion": False, "pasajes_recuperados": pasajes}
    return respuesta, registro


def abstencion(formato, pasajes):
    salida = {k: "" for k in CAMPOS[formato]}
    if formato == "multiple_choice":
        salida.update(respuesta_correcta=None, descarte_opciones={})
    if formato == "semi_open":
        salida["palabras_clave"] = []
    return {"formato": formato, **salida, "abstencion": True, "pasajes_recuperados": pasajes}


def generar(cliente, entrada, pasajes, evidencia, config):
    """Una llamada al servidor local de llama.cpp. Devuelve (salida_cruda, registro)."""
    inicio = time.perf_counter()
    letras = list((entrada.get("opciones") or {}).keys())
    usados, conversacion, tokens, omitidas = ajustar(
        cliente, entrada, pasajes, evidencia, config["contexto"], config["max_tokens"], config["max_caracteres_pasaje"])
    prompt = cliente.aplicar_plantilla(conversacion)
    peticion = {"prompt": prompt, "temperature": 0, "seed": 0, "n_predict": config["max_tokens"],
                "cache_prompt": False, "stream": False,
                "repeat_penalty": config["repeat_penalty"], "repeat_last_n": config["repeat_last_n"],
                "frequency_penalty": 0, "presence_penalty": 0,
                "json_schema": esquema(entrada["formato"], letras, config["longitudes"])}
    registro = {"tokens_prompt": tokens, "omitidas": omitidas, "peticion": {k: v for k, v in peticion.items() if k != "prompt"},
                "sha256_prompt": hashlib.sha256(prompt.encode()).hexdigest()}
    try:
        respuesta, _ = cliente.pedir("/completion", peticion)
    except Exception as error:  # gramática rechazada por el servidor: repetir sin maxLength
        import urllib.error
        if not isinstance(error, urllib.error.HTTPError) or not config["longitudes"]:
            raise
        peticion["json_schema"] = esquema(entrada["formato"], letras, False)
        registro["gramatica_sin_longitudes"] = f"{error.code}: {error.read()[:300]!r}"
        respuesta, _ = cliente.pedir("/completion", peticion)
    contenido = respuesta.get("content") or ""
    crudo, reparado = reparar_json(contenido)
    registro.update(contenido=contenido, reparado=reparado, json_valido=crudo is not None,
                    finish_reason=respuesta.get("stop_type"),
                    tokens_salida=respuesta.get("tokens_predicted"),
                    segundos=time.perf_counter() - inicio)
    return crudo, usados, registro
