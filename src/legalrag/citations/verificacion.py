"""Del texto del decoder al objeto de entrega, con las reglas del evaluador oficial.

Cómo puntúa el evaluador (data/oficial/scripts/evaluate.py y citations.py):
- Cerradas: una abstención vale 0 en exactitud. Por eso siempre se entrega una letra.
- Citas: se comparan a nivel de norma, sin artículo. Una cita que no está en el fundamento de
  referencia ni en el texto de los 10 primeros pasajes resta el doble; una que está en los
  pasajes nunca resta. Por eso no se descarta la respuesta: se quita solo la cita sin respaldo
  (`generation.politica.sanear_cita`) y cada pasaje entregado empieza con el nombre de su norma.
- Abstención: 1 si acierta, 0,5 si se abstiene, 0 si falla; pero abstenerse en texto libre deja
  en 0 el juez RAGAS (30 puntos) y las citas de ese ítem. Solo se abstiene sin evidencia.

El posprocesado es el de la versión 05 (`generation.politica.postprocesar`), que con el corpus
anterior subió el puntaje de 11 a 30.
"""
import json
import re

from legalrag.generation.cliente import CAMPOS

CAMPOS_PASAJE = ("doc_id", "inicio", "fin", "texto", "score")


_CLAVE_SIN_COMILLAS = re.compile(r'([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:')


def _candidatos_json(texto):
    """Variantes del texto a intentar, de la más fiel a la más corregida."""
    texto = (texto or "").strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.I)
    texto = re.sub(r"^\{\s*\{", "{", texto)  # "{{" cuando el modelo repite la llave ya abierta
    yield texto
    # Claves sin comillas ({justificacion: "..."}) y comilla de clave cerrada tarde ("referencia_legal:").
    corregido = _CLAVE_SIN_COMILLAS.sub(lambda m: f'{m.group(1)}"{m.group(2)}":', texto)
    # "clave:"[...] o "clave:"123 -> "clave": [...]; "clave:"texto" -> "clave":"texto"
    corregido = re.sub(r'"([A-Za-z_]+):"(?=\s*[\[{\d-])', r'"\1": ', corregido)
    yield re.sub(r'"([A-Za-z_]+):"', r'"\1":"', corregido)


def interpretar_json(texto):
    """Primer objeto JSON del texto del modelo: tolera bloque de código, texto alrededor, claves sin
    comillas, llave repetida, un segundo objeto al final y truncamiento."""
    from legalrag.generation.politica import reparar_json

    decodificador = json.JSONDecoder()
    for candidato in _candidatos_json(texto):
        inicio = candidato.find("{")
        if inicio < 0:
            continue
        try:
            objeto, _ = decodificador.raw_decode(candidato[inicio:])
        except json.JSONDecodeError:
            objeto, _ = reparar_json(candidato[inicio:])
        if isinstance(objeto, dict):
            return objeto
    raise ValueError("La salida no contiene un objeto JSON")


def justificacion_de(texto):
    """Texto de la justificación aunque el JSON venga incompleto."""
    try:
        return _texto(interpretar_json(texto).get("justificacion"))
    except ValueError:
        m = re.search(r'justificacion"?\s*:\s*"(.*?)(?<!\\)"', texto or "", flags=re.S)
        return m.group(1) if m else ""


def letra_de(texto, letras):
    """Letra elegida aunque el JSON no se pueda leer (`"respuesta_correcta": "B"` o sin comillas)."""
    m = re.search(r'respuesta_correcta"?\s*:\s*"?([A-Z])\b', texto or "")
    return m.group(1) if m and m.group(1) in letras else None


def _texto(valor):
    if isinstance(valor, str):
        return valor
    if isinstance(valor, list):
        return " ".join(_texto(v) for v in valor if v not in (None, ""))
    if isinstance(valor, dict):
        return " ".join(_texto(v) for v in valor.values())
    return "" if valor is None else str(valor)


def normalizar_campos(salida, formato):
    """Tipos del esquema aunque el modelo devuelva listas o números donde van textos."""
    salida = dict(salida)
    for campo in CAMPOS[formato]:
        if campo not in salida:
            continue
        valor = salida[campo]
        if campo == "palabras_clave":
            valor = valor if isinstance(valor, list) else re.split(r"\s*[,;]\s*", _texto(valor))
            salida[campo] = [_texto(v).strip() for v in valor if _texto(v).strip()]
        elif campo == "descarte_opciones":
            salida[campo] = {str(k).strip()[:1].upper(): _texto(v) for k, v in valor.items()} \
                if isinstance(valor, dict) else {}
        elif campo == "respuesta_correcta":
            salida[campo] = _texto(valor).strip()[:1].upper() or None
        else:
            salida[campo] = _texto(valor)
    return salida


def pasajes_de_entrega(pasajes, evidencia=None):
    """Campos del esquema; el texto empieza con el encabezado de la norma si el pasaje lo tiene."""
    from legalrag.citations.normas import EvidenciaCorpus

    return [{**{k: p[k] for k in CAMPOS_PASAJE}, "texto": EvidenciaCorpus.texto_entregado(p)} for p in pasajes]


def abstencion(entrada, pasajes, letra=None):
    """Abstención válida para el esquema oficial, con la evidencia conservada."""
    respuesta = {"id": entrada["id"], "formato": entrada["formato"], "abstencion": True,
                 **{campo: "" for campo in CAMPOS[entrada["formato"]]}}
    if entrada["formato"] == "multiple_choice":
        letras = list((entrada.get("opciones") or {"A": ""}).keys())
        respuesta.update(respuesta_correcta=letra if letra in letras else letras[0], descarte_opciones={})
    if entrada["formato"] == "semi_open":
        respuesta["palabras_clave"] = []
    respuesta["pasajes_recuperados"] = pasajes_de_entrega(pasajes)
    return respuesta


CAMPO_FUNDAMENTO = {"multiple_choice": "justificacion", "semi_open": "referencia_legal", "open_ended": "marco_normativo"}


def texto_citable(respuesta):
    """Texto del que el evaluador oficial extrae las citas (evaluate.answer_text)."""
    campos = {"multiple_choice": ("justificacion",), "semi_open": ("respuesta", "referencia_legal"),
              "open_ended": ("marco_normativo", "analisis", "jurisprudencia", "conclusion")}[respuesta["formato"]]
    return " ".join(_texto(respuesta.get(c)) for c in campos)


def citar_respaldo(respuesta, pasajes, evidencia, maximo_abiertas=6):
    """Agrega al fundamento toda norma que el evaluador ya reconoce en los 10 primeros pasajes.

    Incluye las normas que un pasaje *menciona* (no solo la norma a la que pertenece): la referencia
    suele estar ahí. Ninguna puede restar, porque todas figuran en la evidencia. En abiertas el
    campo lo lee el juez RAGAS, así que se limita a `maximo_abiertas`; en semiabiertas va a
    `referencia_legal`, que el juez no lee. Devuelve cuántas normas agregó.
    """
    from legalrag.citations.evidencia import nombre_cuerpo
    from legalrag.citations.normas import EvidenciaCorpus
    from legalrag.generation.politica import _anexar

    citas = evidencia.citas
    respaldo = evidencia.respaldo(pasajes)
    ya = citas.bodies(citas.extract(texto_citable(respuesta)))
    nombres = []
    for pasaje in pasajes[:10]:
        for cuerpo in sorted(citas.bodies(citas.extract(EvidenciaCorpus.texto_entregado(pasaje))), key=str):
            nombre = nombre_cuerpo(cuerpo)
            if cuerpo in ya or not nombre or nombre in nombres or citas.bodies(citas.extract(nombre)) != {cuerpo}:
                continue
            nombres.append(nombre)
    if respuesta["formato"] == "open_ended":
        nombres = nombres[:maximo_abiertas]
    campo = CAMPO_FUNDAMENTO[respuesta["formato"]]
    nuevo = _anexar(respuesta.get(campo, ""), "; ".join(nombres), "Normas de la evidencia: ")
    if not nombres or citas.bodies(citas.extract(nuevo)) - respaldo - ya:
        return 0
    respuesta[campo] = nuevo
    return len(nombres)


def respuesta_final(entrada, crudo, pasajes, evidencia, politica_config, validador_oficial):
    """Devuelve (respuesta, registro). `registro["problema"]` es None si salió sin arreglos."""
    from legalrag.generation.politica import postprocesar

    letras = list((entrada.get("opciones") or {}).keys())
    registro = {"problema": None}
    try:
        salida = normalizar_campos(interpretar_json(crudo), entrada["formato"])
    except ValueError:
        salida, registro["problema"] = {}, "json_invalido"
    if entrada["formato"] == "multiple_choice" and salida.get("respuesta_correcta") not in letras:
        letra = letra_de(crudo, letras)
        if letra:
            salida["respuesta_correcta"] = letra
    modo = str(politica_config.get("citar_evidencia", "todas")).strip().lower()
    if modo not in ("todas", "usadas", "ninguna", "respaldo") and not modo.isdigit():
        raise ValueError(f'citar_evidencia="{politica_config.get("citar_evidencia")}" no es válido: usar '
                         '"respaldo", "todas", "usadas", "ninguna" o un número de normas')
    respaldo_completo = modo == "respaldo"
    politica_base = {**politica_config, "citar_evidencia": "todas" if respaldo_completo else modo}
    respuesta, detalle = postprocesar(entrada, salida, pasajes, evidencia, politica_base)
    respuesta["pasajes_recuperados"] = pasajes_de_entrega(pasajes)
    if respaldo_completo:
        registro["normas_agregadas"] = citar_respaldo(respuesta, pasajes, evidencia,
                                                      politica_config.get("maximo_abiertas", 6))
    registro.update(oraciones_eliminadas=detalle.get("oraciones_eliminadas", []),
                    campos_rellenados=detalle.get("campos_rellenados", []))
    if registro["problema"] is None and (registro["oraciones_eliminadas"] or registro["campos_rellenados"]):
        registro["problema"] = "citas_saneadas" if registro["oraciones_eliminadas"] else "campos_rellenados"
    errores = list(validador_oficial.iter_errors(respuesta))
    if errores:
        registro["problema"] = "esquema_oficial:" + errores[0].message[:120]
        return abstencion(entrada, pasajes, respuesta.get("respuesta_correcta")), registro
    return respuesta, registro


def resumen_json(respuesta):
    return json.dumps({k: v for k, v in respuesta.items() if k != "pasajes_recuperados"}, ensure_ascii=False)
