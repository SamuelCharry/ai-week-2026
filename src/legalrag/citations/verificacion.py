"""Del texto del decoder al objeto de entrega: JSON, citas respaldadas y abstención.

Reglas (enunciado, paso 4 y §6.1):
- Toda norma citada debe figurar en los pasajes recuperados. Una cita sin respaldo y
  fuera del fundamento resta el doble de un acierto, así que se prefiere abstenerse.
- La abstención deja vacíos los campos de contenido y conserva los pasajes, como en el
  ejemplo oficial, para que el jurado vea la evidencia que no alcanzó.
- En cerradas, el esquema oficial solo admite A-D en `respuesta_correcta` aunque su
  descripción mencione null. Con `abstencion: true` el evaluador la puntúa como
  abstención sin mirar la letra, así que se conserva la del modelo o se usa "A".
"""
import json
import re

from legalrag.generation.cliente import CAMPOS, esquema_local

CAMPOS_PASAJE = ("doc_id", "inicio", "fin", "texto", "score")
LETRAS = ("A", "B", "C", "D")


def interpretar_json(texto):
    texto = texto.strip()
    if texto.startswith("```"):
        texto = re.sub(r"^```(?:json)?\s*|\s*```$", "", texto, flags=re.I)
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        inicio, fin = texto.find("{"), texto.rfind("}")
        if inicio < 0 or fin < inicio:
            raise
        return json.loads(texto[inicio:fin + 1])


def pasajes_de_entrega(pasajes):
    return [{k: p[k] for k in CAMPOS_PASAJE} for p in pasajes]


def abstencion(entrada, pasajes, letra=None):
    respuesta = {"id": entrada["id"], "formato": entrada["formato"], "abstencion": True,
                 **{campo: "" for campo in CAMPOS[entrada["formato"]]}}
    if entrada["formato"] == "multiple_choice":
        respuesta.update(respuesta_correcta=letra if letra in LETRAS else "A", descarte_opciones={})
    if entrada["formato"] == "semi_open":
        respuesta["palabras_clave"] = []
    respuesta["pasajes_recuperados"] = pasajes_de_entrega(pasajes)
    return respuesta


def respuesta_final(entrada, crudo, pasajes, manifiesto, validador_oficial):
    """Devuelve (respuesta, problema). `problema` es None si la respuesta sale tal cual."""
    import jsonschema

    from legalrag.evaluation.entrega import auditar_citas

    try:
        salida = interpretar_json(crudo)
        if not isinstance(salida, dict):
            raise ValueError("La salida no es un objeto JSON")
        jsonschema.validate(salida, esquema_local(entrada["formato"]))
    except (ValueError, jsonschema.ValidationError) as error:
        return abstencion(entrada, pasajes), f"json_invalido:{type(error).__name__}"
    letra = salida.get("respuesta_correcta")
    if salida.get("abstencion"):
        return abstencion(entrada, pasajes, letra), "abstencion_del_modelo"
    respuesta = {"id": entrada["id"], "formato": entrada["formato"], **salida,
                 "pasajes_recuperados": pasajes_de_entrega(pasajes)}
    auditoria = auditar_citas(respuesta, manifiesto)
    if auditoria["sin_respaldo"] or auditoria["indeterminadas"]:
        return abstencion(entrada, pasajes, letra), "cita_sin_respaldo_o_indeterminada"
    errores = list(validador_oficial.iter_errors(respuesta))
    if errores:
        return abstencion(entrada, pasajes, letra), "esquema_oficial:" + errores[0].message[:120]
    return respuesta, None
