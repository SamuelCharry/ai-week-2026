import hashlib
import json
import os
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


CAMPOS = {
    "multiple_choice": ["respuesta_correcta", "justificacion", "descarte_opciones"],
    "semi_open": ["respuesta", "palabras_clave", "referencia_legal"],
    "open_ended": ["marco_normativo", "analisis", "jurisprudencia", "conclusion"],
}
FORMATOS = {
    "multiple_choice": "Cerradas: respuesta_correcta (letra), justificacion, descarte_opciones.",
    "semi_open": "Semiabiertas: respuesta (de 3 a 5 oraciones, máximo 150 palabras), palabras_clave, referencia_legal.",
    "open_ended": "Abiertas: marco_normativo, analisis (de 5 a 8 oraciones), jurisprudencia, conclusion.",
}
INSTRUCCIONES = (
    "Requisito mínimo: producción del objeto JSON correspondiente a cada formato, con las claves que se indican a continuación.\n"
    "Las claves del objeto JSON son obligatorias y no admiten modificación, dado que sobre ellas opera el evaluador.\n"
    "Toda norma citada debe proceder de un pasaje efectivamente recuperado.\n"
    "El sistema debe disponer de un mecanismo de abstención, que registre abstencion: true cuando el corpus no proporcione fundamento suficiente."
)


def esquema_local(formato):
    campos = {campo: {"type": "string"} for campo in CAMPOS[formato]}
    campos["abstencion"] = {"type": "boolean"}
    if formato == "multiple_choice":
        campos["respuesta_correcta"] = {"type": ["string", "null"], "enum": ["A", "B", "C", "D", None]}
        campos["descarte_opciones"] = {"type": "object", "additionalProperties": {"type": "string"}}
    if formato == "semi_open":
        campos["palabras_clave"] = {"type": "array", "items": {"type": "string"}}
    return {"type": "object", "properties": campos, "required": list(campos), "additionalProperties": False}


def mensajes(entrada, pasajes):
    from scripts.auxiliares.evaluacion import preparar_entrada

    entrada = preparar_entrada(entrada)
    formato = entrada["formato"]
    if formato not in CAMPOS:
        raise ValueError("Formato desconocido")
    contexto = {"pregunta": entrada, "pasajes_recuperados": pasajes}
    sistema = INSTRUCCIONES + "\n" + FORMATOS[formato] + "\n" + json.dumps(esquema_local(formato), ensure_ascii=False)
    return [{"role": "system", "content": sistema},
            {"role": "user", "content": json.dumps(contexto, ensure_ascii=False, sort_keys=True)}]


class SinRedireccion(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("El servidor local no debe redirigir peticiones")


class ClienteLocal:
    def __init__(self, endpoint, timeout=180):
        url = urllib.parse.urlparse(endpoint)
        if url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("El decoder debe ejecutarse en un endpoint HTTP local")
        if url.username or url.password or url.query or url.fragment:
            raise ValueError("Endpoint local inválido")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}), SinRedireccion())

    def pedir(self, ruta, datos=None):
        cuerpo = None if datos is None else json.dumps(datos, ensure_ascii=False, allow_nan=False).encode("utf-8")
        peticion = urllib.request.Request(self.endpoint + ruta, data=cuerpo, headers={"Content-Type": "application/json"})
        with self.http.open(peticion, timeout=self.timeout) as respuesta:
            raw = respuesta.read()
        return json.loads(raw.decode("utf-8")), raw

    def aplicar_plantilla(self, conversacion):
        plantilla, _ = self.pedir("/apply-template", {"messages": conversacion,
                                                      "chat_template_kwargs": {"enable_thinking": False}})
        return plantilla["prompt"]

    def contar(self, conversacion):
        tokens, _ = self.pedir("/tokenize", {"content": self.aplicar_plantilla(conversacion), "add_special": True,
                                            "parse_special": True})
        return len(tokens["tokens"])


def ajustar_contexto(cliente, entrada, pasajes, contexto=4096, salida=512):
    limite = contexto - salida - 16
    if limite <= 0:
        raise ValueError("No queda espacio para la pregunta")
    elegidos, omitidos = [], []
    tokens = cliente.contar(mensajes(entrada, []))
    if tokens > limite:
        raise ValueError("La pregunta supera el contexto disponible")
    for pasaje in pasajes[:10]:
        if not pasaje.get("texto") or len(pasaje["texto"]) != pasaje["fin"] - pasaje["inicio"]:
            raise ValueError("El pasaje no conserva sus offsets")
        if pasaje.get("recuperar_unidad_completa") and (
                pasaje.get("unidad_inicio") != pasaje["inicio"] or pasaje.get("unidad_fin") != pasaje["fin"]):
            raise ValueError("La evidencia contiene una ventana sin recuperar su unidad completa")
        if any(c in pasaje and pasaje[c] != pasaje[limite] for c, limite in [("unidad_inicio", "inicio"), ("unidad_fin", "fin")]):
            raise ValueError("El pasaje está cortado dentro de su unidad")
        candidato = elegidos + [pasaje]
        cantidad = cliente.contar(mensajes(entrada, candidato))
        if cantidad <= limite:
            elegidos, tokens = candidato, cantidad
        else:
            omitidos.append({"unidad_id": pasaje["unidad_id"], "motivo": "unidad_completa_no_cabe",
                             "tokens_con_unidad": cantidad})
    return elegidos, omitidos, tokens


def abstenerse(formato):
    salida = {campo: "" for campo in CAMPOS[formato]}
    salida["abstencion"] = True
    if formato == "multiple_choice":
        salida.update(respuesta_correcta=None, descarte_opciones={})
    if formato == "semi_open":
        salida["palabras_clave"] = []
    return salida


def controlar_citas(salida, documentos):
    from scripts.auxiliares.evaluacion import auditar_citas

    auditoria = auditar_citas(salida, documentos)
    registro = {"auditoria_citas": auditoria, "abstencion_por_citas": False}
    if auditoria["sin_respaldo"]:
        registro.update(abstencion_por_citas=True, abstencion_programatica=True,
                        motivo="citas_sin_respaldo")
        salida = {"id": salida["id"], "formato": salida["formato"],
                  **abstenerse(salida["formato"]),
                  "pasajes_recuperados": salida.get("pasajes_recuperados", [])}
    return salida, registro


def generar(cliente, entrada, pasajes, contexto=4096, max_tokens=512, gramatica=False,
            abstencion_automatica=True, carpeta_raw=None):
    import jsonschema

    inicio = time.perf_counter()
    usados, omitidos, tokens = ajustar_contexto(cliente, entrada, pasajes, contexto, max_tokens)
    registro = {"id": entrada["id"], "formato": entrada["formato"], "omitidos": omitidos,
                "tokens_prompt_estimados": tokens, "gramatica": gramatica,
                "temperatura": 0, "semilla": 0, "cache_prompt": False,
                "repeat_penalty": 1.0, "frequency_penalty": 0, "presence_penalty": 0,
                "max_tokens": max_tokens, "abstencion_programatica": False}
    if not usados and abstencion_automatica:
        salida = abstenerse(entrada["formato"])
        registro.update(abstencion_programatica=True, motivo="sin_contexto_suficiente", json_valido=True,
                        claves_validas=True, segundos=time.perf_counter() - inicio)
        return {"id": entrada["id"], "formato": entrada["formato"], **salida, "pasajes_recuperados": []}, registro
    conversacion = mensajes(entrada, usados)
    prompt = cliente.aplicar_plantilla(conversacion)
    peticion = {"prompt": prompt, "temperature": 0, "seed": 0,
                "n_predict": max_tokens, "cache_prompt": False, "stream": False,
                "repeat_penalty": 1.0, "frequency_penalty": 0, "presence_penalty": 0,
                "return_tokens": True}
    registro["endpoint_generacion"] = "/completion"
    if gramatica:
        peticion["json_schema"] = esquema_local(entrada["formato"])
    if carpeta_raw:
        carpeta_raw = Path(carpeta_raw)
        carpeta_raw.mkdir(parents=True, exist_ok=True)
        (carpeta_raw / "conversacion.json").write_text(json.dumps(conversacion, ensure_ascii=False, indent=2), encoding="utf-8")
        (carpeta_raw / "prompt.txt").write_bytes(prompt.encode("utf-8"))
        (carpeta_raw / "peticion.json").write_text(json.dumps(peticion, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        respuesta, raw = cliente.pedir("/completion", peticion)
    except urllib.error.HTTPError as error:
        if carpeta_raw:
            (carpeta_raw / "error_http.json").write_bytes(error.read())
        raise
    contenido = respuesta.get("content")
    if contenido is None:
        contenido = ""
    if carpeta_raw:
        (carpeta_raw / "respuesta_http.json").write_bytes(raw)
        (carpeta_raw / "respuesta.txt").write_bytes(contenido.encode("utf-8"))
    registro.update(segundos=time.perf_counter() - inicio, contenido=contenido,
                    sha256_contenido=hashlib.sha256(contenido.encode("utf-8")).hexdigest(),
                    finish_reason=respuesta.get("stop_type"),
                    uso={"prompt_tokens": respuesta.get("tokens_evaluated"),
                         "completion_tokens": respuesta.get("tokens_predicted", len(respuesta.get("tokens", [])))},
                    timings=respuesta.get("timings", {}), json_valido=False, claves_validas=False)
    try:
        salida = json.loads(contenido)
    except json.JSONDecodeError as error:
        registro["error_formato"] = error.msg
        return None, registro
    registro["json_valido"] = True
    errores = list(jsonschema.Draft202012Validator(esquema_local(entrada["formato"])).iter_errors(salida))
    if errores:
        registro["error_formato"] = [e.message for e in errores]
        return None, registro
    registro["claves_validas"] = True
    salida = {"id": entrada["id"], "formato": entrada["formato"], **salida, "pasajes_recuperados": usados}
    return salida, registro


class ServidorLocal:
    def __init__(self, ejecutable, modelo, carpeta, contexto=4096, capas="auto", margen=512, autoajuste=True):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        with socket.socket() as puerto:
            puerto.bind(("127.0.0.1", 0))
            self.puerto = puerto.getsockname()[1]
        self.comando = [str(ejecutable), "--model", str(modelo), "--host", "127.0.0.1", "--port", str(self.puerto),
                        "--ctx-size", str(contexto), "--parallel", "1", "--seed", "0", "--temp", "0",
                        "--repeat-penalty", "1.0", "--threads", "8", "--threads-batch", "8",
                        "--n-gpu-layers", str(capas), "--fit", "on" if autoajuste else "off", "--fit-target", str(margen),
                        "--log-verbosity", "4",
                        "--jinja", "--chat-template-kwargs", '{"enable_thinking":false}']
        self.cliente = ClienteLocal(f"http://127.0.0.1:{self.puerto}")
        self.proceso = None
        self.log = None
        self.contexto = contexto
        self.propiedades = None
        self.capas_gpu = None

    def __enter__(self):
        if not Path(self.comando[0]).is_file() or not Path(self.comando[2]).is_file():
            raise FileNotFoundError("Falta el runtime o el modelo. Ejecutar scripts/descargar_modelos.py")
        self.log = (self.carpeta / "servidor.log").open("wb")
        inicio = time.monotonic()
        entorno = dict(os.environ)
        entorno.pop("OPENROUTER_API_KEY", None)
        self.proceso = subprocess.Popen(self.comando, stdout=self.log, stderr=subprocess.STDOUT, env=entorno,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while time.monotonic() - inicio < 300:
                if self.proceso.poll() is not None:
                    raise RuntimeError("El servidor terminó durante la carga. Revisar servidor.log")
                try:
                    salud, _ = self.cliente.pedir("/health")
                    if salud.get("status") == "ok":
                        self.propiedades, _ = self.cliente.pedir("/props")
                        efectivo = self.propiedades.get("default_generation_settings", {}).get("n_ctx")
                        if efectivo is None:
                            efectivo = self.propiedades.get("n_ctx")
                        if efectivo is None or efectivo < self.contexto:
                            raise RuntimeError("No se confirmó el contexto solicitado en /props")
                        contenido_log = (self.carpeta / "servidor.log").read_text(encoding="utf-8", errors="replace")
                        capas = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", contenido_log)
                        if capas:
                            self.capas_gpu = {"cargadas": int(capas[-1][0]), "total": int(capas[-1][1])}
                        return self
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
            raise TimeoutError("El servidor no terminó de cargar")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, tipo, valor, traza):
        if self.proceso is not None and self.proceso.poll() is None:
            self.proceso.terminate()
            try:
                self.proceso.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proceso.kill()
                self.proceso.wait()
        if self.log:
            self.log.close()
