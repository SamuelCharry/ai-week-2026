"""Ejecución, re-puntuación y comparación de la versión 04.

No modifica nada del 03: lee sus recuperaciones y respuestas y escribe en otra carpeta.
"""
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from legalrag.citations.evidencia import EvidenciaV04
from legalrag.generation import politica as gen

CONFIG_BASE = {
    "decoder": "salamandra-7b-instruct",
    "contexto": 8192,
    "max_tokens": 900,
    "repeat_penalty": 1.1,
    "repeat_last_n": 256,
    "longitudes": True,
    "max_caracteres_pasaje": 1800,
    "politica": {"citar_evidencia": "todas", "abstener_libre": "sin_evidencia"},
}


def _leer(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def _guardar(ruta, contenido):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + ".tmp")
    temporal.write_text(json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporal.replace(ruta)


def _hash(datos):
    return hashlib.sha256(json.dumps(datos, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def muestra(raiz):
    from legalrag.evaluation.oficial import cargar_muestra
    return cargar_muestra(raiz)


def recuperacion_03(carpeta_recuperacion, variante):
    """{id: pasajes} tal como los guardó el 03."""
    filas = _leer(Path(carpeta_recuperacion) / "recuperaciones" / f"{variante}.json")
    return {f["id"]: f["pasajes"] for f in filas}


def evidencia(raiz):
    return EvidenciaV04(raiz, Path(raiz) / "data/processed/corpus")


# ---------------------------------------------------------------- evaluación

def evaluar(raiz, carpeta, respuestas):
    """Evaluador oficial (sin juez) + validación de esquema + diagnóstico por pregunta."""
    raiz, carpeta = Path(raiz), Path(carpeta)
    preguntas, _ = muestra(raiz)
    respuestas = sorted(respuestas, key=lambda r: r["id"])
    carpeta.mkdir(parents=True, exist_ok=True)
    submission = carpeta / "submissions.jsonl"
    submission.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in respuestas), encoding="utf-8")
    try:
        from legalrag.evaluation.entrega import validar_esquema
        errores = validar_esquema(respuestas, raiz / "data/oficial/schema/submission.schema.json")
    except RuntimeError:
        errores = None
    reporte = carpeta / "evaluacion_sin_ragas.json"
    proceso = subprocess.run([sys.executable, str(raiz / "data/oficial/scripts/evaluate.py"), "--submission",
                              str(submission.resolve()), "--split", "sample", "--out", str(reporte.resolve())],
                             capture_output=True, text=True, encoding="utf-8", cwd=raiz / "data/oficial")
    if proceso.returncode:
        raise RuntimeError("El evaluador oficial falló:\n" + proceso.stderr[-2000:])
    resultado = _leer(reporte)
    ev = evidencia(raiz)
    citas = ev.citas
    claves = {q["id"]: q for q in preguntas}
    diagnostico = []
    for r in respuestas:
        q = claves[r["id"]]
        texto = _texto_citas(r)
        ref = citas.extract(q.get("legal_basis") or "")
        puntaje = citas.score(texto, q.get("legal_basis") or "", ev.respaldo(r.get("pasajes_recuperados")))
        fila = {"id": r["id"], "area": q["area"], "formato": q["formato"], "abstencion": r["abstencion"],
                "respuesta_modelo": r.get("respuesta_correcta"), "respuesta_oficial": q.get("respuesta_correcta"),
                "acierto_cerrada": (not r["abstencion"] and r.get("respuesta_correcta") == q.get("respuesta_correcta"))
                if q["formato"] == "multiple_choice" else None,
                "ref_citas": len(citas.bodies(ref)),
                **{f"citas_{k}": v for k, v in puntaje.items() if k != "detalle"},
                "citas_sin_respaldo_detalle": puntaje["detalle"]["citas_sin_respaldo"]}
        diagnostico.append(fila)
    _guardar(carpeta / "diagnostico.json", diagnostico)
    resumen = {"total_50": resultado["total_automatico"]["obtenidos"],
               "cerradas_20": resultado["cerradas"]["puntos"], "aciertos_cerradas": resultado["cerradas"]["aciertos"],
               "n_cerradas": resultado["cerradas"]["n"],
               "citas_20": resultado["citas"]["puntos"], "recall_citas": resultado["citas"]["recall_citas_ponderado"],
               "tasa_sin_respaldo": resultado["citas"]["tasa_sin_respaldo"],
               "abstencion_10": resultado["abstencion"]["puntos"],
               "abstenciones": sum(r["abstencion"] for r in respuestas),
               "errores_validacion_oficial": resultado["validacion"]["errores"],
               "errores_esquema": None if errores is None else len(errores)}
    _guardar(carpeta / "resumen.json", resumen)
    return resumen, resultado, diagnostico


def _texto_citas(r):
    f = r["formato"]
    if f == "multiple_choice":
        return r.get("justificacion") or ""
    if f == "semi_open":
        return " ".join(str(r.get(k) or "") for k in ("respuesta", "referencia_legal"))
    return " ".join(str(r.get(k) or "") for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion"))


# ---------------------------------------------------------- re-puntuar el 03

def repuntuar_03(raiz, carpeta_ejecucion_03, carpeta_recuperacion, variantes, salida, politica=None):
    """Aplica las reglas nuevas de post-proceso a las salidas crudas del 03, sin GPU.

    Mide cuánto aporta cada regla por sí sola. No puede recuperar una letra que el
    modelo nunca eligió (respuesta_correcta null en el 03).
    """
    raiz, salida = Path(raiz), Path(salida)
    politica = politica or CONFIG_BASE["politica"]
    preguntas, entradas = muestra(raiz)
    ev = evidencia(raiz)
    resumenes = {}
    for variante in variantes:
        recup = recuperacion_03(carpeta_recuperacion, variante)
        respuestas, notas = [], []
        for entrada in entradas:
            fila = _leer(Path(carpeta_ejecucion_03) / variante / "respuestas" / f"{entrada['id']}.json")
            contenido = fila["registro"].get("contenido") or ""
            crudo, reparado = gen.reparar_json(contenido)
            pasajes, _ = ev.reconstruir(recup[entrada["id"]])
            sin_letra = (entrada["formato"] == "multiple_choice"
                         and (crudo or {}).get("respuesta_correcta") not in (entrada.get("opciones") or {}))
            if crudo is None or sin_letra:
                # Sin contenido utilizable: se conserva la respuesta del 03 tal cual.
                respuesta = dict(fila["respuesta"])
                motivo = "sin_json" if crudo is None else "cerrada_sin_letra"
            else:
                respuesta, registro = gen.postprocesar(entrada, crudo, pasajes, ev, politica)
                motivo = "reparado" if reparado else "ok"
            respuestas.append(respuesta)
            notas.append({"id": entrada["id"], "formato": entrada["formato"], "motivo": motivo,
                          "abstencion_03": fila["respuesta"]["abstencion"], "abstencion_nueva": respuesta["abstencion"]})
        resumen, _, _ = evaluar(raiz, salida / variante, respuestas)
        _guardar(salida / variante / "notas.json", notas)
        resumenes[variante] = resumen
    return resumenes


# ------------------------------------------------------------- ejecución GPU

def ejecutar(raiz, carpeta_recuperacion, resultados, variantes, config=None, cache_modelos=None,
             nombre="", solo_ids=None, subcarpeta="v04"):
    from legalrag.generation.runtime import hardware, preparar_decoder_persistente, runtime
    from legalrag.generation.cliente import ServidorLocal

    raiz, resultados = Path(raiz).resolve(), Path(resultados)
    config = {**CONFIG_BASE, **(config or {})}
    catalogo = _leer(raiz / "configs/modelos.json")
    ficha = next(f for f in catalogo["decoders"] if f["nombre"] == config["decoder"])
    if ficha["parametros"] > catalogo.get("limite_parametros_decoder", 8_000_000_000):
        raise ValueError("El decoder supera el límite de parámetros")
    preguntas, entradas = muestra(raiz)
    if solo_ids:
        entradas = [e for e in entradas if e["id"] in set(solo_ids)]
    ev = evidencia(raiz)
    from legalrag.evaluation.oficial import codigo
    # Solo el código que decide la respuesta: generación, evidencia y experimentos.
    firma = _hash({"config": config, "recuperacion": str(carpeta_recuperacion), "variantes": variantes,
                   "codigo": {r: hashlib.sha256((raiz / r).read_bytes()).hexdigest()
                              for r in codigo(raiz, ["generation", "citations", "experimentos"])}})
    nombre = nombre or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    carpeta = resultados / subcarpeta / nombre
    if (carpeta / "configuracion.json").is_file() and _leer(carpeta / "configuracion.json")["firma"] != firma:
        raise ValueError("La configuración o el código cambió. Usar otro nombre de ejecución")
    _guardar(carpeta / "configuracion.json", {"firma": firma, "config": config, "variantes": variantes,
                                              "recuperacion": str(carpeta_recuperacion), "hardware": hardware()})
    servidor, fuente = runtime(raiz, catalogo)
    modelo = preparar_decoder_persistente(raiz, ficha, catalogo, servidor, fuente, cache=cache_modelos)
    with ServidorLocal(servidor, modelo, carpeta / "servidor", contexto=config["contexto"]) as motor:
        motor.cliente.timeout = 600
        if not motor.capas_gpu or motor.capas_gpu["cargadas"] == 0:
            raise RuntimeError("El decoder no cargó capas en GPU. Revisar servidor/servidor.log")
        for variante in variantes:
            recup = recuperacion_03(carpeta_recuperacion, variante)
            respuestas = []
            for n, entrada in enumerate(entradas, 1):
                destino = carpeta / variante / "respuestas" / f"{entrada['id']}.json"
                if destino.is_file():
                    guardada = _leer(destino)
                    if guardada.get("firma") == firma:
                        respuestas.append(guardada["respuesta"])
                        continue
                pasajes, avisos = ev.reconstruir(recup[entrada["id"]])
                inicio = time.perf_counter()
                crudo, usados, registro = gen.generar(motor.cliente, entrada, pasajes, ev, config)
                if crudo is None and registro.get("finish_reason") == "limit":
                    # Segundo intento con más penalización; solo para salidas truncadas.
                    registro["primer_intento"] = {k: registro[k] for k in ("contenido", "finish_reason")}
                    crudo, usados, registro2 = gen.generar(motor.cliente, entrada, pasajes, ev,
                                                           {**config, "repeat_penalty": config["repeat_penalty"] + 0.1})
                    registro.update(registro2, reintento=True)
                respuesta, post = gen.postprocesar(entrada, crudo, usados, ev, config["politica"])
                respuesta["latencia_ms"] = int((time.perf_counter() - inicio) * 1000)
                _guardar(destino, {"firma": firma, "respuesta": respuesta, "registro": registro,
                                   "postproceso": post, "avisos": avisos})
                respuestas.append(respuesta)
                print(f"{variante}: {n}/{len(entradas)}", flush=True)
            if not solo_ids:
                evaluar(raiz, carpeta / variante, respuestas)
    return carpeta


# ----------------------------------------------------------------- comparación

def comparar(raiz, carpeta_03, carpeta_04, variante):
    """Tabla pareada por pregunta: qué se arregló, qué se rompió, qué quedó igual."""
    import pandas as pd

    raiz = Path(raiz)
    d04 = {d["id"]: d for d in _leer(Path(carpeta_04) / variante / "diagnostico.json")}
    ruta03 = Path(carpeta_03) / variante / "diagnostico_por_pregunta.json"
    d03 = {d["id"]: d for d in _leer(ruta03)}
    filas = []
    for qid, b in d04.items():
        a = d03.get(qid, {})
        ca = a.get("citas", {}) or {}
        fila = {"id": qid, "formato": b["formato"], "area": b["area"],
                "abst_03": a.get("abstencion"), "abst_04": b["abstencion"],
                "acierto_03": a.get("acierto_cerrada"), "acierto_04": b["acierto_cerrada"],
                "citas_ok_03": ca.get("aciertos"), "citas_ok_04": b.get("citas_aciertos"),
                "sin_respaldo_04": b.get("citas_citas_sin_respaldo")}
        if b["formato"] == "multiple_choice":
            fila["cambio"] = {(True, True): "igual_bien", (False, False): "igual_mal",
                              (False, True): "arreglada", (True, False): "rota"}[(bool(a.get("acierto_cerrada")), bool(b["acierto_cerrada"]))]
        else:
            antes, despues = (ca.get("aciertos") or 0) > 0, (b.get("citas_aciertos") or 0) > 0
            fila["cambio"] = {(True, True): "igual_bien", (False, False): "igual_mal",
                              (False, True): "arreglada", (True, False): "rota"}[(antes, despues)]
        filas.append(fila)
    return pd.DataFrame(filas).sort_values(["formato", "id"])


def comparar_versiones(versiones, variante):
    """Tabla de puntajes por versión. versiones: {etiqueta: carpeta_ejecucion}.

    Acepta ejecuciones del 03 (resumen.csv) y del 04/05 (<variante>/resumen.json).
    """
    import pandas as pd

    filas = []
    for etiqueta, carpeta in versiones.items():
        carpeta = Path(carpeta)
        if (carpeta / variante / "resumen.json").is_file():
            r = _leer(carpeta / variante / "resumen.json")
            ragas = carpeta / variante / "evaluacion_con_ragas.json"
            texto_libre = _leer(ragas)["correccion_ragas"].get("puntos") if ragas.is_file() else None
            filas.append({"version": etiqueta, "total_50": r["total_50"], "cerradas_20": r["cerradas_20"],
                          "citas_20": r["citas_20"], "abstencion_10": r["abstencion_10"],
                          "texto_libre_30": texto_libre,
                          "total_80": r["total_50"] + texto_libre if texto_libre is not None else None,
                          "abstenciones": r["abstenciones"], "errores_esquema": r["errores_esquema"]})
        elif (carpeta / "resumen.csv").is_file():
            r = pd.read_csv(carpeta / "resumen.csv").set_index("variante").loc[variante]
            filas.append({"version": etiqueta, "total_50": r["total_50"], "cerradas_20": r["cerradas_20"],
                          "citas_20": r["citas_20"], "abstencion_10": r["abstencion_10"],
                          "texto_libre_30": r.get("ragas_30"), "total_80": r.get("total_80"),
                          "abstenciones": r["abstenciones"], "errores_esquema": r["errores_esquema"]})
    return pd.DataFrame(filas).set_index("version")
