"""Responde un archivo de preguntas y escribe la entrega en JSONL.

Cada respuesta se guarda al terminar, junto a la salida, para reanudar una ejecución
interrumpida. Una respuesta guardada solo se reutiliza si la configuración y la
pregunta no cambiaron. Al final valida la entrega con el esquema oficial.

Uso:
    python -m legalrag answer --split sample
    python -m legalrag answer --split test
    python -m legalrag answer --entrada preguntas.jsonl --salida respuestas.jsonl
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

from legalrag.evaluation.entrega import cargar_jsonl, preparar_entrada, validar_esquema
from legalrag.evaluation.oficial import guardar_json
from legalrag.agent.componentes import cargar
from legalrag.config import CONFIG, RAIZ, leer_config


def _firma(datos):
    return hashlib.sha256(json.dumps(datos, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _percentil(valores, p):
    ordenados = sorted(valores)
    return ordenados[min(len(ordenados) - 1, int(p * len(ordenados)))]


def responder_lote(raiz, entrada, salida, config, sistema=None, reanudar=True, ids=None):
    """Responde cada pregunta de `entrada` y escribe `salida`. Devuelve el resumen.

    sistema: instancia ya abierta; si falta, se crea con `implementacion` y se abre aquí.
    """
    raiz, entrada, salida = Path(raiz), Path(entrada), Path(salida)
    # preparar_entrada deja solo los campos de la pregunta y rechaza respuestas o legal_basis.
    entradas = [preparar_entrada(p) for p in cargar_jsonl(entrada)]
    if ids:
        entradas = [e for e in entradas if e["id"] in set(ids)]
    carpeta = salida.with_name(salida.stem + "_respuestas")
    firma_config = _firma({k: v for k, v in config.items() if k != "notas"})
    propio = sistema is None
    if propio:
        sistema = cargar(raiz, config)
        sistema.abrir()
    respuestas, latencias, errores = [], [], []

    try:
        for n, pregunta in enumerate(entradas, 1):
            destino = carpeta / f"{pregunta['id']}.json"
            firma = _firma({"config": firma_config, "entrada": pregunta})
            if reanudar and destino.is_file():
                guardada = json.loads(destino.read_text(encoding="utf-8"))
                if guardada.get("firma") == firma:
                    respuestas.append(guardada["respuesta"])
                    latencias.append(guardada["segundos"])
                    continue
            inicio = time.perf_counter()
            try:
                pasajes = sistema.recuperar(pregunta)
                respuesta = sistema.responder(pregunta, pasajes)
                if respuesta.get("id") != pregunta["id"] or respuesta.get("formato") != pregunta["formato"]:
                    raise ValueError("La respuesta cambió el id o el formato")
                json.dumps(respuesta, allow_nan=False)
            except NotImplementedError:
                raise
            except Exception as error:  # una pregunta fallida no detiene la tanda
                errores.append({"id": pregunta["id"], "error": type(error).__name__, "detalle": str(error)})
                print(f"{n}/{len(entradas)} id={pregunta['id']}: {type(error).__name__}: {error}", flush=True)
                continue
            segundos = time.perf_counter() - inicio
            guardar_json(destino, {"firma": firma, "respuesta": respuesta, "segundos": segundos,
                                   "problema": getattr(sistema, "ultimo_problema", None),
                                   "registro": getattr(sistema, "ultimo_registro", None),
                                   "traza_recuperacion": getattr(getattr(sistema, "recuperador", None), "ultima_traza", None)})
            respuestas.append(respuesta)
            latencias.append(segundos)
            print(f"{n}/{len(entradas)} id={pregunta['id']} {segundos:.1f} s", flush=True)
    finally:
        if propio:
            sistema.cerrar()

    salida.parent.mkdir(parents=True, exist_ok=True)
    temporal = salida.with_name(salida.name + ".tmp")
    temporal.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in respuestas), encoding="utf-8")
    temporal.replace(salida)
    esquema = raiz / config["oficial"] / "schema/submission.schema.json"
    resumen = {
        "entrada": str(entrada), "salida": str(salida), "preguntas": len(entradas),
        "respondidas": len(respuestas), "abstenciones": sum(bool(r.get("abstencion")) for r in respuestas),
        "errores": errores, "errores_esquema": validar_esquema(respuestas, esquema) if esquema.is_file() else None,
        "segundos_promedio": statistics.fmean(latencias) if latencias else None,
        "segundos_p95": _percentil(latencias, 0.95) if latencias else None,
        "presupuesto_segundos": config.get("presupuesto_segundos"),
    }
    guardar_json(salida.with_name(salida.stem + "_resumen.json"), resumen)
    return resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    ap.add_argument("--entrada", type=Path, help="reemplaza la entrada del split")
    ap.add_argument("--salida", type=Path, help="reemplaza la salida del split")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--sin-reanudar", action="store_true", help="ignora las respuestas guardadas")
    ap.add_argument("--ids", nargs="+", type=int, help="solo estas preguntas (prueba corta)")
    args = ap.parse_args()

    config = leer_config(args.config)
    entrada = args.entrada or RAIZ / config["entradas"][args.split]
    salida = args.salida or RAIZ / config["salidas"][args.split]
    resumen = responder_lote(RAIZ, entrada, salida, config, reanudar=not args.sin_reanudar, ids=args.ids)
    print(json.dumps({k: v for k, v in resumen.items() if k != "errores_esquema"}, ensure_ascii=False, indent=2))
    if resumen["errores_esquema"]:
        print(f"{len(resumen['errores_esquema'])} errores de esquema; detalle en el resumen.")
    if resumen["errores"] or resumen["errores_esquema"] or resumen["respondidas"] < resumen["preguntas"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
