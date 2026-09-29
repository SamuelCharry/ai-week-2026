"""Experimento 05: corpus ampliado, dos políticas de generación y un decoder alterno.

Reutiliza la ejecución del 04 (`legalrag.experimentos.v04.ejecutar`) y solo cambia la
configuración, para que la comparación aísle una cosa a la vez:

    A  corpus nuevo, generación idéntica al 04      -> cuánto aporta el corpus
    B  además saneo por cita y fundamento con usadas -> cuánto aportan los arreglos
    C  además decoder Qwen3-8B                       -> cuánto aporta el modelo

La corrida C queda detrás de una bandera. El enunciado sugiere Qwen/Qwen3-8B por
nombre en su sección 3.1, pero fija el límite en 8.000 millones de parámetros y ese
modelo tiene 8.190.735.360: por eso está en `excluidos` de configs/modelos.json.
Hay que confirmarlo con el organizador antes de usarlo en la entrega.

Mide además los segundos por pregunta contra el presupuesto de 22 s del sábado
(992 preguntas en unas seis horas).

Uso: python -m legalrag.experimentos.v05 --help
"""
import json
import sys
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from legalrag.experimentos.v04 import CONFIG_BASE, _guardar, _leer, ejecutar  # noqa: E402,F401

SEGUNDOS_POR_PREGUNTA = 22.0  # 992 preguntas en ~6 horas, según el enunciado
DECODER_8B = "qwen3-8b"

CONFIG_A = {**CONFIG_BASE, "max_tokens": 900, "repeat_penalty": 1.1,
            "politica": {"citar_evidencia": "todas", "abstener_libre": "sin_evidencia", "saneo": "oracion"}}
CONFIG_B = {**CONFIG_A,
            "politica": {"citar_evidencia": "usadas", "abstener_libre": "sin_evidencia", "saneo": "cita"}}
CONFIG_C = {**CONFIG_B, "decoder": DECODER_8B}

CORRIDAS = {"A": ("05A_corpus", CONFIG_A), "B": ("05B_politicas", CONFIG_B), "C": ("05C_decoder", CONFIG_C)}


def habilitar_decoder_8b(raiz=RAIZ, confirmado=False):
    """Mueve Qwen3-8B de `excluidos` a `decoders` y sube el límite de parámetros.

    confirmado: hay que ponerlo en True a mano, y solo después de que el organizador
    confirme que el modelo que su propio enunciado sugiere es admisible pese a
    superar el límite que el mismo enunciado fija. Escribe configs/modelos.json.
    """
    if not confirmado:
        raise RuntimeError(
            "Qwen3-8B tiene 8.190.735.360 parámetros y el reto fija el límite en 8.000 millones. "
            "El enunciado lo sugiere por nombre, así que hay que preguntarle al organizador. "
            "Con su confirmación, volver a llamar con confirmado=True.")
    ruta = Path(raiz) / "configs/modelos.json"
    catalogo = _leer(ruta)
    if any(d["nombre"] == DECODER_8B for d in catalogo["decoders"]):
        return catalogo
    excluido = next((e for e in catalogo["excluidos"] if e["repo_id"] == "Qwen/Qwen3-8B"), None)
    if excluido is None:
        raise RuntimeError("Qwen/Qwen3-8B no está en configs/modelos.json")
    catalogo["decoders"].append({**{k: v for k, v in excluido.items() if k != "motivo"},
                                 "nombre": DECODER_8B, "recomendado": False, "prioridad": 9,
                                 "admitido_por": "confirmación del organizador, pendiente de registrar"})
    catalogo["limite_parametros_decoder"] = max(catalogo["limite_parametros_decoder"], excluido["parametros"])
    _guardar(ruta, catalogo)
    return catalogo


def correr(raiz, carpeta_recuperacion, resultados, variantes, corrida="A", solo_ids=None,
           cache_modelos=None, confirmado_8b=False, decoder=None):
    """Lanza una de las tres corridas. Devuelve la carpeta de resultados.

    decoder: cambia solo el modelo y deja el resto de la configuración igual. Sirve
    para la prueba de humo en una GPU pequeña, donde Salamandra Q4_K_M (4,6 GiB) no
    cabe y `qwen3-4b` (2,4 GiB) sí. Los resultados de una corrida con otro decoder
    no son comparables con los de la entrega.
    """
    if corrida not in CORRIDAS:
        raise ValueError(f"Corrida desconocida: {corrida}")
    nombre, config = CORRIDAS[corrida]
    if decoder:
        config = {**config, "decoder": decoder}
        nombre = f"{nombre}_{decoder}"
    if config["decoder"] == DECODER_8B:
        habilitar_decoder_8b(raiz, confirmado=confirmado_8b)
    if solo_ids:
        nombre = f"prueba_{nombre}"
    return ejecutar(raiz, carpeta_recuperacion, resultados, variantes, config=config,
                    nombre=nombre, solo_ids=solo_ids, subcarpeta="v05", cache_modelos=cache_modelos)


def tiempos(carpeta, variante, carpeta_recuperacion=None):
    """Segundos por pregunta frente al presupuesto, separando recuperación y generación.

    La generación sale de `latencia_ms` de cada respuesta; la recuperación, de las
    métricas que guardó `preparar_recuperaciones` para la misma variante.
    """
    import pandas as pd

    filas = []
    for ruta in sorted((Path(carpeta) / variante / "respuestas").glob("*.json")):
        dato = _leer(ruta)
        filas.append({"id": dato["respuesta"]["id"],
                      "generacion_s": (dato["respuesta"].get("latencia_ms") or 0) / 1000})
    tabla = pd.DataFrame(filas)
    if carpeta_recuperacion:
        ruta = Path(carpeta_recuperacion) / "recuperaciones" / f"{variante}.json"
        if ruta.is_file():
            recup = pd.DataFrame([{"id": f["id"], "recuperacion_s": f["metricas"]["segundos"]}
                                  for f in _leer(ruta)])
            tabla = tabla.merge(recup, on="id", how="left")
    tabla["recuperacion_s"] = tabla.get("recuperacion_s", 0)
    tabla["total_s"] = tabla["generacion_s"] + tabla["recuperacion_s"].fillna(0)
    return tabla


def resumen_tiempos(tabla, presupuesto=SEGUNDOS_POR_PREGUNTA):
    """Una fila con mediana, p95, máximo y cuántas preguntas pasan del presupuesto."""
    import pandas as pd

    serie = tabla["total_s"]
    return pd.DataFrame([{
        "preguntas": len(serie),
        "mediana_s": round(serie.median(), 2),
        "p95_s": round(serie.quantile(0.95), 2),
        "maximo_s": round(serie.max(), 2),
        "presupuesto_s": presupuesto,
        "sobre_presupuesto": int((serie > presupuesto).sum()),
        "margen_mediana_s": round(presupuesto - serie.median(), 2),
        "992_preguntas_h": round(serie.median() * 992 / 3600, 2),
    }])


def registrar_evolucion(raiz, carpeta, variante, etiqueta, cambio, documentos=None, fragmentos=None):
    """Agrega la corrida a configs/evolucion.json, que alimenta la tabla de CORPUS.md."""
    raiz = Path(raiz)
    resumen = _leer(Path(carpeta) / variante / "resumen.json")
    corpus = _leer(raiz / "data/processed/corpus/resumen.json")
    ruta = raiz / "configs/evolucion.json"
    filas = _leer(ruta) if ruta.is_file() else []
    fila = {"fecha": __import__("datetime").date.today().isoformat(), "experimento": etiqueta,
            "documentos": documentos or corpus["documentos"],
            "fragmentos": fragmentos or corpus["fragmentos"],
            "cerradas": resumen["cerradas_20"], "citas": resumen["citas_20"],
            "abstencion": resumen["abstencion_10"], "total": resumen["total_50"],
            "variante": variante, "cambio": cambio}
    filas = [f for f in filas if not (f.get("experimento") == etiqueta and f.get("variante") == variante)]
    filas.append(fila)
    _guardar(ruta, filas)
    return fila


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recuperacion", type=Path, required=True, help="carpeta de recuperación del 05")
    ap.add_argument("--resultados", type=Path, default=RAIZ / "data/experimentos_sistema")
    ap.add_argument("--corrida", choices=sorted(CORRIDAS), default="A")
    ap.add_argument("--variantes", nargs="+", default=["densa"],
                    choices=["densa", "hibrida", "densa_reranker"])
    ap.add_argument("--solo-ids", nargs="*", type=int, default=None, help="prueba de humo")
    ap.add_argument("--confirmado-8b", action="store_true",
                    help="el organizador confirmó que Qwen3-8B es admisible")
    args = ap.parse_args()

    carpeta = correr(RAIZ, args.recuperacion, args.resultados, args.variantes, corrida=args.corrida,
                     solo_ids=args.solo_ids, confirmado_8b=args.confirmado_8b)
    print("Resultados:", carpeta)
    for variante in args.variantes:
        tabla = tiempos(carpeta, variante, args.recuperacion)
        print(f"\n{variante}")
        print(resumen_tiempos(tabla).to_string(index=False))


if __name__ == "__main__":
    main()
