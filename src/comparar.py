"""Compara variantes sobre las 50 preguntas de muestra, con el mismo corpus y el mismo índice.

Etapa 1 · recuperación, sin decoder (minutos). Métrica comparable con E06: alguna norma del
fundamento aparece en los 10 primeros fragmentos (respaldo_literal_top10), el MRR, y si aparece en
los pasajes entregados con su encabezado (respaldo_en_pasajes). El legal_basis solo se lee al puntuar.

    python3 src/comparar.py --recuperacion
    python3 src/comparar.py --recuperacion --encoders bge-m3 e5-large qwen3-emb-0.6b

Etapa 2 · sistema completo con el evaluador oficial (cerradas 20, citas 20, abstención 10; con
--ragas también texto libre 30). Cada variante guarda sus respuestas y se reanuda si se corta.

    python3 src/comparar.py --sistema
    python3 src/comparar.py --sistema --variantes qwen25-7b qwen3-4b-2507
    python3 src/comparar.py --sistema --ids 51 79 140      # prueba corta

Las variantes comparten un caché de generaciones (data/comparacion/cache_generaciones.sqlite): con greedy, el
mismo modelo y el mismo prompt dan la misma salida, así que una variante solo genera las preguntas en las que
su cambio altera el prompt (p. ej. la expansión, solo donde la evidencia era débil). --sin-cache lo apaga.

Resultados en data/comparacion/: recuperacion.csv, sistema.csv y una carpeta por variante.
Otros encoders solo se miden si su índice existe (E06 los construye en
data/experimentos/corpus_definitivo/indices/<modelo>/).
"""
import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

SALIDA = RAIZ / "data/comparacion"

ENCODERS = {
    "bge-m3": {"repo_id": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"},
    "e5-large": {"repo_id": "intfloat/multilingual-e5-large", "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"},
    "qwen3-emb-0.6b": {"repo_id": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"},
    "qwen3-emb-4b": {"repo_id": "Qwen/Qwen3-Embedding-4B", "revision": "5cf2132abc99cad020ac570b19d031efec650f2b"},
}

# Cambia una cosa a la vez respecto a la fila anterior.
VARIANTES_RECUPERACION = {
    "bm25": {"usar_denso": False, "usar_reranker": False, "enrutar_normas": False},
    "denso": {"usar_bm25": False, "usar_reranker": False, "enrutar_normas": False},
    "hibrido": {"usar_reranker": False, "enrutar_normas": False},
    "hibrido+reranker (R03)": {"enrutar_normas": False},
    "+enrutamiento": {"enrutar_normas": True},
    "+enrutamiento+tema": {"enrutar_normas": True, "consulta_con_tema": True},
    # Sistema actual (enrutamiento) más los ajustes para que la ley correcta no quede fuera de la evidencia.
    "+reserva2": {"enrutar_normas": True, "reservar_nombradas": 2},
    "+tope3": {"enrutar_normas": True, "max_por_documento": 3},
    "+normativos2": {"enrutar_normas": True, "min_normativos": 2},
    "+reserva2+tope3+normativos2": {"enrutar_normas": True, "reservar_nombradas": 2, "max_por_documento": 3,
                                    "min_normativos": 2},
}
SOLO_DENSO = ("denso", "hibrido+reranker (R03)", "+enrutamiento")

DECODERS = {
    "qwen25-7b": {"repo_id": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                  "parametros": 7615616512, "licencia": "apache-2.0"},
    "qwen3-4b-2507": {"repo_id": "Qwen/Qwen3-4B-Instruct-2507", "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                      "parametros": 4022468096, "licencia": "apache-2.0"},
    "salamandra-7b": {"repo_id": "BSC-LT/salamandra-7b-instruct", "revision": "a3ed5452fafb3698a0423b1a18bfb5888f4e611b",
                      "parametros": 7768117248, "licencia": "apache-2.0"},
    "mistral-7b": {"repo_id": "mistralai/Mistral-7B-Instruct-v0.3", "revision": "c170c708c41dac9275d15a8fff4eca08d52bab71",
                   "parametros": 7248023552, "licencia": "apache-2.0"},
    "phi4-mini": {"repo_id": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
                  "parametros": 3836021760, "licencia": "mit"},
    # El enunciado lo sugiere por nombre en la §3.1 aunque tiene 8.190.735.360 parámetros.
    "qwen3-8b": {"repo_id": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                 "parametros": 8190735360, "licencia": "apache-2.0",
                 "admitido_por_enunciado": "§3.1 del enunciado: opción sugerida Qwen/Qwen3-8B"},
    # Con licencia que hay que aceptar en Hugging Face (huggingface-cli login).
    "gemma3-4b": {"repo_id": "google/gemma-3-4b-it", "revision": "093f9f388b31de276ce2de164bdc2081324b9767",
                  "parametros": 4300079472, "licencia": "gemma"},
    "llama32-3b": {"repo_id": "meta-llama/Llama-3.2-3B-Instruct", "revision": "0cb88a4f764b7a12671c53f0838cd831a0843b95",
                   "parametros": 3212749824, "licencia": "llama3.2"},
}

QWEN3_DIRECTA = {"generacion.decoder": DECODERS["qwen3-8b"], "generacion.letra_por_probabilidad": True}

# Variantes del sistema: rutas "seccion.clave" sobre configs/sistema.json.
VARIANTES_SISTEMA = {
    "qwen25-7b": {},
    "qwen3-4b-2507": {"generacion.decoder": DECODERS["qwen3-4b-2507"]},
    "salamandra-7b": {"generacion.decoder": DECODERS["salamandra-7b"]},
    # Qwen3-8B sobre la mejor configuración (38,43): letra razonada y recuperación en texto libre.
    "qwen3-8b": {"generacion.decoder": DECODERS["qwen3-8b"], "generacion.letra_por_probabilidad": "razonada"},
    "qwen3-8b-letra-directa": {"generacion.decoder": DECODERS["qwen3-8b"], "generacion.letra_por_probabilidad": True},
    # Permutaciones de las opciones (sesgo por posición) y descarte POE sobre Qwen3-8B con letra directa.
    "qwen3-8b-permutado": {"generacion.decoder": DECODERS["qwen3-8b"], "generacion.letra_por_probabilidad": True,
                           "generacion.permutar_opciones": True},
    "qwen3-8b-descarte": {"generacion.decoder": DECODERS["qwen3-8b"], "generacion.letra_por_probabilidad": True,
                          "generacion.descarte_mantener": 2},
    "qwen3-8b-permutado-descarte": {"generacion.decoder": DECODERS["qwen3-8b"], "generacion.letra_por_probabilidad": True,
                                    "generacion.permutar_opciones": True, "generacion.descarte_mantener": 2},
    # Pasos extra del agente sobre qwen3-8b-letra-directa (40,00): expansión HyDE/Query2doc activada como en
    # CRAG (solo texto libre), reintento de JSON inválido y verificación en cadena (CoVe). Ver generation.pasos.
    "qwen3-8b-expansion-debil": {**QWEN3_DIRECTA, "recuperacion.expansion": "debil",
                                 "recuperacion.umbral_evidencia_debil": 0.0},
    "qwen3-8b-expansion-siempre": {**QWEN3_DIRECTA, "recuperacion.expansion": "siempre"},
    "qwen3-8b-regenerar-json": {**QWEN3_DIRECTA, "generacion.regenerar_json": True},
    "qwen3-8b-cove": {**QWEN3_DIRECTA, "generacion.verificar": True},
    "qwen3-8b-mejoras": {**QWEN3_DIRECTA, "recuperacion.expansion": "debil", "recuperacion.umbral_evidencia_debil": 0.0,
                         "generacion.regenerar_json": True},
    "qwen3-8b-mejoras-cove": {**QWEN3_DIRECTA, "recuperacion.expansion": "debil",
                              "recuperacion.umbral_evidencia_debil": 0.0, "generacion.regenerar_json": True,
                              "generacion.verificar": True},
    "mistral-7b": {"generacion.decoder": DECODERS["mistral-7b"]},
    "phi4-mini": {"generacion.decoder": DECODERS["phi4-mini"]},
    "gemma3-4b": {"generacion.decoder": DECODERS["gemma3-4b"]},
    "llama32-3b": {"generacion.decoder": DECODERS["llama32-3b"]},
    # Evidencia completa: los 10 pasajes recuperados se entregan siempre; con más contexto el modelo los lee todos.
    "todos10": {"generacion.entregar_todos": True},
    "todos10-ctx10k": {"generacion.entregar_todos": True, "generacion.contexto": 10240},
    "todos10-ctx10k-recuperacion": {"generacion.entregar_todos": True, "generacion.contexto": 10240,
                                    "recuperacion.reservar_nombradas": 2, "recuperacion.max_por_documento": 3,
                                    "recuperacion.min_normativos": 2},
    # Letra razonada (explícita, no depende de configs/sistema.json) con los ajustes de recuperación solo en texto libre.
    "razonada-recuperacion-texto-libre": {"generacion.letra_por_probabilidad": "razonada",
                                          "recuperacion.reservar_nombradas": 2, "recuperacion.max_por_documento": 3,
                                          "recuperacion.min_normativos": 2, "recuperacion.ajustes_solo_texto_libre": True},
    "qwen25-7b-sin-enrutar": {"recuperacion.enrutar_normas": False},
    # Fundamento con las normas dueñas de los pasajes, sin las que los pasajes mencionan (corrida de 30,75).
    "qwen25-7b-citar-todas": {"generacion.politica": {"citar_evidencia": "todas", "abstener_libre": "sin_evidencia",
                                                      "saneo": "cita"}},
    # Fundamento solo con las normas que el modelo nombró (la política de la primera corrida en la 4090: 29,07).
    "qwen25-7b-citar-usadas": {"generacion.politica": {"citar_evidencia": "usadas", "abstener_libre": "sin_evidencia",
                                                       "saneo": "cita"}},
    "qwen3-4b-2507-citar-usadas": {"generacion.decoder": DECODERS["qwen3-4b-2507"],
                                   "generacion.politica": {"citar_evidencia": "usadas", "abstener_libre": "sin_evidencia",
                                                           "saneo": "cita"}},
    # Razona primero (justificación) y elige la letra después, comparando A-D con ese razonamiento escrito.
    "qwen25-7b-letra-razonada": {"generacion.letra_por_probabilidad": "razonada"},
    "qwen3-4b-2507-letra-razonada": {"generacion.decoder": DECODERS["qwen3-4b-2507"],
                                     "generacion.letra_por_probabilidad": "razonada"},
    "qwen25-7b-saneo-oracion": {"generacion.politica": {"citar_evidencia": "todas", "abstener_libre": "sin_evidencia",
                                                        "saneo": "oracion"}},
}


def aplicar(config, cambios):
    config = copy.deepcopy(config)
    for ruta, valor in cambios.items():
        destino = config
        *partes, ultima = ruta.split(".")
        for parte in partes:
            destino = destino[parte]
        destino[ultima] = valor
    return config


def escribir_csv(ruta, filas):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    columnas = list(dict.fromkeys(k for f in filas for k in f))
    with ruta.open("w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def imprimir(filas, columnas):
    anchos = {c: max(len(c), *(len(str(f.get(c, ""))) for f in filas)) for c in columnas}
    print("  ".join(c.ljust(anchos[c]) for c in columnas))
    for f in filas:
        print("  ".join(str(f.get(c, "")).ljust(anchos[c]) for c in columnas))


def indice_de(encoder):
    return RAIZ / "data/experimentos/corpus_definitivo/indices" / encoder["repo_id"].replace("/", "__")


# ----------------------------------------------------------------- etapa 1

def comparar_recuperacion(config, encoders, ids):
    from legalrag.evaluation.entrega import cargar_jsonl
    from legalrag.evaluation.recuperacion import evaluar
    from legalrag.retrieval.hibrido import RecuperadorHibrido

    preguntas = [p for p in cargar_jsonl(RAIZ / config["entradas"]["sample"]) if not ids or p["id"] in ids]
    filas = []
    for nombre in encoders:
        ficha = ENCODERS[nombre]
        carpeta = indice_de(ficha)
        if not (carpeta / "complete.json").is_file():
            print(f"\n[{nombre}] sin índice en {carpeta}: se omite")
            continue
        rec = {**config["recuperacion"], "encoder": ficha,
               "max_tokens_encoder": 512 if "e5" in nombre else config["recuperacion"]["max_tokens_encoder"],
               "indice_denso": str((carpeta / "index.faiss").relative_to(RAIZ)),
               "indice_meta": str((carpeta / "complete.json").relative_to(RAIZ))}
        recuperador = RecuperadorHibrido(RAIZ, rec)
        print(f"\n[{nombre}] cargando índice y modelos...", flush=True)
        recuperador.abrir()
        try:
            for variante, cambios in VARIANTES_RECUPERACION.items():
                if nombre != "bge-m3" and variante not in SOLO_DENSO:
                    continue
                recuperador.config = {**rec, **cambios}
                resultado = evaluar(recuperador, preguntas, recuperador.citaciones)
                fila = {"encoder": nombre, "variante": variante,
                        **{k: round(v, 3) if isinstance(v, float) else v for k, v in resultado.items()
                           if k not in ("detalle", "referencia_E06_R03")}}
                filas.append(fila)
                fallan = [d["id"] for d in resultado["detalle"] if not d["respaldo_en_pasajes"]]
                fila["sin_norma_en_pasajes"] = " ".join(map(str, fallan))
                print(f"  {variante:30} top10={fila['respaldo_literal_top10']} MRR={fila['MRR_cita_top10']} "
                      f"pasajes={fila['respaldo_en_pasajes']} {fila['segundos_promedio']} s · sin la norma: {fallan}",
                      flush=True)
                (SALIDA / "recuperacion").mkdir(parents=True, exist_ok=True)
                (SALIDA / "recuperacion" / f"{nombre}__{variante.replace(' ', '_')}.json").write_text(
                    json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
        finally:
            recuperador.cerrar()
    escribir_csv(SALIDA / "recuperacion.csv", filas)
    print("\nRecuperación (referencia E06 R03: top10 0,854 · MRR 0,546)")
    imprimir(filas, ["encoder", "variante", "respaldo_literal_top10", "MRR_cita_top10", "respaldo_en_pasajes",
                     "segundos_promedio", "sin_norma_en_pasajes"])


# ----------------------------------------------------------------- etapa 2

def comparar_sistema(config, variantes, ids, ragas, cache=True):
    from legalrag.agent.pipeline import responder_lote

    filas = []
    for nombre in variantes:
        variante = aplicar(config, VARIANTES_SISTEMA[nombre])
        # Las respuestas guardadas se reutilizan solo si la configuración es la misma; con la huella del índice,
        # ampliar el corpus (ingestion.agregar_puntuales) invalida las respuestas viejas sin borrar carpetas.
        fragmentos = (RAIZ / variante["recuperacion"]["fragmentos"]).stat()
        variante["recuperacion"]["huella_indice"] = f"{fragmentos.st_size}-{int(fragmentos.st_mtime)}"
        if cache:
            # Las variantes comparten las generaciones idénticas (mismo modelo y mismo prompt, greedy): cada una
            # solo genera las preguntas en las que su cambio altera el prompt.
            SALIDA.mkdir(parents=True, exist_ok=True)
            variante["generacion"]["cache_generaciones"] = str(SALIDA / "cache_generaciones.sqlite")
        salida = SALIDA / nombre / "submissions.jsonl"
        print(f"\n===== {nombre}: {variante['generacion']['decoder']['repo_id']}", flush=True)
        inicio = time.perf_counter()
        resumen = responder_lote(RAIZ, RAIZ / config["entradas"]["sample"], salida, variante, ids=ids)
        fila = {"variante": nombre, "respondidas": resumen["respondidas"], "abstenciones": resumen["abstenciones"],
                "errores": len(resumen["errores"]), "errores_esquema": len(resumen["errores_esquema"] or []),
                "s_por_pregunta": round(resumen["segundos_promedio"] or 0, 1),
                "minutos": round((time.perf_counter() - inicio) / 60, 1)}
        problemas, pasos = {}, {}
        for archivo in salida.with_name(salida.stem + "_respuestas").glob("*.json"):
            guardada = json.loads(archivo.read_text(encoding="utf-8"))
            problema = (guardada.get("problema") or "limpia").split(":")[0]
            problemas[problema] = problemas.get(problema, 0) + 1
            registro = guardada.get("registro") or {}
            # Cuántas veces actuó cada paso del agente (generation.pasos) y cuántas cambió la respuesta.
            for paso, actuo in (("expandidas", (registro.get("expansion") or {}).get("hipotesis")),
                                ("verificadas", registro.get("verificacion")),
                                ("verificacion_aceptada", (registro.get("verificacion") or {}).get("aceptada")),
                                ("reintentos_json", registro.get("reintento_json")),
                                ("reintento_mejoro", (registro.get("reintento_json") or {}).get("problema_despues")
                                 != (registro.get("reintento_json") or {}).get("problema_antes")
                                 if registro.get("reintento_json") else None)):
                if actuo:
                    pasos[paso] = pasos.get(paso, 0) + 1
        fila["problemas"] = json.dumps(problemas, ensure_ascii=False)
        fila["pasos"] = json.dumps(pasos, ensure_ascii=False)
        if not ids:
            reporte = salida.with_name("reporte.json")
            subprocess.run([sys.executable, str(RAIZ / config["oficial"] / "scripts/evaluate.py"), "--submission",
                            str(salida), "--split", "sample", "--out", str(reporte)] + (["--ragas"] if ragas else []),
                           check=True, capture_output=True, text=True)
            datos = json.loads(reporte.read_text(encoding="utf-8"))
            fila.update(cerradas=datos["cerradas"]["puntos"], aciertos_cerradas=datos["cerradas"]["aciertos"],
                        citas=datos["citas"]["puntos"], recall_citas=datos["citas"]["recall_citas_ponderado"],
                        sin_respaldo=datos["citas"]["tasa_sin_respaldo"], abstencion=datos["abstencion"]["puntos"],
                        ragas=(datos.get("correccion_ragas") or {}).get("puntos"),
                        total=datos["total_automatico"]["obtenidos"], posibles=datos["total_automatico"]["posibles"])
        filas.append(fila)
        escribir_csv(SALIDA / "sistema.csv", filas)
    print("\nSistema")
    imprimir(filas, ["variante", "total", "posibles", "cerradas", "aciertos_cerradas", "citas", "recall_citas",
                     "sin_respaldo", "abstencion", "ragas", "abstenciones", "s_por_pregunta", "problemas", "pasos"])


def main():
    from legalrag.config import leer_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recuperacion", action="store_true", help="etapa 1: recuperación sin decoder")
    ap.add_argument("--sistema", action="store_true", help="etapa 2: sistema completo con evaluador oficial")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--variantes", nargs="+", default=["qwen25-7b", "qwen3-4b-2507"], choices=list(VARIANTES_SISTEMA))
    ap.add_argument("--ids", nargs="+", type=int, help="solo estas preguntas (prueba corta)")
    ap.add_argument("--ragas", action="store_true", help="incluye el juez de texto libre (OPENROUTER_API_KEY)")
    ap.add_argument("--sin-cache", action="store_true",
                    help="genera todo de nuevo en cada variante (por defecto reutiliza generaciones con el mismo prompt)")
    args = ap.parse_args()
    if not (args.recuperacion or args.sistema):
        ap.error("indicar --recuperacion, --sistema o ambas")
    config = leer_config()
    if args.recuperacion:
        comparar_recuperacion(config, args.encoders, args.ids)
    if args.sistema:
        comparar_sistema(config, args.variantes, args.ids, args.ragas, cache=not args.sin_cache)


if __name__ == "__main__":
    main()
