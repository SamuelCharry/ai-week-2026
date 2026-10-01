"""Compara variantes sobre las 50 preguntas de muestra, con el mismo corpus y el mismo índice.

Etapa 1 · recuperación, sin decoder (minutos). Métrica comparable con E06: alguna norma del
fundamento aparece en los 10 primeros fragmentos (respaldo_literal_top10), el MRR, y si aparece en
los pasajes entregados con su encabezado (respaldo_en_pasajes). El legal_basis solo se lee al puntuar.

    python3 src/comparar.py --recuperacion
    python3 src/comparar.py --recuperacion --encoders bge-m3 e5-large qwen3-emb-0.6b
    python3 src/comparar.py --rerankers bge qwen3-0.6b qwen3-4b   # mismo recorrido, solo cambia el reranker

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
import gc
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

RERANKERS = {
    "bge": {"repo_id": "BAAI/bge-reranker-v2-m3", "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"},
    "qwen3-0.6b": {"repo_id": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
                   "parametros": 595776512, "licencia": "apache-2.0"},
    # ~8 GB en bf16: no cabe junto a Qwen3-8B en bf16 en 24 GB; se mide primero solo en recuperación.
    "qwen3-4b": {"repo_id": "Qwen/Qwen3-Reranker-4B", "revision": "22e683669bc0f0bd69640a1354a6d0aebcfeede5",
                 "parametros": 4021784576, "licencia": "apache-2.0"},
}

VERIFICADOR_NLI = {"modelo": {"repo_id": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
                              "revision": "b5113eb38ab63efdd7f280f8c144ea8b13f978ce", "parametros": 278812163,
                              "licencia": "mit"},
                   "modo": "registrar", "umbral_implica": 0.5, "umbral_contradice": 0.9}

# Segundos agentes (otra familia que Qwen, ≤ 8.000 M). Llama-3.1-8B pide aceptar su licencia en Hugging Face
# (huggingface-cli login); Salamandra-7B no, y es nativo en español. Ambos los sugiere el enunciado (§3.1).
SEGUNDOS = {
    "llama31-8b": {"repo_id": "meta-llama/Llama-3.1-8B-Instruct", "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
                   "parametros": 8030261248, "licencia": "llama3.1",
                   "admitido_por_enunciado": "§3.1 del enunciado: opción sugerida meta-llama/Llama-3.1-8B-Instruct"},
    "salamandra-7b": DECODERS["salamandra-7b"],
}


def multiagente(segundo, juez=True, opinion=True):
    return {**QWEN3_DIRECTA, "agentes": {"segundo": SEGUNDOS[segundo], "juez_evidencia": juez, "segunda_opinion": opinion}}


# Agente reformulador: en cada pregunta de texto libre lista las normas aplicables; se buscan por nombre y sus
# artículos entran como candidatos al reranker (auditoría: 58 y 247 se pierden antes del reranker).
REFORMULADOR = {"recuperacion.expansion": "siempre", "recuperacion.agente_expansion": "normas",
                "recuperacion.expansion_articulos": True, "recuperacion.max_tokens_hipotesis": 120}

# Expansión solo cuando el reranker no encontró nada convincente y la pregunta no nombra una norma del corpus
# (Adaptive-RAG); en la corrida anterior el criterio "ambos" la activó en 26 de 35 preguntas de texto libre.
CALIBRADA = {**QWEN3_DIRECTA, "recuperacion.expansion": "debil", "recuperacion.criterio_evidencia_debil": "puntaje",
             "recuperacion.umbral_evidencia_debil": 0.0, "recuperacion.expansion_sin_norma_nombrada": True,
             "generacion.regenerar_json": True}

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
    # Ronda 2: expansión calibrada, recuperación por opción en cerradas, verificador NLI y reranker Qwen3.
    "qwen3-8b-calibrada": CALIBRADA,
    "qwen3-8b-calibrada-opciones": {**CALIBRADA, "recuperacion.recuperar_por_opcion": True},
    "qwen3-8b-calibrada-opciones-nli": {**CALIBRADA, "recuperacion.recuperar_por_opcion": True,
                                        "generacion.verificador_nli": VERIFICADOR_NLI},
    "qwen3-8b-calibrada-opciones-rr06": {**CALIBRADA, "recuperacion.recuperar_por_opcion": True,
                                         "recuperacion.reranker": RERANKERS["qwen3-0.6b"]},
    # Agente calculadora (generation.calculadora): montos de la pregunta en SMMLV y UVT con los decretos del corpus.
    "qwen3-8b-calculadora": {**QWEN3_DIRECTA, "generacion.calculadora": True},
    # Ronda 4 (tras la auditoría de fallas): herramientas y reformulador, sobre qwen3-8b-letra-directa.
    "qwen3-8b-normalizador": {**QWEN3_DIRECTA, "generacion.normalizador_citas": True},
    "qwen3-8b-reformulador": {**QWEN3_DIRECTA, **REFORMULADOR},
    "qwen3-8b-agentes": {**QWEN3_DIRECTA, **REFORMULADOR, "generacion.calculadora": True,
                         "generacion.normalizador_citas": True},
    "qwen3-8b-agentes-cerradas": {**QWEN3_DIRECTA, **REFORMULADOR, "recuperacion.expansion_cerradas": True,
                                  "generacion.calculadora": True, "generacion.normalizador_citas": True},
    # Redacción alineada con la métrica de texto libre (politica.INSTRUCCIONES_DIRECTAS); se mide con RAGAS≈.
    "qwen3-8b-agentes-directo": {**QWEN3_DIRECTA, **REFORMULADOR, "generacion.calculadora": True,
                                 "generacion.normalizador_citas": True, "generacion.estilo": "directo",
                                 "generacion.politica.maximo_abiertas": 3},
    # Ronda 3: multiagente por etapas sobre qwen3-8b-letra-directa (agent.componentes.preparar_lote).
    "multiagente-llama": multiagente("llama31-8b"),
    "multiagente-llama-juez": multiagente("llama31-8b", opinion=False),
    "multiagente-llama-opinion": multiagente("llama31-8b", juez=False),
    "multiagente-salamandra": multiagente("salamandra-7b"),
    "multiagente-salamandra-juez": multiagente("salamandra-7b", opinion=False),
    "multiagente-salamandra-opinion": multiagente("salamandra-7b", juez=False),
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

FORMATO_CORTO = {"multiple_choice": "cerrada", "semi_open": "semiabierta", "open_ended": "abierta"}
NOMBRE_PROBLEMA = {"json_invalido": "JSON inválido", "campos_rellenados": "campos rellenados",
                   "citas_saneadas": "citas saneadas", "sin_evidencia_en_contexto": "sin evidencia",
                   "esquema_oficial": "esquema oficial"}


def evaluador_oficial(config):
    """evaluate.py oficial, sin modificar, para puntuar pregunta por pregunta con sus mismas funciones."""
    import importlib.util

    scripts = RAIZ / config["oficial"] / "scripts"
    ruta = scripts / "evaluate.py"
    if not ruta.is_file():
        sys.exit(f"No está el evaluador oficial en {ruta}.\n¿Se está corriendo en la carpeta correcta del proyecto "
                 f"(la que tiene data/oficial, el índice y el corpus)? Carpeta actual del proyecto: {RAIZ}")
    sys.path.insert(0, str(scripts))  # evaluate.py importa sus vecinos citations.py y common.py
    spec = importlib.util.spec_from_file_location("evaluador_oficial", ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def por_pregunta(evaluador, salida, muestra):
    """{id: resultado} con las reglas del evaluador oficial (cerradas, citas y abstención por ítem)."""
    subs = {s["id"]: s for s in evaluador.read_jsonl(salida)} if salida.is_file() else {}
    citas, resultado = evaluador.citations, {}
    for pregunta in muestra:
        s = subs.get(pregunta["id"])
        fila = {"id": pregunta["id"], "formato": FORMATO_CORTO[pregunta["formato"]], "respondida": s is not None,
                "abstencion": bool(s and s.get("abstencion"))}
        if pregunta["formato"] == "multiple_choice" and pregunta["id"] not in evaluador.FLAWED_IDS:
            fila["letra"] = (s or {}).get("respuesta_correcta")
            fila["cerrada"] = bool(s) and not fila["abstencion"] and fila["letra"] == pregunta["respuesta_correcta"]
        referencia = citas.extract(pregunta.get("legal_basis") or "")
        if referencia:
            fila.update(citas_acertadas=0.0, citas_ref=len(citas.bodies(referencia)), citas_sin_respaldo=0)
            if s and not fila["abstencion"]:
                r = citas.score(evaluador.answer_text(s), pregunta.get("legal_basis") or "",
                                evaluador.citas_respaldadas(s))
                fila.update(citas_acertadas=r["aciertos_respaldados"] + 0.5 * r["aciertos_sin_respaldo"],
                            citas_ref=r["n_ref"], citas_sin_respaldo=r["citas_sin_respaldo"])
        resultado[pregunta["id"]] = fila
    return resultado


def cambios(base, otra):
    """[(signo, id, formato, detalle)] de lo que cambió frente a la base: + mejora, - empeora, ~ neutro."""
    lineas = []
    for qid, b in base.items():
        o = otra.get(qid) or {}
        if "cerrada" in b and (b["cerrada"], b["letra"]) != (o.get("cerrada"), o.get("letra")):
            signo = "~" if b["cerrada"] == o.get("cerrada") else "+" if o.get("cerrada") else "-"
            lineas.append((signo, qid, b["formato"], f"letra {b['letra']} → {o.get('letra')} "
                                                     f"({'correcta' if o.get('cerrada') else 'incorrecta'})"))
        if "citas_ref" in b:
            antes = (b["citas_acertadas"], b["citas_sin_respaldo"])
            despues = (o.get("citas_acertadas", 0.0), o.get("citas_sin_respaldo", 0))
            if antes != despues:
                delta = despues[0] - antes[0] - 2 * (despues[1] - antes[1])
                signo = "+" if delta > 0 else "-" if delta < 0 else "~"
                extra = f", sin respaldo {antes[1]} → {despues[1]}" if antes[1] != despues[1] else ""
                lineas.append((signo, qid, b["formato"], f"citas {antes[0]:g}/{b['citas_ref']} → "
                                                         f"{despues[0]:g}/{b['citas_ref']}{extra}"))
        if b["abstencion"] != o.get("abstencion"):
            lineas.append(("~", qid, b["formato"], "ahora se abstiene" if o.get("abstencion") else "ya no se abstiene"))
    return sorted(lineas, key=lambda l: ("+-~".index(l[0]), l[1]))


def compacto(conteos, nombres):
    partes = [f"{nombres.get(k, k)} {v}" for k, v in sorted(conteos.items(), key=lambda kv: -kv[1]) if k != "limpia"]
    return " · ".join(partes) or "—"


def informe(filas, detalles):
    """Tabla principal, cambios por pregunta frente a la primera variante y fallas que ninguna resuelve."""
    lineas = []
    base = filas[0]
    tabla = []
    for f in filas:
        pasos = json.loads(f["pasos"])
        texto_pasos = []
        if pasos.get("expandidas"):
            texto_pasos.append(f"expandidas {pasos['expandidas']}")
        if pasos.get("reintentos_json"):
            texto_pasos.append(f"reintentos JSON {pasos['reintentos_json']} (mejoraron {pasos.get('reintento_mejoro', 0)})")
        if pasos.get("verificadas"):
            texto_pasos.append(f"verificadas {pasos['verificadas']} (aceptadas {pasos.get('verificacion_aceptada', 0)})")
        if pasos.get("juzgadas"):
            texto_pasos.append(f"juez: {pasos['juzgadas']} preguntas, descartó {pasos.get('pasajes_descartados', 0)} pasajes")
        if pasos.get("segunda_opinion"):
            texto_pasos.append(f"segunda opinión: {pasos['segunda_opinion']} cerradas, cambió la letra en "
                               f"{pasos.get('opinion_cambio_letra', 0)}")
        if pasos.get("nli_oraciones"):
            texto_pasos.append(f"NLI: respaldadas {pasos.get('nli_respaldadas', 0)}/{pasos['nli_oraciones']} oraciones, "
                               f"contradichas {pasos.get('nli_contradichas', 0)}, quitadas {pasos.get('nli_quitadas', 0)}")
        delta = None
        if f.get("total") is not None and base.get("total") is not None and f is not base:
            delta = f"{f['total'] - base['total']:+.2f}"
        tabla.append({"variante": f["variante"], "total": f.get("total", "—"), "Δ base": delta or "—",
                      "cerradas": f"{f['aciertos_cerradas']}/{f['n_cerradas']}" if "n_cerradas" in f else "—",
                      "citas": f.get("citas", "—"), "recall": f.get("recall_citas", "—"),
                      "abstención": f.get("abstencion", "—"), **({"ragas": f["ragas"]} if f.get("ragas") is not None else {}),
                      **({"RAGAS≈": f["ragas_aprox"]} if f.get("ragas_aprox") is not None else {}),
                      "s/preg": f["s_por_pregunta"], "min": f["minutos"],
                      "problemas": compacto(json.loads(f["problemas"]), NOMBRE_PROBLEMA),
                      "pasos del agente": " · ".join(texto_pasos) or "—"})
    columnas = list(dict.fromkeys(k for t in tabla for k in t))
    anchos = {c: max(len(c), *(len(str(t.get(c, ""))) for t in tabla)) for c in columnas}
    numericas = {"total", "Δ base", "cerradas", "citas", "recall", "abstención", "ragas", "RAGAS≈", "s/preg", "min"}
    formato = lambda c, v: str(v).rjust(anchos[c]) if c in numericas else str(v).ljust(anchos[c])  # noqa: E731
    lineas += ["", "SISTEMA (muestra de 50, evaluador oficial)", "",
               "  ".join(formato(c, c) for c in columnas), "  ".join("-" * anchos[c] for c in columnas)]
    lineas += ["  ".join(formato(c, t.get(c, "")) for c in columnas) for t in tabla]

    nombre_base = base["variante"]
    for f in filas[1:]:
        lista = cambios(detalles[nombre_base], detalles[f["variante"]])
        lineas += ["", f"{f['variante']} frente a {nombre_base}: "
                   f"{sum(l[0] == '+' for l in lista)} mejoras, {sum(l[0] == '-' for l in lista)} empeoras, "
                   f"{sum(l[0] == '~' for l in lista)} cambios neutros"]
        lineas += [f"  {signo} {qid:>4} {formato_q:<11} {detalle}" for signo, qid, formato_q, detalle in lista]
        if not lista:
            lineas.append("  (mismas respuestas puntuables en todas las preguntas)")

    todas = list(detalles.values())
    cerradas = [q for q, d in todas[0].items() if "cerrada" in d and not any(v[q]["cerrada"] for v in todas)]
    sin_citas = [q for q, d in todas[0].items() if "citas_ref" in d and not any(v[q]["citas_acertadas"] for v in todas)]
    parciales = [q for q, d in todas[0].items() if "citas_ref" in d and q not in sin_citas
                 and not any(v[q]["citas_acertadas"] >= v[q]["citas_ref"] for v in todas)]
    lineas += ["", "Sin resolver en ninguna variante",
               f"  cerradas incorrectas:            {', '.join(map(str, cerradas)) or '—'}",
               f"  ninguna norma del fundamento:    {', '.join(map(str, sin_citas)) or '—'}",
               f"  fundamento citado en parte:      {', '.join(map(str, parciales)) or '—'}",
               "", "+ mejora · - empeora · ~ cambia sin efecto en el puntaje. Detalle por pregunta en "
               "data/comparacion/<variante>/por_pregunta.csv",
               "RAGAS≈: aproximación local de los 30 puntos de texto libre (e5-large del evaluador + NLI); ordena "
               "variantes, no reemplaza al juez oficial. Detalle en data/comparacion/<variante>/ragas_local.json"]
    return "\n".join(l.rstrip() for l in lineas)


def comparar_rerankers(config, nombres, ids):
    """Etapa 1b: el sistema de recuperación actual con cada reranker, con y sin recuperación por opción.
    Sin decoder, así que el reranker de 4B cabe en la GPU."""
    from legalrag.evaluation.entrega import cargar_jsonl
    from legalrag.evaluation.recuperacion import evaluar
    from legalrag.retrieval.hibrido import RecuperadorHibrido, crear_reordenador

    preguntas = [p for p in cargar_jsonl(RAIZ / config["entradas"]["sample"]) if not ids or p["id"] in ids]
    rec = config["recuperacion"]
    recuperador = RecuperadorHibrido(RAIZ, {**rec, "usar_reranker": False})
    print("Cargando índice y encoder...", flush=True)
    recuperador.abrir()
    filas = []
    try:
        for nombre in nombres:
            # Libera el reranker anterior antes de cargar el siguiente (el de 4B ocupa ~8 GB).
            recuperador.reordenador = None
            gc.collect()
            if "torch" in sys.modules:
                sys.modules["torch"].cuda.empty_cache()
            print(f"\n[{nombre}] cargando {RERANKERS[nombre]['repo_id']}...", flush=True)
            recuperador.reordenador = crear_reordenador(RERANKERS[nombre], rec["dtype"], rec["max_tokens_reranker"],
                                                        rec["lote_reranker"], rec.get("dispositivo", "cuda"))
            for variante, cambios in (("sistema", {}), ("+por_opcion", {"recuperar_por_opcion": True})):
                recuperador.config = {**rec, "reranker": RERANKERS[nombre], **cambios}
                resultado = evaluar(recuperador, preguntas, recuperador.citaciones)
                fallan = [d["id"] for d in resultado["detalle"] if not d["respaldo_en_pasajes"]]
                fila = {"reranker": nombre, "variante": variante,
                        **{k: round(v, 3) if isinstance(v, float) else v for k, v in resultado.items()
                           if k not in ("detalle", "referencia_E06_R03")},
                        "sin_norma_en_pasajes": " ".join(map(str, fallan))}
                filas.append(fila)
                print(f"  {variante:12} top10={fila['respaldo_literal_top10']} MRR={fila['MRR_cita_top10']} "
                      f"pasajes={fila['respaldo_en_pasajes']} {fila['segundos_promedio']} s · sin la norma: {fallan}",
                      flush=True)
    finally:
        recuperador.cerrar()
    escribir_csv(SALIDA / "rerankers.csv", filas)
    print("\nRerankers (mismo índice y mismo recorrido; solo cambia el reordenamiento)")
    imprimir(filas, ["reranker", "variante", "respaldo_literal_top10", "MRR_cita_top10", "respaldo_en_pasajes",
                     "segundos_promedio", "sin_norma_en_pasajes"])


def ragas_local(filas, muestra, evaluador):
    """Columna RAGAS≈ (legalrag.evaluation.ragas_local): aproximación gratuita de los 30 puntos de texto libre."""
    from legalrag.evaluation.ragas_local import RagasLocal

    print("\nAproximando RAGAS en local (e5-large del evaluador + NLI)...", flush=True)
    juez = RagasLocal(VERIFICADOR_NLI["modelo"])
    juez.abrir()
    try:
        for fila in filas:
            salida = SALIDA / fila["variante"] / "submissions.jsonl"
            if not salida.is_file():
                continue
            resultado = juez.evaluar({s["id"]: s for s in evaluador.read_jsonl(salida)}, muestra)
            fila["ragas_aprox"] = resultado["puntos_aprox"]
            salida.with_name("ragas_local.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
    finally:
        juez.cerrar()


def comparar_sistema(config, variantes, ids, ragas, cache=True, aproximar_ragas=True):
    from legalrag.agent.pipeline import responder_lote

    evaluador = evaluador_oficial(config)
    muestra = [p for p in evaluador.read_jsonl(RAIZ / config["entradas"]["sample"]) if not ids or p["id"] in ids]
    filas, detalles = [], {}
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
            if registro.get("juez_evidencia"):
                pasos["juzgadas"] = pasos.get("juzgadas", 0) + 1
                pasos["pasajes_descartados"] = pasos.get("pasajes_descartados", 0) + sum(
                    j["si"] < 0.5 for j in registro["juez_evidencia"])
            if registro.get("letras_segundo") and registro.get("letras_principal"):
                pasos["segunda_opinion"] = pasos.get("segunda_opinion", 0) + 1
                principal = registro["letras_principal"]
                final = registro.get("probabilidades_letras") or principal
                if max(sorted(principal), key=principal.get) != max(sorted(final), key=final.get):
                    pasos["opinion_cambio_letra"] = pasos.get("opinion_cambio_letra", 0) + 1
            nli = registro.get("verificacion_nli") or {}
            for clave in ("oraciones", "respaldadas", "contradichas"):
                if nli.get(clave):
                    pasos[f"nli_{clave}"] = pasos.get(f"nli_{clave}", 0) + nli[clave]
            if nli.get("quitadas"):
                pasos["nli_quitadas"] = pasos.get("nli_quitadas", 0) + len(nli["quitadas"])
        fila["problemas"] = json.dumps(problemas, ensure_ascii=False)
        fila["pasos"] = json.dumps(pasos, ensure_ascii=False)
        if not ids:
            reporte = salida.with_name("reporte.json")
            subprocess.run([sys.executable, str(RAIZ / config["oficial"] / "scripts/evaluate.py"), "--submission",
                            str(salida), "--split", "sample", "--out", str(reporte)] + (["--ragas"] if ragas else []),
                           check=True, capture_output=True, text=True)
            datos = json.loads(reporte.read_text(encoding="utf-8"))
            fila.update(cerradas=datos["cerradas"]["puntos"], aciertos_cerradas=datos["cerradas"]["aciertos"],
                        n_cerradas=datos["cerradas"]["n"],
                        citas=datos["citas"]["puntos"], recall_citas=datos["citas"]["recall_citas_ponderado"],
                        sin_respaldo=datos["citas"]["tasa_sin_respaldo"], abstencion=datos["abstencion"]["puntos"],
                        ragas=(datos.get("correccion_ragas") or {}).get("puntos"),
                        total=datos["total_automatico"]["obtenidos"], posibles=datos["total_automatico"]["posibles"])
        filas.append(fila)
        detalles[nombre] = por_pregunta(evaluador, salida, muestra)
        escribir_csv(salida.with_name("por_pregunta.csv"), list(detalles[nombre].values()))
        escribir_csv(SALIDA / "sistema.csv", filas)
    if aproximar_ragas and not ragas:
        ragas_local(filas, muestra, evaluador)
        escribir_csv(SALIDA / "sistema.csv", filas)
    texto = informe(filas, detalles)
    (SALIDA / "resumen.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto)
    print(f"\n(guardado en {(SALIDA / 'resumen.txt').relative_to(RAIZ)})")


def main():
    from legalrag.config import leer_config

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recuperacion", action="store_true", help="etapa 1: recuperación sin decoder")
    ap.add_argument("--sistema", action="store_true", help="etapa 2: sistema completo con evaluador oficial")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--variantes", nargs="+", default=["qwen25-7b", "qwen3-4b-2507"], choices=list(VARIANTES_SISTEMA))
    ap.add_argument("--ids", nargs="+", type=int, help="solo estas preguntas (prueba corta)")
    ap.add_argument("--ragas", action="store_true", help="incluye el juez de texto libre (OPENROUTER_API_KEY)")
    ap.add_argument("--rerankers", nargs="+", choices=list(RERANKERS),
                    help="etapa 1b: compara rerankers sobre la recuperación del sistema (sin decoder)")
    ap.add_argument("--sin-ragas-local", action="store_true", help="no calcula la aproximación local de RAGAS")
    ap.add_argument("--sin-cache", action="store_true",
                    help="genera todo de nuevo en cada variante (por defecto reutiliza generaciones con el mismo prompt)")
    args = ap.parse_args()
    if not (args.recuperacion or args.sistema or args.rerankers):
        ap.error("indicar --recuperacion, --rerankers, --sistema o varias")
    config = leer_config()
    if args.recuperacion:
        comparar_recuperacion(config, args.encoders, args.ids)
    if args.rerankers:
        comparar_rerankers(config, list(dict.fromkeys(args.rerankers)), args.ids)
    if args.sistema:
        # Una variante repetida en el comando se corre una sola vez.
        comparar_sistema(config, list(dict.fromkeys(args.variantes)), args.ids, args.ragas, cache=not args.sin_cache,
                         aproximar_ragas=not args.sin_ragas_local)


if __name__ == "__main__":
    main()
