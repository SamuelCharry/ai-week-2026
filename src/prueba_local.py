"""Prueba de punta a punta en miniatura, sin GPU (CPU y ~3 GB de RAM libres).

Recorre el mismo código de la entrega sobre un mini-corpus real de data/raw y ~10 preguntas de
muestra, con un modelo en memoria a la vez:

  1. corpus      normas del fundamento de las preguntas elegidas + documentos de relleno, preparadas
                 con preprocessing.preparar_corpus; inventario con preprocessing.inventario
  2. lexico      chunks.sqlite + BM25 (mismo código que E06)
  3. denso       vectores BGE-M3 reales en CPU e índice FAISS
  4. recuperar   variantes de recuperación (bm25, denso, híbrido, + reranker BGE, + enrutamiento,
                 + tema) con el indicador de E06; guarda los pasajes de la variante del sistema
  5. generar     un decoder pequeño en CPU (Qwen2.5-0.5B-Instruct) con el prompt y el posprocesado
                 de la entrega: prueba JSON, letra, saneo de citas, esquema y abstención
  6. puntaje     funciones del evaluador oficial restringidas a esas preguntas

El 0,5B no mide la calidad del 7B: mide que todo el recorrido funciona y en qué se pierden puntos.
Las preguntas del mini-corpus se eligen mirando su legal_basis solo para saber qué normas deben
estar en el corpus de prueba; ese texto nunca entra a la consulta, al índice ni al prompt.

    python3 src/prueba_local.py                  # todas las etapas (se reanuda donde iba)
    python3 src/prueba_local.py --desde generar  # repetir desde una etapa
    python3 src/prueba_local.py --decoder Qwen/Qwen2.5-1.5B-Instruct
    python3 src/prueba_local.py --conjunto amplio  # 19 preguntas: más formatos y áreas, y preguntas
                                                   # cuya norma NO está en el corpus (evidencia insuficiente)
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

RAIZ_PRUEBA = RAIZ / "data/prueba_local"
# Normas pequeñas: en CPU BGE-M3 tarda ~4 s por fragmento (el CGP solo serían 1.042 fragmentos).
CONJUNTOS = {
    "base": [290, 352, 79, 472, 960, 247],
    # + cerradas, sentencias de la Corte Constitucional, una abierta de insolvencia y preguntas cuyo
    # fundamento no está en el corpus (doctrina, Consejo de Estado, Corte Suprema, caso extranjero):
    # ahí se ve qué hace el sistema cuando la evidencia no alcanza.
    "amplio": [290, 352, 79, 472, 960, 247, 308, 671, 674, 140, 991, 272, 24, 142, 168, 190, 697, 857, 513],
}
BASE, PREGUNTAS = RAIZ_PRUEBA, CONJUNTOS["base"]
RELLENO = 5
CACHE_VECTORES = RAIZ_PRUEBA / "cache_vectores.npz"
ETAPAS = ["corpus", "lexico", "denso", "recuperar", "generar", "puntaje"]
VARIANTES = {
    "bm25": {"usar_denso": False, "usar_reranker": False, "enrutar_normas": False},
    "denso": {"usar_bm25": False, "usar_reranker": False, "enrutar_normas": False},
    "hibrido": {"usar_reranker": False, "enrutar_normas": False},
    "hibrido+reranker": {"enrutar_normas": False},
    "+enrutamiento": {"enrutar_normas": True},
    "+enrutamiento+tema": {"enrutar_normas": True, "consulta_con_tema": True},
}
VARIANTE_SISTEMA = "+enrutamiento"


def paso(texto):
    print(f"\n=== {texto}", flush=True)


def config_local(decoder):
    from legalrag.config import leer_config

    config = leer_config()
    base = BASE.relative_to(RAIZ).as_posix()
    ejecucion = f"{base}/ejecucion"
    config["recuperacion"].update(
        fragmentos=f"{ejecucion}/chunks.sqlite", indice_denso=f"{ejecucion}/indices/BAAI__bge-m3/index.faiss",
        indice_meta=f"{ejecucion}/indices/BAAI__bge-m3/complete.json",
        textos=f"{base}/corpus_preparado/textos",
        manifiesto=f"{base}/release/corpus_manifest.json",
        dispositivo="cpu", dtype="float32", bm25_top=50, denso_top=50, rerank_top=10, lote_reranker=4,
        max_pasajes=6)
    config["generacion"].update(decoder=decoder, dispositivo="cpu", dtype="float32", contexto=4096,
                                max_nuevos_tokens=500)
    return config


def preguntas_completas():
    return {p["id"]: p for p in map(json.loads, (RAIZ / "data/oficial/data/sample_50.jsonl").open(encoding="utf-8"))
            if p["id"] in PREGUNTAS}


# --------------------------------------------------------------------------- 1

def etapa_corpus(config):
    from legalrag.citations.normas import EvidenciaCorpus
    from legalrag.evaluation.oficial import cargar_citaciones
    from legalrag.preprocessing.inventario import fijar_inventario

    paso("1. Mini-corpus desde data/raw")
    crudo = json.loads((RAIZ / "data/raw/manifest.json").read_text(encoding="utf-8"))
    evidencia = EvidenciaCorpus(cargar_citaciones(RAIZ), crudo)
    necesarios = set()
    for pregunta in preguntas_completas().values():
        necesarios |= set(evidencia.normas_de(pregunta.get("legal_basis") or ""))

    def tamano(doc_id):
        return sum(f.stat().st_size for f in (RAIZ / "data/raw" / doc_id).glob("*") if f.is_file())

    # Relleno determinista: leyes y decretos pequeños que no son del fundamento.
    candidatos = sorted((d["doc_id"] for d in crudo if d["doc_id"] not in necesarios
                         and str(d.get("tipo")).lower() in ("ley", "decreto") and tamano(d["doc_id"]) < 150_000),
                        key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    ids = sorted(necesarios) + candidatos[:RELLENO]
    print(f"  {len(necesarios)} documentos del fundamento + {RELLENO} de relleno: {', '.join(sorted(necesarios))}")
    preparado = BASE / "corpus_preparado"
    entorno = {"PYTHONPATH": str(RAIZ / "src")}
    import os
    codigo = subprocess.run([sys.executable, "-m", "legalrag.preprocessing.preparar_corpus", "--raw", str(RAIZ / "data/raw"),
                             "--output", str(preparado), "--workers", "2", "--ids", *ids],
                            cwd=RAIZ, env={**os.environ, **entorno}, capture_output=True, text=True)
    if not (preparado / "documentos.jsonl").is_file():
        raise SystemExit("La preparación falló:\n" + codigo.stderr[-2000:])
    release = RAIZ / config["recuperacion"]["manifiesto"]
    fijar_inventario(preparado, release.parent, RAIZ)


# --------------------------------------------------------------------------- 2

def etapa_lexico(config):
    from legalrag.experimentos import corpus_definitivo as e06

    paso("2. Fragmentos y BM25")
    rec = config["recuperacion"]
    e06.RELEASE = (RAIZ / rec["manifiesto"]).parent
    e06.RUNS = (RAIZ / rec["fragmentos"]).parent
    print("  ", e06.build_lexical(force=True))


# --------------------------------------------------------------------------- 3

def etapa_denso(config):
    import faiss
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    paso("3. Vectores BGE-M3 en CPU")
    rec = config["recuperacion"]
    ficha = rec["encoder"]
    conexion = sqlite3.connect(RAIZ / rec["fragmentos"])
    filas = conexion.execute("SELECT id, texto_busqueda FROM chunks ORDER BY id").fetchall()
    conexion.close()
    # Caché por texto (compartida entre conjuntos): solo se codifica lo que no se ha visto.
    def clave(texto):
        return hashlib.sha1(f"{ficha['revision']}\n{texto}".encode()).hexdigest()

    cache = dict(np.load(CACHE_VECTORES)) if CACHE_VECTORES.is_file() else {}
    faltan = [t for _, t in filas if clave(t) not in cache]
    print(f"  {len(filas)} fragmentos; {len(filas) - len(faltan)} ya en caché, {len(faltan)} por codificar", flush=True)
    inicio = time.perf_counter()
    if faltan:
        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], revision=ficha["revision"])
        modelo = AutoModel.from_pretrained(ficha["repo_id"], revision=ficha["revision"], dtype=torch.float32).eval()
        for i in range(0, len(faltan), 8):
            lote = faltan[i:i + 8]
            tokens = tokenizer(lote, padding=True, truncation=True, max_length=1024, return_tensors="pt")
            with torch.inference_mode():
                cls = modelo(**tokens).last_hidden_state[:, 0].float()
            for texto, vector in zip(lote, torch.nn.functional.normalize(cls, p=2, dim=1).numpy()):
                cache[clave(texto)] = vector
            if (i // 8) % 25 == 0 or i + 8 >= len(faltan):
                print(f"  {i + len(lote)}/{len(faltan)} codificados · {time.perf_counter() - inicio:.0f} s", flush=True)
                np.savez(CACHE_VECTORES, **cache)
        del modelo
    matriz = np.vstack([cache[clave(t)] for _, t in filas]).astype("float32")
    indice = faiss.IndexFlatIP(matriz.shape[1])
    indice.add(matriz)
    destino = RAIZ / rec["indice_denso"]
    destino.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(indice, str(destino))
    (RAIZ / rec["indice_meta"]).write_text(json.dumps({"modelo": ficha["repo_id"], "revision": ficha["revision"],
                                                       "chunks": len(filas), "dispositivo": "cpu"}), encoding="utf-8")
    print(f"  índice de {len(filas)} vectores en {time.perf_counter() - inicio:.0f} s")


# --------------------------------------------------------------------------- 4

class ReordenadorConCache:
    """El reranker en CPU tarda segundos por par: cada (consulta, fragmento) se puntúa una sola vez."""

    def __init__(self, reordenador):
        self.reordenador, self.cache = reordenador, {}

    def puntuar(self, texto, candidatos):
        faltan = [c for c in dict.fromkeys(candidatos) if (texto, c) not in self.cache]
        for c, p in zip(faltan, self.reordenador.puntuar(texto, faltan) if faltan else []):
            self.cache[(texto, c)] = p
        return [self.cache[(texto, c)] for c in candidatos]


class VectoresFijos:
    """Encoder ya consultado: devuelve el vector calculado antes de liberar BGE-M3."""

    def __init__(self, vectores):
        self.vectores = vectores

    def codificar(self, texto):
        return self.vectores[texto]


def etapa_recuperar(config):
    import faiss

    from legalrag.evaluation.entrega import preparar_entrada
    from legalrag.evaluation.oficial import guardar_json
    from legalrag.evaluation.recuperacion import evaluar
    from legalrag.retrieval.hibrido import EncoderConsultas, RecuperadorHibrido, Reordenador, consulta

    paso("4. Recuperación: variantes con el indicador de E06")
    rec = config["recuperacion"]
    preguntas = list(preguntas_completas().values())
    recuperador = RecuperadorHibrido(RAIZ, rec)
    recuperador.abrir(modelos=False)
    textos = {consulta(preparar_entrada(p), tema) for p in preguntas for tema in (False, True)}
    encoder = EncoderConsultas(rec["encoder"], "float32", rec["max_tokens_encoder"], "cpu")
    recuperador.encoder = VectoresFijos({t: encoder.codificar(t) for t in textos})
    del encoder
    gc.collect()
    recuperador.indice = faiss.read_index(str(RAIZ / rec["indice_denso"]))
    print("  cargando el reranker BGE-v2-m3...", flush=True)
    recuperador.reordenador = ReordenadorConCache(
        Reordenador(rec["reranker"], "float32", rec["max_tokens_reranker"], rec["lote_reranker"], "cpu"))
    filas = []
    for nombre, cambios in VARIANTES.items():
        recuperador.config = {**rec, **cambios}
        resultado = evaluar(recuperador, preguntas, recuperador.citaciones)
        filas.append((nombre, resultado))
        print(f"  {nombre:22} top10={resultado['respaldo_literal_top10']:.3f} MRR={resultado['MRR_cita_top10']:.3f} "
              f"pasajes={resultado['respaldo_en_pasajes']:.3f} · {resultado['segundos_promedio']:.1f} s/pregunta", flush=True)
    recuperador.config = {**rec, **VARIANTES[VARIANTE_SISTEMA]}
    pasajes = {}
    for pregunta in preguntas:
        entrada = preparar_entrada(pregunta)
        pasajes[pregunta["id"]] = recuperador.buscar(entrada)
    guardar_json(BASE / "recuperacion.json", {"variantes": {n: r for n, r in filas}, "variante_sistema": VARIANTE_SISTEMA})
    guardar_json(BASE / "pasajes.json", {str(k): v for k, v in pasajes.items()})
    recuperador.cerrar()


# --------------------------------------------------------------------------- 5

def etapa_generar(config):
    import torch

    from legalrag.agent.componentes import Sistema
    from legalrag.agent.pipeline import responder_lote
    from legalrag.citations.normas import EvidenciaCorpus
    from legalrag.evaluation.oficial import cargar_citaciones

    paso(f"5. Generación con {config['generacion']['decoder']['repo_id']} en CPU")
    torch.set_num_threads(4)
    pasajes = {int(k): v for k, v in json.loads((BASE / "pasajes.json").read_text(encoding="utf-8")).items()}
    manifiesto = json.loads((RAIZ / config["recuperacion"]["manifiesto"]).read_text(encoding="utf-8"))
    sistema = Sistema(RAIZ, config)
    sistema.recuperador.evidencia = EvidenciaCorpus(cargar_citaciones(RAIZ), manifiesto)
    sistema.recuperar = lambda entrada: pasajes[entrada["id"]]
    import jsonschema
    esquema = json.loads((RAIZ / "data/oficial/schema/submission.schema.json").read_text(encoding="utf-8"))
    sistema.validador = jsonschema.validators.validator_for(esquema)(esquema)
    sistema.decoder.abrir()
    resumen = responder_lote(RAIZ, RAIZ / "data/oficial/data/sample_50.jsonl", BASE / "submissions.jsonl", config,
                             sistema=sistema, ids=PREGUNTAS)
    print(json.dumps({k: v for k, v in resumen.items() if k not in ("errores_esquema",)}, ensure_ascii=False, indent=1))
    if resumen["errores_esquema"]:
        print("Errores de esquema:", resumen["errores_esquema"][:5])


# --------------------------------------------------------------------------- 6

def etapa_puntaje(config):
    paso("6. Puntaje con el evaluador oficial (solo estas preguntas)")
    scripts = RAIZ / "data/oficial/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("evaluate_oficial", scripts / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    filas = ev.read_jsonl(RAIZ / "data/oficial/data/sample_50.jsonl")
    key = {r["id"]: {"respuesta_correcta": r.get("respuesta_correcta"), "respuesta_esperada": r.get("respuesta_esperada"),
                     "legal_basis": r.get("legal_basis"), "pregunta": r["pregunta"], "formato": r["formato"]}
           for r in filas if r["id"] in PREGUNTAS}
    cerradas = [q for q, k in key.items() if k["formato"] == "multiple_choice" and q not in ev.FLAWED_IDS]
    subs = {s["id"]: s for s in ev.read_jsonl(BASE / "submissions.jsonl")}
    problemas = ev.validate(list(subs.values()), set(key))
    reporte = {"validacion": problemas, "cerradas": ev.score_closed(subs, key, cerradas),
               "citas": ev.score_citations(subs, key), "abstencion": ev.score_abstention(subs, key, set(cerradas))}
    total = sum(reporte[k]["puntos"] for k in ("cerradas", "citas", "abstencion"))
    from legalrag.citations.normas import EvidenciaCorpus
    manifiesto = json.loads((RAIZ / config["recuperacion"]["manifiesto"]).read_text(encoding="utf-8"))
    en_corpus = EvidenciaCorpus(ev.citations, manifiesto)
    grupos = {"cerradas": [q for q in key if key[q]["formato"] == "multiple_choice"],
              "semiabiertas": [q for q in key if key[q]["formato"] == "semi_open"],
              "abiertas": [q for q in key if key[q]["formato"] == "open_ended"],
              "norma en el corpus": [q for q in key if en_corpus.normas_de(key[q]["legal_basis"] or "")],
              "norma fuera del corpus": [q for q in key if not en_corpus.normas_de(key[q]["legal_basis"] or "")]}
    reporte["grupos"] = {}
    print("  por grupo (puntos de cada componente sobre sus propias preguntas):")
    for nombre, ids in grupos.items():
        if not ids:
            continue
        clave_g = {q: key[q] for q in ids}
        subs_g = {q: s for q, s in subs.items() if q in ids}
        cerradas_g = [q for q in cerradas if q in ids]
        g = {"n": len(ids), "cerradas": ev.score_closed(subs_g, clave_g, cerradas_g)["puntos"] if cerradas_g else None,
             "citas": ev.score_citations(subs_g, clave_g)["puntos"],
             "abstencion": ev.score_abstention(subs_g, clave_g, set(cerradas_g))["puntos"],
             "abstuvo": sum(bool(subs_g[q].get("abstencion")) for q in ids if q in subs_g)}
        reporte["grupos"][nombre] = g
        print(f"    {nombre:24} n={g['n']:2}  cerradas={g['cerradas']}  citas={g['citas']}  "
              f"abstención={g['abstencion']}  se abstuvo en {g['abstuvo']}")
    print(f"  cerradas {reporte['cerradas']['puntos']}/20 ({reporte['cerradas']['aciertos']}/{reporte['cerradas']['n']})  "
          f"citas {reporte['citas']['puntos']}/20 (recall {reporte['citas']['recall_citas_ponderado']}, "
          f"sin respaldo {reporte['citas']['tasa_sin_respaldo']})  abstención {reporte['abstencion']['puntos']}/10  "
          f"=> {total:.2f}/50 en {len(subs)} preguntas")
    print("  validación:", problemas or "sin errores")
    for qid in PREGUNTAS:
        respuesta = subs.get(qid)
        if not respuesta:
            continue
        guardada = json.loads((BASE / "submissions_respuestas" / f"{qid}.json").read_text(encoding="utf-8"))
        extra = f" letra={respuesta.get('respuesta_correcta')} correcta={key[qid]['respuesta_correcta']}" \
            if respuesta["formato"] == "multiple_choice" else ""
        citas = sorted(ev.citations.bodies(ev.citations.extract(ev.answer_text(respuesta))))
        print(f"  {qid:5} {respuesta['formato']:15} abst={respuesta['abstencion']!s:5} problema={guardada.get('problema')}"
              f"{extra} {guardada['segundos']:.0f}s citas={citas[:3]}")
    (BASE / "reporte.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", choices=ETAPAS, default=None, help="rehace desde esta etapa")
    ap.add_argument("--hasta", choices=ETAPAS, default="puntaje")
    ap.add_argument("--decoder", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--conjunto", choices=list(CONJUNTOS), default="base")
    args = ap.parse_args()
    global BASE, PREGUNTAS
    BASE = RAIZ_PRUEBA if args.conjunto == "base" else RAIZ_PRUEBA / args.conjunto
    PREGUNTAS = CONJUNTOS[args.conjunto]

    from huggingface_hub import model_info

    info = model_info(args.decoder)
    parametros = sum(v for v in (info.safetensors.parameters or {}).values()) if info.safetensors else 0
    decoder = {"repo_id": args.decoder, "revision": info.sha, "parametros": parametros or 1, "licencia": "apache-2.0"}
    config = config_local(decoder)
    listas = {"corpus": (RAIZ / config["recuperacion"]["manifiesto"]).is_file(),
              "lexico": (RAIZ / config["recuperacion"]["fragmentos"]).with_name("chunks_complete.json").is_file(),
              "denso": (RAIZ / config["recuperacion"]["indice_meta"]).is_file(),
              "recuperar": (BASE / "pasajes.json").is_file(), "generar": False, "puntaje": False}
    rehacer = False
    for etapa in ETAPAS[:ETAPAS.index(args.hasta) + 1]:
        rehacer = rehacer or etapa == args.desde
        if listas[etapa] and not rehacer:
            print(f"(etapa {etapa} ya hecha)")
            continue
        inicio = time.perf_counter()
        globals()[f"etapa_{etapa}"](copy.deepcopy(config))
        print(f"  etapa {etapa}: {time.perf_counter() - inicio:.0f} s", flush=True)
        gc.collect()


if __name__ == "__main__":
    main()
