"""Cerberus de punta a punta, en un solo comando: la configuración de entrega (configs/sistema.json).

BM25 + BGE-M3 con RRF, reranker BGE-v2-m3 y tres agentes alrededor de Qwen3-8B (agent.componentes).

Para iterar sobre Cerberus con los datos ya en su lugar (sin caché: tiempos y respuestas reales):

    python3 src/main.py --datos data                    # muestra + evaluador + cambios frente a la corrida anterior
    python3 src/main.py --datos data --comparar-con data/comparacion/entrega/submissions.jsonl

    python3 src/main.py --datos datos --prueba          # 3 preguntas, para medir y revisar
    python3 src/main.py --datos datos                   # las 50 de muestra + evaluador oficial
    python3 src/main.py --datos datos --split test      # las 992 del sábado -> submissions.jsonl
    python3 src/main.py --datos datos --solo-preparar   # solo revisa entorno, datos e índice

    python3 src/main.py --desde-raw --solo-corpus       # desde data/raw: textos, inventario y BM25 (sin GPU)
    python3 src/main.py --desde-raw --prueba            # desde data/raw hasta 3 respuestas (con GPU)

Pasos:
  1. Entorno: Python, GPU con CUDA y paquetes.
  2. Datos (--datos): busca dentro de --datos los textos del corpus, el manifiesto de corpus_eval_v1,
     el catálogo de modelos y, si existen, chunks.sqlite y el índice de E06. Los enlaza en las
     rutas de configs/sistema.json (si no se puede enlazar, los copia).
     Datos (--desde-raw): prepara los originales de data/raw como textos canónicos
     (preprocessing.preparar_corpus, reanudable) y arma el inventario data/releases/corpus_eval_v1
     con el mismo formato que ingestion.fijar_corpus_evaluacion, sin sus reportes de perfilado.
  3. Índice: si falta chunks.sqlite o el índice FAISS de BGE-M3, los construye (tarda; si se
     corta, volver a correr el mismo comando continúa desde el último punto de control).
  4. Respuestas: una por pregunta, guardadas al terminar; si se corta, se reanuda.
  5. Evaluación: el evaluador oficial sobre la muestra (no en --prueba ni en test).
"""
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

PRUEBA = [51, 79, 140]
PAQUETES = {"torch": "torch", "transformers": "transformers", "accelerate": "accelerate",
            "faiss": "faiss-cpu", "jsonschema": "jsonschema", "numpy": "numpy", "psutil": "psutil"}


def paso(numero, texto):
    print(f"\n=== {numero}. {texto}", flush=True)


def salir(mensaje):
    print(f"\nERROR: {mensaje}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------ 1. entorno

def comprobar_entorno(gpu=True):
    paso(1, "Entorno")
    if sys.version_info < (3, 10):
        salir(f"Se necesita Python 3.10 o superior; este es {sys.version.split()[0]}")
    faltan = []
    requeridos = PAQUETES if gpu else {"bs4": "beautifulsoup4", "ftfy": "ftfy", "pdfplumber": "pdfplumber"}
    for modulo, paquete in requeridos.items():
        try:
            __import__(modulo)
        except ImportError:
            faltan.append(paquete)
    if faltan:
        salir("Faltan paquetes: " + ", ".join(faltan) + "\n"
              "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
              "  pip install -r requirements.txt -r data/oficial/scripts/requirements-evaluador.txt")
    if not gpu:
        print(f"Python {sys.version.split()[0]} · sin GPU (solo corpus y BM25)")
        return
    import torch
    import transformers
    if not torch.cuda.is_available():
        salir("PyTorch no ve una GPU con CUDA. Instalar torch con cu128 y revisar el driver (nvidia-smi).")
    nombre = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"Python {sys.version.split()[0]} · torch {torch.__version__} · transformers {transformers.__version__}")
    print(f"GPU: {nombre}, {vram:.1f} GiB")
    if vram < 20:
        salir("Se necesitan al menos 20 GiB de VRAM (RTX 4090 o A100).")
    if not transformers.__version__.startswith("5."):
        print("AVISO: E06 construyó el índice con transformers 5.3; esta versión puede dar resultados distintos.")


# ------------------------------------------------------------------- 2. datos

def buscar(base, condicion):
    for carpeta, subcarpetas, archivos in os.walk(base):
        subcarpetas[:] = [s for s in subcarpetas if s not in (".git", "__pycache__", ".venv")]
        actual = Path(carpeta)
        if condicion(actual, archivos):
            return actual
    return None


def enlazar(origen, destino):
    """Deja `destino` apuntando a `origen`. No toca un destino real que ya exista."""
    destino = RAIZ / destino
    if destino.is_symlink() or destino.exists():
        if destino.resolve() == Path(origen).resolve():
            return "ya estaba"
        if not destino.is_symlink():
            return "se conserva el existente"
        destino.unlink()
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        destino.symlink_to(Path(origen).resolve(), target_is_directory=Path(origen).is_dir())
        return "enlazado"
    except OSError:  # Windows sin permiso de enlaces
        (shutil.copytree if Path(origen).is_dir() else shutil.copy2)(origen, destino)
        return "copiado"


def preparar_datos(datos, config):
    paso(2, f"Datos en {datos}")
    if not datos.is_dir():
        salir(f"No existe la carpeta {datos}")
    rec = config["recuperacion"]
    encontrados = {
        "textos": buscar(datos, lambda c, a: c.name == "textos" and c.parent.name == "corpus_preparado"),
        "release": buscar(datos, lambda c, a: c.name == "corpus_eval_v1" and "corpus_manifest.json" in a),
        "catalogo": buscar(datos, lambda c, a: "modelos_verificados.json" in a),
        "ejecucion": buscar(datos, lambda c, a: "chunks.sqlite" in a),
    }
    if encontrados["textos"] is None and not (RAIZ / rec["textos"]).is_dir():
        salir("No encontré corpus_preparado/textos dentro de la carpeta de datos.")
    if encontrados["release"] is None and not (RAIZ / rec["manifiesto"]).is_file():
        salir("No encontré corpus_eval_v1/corpus_manifest.json dentro de la carpeta de datos.")
    destinos = {
        "textos": ("data/processed/corpus_preparado", lambda p: p.parent),
        "release": ("data/releases/corpus_eval_v1", lambda p: p),
        "catalogo": ("reports/reporte_evaluacion/modelos_verificados.json", lambda p: p / "modelos_verificados.json"),
        "ejecucion": (str(Path(rec["fragmentos"]).parent), lambda p: p),
    }
    for clave, ruta in encontrados.items():
        destino, origen = destinos[clave]
        if ruta is None:
            print(f"  {clave:10} no está en los datos")
            continue
        print(f"  {clave:10} {origen(ruta)} -> {destino} ({enlazar(origen(ruta), destino)})")
    textos = RAIZ / rec["textos"]
    print(f"  textos del corpus: {sum(1 for _ in textos.glob('*.txt'))}")


def preparar_desde_raw(config, workers):
    paso(2, "Datos desde data/raw")
    raw = RAIZ / "data/raw"
    if not (raw / "manifest.json").is_file():
        salir("No está data/raw/manifest.json.")
    rec = config["recuperacion"]
    preparado = RAIZ / "data/processed/corpus_preparado"
    release = RAIZ / rec["manifiesto"]
    if release.is_file():
        print(f"  inventario ya existe: {release.parent} (borrarlo para rehacerlo)")
    else:
        print(f"  Preparando textos canónicos con {workers} procesos (reanudable; puede tardar)...", flush=True)
        entorno = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent)}
        codigo = subprocess.run([sys.executable, "-m", "legalrag.preprocessing.preparar_corpus", "--raw", str(raw),
                                 "--output", str(preparado), "--workers", str(workers)], cwd=RAIZ, env=entorno).returncode
        if not (preparado / "resumen.json").is_file():
            salir("La preparación falló antes de terminar (ver el error arriba); el mismo comando la reanuda.")
        resumen = json.loads((preparado / "resumen.json").read_text(encoding="utf-8"))
        if codigo not in (0, 1) or resumen["modo"] != "completo":
            salir("La preparación no terminó; volver a correr el mismo comando la reanuda.")
        if resumen["errores"]:
            print(f"  AVISO: {resumen['errores']} documentos no se pudieron extraer; quedan fuera (ver pendientes.csv).")
        from legalrag.preprocessing.inventario import fijar_inventario
        fijar_inventario(preparado, release.parent, RAIZ)
    catalogo = RAIZ / "reports/reporte_evaluacion/modelos_verificados.json"
    if not catalogo.is_file():
        catalogo.parent.mkdir(parents=True, exist_ok=True)
        fichas = [rec["encoder"], rec["reranker"], config["generacion"]["decoder"]]
        catalogo.write_text(json.dumps([{**f, "cumple_limite_8000000000": f.get("parametros", 0) <= 8_000_000_000}
                                        for f in fichas], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  catálogo de modelos escrito desde configs/sistema.json: {catalogo}")


# ------------------------------------------------------------------ 3. índice

def estado_indice(rec):
    fragmentos, indice, meta = (RAIZ / rec[k] for k in ("fragmentos", "indice_denso", "indice_meta"))
    return fragmentos.is_file() and indice.is_file() and meta.is_file()


def apartar_si_otro_inventario(rec):
    """Si los fragmentos existentes son de otro inventario, se aparta la carpeta (no se borra) y se reconstruye.

    Pasa cuando se copió un índice de otra corrida (p. ej. el de E06 en Colab) pero el inventario
    data/releases/corpus_eval_v1 se armó aquí: mezclarlos daría pasajes de un corpus con vectores de otro.
    """
    carpeta = (RAIZ / rec["fragmentos"]).parent
    marcador = carpeta / "chunks_complete.json"
    release = (RAIZ / rec["manifiesto"]).parent / "snapshot.json"
    if not marcador.is_file() or not release.is_file():
        return
    anterior = json.loads(marcador.read_text(encoding="utf-8")).get("snapshot_id", "")
    actual = json.loads(release.read_text(encoding="utf-8"))["snapshot_id"]
    if anterior == actual:
        return
    destino = carpeta.with_name(f"{carpeta.name}_inventario_{anterior[:8]}")
    n = 1
    while destino.exists():
        destino, n = carpeta.with_name(f"{carpeta.name}_inventario_{anterior[:8]}_{n}"), n + 1
    carpeta.rename(destino)
    print(f"  AVISO: el índice existente era de otro inventario ({anterior[:8]} ≠ {actual[:8]}).\n"
          f"  Se apartó en {destino.relative_to(RAIZ)} (no se borró) y se reconstruye para el inventario actual.")


def preparar_indice(config, solo_lexico=False):
    paso(3, "Índice (BM25 en SQLite + BGE-M3 en FAISS)")
    rec = config["recuperacion"]
    apartar_si_otro_inventario(rec)
    if solo_lexico:
        from legalrag.experimentos import corpus_definitivo as e06
        print("  Segmentación y BM25 (sin GPU)...", flush=True)
        print("  ", e06.build_lexical())
        return
    if not estado_indice(rec):
        if not (RAIZ / "reports/reporte_evaluacion/modelos_verificados.json").is_file():
            salir("Para construir el índice falta modelos_verificados.json (viene en el paquete del corpus).")
        from legalrag.experimentos import corpus_definitivo as e06
        print("No está el índice: se construye con el código de E06.")
        print("  Segmentación y BM25 (unos minutos)...", flush=True)
        print("  ", e06.build_lexical())
        print("  Vectores BGE-M3 de todos los fragmentos (puede tardar horas; se reanuda si se corta)...", flush=True)
        print("  ", e06.build_dense(rec["encoder"]["repo_id"]))
    import faiss
    total = sqlite3.connect(RAIZ / rec["fragmentos"]).execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vectores = faiss.read_index(str(RAIZ / rec["indice_denso"])).ntotal
    meta = json.loads((RAIZ / rec["indice_meta"]).read_text(encoding="utf-8"))
    print(f"  fragmentos {total:,} · vectores {vectores:,} · encoder {meta['modelo']} {meta['revision'][:8]}")
    if total != vectores:
        salir("chunks.sqlite y el índice FAISS no tienen los mismos fragmentos: son de corridas distintas.")
    if meta["revision"] != rec["encoder"]["revision"]:
        salir("El índice se construyó con otra revisión de BGE-M3 que la de configs/sistema.json.")


# -------------------------------------------------------- 4 y 5. responder y evaluar

def responder(config, split, ids):
    from legalrag.agent.pipeline import responder_lote

    paso(4, f"Respuestas ({'prueba: ' + ', '.join(map(str, ids)) if ids else split})")
    entrada = RAIZ / config["entradas"][split]
    if not entrada.is_file():
        salir(f"No está {entrada}. Para el sábado, copiar ahí test_992.jsonl.")
    salida = RAIZ / ("data/reproduccion/prueba.jsonl" if ids else config["salidas"][split])
    print(f"La primera vez descarga {config['generacion']['decoder']['repo_id']}, BGE-M3 y el reranker (~20 GB).",
          flush=True)
    resumen = responder_lote(RAIZ, entrada, salida, config, ids=ids)
    print(json.dumps({k: v for k, v in resumen.items() if k != "errores_esquema"}, ensure_ascii=False, indent=2))
    if resumen["errores_esquema"]:
        print(f"{len(resumen['errores_esquema'])} errores de esquema; detalle en {salida.stem}_resumen.json")
    problemas = {}
    for archivo in salida.with_name(salida.stem + "_respuestas").glob("*.json"):
        problema = json.loads(archivo.read_text(encoding="utf-8")).get("problema") or "respondida"
        problemas[problema] = problemas.get(problema, 0) + 1
    print("Resultado por pregunta:", json.dumps(problemas, ensure_ascii=False))
    return salida, resumen


def evaluar(salida, ragas):
    paso(5, "Evaluador oficial")
    reporte = salida.with_name(salida.stem + "_reporte.json")
    comando = [sys.executable, str(RAIZ / "data/oficial/scripts/evaluate.py"), "--submission", str(salida),
               "--split", "sample", "--out", str(reporte)] + (["--ragas"] if ragas else [])
    subprocess.run(comando, check=True)
    print("Reporte:", reporte)


def iteracion(config, salida, comparar_con, aproximar_ragas):
    """Resumen de la iteración: puntaje, cambios pregunta por pregunta frente a la corrida anterior y RAGAS≈."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("comparar", RAIZ / "src/comparar.py")
    comparar = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comparar)
    ev = comparar.evaluador_oficial(config)
    muestra = ev.read_jsonl(RAIZ / config["entradas"]["sample"])
    reporte = json.loads(salida.with_name(salida.stem + "_reporte.json").read_text(encoding="utf-8"))
    lineas = ["", "ITERACIÓN (muestra de 50, evaluador oficial)",
              f"  total {reporte['total_automatico']['obtenidos']} · cerradas {reporte['cerradas']['aciertos']}/"
              f"{reporte['cerradas']['n']} ({reporte['cerradas']['puntos']}) · citas {reporte['citas']['puntos']} "
              f"(recall {reporte['citas']['recall_citas_ponderado']}) · abstención {reporte['abstencion']['puntos']}"]
    actual = comparar.por_pregunta(ev, salida, muestra)
    previo = comparar.por_pregunta(ev, comparar_con, muestra) if comparar_con and comparar_con.is_file() else None
    if previo:
        cambios = comparar.cambios(previo, actual)
        lineas.append(f"  frente a {comparar_con.relative_to(RAIZ) if comparar_con.is_relative_to(RAIZ) else comparar_con}: "
                      f"{sum(c[0] == '+' for c in cambios)} mejoras, {sum(c[0] == '-' for c in cambios)} empeoras")
        lineas += [f"    {signo} {qid:>4} {formato:<11} {detalle}" for signo, qid, formato, detalle in cambios]
    if aproximar_ragas:
        from legalrag.evaluation.ragas_local import RagasLocal

        juez = RagasLocal(comparar.VERIFICADOR_NLI["modelo"])
        juez.abrir()
        try:
            def puntos(ruta):
                return juez.evaluar({r["id"]: r for r in ev.read_jsonl(ruta)}, muestra)["puntos_aprox"]
            texto = f"  RAGAS≈ {puntos(salida)} de 30"
            if previo:
                texto += f" (anterior {puntos(comparar_con)})"
            lineas.append(texto + " · aproximación local; el juez oficial es --ragas")
        finally:
            juez.cerrar()
    texto = "\n".join(lineas)
    print(texto)
    salida.with_name("iteracion.txt").write_text(texto + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos", help="carpeta donde se descomprimieron los datos")
    ap.add_argument("--desde-raw", action="store_true", help="prepara el corpus desde data/raw en vez de usar --datos")
    ap.add_argument("--solo-corpus", action="store_true", help="con --desde-raw: textos, inventario y BM25, sin GPU")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="procesos de la preparación")
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    ap.add_argument("--prueba", action="store_true", help=f"solo las preguntas {PRUEBA}")
    ap.add_argument("--ids", nargs="+", type=int, help="solo estas preguntas")
    ap.add_argument("--solo-preparar", action="store_true", help="revisa entorno, datos e índice y termina")
    ap.add_argument("--ragas", action="store_true", help="evalúa también texto libre (OPENROUTER_API_KEY)")
    ap.add_argument("--comparar-con", type=Path,
                    help="entrega anterior para ver cambios pregunta por pregunta (por defecto, la corrida previa)")
    ap.add_argument("--sin-ragas-local", action="store_true", help="no calcula la aproximación local de RAGAS")
    args = ap.parse_args()

    from legalrag.config import leer_config

    os.chdir(RAIZ)
    config = leer_config()
    if args.solo_corpus and not args.desde_raw:
        salir("--solo-corpus va con --desde-raw.")
    comprobar_entorno(gpu=not args.solo_corpus)
    if args.desde_raw:
        preparar_desde_raw(config, args.workers)
    else:
        preparar_datos(args.datos.resolve(), config)
    preparar_indice(config, solo_lexico=args.solo_corpus)
    if args.solo_corpus:
        print("\nCorpus e índice BM25 listos. En la máquina con GPU: python3 src/main.py --desde-raw --prueba")
        return
    if args.solo_preparar:
        print("\nListo para responder.")
        return
    ids = args.ids or (PRUEBA if args.prueba else None)
    comparar_con = args.comparar_con.resolve() if args.comparar_con else None
    if args.split == "sample" and not ids and comparar_con is None:
        # La corrida previa queda como referencia de la iteración.
        previa = RAIZ / config["salidas"]["sample"]
        if previa.is_file():
            comparar_con = previa.with_name(previa.stem + "_anterior.jsonl")
            shutil.copy2(previa, comparar_con)
    salida, resumen = responder(config, args.split, ids)
    if args.split == "sample" and not ids and resumen["respondidas"] == resumen["preguntas"]:
        evaluar(salida, args.ragas)
        iteracion(config, salida, comparar_con, not args.sin_ragas_local and not args.ragas)
    elif args.split == "test":
        print(f"\nEntrega: {salida}. Revisar que 'respondidas' sea 992 y que no haya errores de esquema.")


if __name__ == "__main__":
    main()
