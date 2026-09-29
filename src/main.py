"""R03 del reporte de punta a punta, en un solo comando (rama r03-reporte).

BM25 + BGE-M3 con RRF, reranker BGE-v2-m3 y Qwen2.5-7B-Instruct, con ventanas de 384 tokens
(solapamiento 64) vinculadas al artículo o sección íntegros.

    python3 src/main.py --desde-raw --solo-recuperacion  # R03 sin decoder: respaldo de citas vs E06

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
        # El índice de E06 (ventanas de 1.500 caracteres) no sirve para R03, que construye el suyo.
        "ejecucion": None if "segmentacion" in rec else buscar(datos, lambda c, a: "chunks.sqlite" in a),
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
        fijar_inventario(preparado, release.parent)
    catalogo = RAIZ / "reports/reporte_evaluacion/modelos_verificados.json"
    if not catalogo.is_file():
        catalogo.parent.mkdir(parents=True, exist_ok=True)
        fichas = [rec["encoder"], rec["reranker"], config["generacion"]["decoder"]]
        catalogo.write_text(json.dumps([{**f, "cumple_limite_8000000000": f.get("parametros", 0) <= 8_000_000_000}
                                        for f in fichas], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  catálogo de modelos escrito desde configs/sistema.json: {catalogo}")


def fijar_inventario(preparado, destino):
    """Inventario evaluable: los documentos extraídos y aptos para búsqueda, con texto y SHA-256."""
    import hashlib
    from datetime import datetime, timezone

    campos = ("doc_id", "titulo", "tipo", "numero", "anio", "fuente", "url", "fecha_consulta", "organo_emisor",
              "areas", "temas", "nivel", "vigencia", "vigencia_fuente", "licencia_fuente", "redistribuir_raw",
              "edicion_con_anotaciones", "origen_ampliacion", "areas_por_epigrafe", "sha256_texto", "caracteres",
              "estado_extraccion", "avisos")
    elegidos, excluidos, restricciones = [], [], []
    for linea in (preparado / "documentos.jsonl").read_text(encoding="utf-8").splitlines():
        registro = json.loads(linea)
        fila = {k: registro.get(k) for k in campos}
        fila["texto_archivo"] = (Path("data/processed/corpus_preparado") / (registro.get("texto_archivo") or "")).as_posix()
        fila["restricciones_especificas"] = sorted(set(registro.get("pendientes_preparacion", [])))
        fila["archivos_raw"] = registro["archivos_raw"]
        if registro["estado_extraccion"] == "error" or registro.get("apta_para_busqueda") is False:
            excluidos.append(fila)
        else:
            elegidos.append(fila)
            if fila["restricciones_especificas"]:
                restricciones.append({"doc_id": fila["doc_id"], "restricciones": fila["restricciones_especificas"]})
    destino.mkdir(parents=True, exist_ok=True)
    archivos = []
    for nombre, contenido in (("corpus_manifest.json", sorted(elegidos, key=lambda r: r["doc_id"])),
                              ("excluidos.json", sorted(excluidos, key=lambda r: r["doc_id"])),
                              ("restricciones.json", restricciones)):
        ruta = destino / nombre
        ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        archivos.append({"archivo": nombre, "sha256": hashlib.sha256(ruta.read_bytes()).hexdigest()})
    snapshot = {"version": destino.name,
                "snapshot_id": hashlib.sha256(json.dumps(archivos, sort_keys=True).encode()).hexdigest(),
                "fecha_utc": datetime.now(timezone.utc).isoformat(),
                "tipo_snapshot": "desde_data_raw_con_src_main_sin_perfilado",
                "archivos_version": archivos, "documentos_evaluables": len(elegidos),
                "documentos_excluidos": len(excluidos)}
    (destino / "snapshot.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  inventario: {len(elegidos)} documentos evaluables, {len(excluidos)} excluidos -> {destino}")


# ------------------------------------------------------------------ 3. índice

def estado_indice(rec):
    fragmentos, indice, meta = (RAIZ / rec[k] for k in ("fragmentos", "indice_denso", "indice_meta"))
    return fragmentos.is_file() and indice.is_file() and meta.is_file()


def preparar_indice(config, solo_lexico=False):
    paso(3, "Índice (BM25 en SQLite + BGE-M3 en FAISS)")
    rec = config["recuperacion"]
    from legalrag.experimentos import corpus_definitivo as e06

    def lexico():
        if "segmentacion" in rec:
            from legalrag.indexing import r03
            print("  Segmentación R03 (ventanas por tokens) y BM25, sin GPU...", flush=True)
            e06.RUNS = (RAIZ / rec["fragmentos"]).parent
            return r03.construir(RAIZ, rec)
        return e06.build_lexical()

    if solo_lexico:
        print("  ", lexico())
        return
    if not estado_indice(rec):
        if not (RAIZ / "reports/reporte_evaluacion/modelos_verificados.json").is_file():
            salir("Para construir el índice falta modelos_verificados.json (viene en el paquete del corpus).")
        print("No está el índice: se construye.")
        print("  ", lexico())
        e06.RUNS = (RAIZ / rec["fragmentos"]).parent
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
    print("La primera vez descarga Qwen2.5-7B, BGE-M3 y el reranker (~20 GB).", flush=True)
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


def medir_recuperacion(config):
    from legalrag.evaluation.entrega import cargar_jsonl
    from legalrag.evaluation.oficial import cargar_citaciones, guardar_json
    from legalrag.evaluation.recuperacion import evaluar as evaluar_recuperacion
    from legalrag.retrieval.hibrido import RecuperadorHibrido

    paso(4, "Recuperación R03 sin decoder (50 preguntas de muestra)")
    recuperador = RecuperadorHibrido(RAIZ, config["recuperacion"])
    recuperador.abrir()
    try:
        preguntas = cargar_jsonl(RAIZ / config["entradas"]["sample"])
        resultado = evaluar_recuperacion(recuperador, preguntas, cargar_citaciones(RAIZ))
    finally:
        recuperador.cerrar()
    ruta = RAIZ / "data/reproduccion/recuperacion_r03.json"
    guardar_json(ruta, resultado)
    print(json.dumps({k: v for k, v in resultado.items() if k != "detalle"}, ensure_ascii=False, indent=2))
    print("Detalle por pregunta:", ruta)


def evaluar(salida, ragas):
    paso(5, "Evaluador oficial")
    reporte = salida.with_name(salida.stem + "_reporte.json")
    comando = [sys.executable, str(RAIZ / "data/oficial/scripts/evaluate.py"), "--submission", str(salida),
               "--split", "sample", "--out", str(reporte)] + (["--ragas"] if ragas else [])
    subprocess.run(comando, check=True)
    print("Reporte:", reporte)


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
    ap.add_argument("--solo-recuperacion", action="store_true", help="mide R03 sin decoder y termina")
    ap.add_argument("--ragas", action="store_true", help="evalúa también texto libre (OPENROUTER_API_KEY)")
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
    if args.solo_recuperacion:
        medir_recuperacion(config)
        return
    ids = args.ids or (PRUEBA if args.prueba else None)
    salida, resumen = responder(config, args.split, ids)
    if args.split == "sample" and not ids and resumen["respondidas"] == resumen["preguntas"]:
        evaluar(salida, args.ragas)
    elif args.split == "test":
        print(f"\nEntrega: {salida}. Revisar que 'respondidas' sea 992 y que no haya errores de esquema.")


if __name__ == "__main__":
    main()
