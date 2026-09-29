import argparse
import atexit
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))
from legalrag.encoding.encoder import Encoder
from legalrag.indexing.indice import cargar_corpus, construir_indice
from legalrag.comun import escribir_jsonl, hash_json, sha256
from legalrag.evaluation.sondas import crear_sondas, evaluar_sondas, resumir, seleccionar_muestra

parser = argparse.ArgumentParser()
parser.add_argument("--modelos", nargs="*")
parser.add_argument("--tamanos", nargs="+", type=int, default=[192, 320, 448])
parser.add_argument("--por-area", type=int, default=40)
parser.add_argument("--device", default="cuda")
parser.add_argument("--precision", default="float16")
parser.add_argument("--batch-size", type=int, default=4)
args = parser.parse_args()

catalogo = json.loads(Path("configs/modelos.json").read_text(encoding="utf-8"))
unidades, _ = cargar_corpus("data/processed/corpus")
seleccion = seleccionar_muestra(unidades, args.por_area)
ejecucion = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
carpeta = Path("data/experimentos/encoders") / ejecucion
reportes = Path("reports/encoders") / ejecucion
carpeta.mkdir(parents=True, exist_ok=False)
reportes.mkdir(parents=True, exist_ok=False)
seleccion_path = carpeta / "seleccion.json"
seleccion_path.write_text(json.dumps(seleccion, indent=2), encoding="utf-8")
unidades = [u for u in unidades if u["unidad_id"] in set(seleccion)]
sondas = crear_sondas(unidades)
escribir_jsonl(reportes / "sondas.jsonl", sondas)
estado = {"estado": "en_ejecucion", "reportes": reportes.as_posix(), "datos": carpeta.as_posix(),
          "sha256_corpus": hash_json(unidades), "sha256_catalogo": sha256("configs/modelos.json"),
          "sha256_codigo": {p: sha256(p) for p in ["src/legalrag/experimentos/comparar_encoders.py", "src/legalrag/indexing/indice.py", "src/legalrag/evaluation/sondas.py"]},
          "modelos": args.modelos or [f["nombre"] for f in catalogo["encoders"]], "tamanos": args.tamanos}
Path("reports/f_encoders_ultima.json").write_text(json.dumps(estado, indent=2), encoding="utf-8")


def registrar_salida():
    if estado["estado"] == "en_ejecucion":
        estado["estado"] = "incompleto"
        Path("reports/f_encoders_ultima.json").write_text(json.dumps(estado, indent=2), encoding="utf-8")


atexit.register(registrar_salida)
print(f"Muestra: {len(unidades)} artículos, {len(sondas)} sondas", flush=True)

for ficha in sorted(catalogo["encoders"], key=lambda x: x["prioridad"]):
    if args.modelos and ficha["nombre"] not in args.modelos:
        continue
    nombre = ficha["nombre"]
    print("Cargando " + nombre, flush=True)
    inicio = time.perf_counter()
    encoder = Encoder(ficha, args.device, args.precision)
    carga_s = time.perf_counter() - inicio
    encoder.encode([unidades[0]["texto"][:120]], "query")
    for tamano in args.tamanos:
        config = {"encoder": ficha, "corpus": "data/processed/corpus",
                  "seleccion": seleccion_path.as_posix(), "tamano_tokens": tamano,
                  "solapamiento_tokens": 32, "device": args.device, "precision": args.precision,
                  "batch_size": args.batch_size, "hibrido": False,
                  "salida": f"data/index/experimentos/{ejecucion}/{nombre}_{tamano}"}
        (carpeta / f"{nombre}_{tamano}.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        manifiesto, recuperador = construir_indice(config, encoder)
        for hibrido in [False, True]:
            from legalrag.retrieval.recuperador import BM25
            recuperador.bm25 = BM25([v["texto_busqueda"] for v in recuperador.ventanas]) if hibrido else None
            resultados = evaluar_sondas(recuperador, sondas)
            metodo = "hibrido" if hibrido else "denso"
            prefijo = str(reportes / f"f_encoder_{nombre}_{tamano}_{metodo}")
            escribir_jsonl(prefijo + "_sondas.jsonl", resultados)
            resumen = {"modelo": nombre, "recomendado": ficha["recomendado"], "tamano_tokens": tamano,
                       "metodo": metodo, **resumir(resultados), "carga_s": carga_s,
                       **{k: manifiesto[k] for k in ["segundos", "ventanas", "unidades", "vram_pico_mib", "ram_rss_mib", "sha256_indice"]}}
            Path(prefijo + ".json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
            print(json.dumps(resumen), flush=True)
        del recuperador
        gc.collect()
    del encoder
    gc.collect()
    import torch
    torch.cuda.empty_cache()

estado["estado"] = "completo"
Path("reports/f_encoders_ultima.json").write_text(json.dumps(estado, indent=2), encoding="utf-8")
