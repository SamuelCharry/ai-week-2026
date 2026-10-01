"""Reconstruye chunks.sqlite e index.faiss desde los textos ya preparados, sin volver a extraer el corpus.

Para cuando cambia la segmentación (preprocessing.ingesta) o se reemplazan documentos del inventario
(ingestion.agregar_puntuales --sin-indice): vuelve a partir los textos de data/processed/corpus_preparado
con el código actual y vectoriza todo con el encoder de configs/sistema.json (código de E06). No descarga
ni extrae nada; la extracción de los ~19.000 documentos no cambia.

    python3 src/reconstruir_indice.py            # aparta el índice actual y construye uno nuevo
    python3 src/reconstruir_indice.py --borrar   # borra el índice actual (si falta disco)

Si se corta, el mismo comando continúa: los fragmentos ya hechos se conservan y la vectorización sigue desde
su último punto de control. Tarda horas (1,7 millones de fragmentos con BGE-M3 en la 4090).
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--borrar", action="store_true", help="borra el índice actual en vez de apartarlo")
    args = ap.parse_args()

    from legalrag.config import leer_config
    from legalrag.experimentos import corpus_definitivo as e06

    rec = leer_config()["recuperacion"]
    carpeta = (RAIZ / rec["fragmentos"]).parent
    if carpeta.resolve() != e06.RUNS.resolve() or (RAIZ / rec["manifiesto"]).parent.resolve() != e06.RELEASE.resolve():
        sys.exit(f"configs/sistema.json apunta a {carpeta} y E06 construye en {e06.RUNS}: no coinciden")
    if not e06.MODEL_CATALOG.is_file():
        e06.MODEL_CATALOG.parent.mkdir(parents=True, exist_ok=True)
        e06.MODEL_CATALOG.write_text(json.dumps([rec["encoder"]], ensure_ascii=False, indent=2), encoding="utf-8")
    indice = e06._index_dir(rec["encoder"]["repo_id"])
    marca = carpeta / "reconstruccion_en_curso.json"
    if marca.is_file():
        print("Continuando la reconstrucción anterior.")
    elif carpeta.exists():
        if args.borrar:
            shutil.rmtree(carpeta)
            print(f"Índice anterior borrado: {carpeta.relative_to(RAIZ)}")
        else:
            destino = carpeta.with_name(f"{carpeta.name}_anterior_{time.strftime('%Y%m%d_%H%M')}")
            carpeta.rename(destino)
            print(f"Índice anterior apartado en {destino.relative_to(RAIZ)} (bórrenlo cuando el nuevo funcione)")
    carpeta.mkdir(parents=True, exist_ok=True)
    marca.write_text(json.dumps({"inicio": time.strftime("%Y-%m-%d %H:%M")}), encoding="utf-8")

    inicio = time.perf_counter()
    print("1/2 Fragmentos y BM25 con la segmentación actual...", flush=True)
    print("   ", e06.build_lexical(force=not (carpeta / "chunks_complete.json").is_file()))
    print("2/2 Vectores BGE-M3 (horas; se reanuda desde el último punto de control)...", flush=True)
    print("   ", e06.build_dense(rec["encoder"]["repo_id"]))
    import faiss
    import sqlite3
    total = sqlite3.connect(carpeta / "chunks.sqlite").execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    vectores = faiss.read_index(str(indice / "index.faiss")).ntotal
    if total != vectores:
        sys.exit(f"Fragmentos {total:,} y vectores {vectores:,} no coinciden: volver a correr el comando")
    marca.unlink()
    print(f"\nListo en {(time.perf_counter() - inicio) / 3600:.1f} h: {total:,} fragmentos = {vectores:,} vectores")


if __name__ == "__main__":
    main()
