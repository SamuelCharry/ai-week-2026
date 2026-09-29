"""Punto de entrada: python -m legalrag <comando> [argumentos del comando].

    ingest     descarga los originales desde las URL del inventario (ingestion.reconstruir)
    prepare    extrae texto canónico con procedencia (preprocessing.preparar_corpus)
    index      construye y congela el índice FAISS (indexing.construir)
    answer     responde un split y escribe la entrega JSONL (agent.pipeline)
    run        reproducción completa sobre la muestra y evaluador oficial (agent.reproducir)
    evaluate   evaluador oficial sobre una entrega (data/oficial/scripts/evaluate.py)
    serve      servicio POST /preguntar para la interfaz (agent.servicio)

`python -m legalrag <comando> --help` muestra los argumentos de cada uno.
"""
import runpy
import subprocess
import sys

from legalrag.config import OFICIAL

COMANDOS = {
    "ingest": "legalrag.ingestion.reconstruir",
    "prepare": "legalrag.preprocessing.preparar_corpus",
    "index": "legalrag.indexing.construir",
    "answer": "legalrag.agent.pipeline",
    "run": "legalrag.agent.reproducir",
    "serve": "legalrag.agent.servicio",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in {*COMANDOS, "evaluate"}:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    comando, resto = argv[0], argv[1:]
    if comando == "evaluate":
        # El evaluador oficial importa sus módulos vecinos; se ejecuta como script propio.
        sys.exit(subprocess.run([sys.executable, str(OFICIAL / "scripts/evaluate.py"), *resto]).returncode)
    sys.argv = [f"legalrag {comando}", *resto]
    runpy.run_module(COMANDOS[comando], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
