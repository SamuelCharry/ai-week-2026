"""Compara dos entregas ya generadas sobre la muestra, sin volver a generar: evaluador oficial, cambios pregunta
por pregunta y RAGAS≈ de ambas (src/main.py hace lo mismo al terminar cada corrida).

    python3 src/comparar_entregas.py ANTERIOR.jsonl NUEVA.jsonl
    python3 src/comparar_entregas.py data/comparacion/entrega/submissions.jsonl data/reproduccion/submissions_sample.jsonl
"""
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anterior", type=Path)
    ap.add_argument("nueva", type=Path)
    ap.add_argument("--sin-ragas-local", action="store_true")
    args = ap.parse_args()

    from legalrag.config import leer_config

    config = leer_config()
    for entrega in (args.anterior, args.nueva):
        if not entrega.is_file():
            sys.exit(f"No está {entrega}")
    reporte = args.nueva.with_name(args.nueva.stem + "_reporte.json")
    subprocess.run([sys.executable, str(RAIZ / config["oficial"] / "scripts/evaluate.py"), "--submission",
                    str(args.nueva), "--split", "sample", "--out", str(reporte)], check=True, capture_output=True)
    spec = importlib.util.spec_from_file_location("main_entrega", RAIZ / "src/main.py")
    principal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(principal)
    principal.iteracion(config, args.nueva.resolve(), args.anterior.resolve(), not args.sin_ragas_local)


if __name__ == "__main__":
    main()
