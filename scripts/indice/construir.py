"""Construye el índice FAISS a partir de un archivo de configuración.

Uso: python -m scripts.indice.construir --config configs/indice.json [--congelar]
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.indice.recuperacion import construir_indice  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--salida")
    parser.add_argument("--congelar", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.salida:
        config["salida"] = args.salida
    resumen, _ = construir_indice(config)
    if args.congelar:
        Path(config["salida"], "CONGELADO.json").write_text(json.dumps({
            "sha256_indice": resumen["sha256_indice"], "sha256_corpus": resumen["sha256_corpus"],
            "sha256_configuracion": resumen["sha256_configuracion"]}, indent=2), encoding="utf-8")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
