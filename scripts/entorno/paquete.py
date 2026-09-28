"""Empaqueta el proyecto para Colab en una ruta fija.

Un solo mecanismo y un solo destino: data/colab/ai-week.zip. Antes había un ZIP por
experimento (ai-week-colab, ai-week-experimentos, ai-week-v04) y había que recordar
cuál subir. Los anteriores se conservan; este es el que se regenera.

Incluye notebooks, scripts, configs, los originales de data/raw, el corpus procesado
y el material oficial. No incluye llaves, pesos de modelos ni resultados.

Uso: python -m scripts.entorno.paquete [--nombre otro.zip]
"""
import argparse
import sys
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.entorno.runtime import crear_paquete, sha256_archivo  # noqa: E402

NOMBRE = "ai-week.zip"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nombre", default=NOMBRE, help=f"nombre del ZIP (por defecto {NOMBRE})")
    args = ap.parse_args()

    destino = crear_paquete(RAIZ, args.nombre)
    print("Paquete:", destino)
    print("Tamaño:", round(destino.stat().st_size / 2**20, 1), "MiB")
    print("sha256:", sha256_archivo(destino))
    print("\nSubirlo a Mi unidad/AIWEEK sin descomprimir.")


if __name__ == "__main__":
    main()
