"""Ejecuta un notebook de punta a punta y guarda sus salidas.

Evita que Windows suspenda la máquina durante la corrida.

Uso: python -m legalrag.experimentos.ejecutar_notebook notebooks/basicos/00_eda.ipynb
"""
import argparse
import ctypes
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("--kernel", default="ai-week")
    args = parser.parse_args()
    ruta = args.notebook.resolve()
    nb = nbformat.read(ruta, as_version=4)
    cliente = NotebookClient(nb, timeout=None, kernel_name=args.kernel,
                             resources={"metadata": {"path": str(RAIZ)}})
    mantener_activo = sys.platform == "win32"
    if mantener_activo:
        if not ctypes.windll.kernel32.SetThreadExecutionState(0x80000001):
            raise OSError("No se pudo evitar la suspensión automática durante la ejecución")
    try:
        cliente.execute()
    finally:
        if mantener_activo:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        nbformat.write(nb, ruta)
    print("Ejecutado:", ruta.name)


if __name__ == "__main__":
    main()
