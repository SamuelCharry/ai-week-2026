import argparse
import ctypes
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

parser = argparse.ArgumentParser()
parser.add_argument("notebook", type=Path)
parser.add_argument("--kernel", default="ai-week")
args = parser.parse_args()
raiz = Path(__file__).resolve().parents[1]
ruta = args.notebook.resolve()
nb = nbformat.read(ruta, as_version=4)
cliente = NotebookClient(nb, timeout=None, kernel_name=args.kernel,
                         resources={"metadata": {"path": str(raiz)}})
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
