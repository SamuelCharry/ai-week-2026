"""Prueba ligera de ejecución integral, sin descargar modelos neuronales."""
from pathlib import Path
import re

import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
for name in ("01_comparacion_encoders_chunking.ipynb", "02_comparacion_recuperacion_faiss.ipynb"):
    nb = nbformat.read(root / "notebooks" / name, as_version=4)
    if name.startswith("01"):
        for cell in nb.cells:
            if cell.cell_type == "code" and "encoder_names = " in cell.source:
                cell.source = re.sub(r"^encoder_names = .*", 'encoder_names = ["tfidf"]', cell.source, flags=re.M)
    else:
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.source = cell.source.replace("ACTIVAR_CROSS_ENCODER = True", "ACTIVAR_CROSS_ENCODER = False")
    NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": str(root)}}).execute()
    print(name, "OK")
