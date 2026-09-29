"""Configuración congelada del sistema entregado.

Todas las etapas leen de aquí la raíz del repositorio y la configuración de
`configs/sistema.json`. Para la entrega, ese archivo se congela junto al índice.
"""
import json
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
CONFIG = RAIZ / "configs/sistema.json"
OFICIAL = RAIZ / "data/oficial"


def leer_config(ruta=CONFIG):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))
