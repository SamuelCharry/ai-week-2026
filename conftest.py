"""Pone la raíz del proyecto en sys.path para que `scripts.*` se importe.

Sin esto, tres de las pruebas solo corrían desde la raíz del repo.
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))
