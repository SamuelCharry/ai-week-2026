"""Pone src/ en sys.path para que `legalrag.*` se importe sin instalar el paquete."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
