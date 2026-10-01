"""Acceso al paquete src/legalrag desde la raíz del repositorio."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "src" / "legalrag")]
