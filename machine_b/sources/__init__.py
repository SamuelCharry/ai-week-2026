"""Scrapers por fuente. Cada módulo expone fetch(entry, http) -> list[Document]."""
from . import corte_constitucional, suin, senado, dian, sic  # noqa: F401

REGISTRY = {
    "corte_constitucional": corte_constitucional,
    "suin": suin,
    "senado": senado,
    "dian": dian,
    "sic": sic,
}
