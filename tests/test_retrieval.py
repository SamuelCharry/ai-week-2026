"""Recuperación léxica (BM25) sobre la Ley 1581 de 2012. La parte densa (bge-m3)
necesita descargar el encoder y se ejecuta en `python cli.py demo`."""
from legalrag.retrieve import Retriever

ALIASES = {"ley 1581 de 2012": "ley 1581 de 2012"}


def top_articles(pasajes, query, k=3):
    return [p["articulo"] for p in Retriever(pasajes, aliases=ALIASES).search(query, k=k)]


def test_consulta_en_lenguaje_natural(pasajes):
    assert top_articles(pasajes, "¿En qué casos no es necesaria la autorización del Titular?")[0] == "10"
    assert "15" in top_articles(pasajes, "término máximo para atender el reclamo")
    assert "5" in top_articles(pasajes, "¿qué son los datos sensibles?")


def test_referencia_explicita_va_primero(pasajes):
    assert top_articles(pasajes, "¿Qué dice el artículo 26 de la Ley 1581 de 2012?", k=1) == ["26"]


def test_determinista(pasajes):
    q = "deberes de los encargados del tratamiento"
    a = Retriever(pasajes, aliases=ALIASES).search(q)
    b = Retriever(pasajes, aliases=ALIASES).search(q)
    assert [(p["chunk_id"], p["score"]) for p in a] == [(p["chunk_id"], p["score"]) for p in b]
