"""Recuperación híbrida: BM25 + vectores densos, fusionados por rango (RRF).

La búsqueda densa se basa en src/retrieval/search.py de CODEFEST Ad Astra 2026
(autor en el historial git: Juanesillo). Se añadieron BM25, la fusión y la
búsqueda directa cuando la consulta nombra un artículo ("artículo 26 de la
Ley 1581 de 2012"), porque los embeddings no distinguen bien números.
Todo es determinista: FAISS exacto, BM25 y desempates por posición.
"""
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from legalrag.citations import extract_citations, fold
from legalrag.index import read_jsonl
from legalrag.segment import indexed_text

RRF_K = 60


def _stem(w: str) -> str:
    """Normalización mínima de plurales: datos/dato, sensibles/sensible,
    autorizaciones/autorización, legales/legal quedan iguales."""
    if len(w) > 3 and w.endswith("s"):
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


def tokenize(texto: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"\w+", fold(texto))]


class Retriever:
    def __init__(self, pasajes: list[dict], faiss_index=None, aliases: dict[str, str] | None = None):
        self.pasajes = pasajes
        self.faiss_index = faiss_index
        self.aliases = aliases or {}
        self.bm25 = BM25Okapi([tokenize(indexed_text(p)) for p in pasajes])

    @classmethod
    def from_dir(cls, index_dir: str, aliases=None, dense: bool = True):
        d = Path(index_dir)
        pasajes = read_jsonl(d / "pasajes.jsonl")
        faiss_index = None
        if dense:
            import faiss
            faiss_index = faiss.read_index(str(d / "index.faiss"))
        return cls(pasajes, faiss_index, aliases)

    def _dense_ranking(self, query: str, n: int) -> list[int]:
        from legalrag.index import encode_texts
        _, ids = self.faiss_index.search(encode_texts([query], batch_size=1), n)
        return [int(i) for i in ids[0] if i != -1]

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [i for i in order[:n] if scores[i] > 0]

    def _explicit(self, query: str) -> list[int]:
        """Pasajes del artículo que la consulta nombra explícitamente."""
        hits = []
        for c in extract_citations(query, self.aliases):
            if c["norma"] is None:
                continue  # sin norma, "artículo 10" es ambiguo en un corpus grande
            hits += [i for i, p in enumerate(self.pasajes)
                     if p["articulo"] == c["articulo"] and p.get("norma_key") == c["norma"]]
        return list(dict.fromkeys(hits))

    def search(self, query: str, k: int = 10, candidatos: int = 50) -> list[dict]:
        rankings = [self._bm25_ranking(query, candidatos)]
        if self.faiss_index is not None:
            rankings.append(self._dense_ranking(query, candidatos))
        fused: dict[int, float] = {}
        for ranking in rankings:
            for r, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        explicit = self._explicit(query)
        order = explicit + sorted((i for i in fused if i not in explicit), key=lambda i: (-fused[i], i))
        top_score = max(fused.values(), default=0.0)
        out = []
        for i in order[:k]:
            # los pasajes pedidos por referencia explícita van primero con el puntaje máximo
            score = fused.get(i, 0.0) if i not in explicit else max(top_score, fused.get(i, 0.0))
            out.append({**self.pasajes[i], "score": round(score, 6)})
        return out
