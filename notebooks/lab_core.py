"""Funciones compartidas por los dos notebooks de comparación RAG.

No usa las etiquetas de relevancia para construir chunks, embeddings ni índices.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_sample():
    return json.loads((HERE / "sample_ley1581.json").read_text(encoding="utf-8"))


def fold(text):
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def tokens(text):
    return re.findall(r"\b\w+\b", fold(text))


def flatten_html(html):
    """Ejemplo de aplanado: preserva encabezados, filas y celdas como líneas."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    for row in soup.select("tr"):
        row.replace_with(soup.new_string(" | ".join(c.get_text(" ", strip=True) for c in row.select("th,td")) + "\n"))
    for tag in soup.select("h1,h2,h3,p,li"):
        tag.insert_after(soup.new_string("\n"))
    lines = [re.sub(r"\s+", " ", x).strip() for x in soup.get_text().splitlines()]
    return "\n".join(x for x in lines if x)


def extract_pdf(path):
    """PDF con capa de texto. Páginas sin texto requieren OCR y revisión humana."""
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        return [{"page": i + 1, "text": page.extract_text() or ""} for i, page in enumerate(pdf.pages)]


def category(doc):
    # Categorías de la estructura documental; reglas declaradas antes de evaluar.
    n = doc["article"]
    if n <= 4: return "fundamentos"
    if n <= 7: return "datos_especiales"
    if n <= 16: return "derechos_tramites"
    return "deberes_autoridad"


def classify_query(query):
    """Enrutamiento conservador; None significa buscar en todo el corpus."""
    q = fold(query)
    if any(x in q for x in ("encargado del tratamiento", "responsable del tratamiento", "superintendencia", "autoridad")):
        if "significa" not in q and "informar" not in q:
            return "deberes_autoridad"
    if any(x in q for x in ("sensible", "biometric", "ninos", "ninas", "adolescentes")):
        return "datos_especiales"
    if any(x in q for x in ("consulta", "reclamo", "titular", "autorizacion", "suministrar", "queja")):
        return "derechos_tramites"
    if any(x in q for x in ("objeto", "aplica", "exceptuad", "principio", "dato personal")):
        return "fundamentos"
    return None


def _units(text, mode):
    """Spans sobre el texto intacto: no se pierden posiciones de cita."""
    if mode == "word":
        return [m.span() for m in re.finditer(r"\S+", text)]
    if mode == "paragraph":
        return [(m.start(), m.end()) for m in re.finditer(r"\S[\s\S]*?(?=\n\s*\n|\Z)", text)]
    if mode == "sentence":
        spans = []
        for m in re.finditer(r"[^.!?;]+[.!?;]?", text):
            if m.group().strip():
                s, e = m.span()
                while s < e and text[s].isspace(): s += 1
                while e > s and text[e-1].isspace(): e -= 1
                spans.append((s, e))
        return spans
    raise ValueError(mode)


def _pack(text, units, max_words, overlap_words=0):
    """Empaca unidades; divide unidades gigantes por palabras."""
    expanded = []
    for s, e in units:
        if len(tokens(text[s:e])) <= max_words:
            expanded.append((s, e))
        else:
            words = _units(text[s:e], "word")
            for j in range(0, len(words), max_words):
                part = words[j:j+max_words]
                expanded.append((s+part[0][0], s+part[-1][1]))
    out, i = [], 0
    while i < len(expanded):
        start = i
        end = expanded[i][1]
        i += 1
        while i < len(expanded) and len(tokens(text[expanded[start][0]:expanded[i][1]])) <= max_words:
            end = expanded[i][1]
            i += 1
        out.append((expanded[start][0], end))
        if overlap_words and i < len(expanded):
            back = i - 1
            while back > start and len(tokens(text[expanded[back-1][0]:end])) <= overlap_words:
                back -= 1
            i = max(start + 1, back)
    return out


STRATEGIES = {
    "fijo": ("word", 90, 0, False),
    "fijo_solape": ("word", 90, 20, False),
    "oraciones": ("sentence", 110, 0, False),
    "parrafos": ("paragraph", 130, 0, False),
    "jerarquico": ("paragraph", 130, 0, True),
}


def make_chunks(docs, strategy):
    mode, size, overlap, hierarchical = STRATEGIES[strategy]
    chunks = []
    for doc in docs:
        spans = _pack(doc["text"], _units(doc["text"], mode), size, overlap)
        for part, (s, e) in enumerate(spans, 1):
            body = doc["text"][s:e]
            heading = f"{doc['source']}, artículo {doc['article']}"
            if hierarchical:
                heading += f". Sección: {doc['section']}"
            chunks.append({"chunk_id": f"{doc['doc_id']}:{part}", "doc_id": doc["doc_id"],
                           "article": doc["article"], "category": category(doc),
                           "section": doc["section"], "text": body,
                           "indexed_text": heading + "\n" + body,
                           "start": doc["source_start"] + s, "end": doc["source_start"] + e})
    return chunks


def chunk_diagnostics(docs, chunks):
    from collections import defaultdict
    by_doc = defaultdict(list)
    for c in chunks: by_doc[c["doc_id"]].append(c)
    coverage, duplicates, lengths, boundaries = [], [], [], []
    for doc in docs:
        marks = np.zeros(len(doc["text"]), dtype=np.int16)
        for c in by_doc[doc["doc_id"]]:
            s = c["start"] - doc["source_start"]
            e = c["end"] - doc["source_start"]
            marks[s:e] += 1
            lengths.append(len(tokens(c["text"])))
            starts_clean = s == 0 or doc["text"][:s].rstrip().endswith((".", "!", "?", ";", ":"))
            ends_clean = e == len(doc["text"]) or doc["text"][:e].rstrip().endswith((".", "!", "?", ";", ":"))
            boundaries.append((starts_clean + ends_clean) / 2)
        nonspace = np.array([not c.isspace() for c in doc["text"]])
        coverage.append(float(np.mean(marks[nonspace] > 0)))
        duplicates.append(float(np.mean(marks[nonspace] > 1)))
    return {"chunks": len(chunks), "palabras_mediana": float(np.median(lengths)),
            "palabras_p95": float(np.percentile(lengths, 95)),
            "cobertura": float(np.mean(coverage)), "redundancia": float(np.mean(duplicates)),
            "max_palabras": max(lengths), "fronteras_logicas": float(np.mean(boundaries))}


class Encoder:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.vectorizer = None

    def fit_documents(self, texts):
        if self.name == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), min_df=1)
            result = self.vectorizer.fit_transform(texts).astype("float32").toarray()
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.name)
            payload = ["passage: " + x for x in texts] if "e5" in self.name else texts
            result = self.model.encode(payload, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        return np.ascontiguousarray(result, dtype="float32")

    def queries(self, texts):
        if self.name == "tfidf":
            result = self.vectorizer.transform(texts).astype("float32").toarray()
        else:
            payload = ["query: " + x for x in texts] if "e5" in self.name else texts
            result = self.model.encode(payload, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        result = np.ascontiguousarray(result, dtype="float32")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-12)


def rank_exact(document_vectors, query_vectors, k):
    scores = query_vectors @ document_vectors.T
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def metrics(rankings, chunks, queries, ks=(1, 3, 5)):
    """Relevancia a nivel artículo: cualquier chunk del artículo cuenta una vez."""
    values = {f"recall@{k}": [] for k in ks}
    values.update({"mrr@5": [], "ndcg@5": []})
    for row, q in zip(rankings, queries):
        relevant = set(q["relevant_doc_ids"])
        seen, doc_order = set(), []
        for i in row:
            d = chunks[int(i)]["doc_id"]
            if d not in seen:
                seen.add(d); doc_order.append(d)
        for k in ks:
            values[f"recall@{k}"].append(len(relevant.intersection(doc_order[:k])) / len(relevant))
        positions = [i+1 for i, d in enumerate(doc_order[:5]) if d in relevant]
        values["mrr@5"].append(1 / positions[0] if positions else 0)
        dcg = sum(1 / np.log2(p + 1) for p in positions)
        idcg = sum(1 / np.log2(p + 1) for p in range(1, min(5, len(relevant))+1))
        values["ndcg@5"].append(float(dcg / idcg) if idcg else 0)
    return {key: float(np.mean(val)) for key, val in values.items()}


def bm25_rank(chunks, queries, k):
    """BM25 local sin dependencia externa, con desempates estables."""
    tok = [tokens(c["indexed_text"]) for c in chunks]
    n = len(tok)
    lengths = np.array([len(t) for t in tok], dtype="float32")
    avg = max(float(lengths.mean()), 1.0)
    postings = {}
    for i, words in enumerate(tok):
        for term in set(words):
            postings.setdefault(term, []).append((i, words.count(term)))
    rankings = []
    for q in queries:
        score = np.zeros(n, dtype="float32")
        for term in set(tokens(q)):
            entries = postings.get(term, [])
            df = len(entries)
            if not df: continue
            idf = np.log(1 + (n - df + .5) / (df + .5))
            for i, tf in entries:
                score[i] += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * lengths[i] / avg))
        rankings.append([int(i) for i in np.argsort(-score, kind="stable")[:k] if score[i] > 0])
    return rankings


def rrf(rankings_a, rankings_b, k, constant=60):
    out = []
    for a, b in zip(rankings_a, rankings_b):
        score = {}
        for ranking in (a, b):
            for r, i in enumerate(ranking, 1): score[int(i)] = score.get(int(i), 0) + 1 / (constant + r)
        out.append(sorted(score, key=lambda i: (-score[i], i))[:k])
    return out


def dedupe_documents(ranking, chunks, k):
    seen, out = set(), []
    for i in ranking:
        d = chunks[int(i)]["doc_id"]
        if d not in seen:
            seen.add(d); out.append(int(i))
        if len(out) == k: break
    return out


def timed_search(index, queries, k, repeats=30):
    """Mediana por consulta, con warm-up; evita incluir la construcción del índice."""
    index.search(queries[:1], k)
    observations = []
    for _ in range(repeats):
        start = time.perf_counter()
        index.search(queries, k)
        observations.append((time.perf_counter() - start) * 1000 / len(queries))
    return float(np.median(observations))
