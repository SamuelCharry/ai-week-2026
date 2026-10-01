from collections import Counter
from dataclasses import replace
import json
import sqlite3
from legalrag.encoding.embed import Encoder
from legalrag.ingestion.coverage_audit import lexical_query
from legalrag.citations.extract import extract_references
from legalrag.io import sha256, source_path


class Retriever:
    def __init__(self, config):
        import faiss
        self.config = config
        path = config.index_dir / "build.json"
        if not path.is_file():
            raise FileNotFoundError("Ejecuta index antes de run.")
        self.build = json.loads(path.read_text(encoding="utf-8"))
        if self.build["config"]["embedding_model"] != config.embedding_model:
            raise ValueError("El encoder no coincide con el índice.")
        revision = self.build.get("encoder_revision")
        self.encoder = Encoder(replace(config, embedding_revision=revision or config.embedding_revision))
        self.databases = {}
        for tier in self.build["particiones"]:
            dbpath = config.index_dir / f"{tier}.sqlite"
            if sha256(dbpath) != self.build["particiones"][tier]["sha256_metadata"]:
                raise ValueError(f"Metadatos alterados: {tier}")
            self.databases[tier] = sqlite3.connect(dbpath.as_uri() + "?mode=ro", uri=True)
        self.index = None
        self.loaded_tier = None
        self.verified = set()
        self.verified_sources = set()
        faiss.omp_set_num_threads(1)

    def _load(self, tier):
        import faiss
        import psutil
        if self.loaded_tier == tier:
            return self.index
        self.index = None
        self.loaded_tier = None
        path = self.config.index_dir / f"{tier}.faiss"
        if path.stat().st_size > psutil.virtual_memory().available * .8:
            raise MemoryError(f"{tier}: FAISS necesita aproximadamente {path.stat().st_size / 2**30:.1f} GiB de RAM.")
        if tier not in self.verified:
            if sha256(path) != self.build["particiones"][tier]["sha256_index"]:
                raise ValueError(f"Índice alterado: {tier}")
            self.verified.add(tier)
        self.index = faiss.read_index(str(path))
        if self.index.d != self.encoder.model.get_sentence_embedding_dimension():
            raise ValueError("Dimensión incompatible con el encoder.")
        self.loaded_tier = tier
        return self.index

    def search(self, question, tier="nucleo", expanded=None):
        index = self._load(tier)
        if index.ntotal == 0:
            return []
        k = min(self.config.top_k_retrieval * 3, index.ntotal)
        db = self.databases[tier]
        ranking_lists = []
        dense_scores = {}
        for query in [question] + ([expanded] if expanded else []):
            scores, ids = index.search(self.encoder.encode([query]), k)
            ranking_lists.append([int(i) for i in ids[0] if i >= 0])
            for i, score in zip(ids[0], scores[0]):
                if i >= 0:
                    dense_scores[int(i)] = max(dense_scores.get(int(i), -1), float(score))
        exact = []
        if self.config.hybrid:
            query = lexical_query(question)
            for ref in extract_references(question):
                if ref.articulo:
                    hits = db.execute("SELECT id FROM chunks WHERE norma=? AND articulo=? ORDER BY id LIMIT ?",
                                      (ref.norma, ref.articulo, k)).fetchall()
                else:
                    hits = db.execute("SELECT lexical.rowid FROM lexical JOIN chunks ON chunks.id=lexical.rowid "
                                      "WHERE lexical MATCH ? AND chunks.norma=? "
                                      "ORDER BY bm25(lexical),lexical.rowid LIMIT ?",
                                      (query, ref.norma, k)).fetchall() if query else []
                    if not hits:
                        hits = db.execute("SELECT id FROM chunks WHERE norma=? ORDER BY id LIMIT ?",
                                          (ref.norma, k)).fetchall()
                exact.extend(h[0] for h in hits)
            if query:
                ranking_lists.append([r[0] for r in db.execute(
                    "SELECT rowid FROM lexical WHERE lexical MATCH ? ORDER BY bm25(lexical),rowid LIMIT ?",
                    (query, k))])
            if exact:
                ranking_lists.append(list(dict.fromkeys(exact)))
        fused = Counter()
        for ranking in ranking_lists:
            for position, vector_id in enumerate(ranking, 1):
                fused[vector_id] += 1 / (self.config.rrf_k + position)
        result = []
        parents_count = Counter()
        for vector_id, score in sorted(fused.items(), key=lambda item: (-item[1], item[0])):
            row = db.execute("SELECT metadata,texto FROM chunks WHERE id=?", (vector_id,)).fetchone()
            passage = json.loads(row[0])
            passage["texto"] = row[1]
            if parents_count[passage["parent_id"]] >= 2:
                continue
            parents_count[passage["parent_id"]] += 1
            passage.update(score=float(score), dense_score=dense_scores.get(vector_id),
                           exact_match=vector_id in exact)
            result.append(passage)
            if len(result) == self.config.top_k_retrieval:
                break
        return result

    def verify_sources(self, passages):
        for passage in passages:
            identity = passage["doc_id"]
            if identity in self.verified_sources:
                continue
            expected = self.build["particiones"][passage["nivel"]]["hashes_texto"][identity]
            if sha256(source_path(self.config, passage)) != expected:
                raise ValueError(f"El texto canónico cambió desde la indexación: {identity}")
            self.verified_sources.add(identity)

    def close(self):
        self.index = None
        for db in self.databases.values():
            db.close()
        self.encoder.close()
