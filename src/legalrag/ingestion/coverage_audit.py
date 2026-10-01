from collections import Counter
import json
import hashlib
from pathlib import Path
import re
import sqlite3
from legalrag.citations.extract import extract_references, norm_identity, fold
from legalrag.chunking.hierarchical import parents
from legalrag.io import corpus_records, source_path, records, write_json, atomic_text, sha256, now


def lexical_query(text):
    words = list(dict.fromkeys(re.findall(r"[a-z0-9]{2,}", fold(text))))
    stop = {"de", "la", "el", "en", "que", "del", "las", "los", "una", "por", "para", "con", "se", "es"}
    return " OR ".join('"' + w + '"' for w in words if w not in stop)[:6000]


def coverage_audit(config, questions_path):
    code_signature = hashlib.sha256()
    for source in (Path(__file__), Path(__file__).parents[1] / "chunking/hierarchical.py",
                   Path(__file__).parents[1] / "citations/extract.py"):
        code_signature.update(source.read_bytes())
    config.reports.mkdir(parents=True, exist_ok=True)
    path = config.reports / "cobertura_lexical.sqlite"
    db = sqlite3.connect(path)
    db.executescript("""
        DROP TABLE IF EXISTS passages; DROP TABLE IF EXISTS norms; DROP TABLE IF EXISTS articles;
        CREATE VIRTUAL TABLE passages USING fts5(norma UNINDEXED, articulo UNINDEXED, texto);
        CREATE TABLE norms(norma TEXT PRIMARY KEY);
        CREATE TABLE articles(norma TEXT, articulo TEXT, PRIMARY KEY(norma,articulo));
    """)
    corpus_signature = hashlib.sha256()
    for i, doc in enumerate(corpus_records(config), 1):
        if doc.get("apta_para_busqueda") is not True:
            continue
        identity = norm_identity(doc)
        db.execute("INSERT OR IGNORE INTO norms VALUES (?)", (identity,))
        text = source_path(config, doc).read_text(encoding="utf-8")
        corpus_signature.update((doc["doc_id"] + ":" + hashlib.sha256(text.encode("utf-8")).hexdigest()).encode())
        for parent in parents(doc, text):
            article = None if parent["atribucion_ambigua"] else parent["numero_articulo"]
            if article:
                db.execute("INSERT OR IGNORE INTO articles VALUES (?,?)",
                           (identity, article))
            # Un párrafo en jurisprudencia, un artículo en normas.
            db.execute("INSERT INTO passages(norma,articulo,texto) VALUES (?,?,?)",
                       (identity, article, doc.get("titulo", "") + "\n" +
                        text[parent["inicio"]:parent["fin"]]))
        if i % 100 == 0:
            db.commit()
            print(f"[coverage] {i} documentos — índice BM25 de auditoría", flush=True)
    db.commit()
    counts = Counter({key: 0 for key in ("norma_ausente", "articulo_ausente", "no_recuperada", "recuperada")})
    details = []
    no_refs = []
    for question_number, question in enumerate(records(questions_path), 1):
        refs = list(dict.fromkeys(extract_references(question["pregunta"]) +
                                 extract_references(question.get("legal_basis") or "")))
        if not refs:
            no_refs.append(question["id"])
        query = lexical_query(question["pregunta"])
        hits = db.execute("SELECT norma,articulo FROM passages WHERE passages MATCH ? "
                          "ORDER BY bm25(passages),rowid LIMIT 20", (query,)).fetchall() if query else []
        for ref in refs:
            norm_exists = db.execute("SELECT 1 FROM norms WHERE norma=?", (ref.norma,)).fetchone()
            article_exists = (db.execute("SELECT 1 FROM articles WHERE norma=? AND articulo=? LIMIT 1",
                                        (ref.norma, ref.articulo)).fetchone() if ref.articulo else norm_exists)
            if not norm_exists:
                category = "norma_ausente"
            elif not article_exists:
                category = "articulo_ausente"
            elif not any(n == ref.norma and (ref.articulo is None or a == ref.articulo) for n, a in hits):
                category = "no_recuperada"
            else:
                category = "recuperada"
            counts[category] += 1
            details.append({"id": question["id"], **ref.dict(), "categoria": category})
        if question_number % 5 == 0:
            print(f"[coverage] {question_number} preguntas — {sum(counts.values())} referencias — {dict(counts)}", flush=True)
    db.close()
    total = sum(counts.values())
    report = {"fecha": now(), "categorias": dict(counts), "citas_evaluadas": total,
              "corpus_sha256": corpus_signature.hexdigest(), "codigo_sha256": code_signature.hexdigest(),
              "preguntas_sha256": sha256(questions_path), "retriever": "SQLite FTS5 BM25", "top_k": 20,
              "cobertura_corpus": (counts["recuperada"] + counts["no_recuperada"]) / total if total else None,
              "recall_citas_top20": counts["recuperada"] / total if total else None,
              "sin_referencia_evaluable": no_refs, "detalle": details,
              "limitacion": "Auditoría de citas explícitas. No mide corrección jurídica ni soporte semántico."}
    write_json(config.reports / "cobertura.json", report)
    with atomic_text(config.reports / "cobertura.md") as stream:
        stream.write("# Cobertura\n\n| Categoría | Citas |\n|---|---:|\n")
        for key, value in counts.items():
            stream.write(f"| {key} | {value} |\n")
        stream.write("\n| Pregunta | Norma | Artículo | Categoría |\n|---|---|---|---|\n")
        for d in details:
            stream.write(f"| {d['id']} | {d['norma']} | {d['articulo'] or '—'} | {d['categoria']} |\n")
        stream.write(f"\nSin referencia evaluable: {no_refs}. No entran en el denominador.\n")
    print(f"[coverage] {dict(counts)} — cobertura: {report['cobertura_corpus']}", flush=True)
    return report
