"""Genera los notebooks versionados; ejecutar tras editar esta plantilla."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent


def notebook(cells, name):
    nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(x) if kind == "md" else nbf.v4.new_code_cell(x)
                                    for kind, x in cells], metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                                                    "language_info": {"name": "python", "version": "3.10"}})
    nbf.validate(nb)
    nbf.write(nb, HERE / name)


common = '''from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

HERE = Path.cwd() / "notebooks" if (Path.cwd() / "notebooks/lab_core.py").exists() else Path.cwd()
if not (HERE / "lab_core.py").exists():
    raise FileNotFoundError("Ejecuta desde la raíz de rag-derecho-colombiano o desde notebooks/")
sys.path.insert(0, str(HERE))
import lab_core as lab
data = lab.load_sample()
docs, queries = data["documents"], data["queries"]
dev = [q for q in queries if q["split"] == "dev"]
test = [q for q in queries if q["split"] == "test"]
print(f"{len(docs)} artículos, {len(dev)} consultas dev, {len(test)} test")'''

encoder = [
("md", """# 01 · Preparación, chunking y encoders

Laboratorio reproducible para el RAG de derecho colombiano. Los **20 documentos** son los artículos 1–20 del fixture local de la Ley 1581 de 2012. Hay 32 consultas, incluidas paráfrasis y una que requiere dos artículos. La unidad de evaluación es el **artículo**, aunque se buscan chunks. Las etiquetas se usan solo para medir.

**Ejecución:** instalar `numpy pandas matplotlib scikit-learn beautifulsoup4 sentence-transformers faiss-cpu nbformat`; ejecutar de arriba abajo. El primer uso de un encoder puede descargar pesos. `TF-IDF` es control léxico, no un encoder neuronal. Las cifras se generan al ejecutar, nunca están precalculadas. El conjunto es deliberadamente pequeño y de una sola ley: sirve para depurar estrategias; repetir el protocolo con consultas representativas y otras normas antes de adoptar una configuración.

**Diseño experimental:** 19 consultas `dev` eligen chunking y encoder; 13 `test` quedan cerradas hasta el notebook 02. No se usa el texto de las consultas para construir los chunks ni para entrenar embeddings."""),
("code", common),
("md", """## 1. Levantamiento y aplanado

La fuente de esta muestra es texto UTF-8 con artículos y secciones. Cada documento conserva `source_start/source_end`; cada chunk conserva posiciones absolutas para comprobar citas. Para HTML, eliminar navegación y scripts, preservar encabezados y convertir tablas en filas con separadores. Para PDF con capa de texto, extraer por página y revisar orden de lectura. Páginas vacías o escaneadas requieren OCR y control de calidad; no conviene inventar texto. Mantener texto bruto, texto procesado, hash, URL, fecha, jurisdicción, norma, artículo, sección y vigencia verificada. No usar encabezados repetidos, pies, firmas ni preguntas de evaluación como contenido indexable.

La categorización inicial aquí se basa en la estructura de la norma. En el corpus completo se deben preservar áreas, tipo de norma y jurisdicción; una clasificación aprendida solo vale si mejora la recuperación en consultas independientes."""),
("code", '''html = "<nav>Menú</nav><h2>Artículo 9</h2><p>Autorización previa del titular.</p><table><tr><th>Campo</th><th>Regla</th></tr><tr><td>Datos</td><td>Consentimiento</td></tr></table>"
print(lab.flatten_html(html))
display(pd.DataFrame([{"doc_id": d["doc_id"], "sección": d["section"], "categoría": lab.category(d), "palabras": len(lab.tokens(d["text"]))} for d in docs]))
source = (HERE.parent / docs[0]["source_file"]).read_text(encoding="utf-8")
assert all(source[d["source_start"]:d["source_end"]] == d["text"] for d in docs)
print("Proveniencia: 20/20 cortes coinciden con la fuente local")'''),
("md", """## 2. Cinco estrategias de chunking

`fijo` usa 90 palabras; `fijo_solape` añade 20 de solape; `oraciones` y `parrafos` empaquetan unidades hasta 110/130 palabras; `jerarquico` usa párrafos y añade sección y artículo al texto indexado. Todas mantienen artículo y posiciones de origen. La referencia explícita al artículo evita que una sección aislada pierda identidad. Si el corpus contiene tablas o artículos largos, se requieren reglas específicas de cada formato y un límite medido con el **tokenizador real** del encoder. Aquí la aproximación por palabras permite comparar la geometría de los chunks."""),
("code", '''strategies = list(lab.STRATEGIES)
chunk_sets = {s: lab.make_chunks(docs, s) for s in strategies}
diagnostics = pd.DataFrame([{"estrategia": s, **lab.chunk_diagnostics(docs, c)} for s, c in chunk_sets.items()])
for s, chunks in chunk_sets.items():
    assert all(source[c["start"]:c["end"]] == c["text"] for c in chunks)
display(diagnostics.round(3))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
diagnostics.plot.bar(x="estrategia", y="chunks", ax=axes[0], legend=False, title="Tamaño del índice")
diagnostics.plot.bar(x="estrategia", y=["cobertura", "redundancia", "fronteras_logicas"], ax=axes[1], title="Cobertura, repetición y límites")
for ax in axes: ax.tick_params(axis="x", rotation=35)
plt.tight_layout(); plt.show()
display(pd.DataFrame([{"estrategia": s, "muestra": chunk_sets[s][1]["text"][:180].replace("\\n", " ")}
                      for s in strategies]))'''),
("md", """## 3. Comparar encoders y recuperación inicial

Por defecto se comparan TF-IDF y `multilingual-e5-base`; activar BGE-M3 para la prueba completa (descarga y CPU más costosos). E5 usa los prefijos `query:` y `passage:`; todos los vectores se normalizan y se comparan por coseno. Se registra tiempo de construcción, tiempo de codificar consultas, dimensión y memoria vectorial. **Recall@k, MRR@5 y nDCG@5** se calculan sobre artículos deduplicados; el ranking inicial se obtiene a nivel chunk. La selección en `dev` pondera Recall@3 y MRR@5, con desempate por menor memoria."""),
("code", '''INCLUIR_BGE_M3 = False  # cambiar a True para la comparación completa
encoder_names = ["tfidf", "intfloat/multilingual-e5-base"] + (["BAAI/bge-m3"] if INCLUIR_BGE_M3 else [])
rows, cache = [], {}
for strategy, chunks in chunk_sets.items():
    texts = [c["indexed_text"] for c in chunks]
    for name in encoder_names:
        t0 = time.perf_counter()
        enc = lab.Encoder(name)
        vectors = enc.fit_documents(texts)
        build_s = time.perf_counter() - t0
        if enc.model is None:
            token_p95, truncation_rate = np.nan, np.nan
        else:
            payload = ["passage: " + text for text in texts] if "e5" in name else texts
            lengths = [len(ids) for ids in enc.model.tokenizer(payload, truncation=False)["input_ids"]]
            token_p95 = float(np.percentile(lengths, 95))
            truncation_rate = float(np.mean(np.array(lengths) > enc.model.max_seq_length))
        t0 = time.perf_counter()
        qvectors = enc.queries([q["question"] for q in queries])
        query_ms = 1000 * (time.perf_counter() - t0) / len(queries)
        dev_pos = [i for i, q in enumerate(queries) if q["split"] == "dev"]
        ranked = lab.rank_exact(vectors, qvectors[dev_pos], len(chunks))
        score = lab.metrics(ranked, chunks, dev)
        key = (strategy, name)
        cache[key] = (chunks, vectors, qvectors)
        rows.append({"estrategia": strategy, "encoder": name, **score,
                     "construcción_s": build_s, "consulta_encoder_ms": query_ms,
                     "dim": vectors.shape[1], "vectores_MiB": vectors.nbytes / 2**20,
                     "tokens_p95": token_p95, "fraccion_truncada": truncation_rate})
results = pd.DataFrame(rows)
display(results.sort_values(["recall@3", "mrr@5"], ascending=False).round(3))'''),
("code", '''fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for name, part in results.groupby("encoder"):
    axes[0].plot(part["estrategia"], part["recall@3"], marker="o", label=name)
axes[0].set(title="Recall@3 en desarrollo", ylim=(0, 1.05)); axes[0].tick_params(axis="x", rotation=35); axes[0].legend(fontsize=8)
for name, part in results.groupby("encoder"):
    axes[1].scatter(part["vectores_MiB"], part["mrr@5"], s=50, label=name)
axes[1].set(xlabel="Memoria de vectores (MiB)", ylabel="MRR@5", title="Calidad frente a memoria")
axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()
results["utilidad_dev"] = .6 * results["recall@3"] + .4 * results["mrr@5"]
winner = results.sort_values(["utilidad_dev", "vectores_MiB", "construcción_s"], ascending=[False, True, True]).iloc[0]
print("Elección exploratoria (solo dev):", winner[["estrategia", "encoder", "utilidad_dev"]].to_dict())'''),
("md", """## 4. Exportar la base de conocimiento experimental

El artefacto conserva alineación exacta entre fila vectorial y metadatos del chunk. El notebook 02 leerá **la misma configuración** y usará el mismo encoder para codificar consultas. Para el corpus completo: sustituir el fixture por manifiestos reales, conservar URL/hash/offsets, comprobar vigencia, crear un conjunto de juicios por dominio y congelar el conjunto de prueba antes de ajustar."""),
("code", '''out = HERE / "artefactos_lab"
out.mkdir(exist_ok=True)
key = (winner["estrategia"], winner["encoder"])
chosen_chunks, chosen_vectors, chosen_qvectors = cache[key]
(out / "chunks.json").write_text(json.dumps(chosen_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
(out / "config.json").write_text(json.dumps({"strategy": key[0], "encoder": key[1], "evaluated_encoders": encoder_names,
                                        "n_docs": len(docs), "n_chunks": len(chosen_chunks)}, ensure_ascii=False, indent=2), encoding="utf-8")
np.savez_compressed(out / "vectors.npz", documents=chosen_vectors, queries=chosen_qvectors)
results.to_csv(out / "comparacion_encoder_dev.csv", index=False)
print("Guardado en", out)
print("No abrir test en este notebook; se reserva para 02.")'''),
]

decoder = [
("md", """# 02 · Índices FAISS, recuperación y reranking

Lee los chunks y vectores elegidos por el notebook 01. Si no existen, genera una configuración de arranque (`jerarquico` + TF-IDF) para poder ejecutarse de forma independiente; para una decisión real, ejecutar primero 01. **Decoder** aquí significa etapa de recuperación que entrega evidencia al generador; el cross-encoder es un reranker, no un modelo generativo.

Instalar también `faiss-cpu`. Los índices aproximados en ~20 documentos solo ilustran el método; comparar latencia y recall ANN en el corpus real. Las consultas `dev` eligen la canalización; `test` se usa una única vez al final."""),
("code", common),
("code", '''import faiss
out = HERE / "artefactos_lab"
saved_config = json.loads((out / "config.json").read_text(encoding="utf-8")) if (out / "config.json").exists() else {}
if (out / "vectors.npz").exists() and any(name != "tfidf" for name in saved_config.get("evaluated_encoders", [])):
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    chunks = json.loads((out / "chunks.json").read_text(encoding="utf-8"))
    loaded = np.load(out / "vectors.npz")
    x, qv = loaded["documents"], loaded["queries"]
else:
    config = {"strategy": "jerarquico", "encoder": "tfidf", "origen": "arranque_independiente"}
    chunks = lab.make_chunks(docs, config["strategy"])
    enc = lab.Encoder(config["encoder"])
    x = enc.fit_documents([c["indexed_text"] for c in chunks])
    qv = enc.queries([q["question"] for q in queries])
x, qv = np.ascontiguousarray(x, dtype="float32"), np.ascontiguousarray(qv, dtype="float32")
assert x.shape[0] == len(chunks) and qv.shape[0] == len(queries)
assert len({c["chunk_id"] for c in chunks}) == len(chunks)
faiss.normalize_L2(x); faiss.normalize_L2(qv)
display(config); print(x.shape, qv.shape)'''),
("md", """## 1. Índices exactos y aproximados

`FlatIP` es la referencia exacta (producto interno = coseno por normalización). `FlatL2` debe producir el mismo orden salvo empates. `IVFFlat` entrena centroides y varía `nprobe`; `HNSW` varía `efSearch`. El ID retornado por FAISS es la fila de `chunks`; **FAISS no almacena la cita ni la categoría**. Se mide recall de vecinos contra FlatIP, Recall@3 de artículos, memoria serializada y mediana de latencia de búsqueda. El tiempo de codificar la consulta se evalúa aparte en 01."""),
("code", '''n, dim = x.shape
flat = faiss.IndexFlatIP(dim); flat.add(x)
l2 = faiss.IndexFlatL2(dim); l2.add(x)
indices = {"FlatIP": flat, "FlatL2": l2}
nlist = max(2, min(8, int(np.sqrt(n))))
ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
ivf.train(x); ivf.add(x)
for probes in sorted({1, min(4, nlist), nlist}):
    clone = faiss.deserialize_index(faiss.serialize_index(ivf))
    clone.nprobe = probes
    indices[f"IVF nprobe={probes}"] = clone
hnsw = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)
hnsw.hnsw.efConstruction = 40; hnsw.add(x)
for ef in (16, 48):
    clone = faiss.deserialize_index(faiss.serialize_index(hnsw))
    clone.hnsw.efSearch = ef
    indices[f"HNSW ef={ef}"] = clone
dev_pos = [i for i, q in enumerate(queries) if q["split"] == "dev"]
test_pos = [i for i, q in enumerate(queries) if q["split"] == "test"]
exact_ids = flat.search(qv[dev_pos], min(10, n))[1]
index_rows = []
for name, index in indices.items():
    _, ids = index.search(qv[dev_pos], n)
    ann_recall = np.mean([len(set(a).intersection(b[:len(a)])) / len(a) for a, b in zip(exact_ids, ids)])
    index_rows.append({"índice": name, "vecinos@10_vs_FlatIP": ann_recall,
                       "latencia_búsqueda_ms": lab.timed_search(index, qv[dev_pos], min(10, n)),
                       "índice_MiB": len(faiss.serialize_index(index)) / 2**20,
                       **lab.metrics(ids, chunks, dev)})
index_table = pd.DataFrame(index_rows)
display(index_table.round(4))
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
index_table.plot.bar(x="índice", y="vecinos@10_vs_FlatIP", ax=axes[0], legend=False, ylim=(0, 1.05))
index_table.plot.scatter(x="latencia_búsqueda_ms", y="recall@3", ax=axes[1], title="Calidad frente a tiempo")
for ax in axes: ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()'''),
("md", """## 2. Clasificación, BM25, fusión y agregación

Se prueban búsqueda densa, BM25 y RRF; luego un filtro de categoría **previo** a la búsqueda. El clasificador por reglas puede fallar, así que `None` busca globalmente y se muestra su exactitud por consulta. Las consultas con cita explícita se pueden enrutar por norma y artículo verificados; aquí se evita usar el `doc_id` verdadero como atajo. Se deduplican documentos por el primer chunk (equivalente a max pooling del puntaje si el ranking se ordena por puntaje)."""),
("code", '''K_CANDIDATOS = min(10, n)
dense = [list(map(int, row)) for row in flat.search(qv, n)[1]]
lexical = lab.bm25_rank(chunks, [q["question"] for q in queries], n)
hybrid = lab.rrf(dense, lexical, n)

def category_rank(i, fallback=True):
    label = lab.classify_query(queries[i]["question"])
    if label is None: return dense[i]
    subset = np.array([j for j, c in enumerate(chunks) if c["category"] == label], dtype=int)
    if len(subset) == 0: return dense[i] if fallback else []
    # Se restringe el universo ANTES de puntuar; en FAISS grande usar IDSelector
    # o índices separados por categoría.
    scores = x[subset] @ qv[i]
    return subset[np.argsort(-scores, kind="stable")].tolist()

filtered = [category_rank(i) for i in range(len(queries))]
query_categories = pd.DataFrame([{"consulta": q["query_id"], "predicha": lab.classify_query(q["question"]),
                                  "real": lab.category(next(d for d in docs if d["doc_id"] == q["relevant_doc_ids"][0]))}
                                 for q in queries])
display(query_categories)
print("Exactitud de clasificación en dev:", np.mean((query_categories.iloc[dev_pos].predicha == query_categories.iloc[dev_pos].real)))
pipelines = {"denso": dense, "BM25": lexical, "RRF": hybrid, "denso+categoría": filtered}
pipeline_rows = []
for name, ranks in pipelines.items():
    pipeline_rows.append({"pipeline": name, **lab.metrics([ranks[i] for i in dev_pos], chunks, dev)})
pipeline_table = pd.DataFrame(pipeline_rows)
display(pipeline_table.round(3))'''),
("md", """## 3. Cross-encoder en candidatos

El reranker recibe pares `(consulta, chunk)` y reordena los primeros 10 candidatos del recuperador. La primera ejecución descarga el modelo multilingüe [mMARCO MiniLM](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1). **No** comparar scores crudos de modelos diferentes; comparar rankings. Si un artículo relevante no llega a los candidatos, el reranker no lo puede rescatar."""),
("code", '''ACTIVAR_CROSS_ENCODER = True
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
if ACTIVAR_CROSS_ENCODER:
    from sentence_transformers import CrossEncoder
    cross = CrossEncoder(RERANK_MODEL, max_length=512)
    def rerank(query, ranking, k=K_CANDIDATOS):
        candidates = list(ranking[:k])
        pairs = [(query, chunks[j]["indexed_text"]) for j in candidates]
        scores = cross.predict(pairs, batch_size=16)
        return [candidates[j] for j in np.argsort(-np.asarray(scores), kind="stable")] + list(ranking[k:])
    t0 = time.perf_counter()
    pipelines["RRF+cross"] = [rerank(q["question"], ranking) for q, ranking in zip(queries, hybrid)]
    rerank_ms = (time.perf_counter() - t0) * 1000 / len(queries)
    print(f"Reranking: {rerank_ms:.1f} ms/consulta, además de búsqueda y encoder")
    pipeline_table = pd.DataFrame([{"pipeline": name, **lab.metrics([ranks[i] for i in dev_pos], chunks, dev)} for name, ranks in pipelines.items()])
    display(pipeline_table.round(3))
else:
    print("Cross-encoder desactivado. Actívalo para comparar calidad y latencia reales.")'''),
("md", """## 4. Elegir en desarrollo y abrir la prueba una vez

El criterio predefinido es 60 % Recall@3 y 40 % MRR@5 en `dev`. Los resultados de `test` son una estimación muy ruidosa (13 consultas). Reportar además fallos por pregunta y distinguir errores de extracción, chunking, candidato, categoría y reranking. En producción también medir P50/P95 con concurrencia, coste de memoria, actualización de índices y evaluación por norma y dominio."""),
("code", '''pipeline_table["utilidad_dev"] = .6 * pipeline_table["recall@3"] + .4 * pipeline_table["mrr@5"]
selected_name = pipeline_table.sort_values(["utilidad_dev", "pipeline"], ascending=[False, True]).iloc[0]["pipeline"]
selected = pipelines[selected_name]
test_result = lab.metrics([selected[i] for i in test_pos], chunks, test)
print("Pipeline elegido en dev:", selected_name)
display(pd.DataFrame([test_result]).round(3))
failures = []
for i in test_pos:
    top = lab.dedupe_documents(selected[i], chunks, 3)
    retrieved = [chunks[j]["doc_id"] for j in top]
    failures.append({"consulta": queries[i]["query_id"], "pregunta": queries[i]["question"],
                     "relevante": queries[i]["relevant_doc_ids"][0], "top3": retrieved,
                     "acierto@3": queries[i]["relevant_doc_ids"][0] in retrieved})
display(pd.DataFrame(failures))
fig, ax = plt.subplots(figsize=(8, 3))
pipeline_table.plot.bar(x="pipeline", y=["recall@3", "mrr@5"], ax=ax, ylim=(0, 1.05), title="Selección en desarrollo")
plt.xticks(rotation=30); plt.tight_layout(); plt.show()
out.mkdir(exist_ok=True)
pd.DataFrame(failures).to_json(out / "errores_test.json", orient="records", force_ascii=False, indent=2)
index_table.to_csv(out / "comparacion_indices_dev.csv", index=False)
pipeline_table.to_csv(out / "comparacion_retrieval_dev.csv", index=False)
print("Resultados guardados en", out)'''),
("md", """## 5. Paso al corpus completo

1. Congelar manifiestos por fuente, hash, fecha, licencia y vigencia comprobada. Extraer PDF/HTML con validación de orden de lectura, tablas y OCR; mantener coordenadas/offsets citables.
2. Construir juicios de relevancia con varias normas, áreas y preguntas sin respuesta. Reservar `test` por documento o norma para reducir fuga; usar varios juicios por pregunta si existen artículos complementarios.
3. Repetir el barrido de chunking y encoder con el tokenizador real y presupuesto de memoria. Inspeccionar visualmente chunks largos, huérfanos, duplicados y fragmentos que separan excepción de regla.
4. Recalibrar FAISS con el número real de vectores y separar tiempo de embedding, búsqueda y reranking. Comprobar ANN recall contra FlatIP. Versionar encoder, normalización, índice y mapeo fila→chunk juntos.
5. Escoger con `dev`; ejecutar `test` una sola vez y pasar los pasajes recuperados al generador con citas y abstención. No inferir calidad de respuesta solo desde Recall de recuperación."""),
]

notebook(encoder, "01_comparacion_encoders_chunking.ipynb")
notebook(decoder, "02_comparacion_recuperacion_faiss.ipynb")
print("Notebooks generados")
