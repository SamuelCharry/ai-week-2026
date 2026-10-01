from pathlib import Path
import json
import sqlite3
import time
from legalrag.io import corpus_records, source_path, partition, sha256, now, write_json, dump_line, inventory_path
from legalrag.chunking.hierarchical import windows
from legalrag.encoding.embed import Encoder
from legalrag.preprocessing.clean import clean_text


def create_database(path):
    db = sqlite3.connect(path)
    db.executescript("""
        PRAGMA journal_mode=DELETE;
        CREATE TABLE chunks (
          id INTEGER PRIMARY KEY, doc_id TEXT, norma TEXT, articulo TEXT,
          parent_id TEXT, inicio INTEGER, fin INTEGER, parent_inicio INTEGER, parent_fin INTEGER,
          encabezado TEXT, texto TEXT, metadata TEXT
        );
        CREATE INDEX by_reference ON chunks(norma, articulo);
        CREATE INDEX by_parent ON chunks(parent_id);
        CREATE VIRTUAL TABLE lexical USING fts5(texto, content='chunks', content_rowid='id',
                                               tokenize='unicode61 remove_diacritics 2');
    """)
    return db


def build_index(config, replace=False):
    inventory = inventory_path(config)
    version_path = inventory.parent / "version.json"
    if inventory.parent.name != "corpus_final" or not version_path.is_file():
        raise RuntimeError("Primero ejecuta prepare, finalize y audit --coverage desde data/raw.")
    version = json.loads(version_path.read_text(encoding="utf-8"))
    if sha256(inventory) != version["sha256"]:
        raise ValueError("El inventario cambió después del cierre. Repite finalize y audit antes de indexar.")
    audit = version.get("auditoria", {})
    if audit.get("estado") != "correcta" or audit.get("inventario_sha256") != version["sha256"]:
        raise RuntimeError("Falta la auditoría de integridad del corpus cerrado. Ejecuta audit --coverage.")
    if sha256(config.root / "corpus_manifest.json") != audit.get("manifiesto_sha256"):
        raise ValueError("El manifiesto cambió después de la auditoría. Repite audit --coverage.")
    import faiss
    from tqdm import tqdm
    config.index_dir.mkdir(parents=True, exist_ok=True)
    if (config.index_dir / "build.json").exists() and not replace:
        raise FileExistsError("Ya existe un índice. Usa --replace para reconstruirlo.")
    if config.schema_file.exists():
        schema_hash = sha256(config.schema_file)
    else:
        raise FileNotFoundError(config.schema_file)
    counts = {"nucleo": 0, "complementario": 0}
    for doc in corpus_records(config):
        if doc.get("apta_para_busqueda") is True:
            counts[partition(doc)] += 1
    encoder = Encoder(config)
    details = {}
    # Un índice en construcción a la vez. No se almacena una matriz duplicada.
    for tier in ("nucleo", "complementario"):
        dimension = encoder.model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dimension)
        db_path = config.index_dir / f"{tier}.sqlite.tmp"
        if db_path.exists():
            db_path.unlink()
        db = create_database(db_path)
        jsonl_path = config.index_dir / f"{tier}.chunks.jsonl.tmp"
        batch = []
        started = time.perf_counter()
        parents_seen = 0
        processed = 0
        file_hashes = {}

        def flush(output):
            if not batch:
                return
            import psutil
            if len(batch) * dimension * 4 + 512 * 1024**2 > psutil.virtual_memory().available:
                raise MemoryError("RAM insuficiente para ampliar FAISS. Reduce el corpus o libera RAM.")
            vectors = encoder.chunks(batch)
            index.add(vectors)
            first = index.ntotal - len(batch)
            for rowid, chunk in enumerate(batch, first):
                db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    rowid, chunk["doc_id"], chunk["norma"], chunk["numero_articulo"], chunk["parent_id"],
                    chunk["inicio"], chunk["fin"], chunk["parent_inicio"], chunk["parent_fin"],
                    chunk["encabezado"], chunk["texto"],
                    json.dumps({k: v for k, v in chunk.items() if k != "texto"}, ensure_ascii=False)))
                db.execute("INSERT INTO lexical(rowid,texto) VALUES (?,?)",
                           (rowid, chunk["encabezado"] + "\n" + clean_text(chunk["texto"])))
                dump_line(output, {"vector_id": rowid, **{k: v for k, v in chunk.items() if k != "texto"}})
            db.commit()
            batch.clear()
            if index.ntotal % 128 == 0:
                elapsed = max(time.perf_counter() - started, .001)
                print(f"[encoding] {tier}: {index.ntotal} chunks — {index.ntotal/elapsed:.1f} chunks/s — "
                      f"documento {min(processed+1, counts[tier])}/{counts[tier]}", flush=True)

        with jsonl_path.open("w", encoding="utf-8") as output:
            iterator = (d for d in corpus_records(config) if d.get("apta_para_busqueda") is True
                        and partition(d) == tier)
            for doc in tqdm(iterator, total=counts[tier], desc=f"[indexing] {tier}", unit="doc"):
                path = source_path(config, doc)
                digest = sha256(path)
                expected_hash = doc.get("sha256_texto") or doc.get("sha256")
                if expected_hash and digest != expected_hash:
                    raise ValueError(f"Texto modificado: {doc['doc_id']}. Revisa el manifiesto.")
                file_hashes[doc["doc_id"]] = digest
                text = path.read_text(encoding="utf-8")
                previous_parent = None
                for chunk in windows(doc, text, encoder.tokenizer, config):
                    chunk.update(url=doc.get("url"), fuente=doc.get("fuente"), vigencia=doc.get("vigencia"),
                                 avisos=doc.get("avisos", []), nivel=tier,
                                 texto_archivo=str(path.relative_to(config.root)).replace("\\", "/"))
                    if chunk["parent_id"] != previous_parent:
                        parents_seen += 1
                        previous_parent = chunk["parent_id"]
                    batch.append(chunk)
                    if len(batch) >= config.batch_size:
                        flush(output)
                processed += 1
                if processed % 50 == 0 or processed == counts[tier]:
                    flush(output)
                    elapsed = time.perf_counter() - started
                    eta = elapsed / max(processed, 1) * (counts[tier] - processed)
                    print(f"[chunking] {parents_seen} unidades — {processed}/{counts[tier]} documentos", flush=True)
                    print(f"[encoding] {index.ntotal} chunks — ETA {int(eta//60)}m {int(eta%60)}s", flush=True)
            flush(output)
        db.close()
        faiss_path = config.index_dir / f"{tier}.faiss.tmp"
        faiss.write_index(index, str(faiss_path))
        total = index.ntotal
        del index
        for temp, final in ((db_path, config.index_dir / f"{tier}.sqlite"),
                            (jsonl_path, config.index_dir / f"{tier}.chunks.jsonl"),
                            (faiss_path, config.index_dir / f"{tier}.faiss")):
            temp.replace(final)
        details[tier] = {"documentos": processed, "unidades": parents_seen, "chunks": total,
                         "dimension": dimension, "hashes_texto": file_hashes,
                         "sha256_index": sha256(config.index_dir / f"{tier}.faiss"),
                         "sha256_metadata": sha256(config.index_dir / f"{tier}.sqlite")}
    revision = encoder.revision
    encoder.close()
    result = {"fecha": now(), "config": config.serializable(), "encoder_revision": revision,
              "schema_sha256": schema_hash, "inventario_sha256": sha256(inventory),
              "particiones": details}
    write_json(config.index_dir / "build.json", result)
    print("[indexing] índices y metadatos guardados", flush=True)
    return result
