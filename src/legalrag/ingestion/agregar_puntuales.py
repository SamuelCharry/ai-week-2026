"""Agrega fuentes puntuales al corpus ya indexado, sin reconstruir el índice.

Son normas que las preguntas dan por sabidas y el corpus no traía (configs/corpus_puntuales.json):
el salario mínimo y el auxilio de transporte de cada año, la UVT de cada año y normas que una
pregunta del banco pide leer. Cada documento pasa por el mismo código que el corpus completo:

    1. descarga de la fuente oficial    ingestion.reconstruir (procesar/guardar; OCR si es escaneo) -> data/raw/<doc_id>/
    2. texto canónico                   preprocessing.preparar_corpus.prepare_one -> corpus_preparado/textos/
    3. registro en el inventario        preprocessing.inventario.fila_inventario -> corpus_manifest.json
    4. fragmentos                       experimentos.corpus_definitivo.chunk_rows -> chunks.sqlite + FTS5
    5. vectores                         encoder de configs/sistema.json con el pooling de la indexación -> index.faiss

Un documento entra solo si la fuente respondió, el texto nombra la norma y contiene su `clave`
(p. ej. "salario mínimo"). Los fragmentos nuevos van al final con ids consecutivos y el índice denso
crece en el mismo orden, así que "posición i del índice = fragmento i + 1" se mantiene. El snapshot_id
no cambia (main.py no aparta el índice); snapshot.json, chunks_complete.json y complete.json guardan
la lista de ampliaciones. Reanudable: un doc_id que ya está en chunks.sqlite no se vuelve a agregar.

Uso:
    python -m legalrag.ingestion.agregar_puntuales                    # todo (el paso 5 usa la GPU)
    python -m legalrag.ingestion.agregar_puntuales --solo-descargar   # pasos 1 y 2, sin tocar el índice
    python -m legalrag.ingestion.agregar_puntuales --solo-raw         # solo a data/raw, para reconstruir desde cero
    python -m legalrag.ingestion.agregar_puntuales --solo decreto_1572_2024 ...
"""
import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from legalrag.config import RAIZ, leer_config

FUENTES = RAIZ / "configs/corpus_puntuales.json"
RAW = RAIZ / "data/raw"
PREPARADO = RAIZ / "data/processed/corpus_preparado"


def escribir_json(ruta, valor, indent=2):
    temporal = Path(str(ruta) + ".tmp")
    temporal.write_text(json.dumps(valor, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
    temporal.replace(ruta)


def descargar(doc):
    """Entrada de data/raw/manifest.json para el documento, o (None, motivo)."""
    from legalrag.ingestion.reconstruir import completo_en_disco, guardar, procesar

    anterior = RAW / doc["doc_id"] / "entrada.json"
    if anterior.is_file():
        entrada = json.loads(anterior.read_text(encoding="utf-8"))
        if completo_en_disco(RAIZ, entrada):
            return entrada, "ya descargado"
    ficha, evaluadas, elegida = procesar(doc)
    if not elegida:
        return None, "; ".join(f"{e['url']}: {e['estado']}" for e in evaluadas) or "sin fuentes"
    entrada = guardar(RAIZ, ficha, elegida, doc.get("fuente"))
    entrada.update(nivel="complementario", **({"excepcion_alcance": doc["excepcion_alcance"]}
                                              if doc.get("excepcion_alcance") else {}))
    escribir_json(anterior, entrada)
    return entrada, "descargado"


def preparar(entrada):
    """Registro de documentos.jsonl (texto canónico escrito en corpus_preparado/textos)."""
    import hashlib

    from legalrag.preprocessing import ingesta, preparar_corpus

    huella = hashlib.sha256(Path(ingesta.__file__).read_bytes() + Path(preparar_corpus.__file__).read_bytes()).hexdigest()
    (PREPARADO / "registros").mkdir(parents=True, exist_ok=True)
    return preparar_corpus.prepare_one(entrada, str(RAW), str(PREPARADO), huella)


def revisar(doc, registro):
    """Motivo por el que el texto no sirve, o None."""
    from legalrag.preprocessing.inventario import evaluable

    if not evaluable(registro):
        return f"extracción: {registro.get('error') or registro.get('pendientes_preparacion')}"
    texto = (PREPARADO / registro["texto_archivo"]).read_text(encoding="utf-8")
    if not re.search(doc["clave"], texto, re.I):
        return f"el texto no contiene «{doc['clave']}»"
    return None


def registrar_en_raw(entradas):
    """Agrega (o reemplaza) las entradas en data/raw/manifest.json, de donde parte main.py --desde-raw."""
    ids = {e["doc_id"] for e in entradas}
    crudo = [d for d in json.loads((RAW / "manifest.json").read_text(encoding="utf-8")) if d["doc_id"] not in ids]
    escribir_json(RAW / "manifest.json", crudo + entradas)
    return len(crudo) + len(entradas)


def actualizar_listas(registros, filas, rec):
    """documentos.jsonl, data/raw/manifest.json y el inventario (corpus_manifest.json + snapshot.json)."""
    ids = {f["doc_id"] for f in filas}
    documentos = PREPARADO / "documentos.jsonl"
    if documentos.is_file():
        lineas = [l for l in documentos.read_text(encoding="utf-8").splitlines()
                  if l.strip() and json.loads(l)["doc_id"] not in ids]
        lineas += [json.dumps(r, ensure_ascii=False) for r in registros]
        temporal = documentos.with_suffix(".jsonl.tmp")
        temporal.write_text("\n".join(sorted(lineas, key=lambda l: json.loads(l)["doc_id"])) + "\n", encoding="utf-8")
        temporal.replace(documentos)
    if (RAW / "manifest.json").is_file():
        registrar_en_raw([json.loads((RAW / r["doc_id"] / "entrada.json").read_text(encoding="utf-8")) for r in registros])
    manifiesto_ruta = RAIZ / rec["manifiesto"]
    manifiesto = [d for d in json.loads(manifiesto_ruta.read_text(encoding="utf-8")) if d["doc_id"] not in ids]
    manifiesto = sorted(manifiesto + filas, key=lambda d: d["doc_id"])
    escribir_json(manifiesto_ruta, manifiesto)
    snapshot_ruta = manifiesto_ruta.parent / "snapshot.json"
    if snapshot_ruta.is_file():
        snapshot = json.loads(snapshot_ruta.read_text(encoding="utf-8"))
        snapshot["documentos_evaluables"] = len(manifiesto)
        snapshot.setdefault("ampliaciones", []).append(
            {"fecha_utc": datetime.now(timezone.utc).isoformat(), "fuente": FUENTES.relative_to(RAIZ).as_posix(),
             "doc_ids": sorted(ids), "nota": "snapshot_id conservado: el índice existente se amplía, no se reconstruye"})
        escribir_json(snapshot_ruta, snapshot)
    return len(manifiesto)


def agregar_al_indice(filas, rec, codificar):
    """Inserta los fragmentos y sus vectores. Devuelve {doc_id: fragmentos agregados}."""
    import faiss
    import numpy as np

    from legalrag.experimentos.corpus_definitivo import INSERT_CHUNK, chunk_rows

    ruta_indice = RAIZ / rec["indice_denso"]
    indice = faiss.read_index(str(ruta_indice))
    conexion = sqlite3.connect(RAIZ / rec["fragmentos"], timeout=120)
    try:
        total, maximo = conexion.execute("SELECT COUNT(*), MAX(id) FROM chunks").fetchone()
        if not total == maximo == indice.ntotal:
            raise RuntimeError(f"Índice inconsistente antes de agregar: {total} fragmentos, id máximo {maximo}, "
                               f"{indice.ntotal} vectores")
        nuevas, por_doc = [], {}
        for fila in filas:
            if conexion.execute("SELECT 1 FROM chunks WHERE doc_id = ? LIMIT 1", (fila["doc_id"],)).fetchone():
                continue
            filas_doc = chunk_rows(fila, (RAIZ / fila["texto_archivo"]).read_text(encoding="utf-8"))
            nuevas += filas_doc
            por_doc[fila["doc_id"]] = len(filas_doc)
        if not nuevas:
            return por_doc
        vectores = np.vstack([codificar(fila[-2]) for fila in nuevas]).astype("float32")
        if vectores.shape[1] != indice.d or not np.isfinite(vectores).all():
            raise RuntimeError("Vectores con otra dimensión o no finitos")
        for posicion, fila in enumerate(nuevas, start=total + 1):
            cursor = conexion.execute(INSERT_CHUNK, fila)
            if cursor.lastrowid != posicion:
                raise RuntimeError("Los ids nuevos no quedaron consecutivos")
            conexion.execute("INSERT INTO fts(rowid,texto_busqueda) VALUES(?,?)", (cursor.lastrowid, fila[-2]))
        indice.add(vectores)
        # El índice se escribe completo antes de confirmar la transacción: si algo falla, ninguno cambia.
        temporal = ruta_indice.with_suffix(".faiss.tmp")
        faiss.write_index(indice, str(temporal))
        conexion.commit()
        temporal.replace(ruta_indice)
    except BaseException:
        conexion.rollback()
        raise
    finally:
        conexion.close()
    nuevo_total = total + len(nuevas)
    ampliacion = {"fecha_utc": datetime.now(timezone.utc).isoformat(), "fragmentos": len(nuevas),
                  "doc_ids": sorted(por_doc)}
    for marcador in (Path(RAIZ / rec["fragmentos"]).parent / "chunks_complete.json", RAIZ / rec["indice_meta"]):
        if marcador.is_file():
            meta = json.loads(marcador.read_text(encoding="utf-8"))
            meta["chunks"] = nuevo_total
            meta.setdefault("ampliaciones", []).append(ampliacion)
            escribir_json(marcador, meta)
    return por_doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--solo", nargs="*", help="doc_id de configs/corpus_puntuales.json")
    ap.add_argument("--solo-descargar", action="store_true", help="descarga y extrae el texto; no toca el índice")
    ap.add_argument("--solo-raw", action="store_true",
                    help="descarga y registra en data/raw/manifest.json, sin tocar inventario ni índice "
                         "(para reconstruir todo desde cero con main.py --desde-raw)")
    args = ap.parse_args(argv)
    rec = leer_config()["recuperacion"]
    docs = json.loads(FUENTES.read_text(encoding="utf-8"))["documentos"]
    if args.solo:
        desconocidos = set(args.solo) - {d["doc_id"] for d in docs}
        if desconocidos:
            ap.error(f"doc_id que no están en {FUENTES.name}: {sorted(desconocidos)}")
        docs = [d for d in docs if d["doc_id"] in args.solo]
    if args.solo_raw and not (RAW / "manifest.json").is_file():
        ap.error("No está data/raw/manifest.json: --solo-raw es para reconstruir desde data/raw")
    manifiesto = RAIZ / rec["manifiesto"]
    en_corpus = set() if args.solo_raw or not manifiesto.is_file() else \
        {d["doc_id"] for d in json.loads(manifiesto.read_text(encoding="utf-8"))}

    from legalrag.preprocessing.inventario import fila_inventario

    registros, filas, entradas, fallas = [], [], [], {}
    for i, doc in enumerate(docs, 1):
        presente = {doc["doc_id"], "co_" + doc["doc_id"]} & en_corpus
        if presente:
            print(f"[{i}/{len(docs)}] {doc['doc_id']}: ya está en el corpus ({sorted(presente)[0]})")
            continue
        entrada, estado = descargar(doc)
        if entrada is None:
            fallas[doc["doc_id"]] = estado
            print(f"[{i}/{len(docs)}] {doc['doc_id']}: FALLA {estado}", flush=True)
            continue
        registro = preparar(entrada)
        motivo = revisar(doc, registro)
        if motivo:
            fallas[doc["doc_id"]] = motivo
            print(f"[{i}/{len(docs)}] {doc['doc_id']}: FALLA {motivo}", flush=True)
            continue
        registros.append(registro)
        entradas.append(entrada)
        filas.append(fila_inventario(registro, PREPARADO, RAIZ))
        print(f"[{i}/{len(docs)}] {doc['doc_id']}: {estado}, {registro['caracteres']:,} caracteres "
              f"({entrada['fuente']})", flush=True)

    print(f"\nListos {len(filas)} de {len(docs)}; fallas {len(fallas)}")
    for doc_id, motivo in fallas.items():
        print(f"  {doc_id}: {motivo}")
    if args.solo_raw and entradas:
        total = registrar_en_raw(entradas)
        print(f"data/raw/manifest.json: {total} documentos. Siguiente: python3 src/main.py --desde-raw --solo-preparar")
    if args.solo_descargar or args.solo_raw or not filas:
        return 1 if fallas else 0

    from legalrag.retrieval.hibrido import EncoderConsultas

    meta = json.loads((RAIZ / rec["indice_meta"]).read_text(encoding="utf-8"))
    if meta["modelo"] != rec["encoder"]["repo_id"] or meta["revision"] != rec["encoder"]["revision"]:
        raise RuntimeError("El índice denso es de otro encoder que el de configs/sistema.json")
    print(f"\nVectorizando con {meta['modelo']} y agregando al índice...", flush=True)
    encoder = EncoderConsultas(rec["encoder"], rec["dtype"], rec["max_tokens_encoder"], rec.get("dispositivo", "cuda"))
    # Los pasajes se indexaron sin prefijo; en BGE-M3 (CLS) el de consultas también es vacío.
    if encoder.prefijo:
        raise RuntimeError("Este encoder usa prefijo de consulta; los pasajes necesitan el de pasaje")
    por_doc = agregar_al_indice(filas, rec, encoder.codificar)
    documentos = actualizar_listas(registros, filas, rec)
    print(f"Agregados {sum(por_doc.values())} fragmentos de {len(por_doc)} documentos; "
          f"inventario: {documentos} documentos")
    return 1 if fallas else 0


if __name__ == "__main__":
    sys.exit(main())
