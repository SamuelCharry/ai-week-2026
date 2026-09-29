"""Índice R03 del reporte de corpus y experimentos (docs/REPORTE_CORPUS_Y_EXPERIMENTOS.md, §4 y §6).

Segmentación: la unidad citable es el padre (artículo íntegro en normas; sección o bloque en
providencias, conceptos y compendios, según `preprocessing.ingesta.segmentar_documento`). Cada
padre se busca por ventanas de 384 tokens del tokenizador de BGE-M3 con solapamiento de 64. La
cabecera literal corta (título | sección | artículo) va dentro de ese presupuesto, así que ninguna
ventana se trunca al codificarla. La evidencia que ve el decoder es el padre, no la ventana.

Escribe `chunks.sqlite` con el mismo esquema que E06 (tabla `chunks` + FTS5 para BM25), de modo que
`retrieval.hibrido.RecuperadorHibrido` y `experimentos.corpus_definitivo.build_dense` sirven igual.
"""
import json
import sqlite3
import time
from pathlib import Path

ESQUEMA = """
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY, doc_id TEXT NOT NULL, titulo TEXT NOT NULL,
  tipo TEXT NOT NULL, articulo TEXT, seccion TEXT, unidad_id TEXT NOT NULL,
  unidad_inicio INTEGER NOT NULL, unidad_fin INTEGER NOT NULL,
  inicio INTEGER NOT NULL, fin INTEGER NOT NULL, texto TEXT NOT NULL,
  texto_busqueda TEXT NOT NULL, avisos TEXT NOT NULL);
CREATE VIRTUAL TABLE fts USING fts5(
  texto_busqueda, content='chunks', content_rowid='id',
  tokenize='unicode61 remove_diacritics 2');
"""


def cabecera(documento, unidad, tokenizer, maximo):
    """Título | sección | artículo, recortada a `maximo` tokens por el final."""
    partes = [documento["titulo"], unidad.get("seccion"),
              "Artículo " + str(unidad["articulo"]) if unidad.get("articulo") else None]
    texto = " | ".join(str(p) for p in partes if p)
    offsets = tokenizer(texto, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    return texto if len(offsets) <= maximo else texto[:offsets[maximo - 1][1]]


def ventanas(texto, inicio, fin, tokenizer, encabezado, tamano, solapamiento):
    """Cortes [a, b) del texto del padre cuyo `encabezado + "\\n" + texto[a:b]` cabe en `tamano` tokens."""
    especiales = len(tokenizer("")["input_ids"])
    presupuesto = tamano - especiales - len(tokenizer(encabezado + "\n", add_special_tokens=False)["input_ids"])
    if presupuesto <= solapamiento:
        raise ValueError("La cabecera no deja espacio para la ventana")
    offsets = [(a, b) for a, b in tokenizer(texto[inicio:fin], add_special_tokens=False,
                                            return_offsets_mapping=True)["offset_mapping"] if b > a]
    posicion = 0
    while posicion < len(offsets):
        final = min(posicion + presupuesto, len(offsets))
        while True:
            a, b = inicio + offsets[posicion][0], inicio + offsets[final - 1][1]
            if len(tokenizer(encabezado + "\n" + texto[a:b])["input_ids"]) <= tamano or final - posicion <= 1:
                break
            final -= 1  # la retokenización del conjunto puede sumar un token en el borde
        yield a, b
        if final == len(offsets):
            break
        posicion = max(posicion + 1, final - solapamiento)


def construir(raiz, config, limite_docs=None):
    """Segmenta el inventario fijado y escribe chunks.sqlite + chunks_complete.json. Idempotente."""
    from transformers import AutoTokenizer

    from legalrag.preprocessing.ingesta import segmentar_documento

    raiz = Path(raiz)
    seg = config["segmentacion"]
    base = raiz / config["fragmentos"]
    marcador = base.with_name("chunks_complete.json")
    release = (raiz / config["manifiesto"]).parent
    snapshot = json.loads((release / "snapshot.json").read_text(encoding="utf-8"))["snapshot_id"]
    if marcador.is_file():
        meta = json.loads(marcador.read_text(encoding="utf-8"))
        if meta["snapshot_id"] != snapshot or meta["ventana_tokens"] != seg["ventana_tokens"]:
            raise ValueError(f"{base.parent} es de otro inventario o segmentación; usar otra carpeta")
        return meta
    base.parent.mkdir(parents=True, exist_ok=True)
    if base.exists():
        base.unlink()
    tokenizer = AutoTokenizer.from_pretrained(config["encoder"]["repo_id"], revision=config["encoder"]["revision"])
    manifiesto = json.loads((raiz / config["manifiesto"]).read_text(encoding="utf-8"))[:limite_docs]
    conexion = sqlite3.connect(base)
    conexion.executescript(ESQUEMA)
    fragmentos = unidades = 0
    inicio_reloj = time.perf_counter()
    for n, documento in enumerate(manifiesto, 1):
        texto = (raiz / documento["texto_archivo"]).read_text(encoding="utf-8")
        # max_chars enorme: un fragmento por unidad, es decir, el padre completo sin cortar.
        for unidad in segmentar_documento(documento, texto, max_chars=10**12, solapamiento=0):
            if not unidad["apta_para_busqueda"] or not unidad["texto"].strip():
                continue
            unidades += 1
            encabezado = cabecera(documento, unidad, tokenizer, seg["max_cabecera_tokens"])
            for a, b in ventanas(texto, unidad["inicio"], unidad["fin"], tokenizer, encabezado,
                                 seg["ventana_tokens"], seg["solapamiento_tokens"]):
                fila = (documento["doc_id"], documento["titulo"], documento["tipo"], unidad.get("articulo"),
                        unidad.get("seccion"), unidad["unidad_id"], unidad["inicio"], unidad["fin"], a, b,
                        texto[a:b], encabezado + "\n" + texto[a:b], json.dumps(unidad["avisos"], ensure_ascii=False))
                cursor = conexion.execute(
                    "INSERT INTO chunks(doc_id,titulo,tipo,articulo,seccion,unidad_id,unidad_inicio,unidad_fin,"
                    "inicio,fin,texto,texto_busqueda,avisos) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", fila)
                conexion.execute("INSERT INTO fts(rowid,texto_busqueda) VALUES(?,?)", (cursor.lastrowid, fila[-2]))
                fragmentos += 1
        if n % 100 == 0:
            conexion.commit()
            print(f"Segmentados {n}/{len(manifiesto)} documentos · {fragmentos:,} ventanas", end="\r", flush=True)
    conexion.execute("CREATE INDEX idx_doc ON chunks(doc_id)")
    conexion.commit()
    conexion.close()
    meta = {"snapshot_id": snapshot, "documentos": len(manifiesto), "unidades": unidades, "chunks": fragmentos,
            "ventana_tokens": seg["ventana_tokens"], "solapamiento_tokens": seg["solapamiento_tokens"],
            "max_cabecera_tokens": seg["max_cabecera_tokens"], "tokenizador": config["encoder"],
            "segundos": round(time.perf_counter() - inicio_reloj, 1), "piloto": limite_docs is not None}
    marcador.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
