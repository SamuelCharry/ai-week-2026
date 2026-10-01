"""Inventario documento por documento del corpus indexado y cifras de CORPUS.md, desde los datos reales.

El enunciado (Paso 5) pide en CORPUS.md, por cada documento: título, fuente, URL, fecha de consulta, número
de artículos o páginas y áreas del banco. Este script lo genera desde el inventario fijado y chunks.sqlite
(sin GPU, segundos) y reescribe el bloque de cifras de CORPUS.md entre sus marcadores, para que la
bitácora nunca quede con números a mano desactualizados.

    python3 src/inventario_corpus.py

Escribe <release>/inventario.csv (va dentro del ZIP del corpus) y actualiza CORPUS.md.
"""
import csv
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
INICIO, FIN = "<!-- cifras:inicio -->", "<!-- cifras:fin -->"


def miles(n):
    return f"{n:,}".replace(",", ".")


def main():
    rec = json.loads((RAIZ / "configs/sistema.json").read_text(encoding="utf-8"))["recuperacion"]
    manifiesto = RAIZ / rec["manifiesto"]
    documentos = json.loads(manifiesto.read_text(encoding="utf-8"))
    conexion = sqlite3.connect(f"file:{RAIZ / rec['fragmentos']}?mode=ro", uri=True)
    por_doc = {d: (f, a) for d, f, a in conexion.execute(
        "SELECT doc_id, COUNT(*), COUNT(DISTINCT articulo) FROM chunks GROUP BY doc_id")}
    total_fragmentos = sum(f for f, _ in por_doc.values())

    salida = manifiesto.with_name("inventario.csv")
    with salida.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["doc_id", "titulo", "tipo", "numero", "anio", "fuente", "url", "fecha_consulta",
                           "articulos", "fragmentos", "caracteres", "areas", "origen"])
        for d in documentos:
            fragmentos, articulos = por_doc.get(d["doc_id"], (0, 0))
            escritor.writerow([d["doc_id"], d.get("titulo"), d.get("tipo"), d.get("numero"), d.get("anio"),
                               d.get("fuente"), d.get("url"), d.get("fecha_consulta"), articulos, fragmentos,
                               d.get("caracteres"), "; ".join(d.get("areas") or []), d.get("origen_ampliacion")])

    tipos = Counter(str(d.get("tipo") or "otro") for d in documentos)
    fuentes = Counter(re.sub(r"\s+-\s+.*", "", str(d.get("fuente") or "sin fuente")) for d in documentos)
    areas = Counter(a for d in documentos for a in (d.get("areas") or []))
    caracteres = sum(d.get("caracteres") or 0 for d in documentos)
    sin_fragmentos = sum(1 for d in documentos if not por_doc.get(d["doc_id"], (0, 0))[0])
    lineas = [INICIO, "",
              f"Cifras generadas por `python3 src/inventario_corpus.py` sobre el corpus indexado "
              f"(`{manifiesto.relative_to(RAIZ).as_posix()}`).", "",
              "| | |", "|---|---:|",
              f"| Documentos en el inventario | {miles(len(documentos))} |",
              f"| Fragmentos indexados (BM25 + BGE-M3) | {miles(total_fragmentos)} |",
              f"| Caracteres de texto canónico | {miles(caracteres)} |",
              f"| Documentos sin fragmentos (texto vacío o no apto) | {miles(sin_fragmentos)} |", "",
              "| Tipo | Documentos |", "|---|---:|",
              *[f"| {t} | {miles(n)} |" for t, n in tipos.most_common(10)], "",
              "| Fuente | Documentos |", "|---|---:|",
              *[f"| {f} | {miles(n)} |" for f, n in fuentes.most_common(10)], "",
              "| Área del banco | Documentos |", "|---|---:|",
              *[f"| {a} | {miles(n)} |" for a, n in areas.most_common(12)], "",
              f"Inventario por documento (título, fuente, URL, fecha de consulta, artículos, áreas): "
              f"`{salida.relative_to(RAIZ).as_posix()}`.", "", FIN]
    corpus_md = RAIZ / "CORPUS.md"
    texto = corpus_md.read_text(encoding="utf-8")
    if INICIO in texto and FIN in texto:
        texto = texto[:texto.index(INICIO)] + "\n".join(lineas) + texto[texto.index(FIN) + len(FIN):]
        corpus_md.write_text(texto, encoding="utf-8")
        print(f"CORPUS.md actualizado: {miles(len(documentos))} documentos, {miles(total_fragmentos)} fragmentos")
    print(f"Inventario por documento: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
