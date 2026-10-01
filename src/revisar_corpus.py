"""Salud de la segmentación del corpus indexado: artículos no detectados y sentencias pegadas en normas.

Lee el inventario, los textos canónicos y chunks.sqlite de configs/sistema.json (sin GPU, 1-2 minutos):

    artículos perdidos   leyes y decretos con muchos encabezados «Artículo N» en el texto pero pocos artículos
                         distintos en los fragmentos (la secuencia se rompió; ver ingesta._siguiente)
    sentencias pegadas   normas largas que traen dentro el texto de una sentencia (p. ej. la Ley 1581 de 2012
                         con la C-748 de 2011 que publica el Senado): esos párrafos quedan como si fueran la ley

    python3 src/revisar_corpus.py
"""
import json
import re
import sqlite3
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ENCABEZADO = re.compile(r"(?im)^\W{0,3}art[íi]culo\s+\d+")
PONENTE = re.compile(r"Magistrad[oa] Ponente|M\. ?P\.")


def main():
    rec = json.loads((RAIZ / "configs/sistema.json").read_text(encoding="utf-8"))["recuperacion"]
    inventario = json.loads((RAIZ / rec["manifiesto"]).read_text(encoding="utf-8"))
    conexion = sqlite3.connect(f"file:{RAIZ / rec['fragmentos']}?mode=ro", uri=True)
    articulos = dict(conexion.execute(
        "SELECT doc_id, COUNT(DISTINCT articulo) FROM chunks WHERE articulo IS NOT NULL GROUP BY doc_id"))
    textos = RAIZ / rec["textos"]
    perdidos, pegadas, normas = [], [], 0
    for doc in inventario:
        if doc.get("tipo") not in ("ley", "decreto", "acto_legislativo"):
            continue
        ruta = textos / f"{doc['doc_id']}.txt"
        if not ruta.is_file():
            continue
        normas += 1
        texto = ruta.read_text(encoding="utf-8", errors="ignore")
        encabezados, detectados = len(ENCABEZADO.findall(texto)), articulos.get(doc["doc_id"], 0)
        if encabezados >= 10 and detectados < 0.5 * encabezados:
            perdidos.append((encabezados - detectados, doc["doc_id"], detectados, encabezados, doc.get("titulo") or ""))
        ponentes = len(PONENTE.findall(texto))
        if ponentes >= 3 and len(texto) > 150_000:
            pegadas.append((len(texto), doc["doc_id"], ponentes, doc.get("titulo") or ""))

    print(f"Normas revisadas: {normas}")
    print(f"\nArtículos perdidos: {len(perdidos)} documentos (detecta menos de la mitad de sus encabezados)")
    for _, doc_id, detectados, encabezados, titulo in sorted(perdidos, reverse=True)[:30]:
        print(f"   {doc_id:28} detecta {detectados:4} de ~{encabezados:4}   {titulo[:60]}")
    print(f"\nCon sentencias pegadas: {len(pegadas)} documentos")
    for caracteres, doc_id, ponentes, titulo in sorted(pegadas, reverse=True)[:20]:
        print(f"   {doc_id:28} {caracteres:>9,} caracteres, {ponentes:3} menciones de magistrado ponente   {titulo[:50]}")
    salida = RAIZ / "data/comparacion/revision_corpus.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps({"articulos_perdidos": [p[1:] for p in sorted(perdidos, reverse=True)],
                                  "sentencias_pegadas": [p[1:] for p in sorted(pegadas, reverse=True)]},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalle completo: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
