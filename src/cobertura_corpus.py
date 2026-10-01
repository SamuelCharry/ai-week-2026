"""Cobertura mínima del corpus: las normas que usa el banco de preguntas según data/oficial/data/seed_targets.json.

Es un piso, no un techo: el material oficial da estas 186 normas como ejemplo y el examen puede pedir otras.
Para lo que no está en la lista, ver el cierre del grafo normativo (python -m legalrag.ingestion.auditoria).

El material oficial lista 186 normas con `items_del_banco`: cuántos ítems del banco las usan como fundamento
(la Constitución, 90; el CGP, 65...). Es la mejor señal disponible de qué pide el examen de 992 preguntas,
que no se publica antes. Para cada norma dice si está en el corpus indexado y si quedó bien partida:

    FALTA        no hay documento del corpus que el extractor oficial lea como esa norma
    SIN ARTÍCULOS está, pero sus fragmentos no tienen artículos detectados (no se puede fijar "artículo N")
    OK           está, con fragmentos y artículos

Ordena por ítems del banco: arriba lo que más puntos arriesga. Sin GPU, segundos.

    python3 src/cobertura_corpus.py
"""
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))


def main():
    from legalrag.citations.normas import EvidenciaCorpus

    spec = importlib.util.spec_from_file_location("citaciones", RAIZ / "data/oficial/scripts/citations.py")
    citas = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(citas)
    rec = json.loads((RAIZ / "configs/sistema.json").read_text(encoding="utf-8"))["recuperacion"]
    evidencia = EvidenciaCorpus(citas, json.loads((RAIZ / rec["manifiesto"]).read_text(encoding="utf-8")))
    conexion = sqlite3.connect(f"file:{RAIZ / rec['fragmentos']}?mode=ro", uri=True)
    metas = json.loads((RAIZ / "data/oficial/data/seed_targets.json").read_text(encoding="utf-8"))["documentos"]

    filas, items_total, items_cubiertos = [], 0, 0
    for meta in metas:
        cuerpo = tuple(meta["canonico"])
        docs = evidencia._por_cuerpo.get(cuerpo, [])
        fragmentos = articulos = 0
        for doc_id in docs:
            f, a = conexion.execute("SELECT COUNT(*), COUNT(DISTINCT articulo) FROM chunks WHERE doc_id = ?",
                                    (doc_id,)).fetchone()
            fragmentos, articulos = fragmentos + f, articulos + a
        es_jurisprudencia = cuerpo[0] == "jurisprudencia"
        estado = ("FALTA" if not docs or not fragmentos else
                  "SIN ARTÍCULOS" if not articulos and not es_jurisprudencia else "OK")
        items = meta.get("items_del_banco") or 0
        items_total += items
        items_cubiertos += items if estado != "FALTA" else 0
        filas.append((estado, items, meta["norma"], cuerpo, docs, fragmentos, articulos))

    print(f"Normas del banco: {len(metas)} · ítems que las usan: {items_total}")
    print(f"Cubiertas por el corpus: {sum(f[0] != 'FALTA' for f in filas)} normas, {items_cubiertos} de {items_total} "
          f"ítems ({items_cubiertos / max(items_total, 1):.1%})\n")
    for estado in ("FALTA", "SIN ARTÍCULOS"):
        grupo = sorted((f for f in filas if f[0] == estado), key=lambda f: -f[1])
        print(f"{estado}: {len(grupo)} normas, {sum(f[1] for f in grupo)} ítems del banco")
        for _, items, norma, cuerpo, docs, fragmentos, articulos in grupo:
            print(f"   {items:3} ítems  {norma[:45]:45} {'/'.join(str(x) for x in cuerpo if x):28} {', '.join(docs)[:60]}")
        print()
    print("OK (las 25 con más ítems):")
    for _, items, norma, cuerpo, docs, fragmentos, articulos in sorted((f for f in filas if f[0] == "OK"),
                                                                        key=lambda f: -f[1])[:25]:
        print(f"   {items:3} ítems  {norma[:45]:45} {fragmentos:5} fragmentos, {articulos:4} artículos  {', '.join(docs)[:50]}")
    salida = RAIZ / "data/comparacion/cobertura_corpus.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps([{"estado": f[0], "items_del_banco": f[1], "norma": f[2], "canonico": f[3],
                                   "docs": f[4], "fragmentos": f[5], "articulos": f[6]} for f in filas],
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalle: {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
