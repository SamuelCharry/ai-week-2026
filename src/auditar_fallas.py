"""¿Dónde se pierde cada norma de referencia que no llega a los pasajes? Corpus frente a recuperación.

Para cada norma del fundamento que la entrega no trae en sus 10 pasajes, sigue el recorrido de la
recuperación con los mismos componentes del sistema y dice en qué etapa se pierde:

    corpus        la norma no está en el inventario, o su texto quedó corto, sin artículos o sin fragmentos
                  -> hay que volver a descargarla o a procesarla (crawler / data/raw)
    candidatos    está bien en el corpus pero ni BM25 ni el denso la ponen entre sus 100 primeros
                  -> la pregunta no se parece al texto de la norma (consulta, expansión, grafo)
    reranker      entra a los candidatos pero el reranker la deja fuera de los pasajes
    pasajes       sí llega (la falla es de generación: estaba y no se citó)

El fundamento de la muestra solo se lee aquí, para auditar; nunca entra a la consulta ni al índice.
No carga el decoder (cabe en la GPU junto a nada más).

    python3 src/auditar_fallas.py --entrega data/comparacion/qwen3-8b-letra-directa/submissions.jsonl
    python3 src/auditar_fallas.py --entrega ... --ids 247 679 748
"""
import argparse
import importlib.util
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
PROFUNDIDAD = 1000  # hasta dónde se busca la norma en cada ranking


def evaluador():
    scripts = RAIZ / "data/oficial/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("evaluador_oficial", scripts / "evaluate.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def puesto(ranking, docs, doc_de):
    """Primer puesto (1..n) de un fragmento de `docs` en el ranking, o None."""
    for n, (fragmento, _) in enumerate(ranking, 1):
        if doc_de.get(fragmento) in docs:
            return n
    return None


def salud_documento(conexion, meta):
    fila = conexion.execute("SELECT COUNT(*), COUNT(DISTINCT articulo) FROM chunks WHERE doc_id = ?",
                            (meta["doc_id"],)).fetchone()
    return {"doc_id": meta["doc_id"], "titulo": (meta.get("titulo") or "")[:70], "fuente": meta.get("fuente"),
            "caracteres": meta.get("caracteres"), "estado": meta.get("estado_extraccion"),
            "avisos": meta.get("avisos") or [], "restricciones": meta.get("restricciones_especificas") or [],
            "fragmentos": fila[0], "articulos": fila[1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrega", type=Path, required=True, help="submissions.jsonl de una corrida sobre la muestra")
    ap.add_argument("--ids", nargs="+", type=int, help="solo estas preguntas")
    args = ap.parse_args()

    from legalrag.config import leer_config
    from legalrag.evaluation.entrega import preparar_entrada
    from legalrag.retrieval.hibrido import RecuperadorHibrido, consulta

    ev = evaluador()
    citas = ev.citations
    muestra = {r["id"]: r for r in ev.read_jsonl(RAIZ / "data/oficial/data/sample_50.jsonl")}
    entrega = {r["id"]: r for r in ev.read_jsonl(args.entrega)}

    # 1. Normas de referencia que faltan en los pasajes entregados.
    faltantes = {}
    for qid, ref in muestra.items():
        if args.ids and qid not in args.ids or qid not in entrega:
            continue
        referencia = citas.bodies(citas.extract(ref.get("legal_basis") or ""))
        en_pasajes = citas.bodies(set().union(*(citas.extract(p.get("texto") or "")
                                                for p in entrega[qid].get("pasajes_recuperados", [])[:10])))
        if referencia - en_pasajes:
            faltantes[qid] = sorted(referencia - en_pasajes, key=str)
    if not faltantes:
        print("Todas las normas de referencia llegan a los pasajes.")
        return

    config = leer_config()
    rec = RecuperadorHibrido(RAIZ, config["recuperacion"])
    print("Cargando índice, encoder y reranker...", flush=True)
    rec.abrir()
    resultados = []
    try:
        with closing(sqlite3.connect(f"file:{RAIZ / config['recuperacion']['fragmentos']}?mode=ro", uri=True)) as con:
            for qid, cuerpos in faltantes.items():
                ref = muestra[qid]
                entrada = preparar_entrada(ref)
                texto = consulta(entrada)
                bm25 = rec.fragmentos.bm25(texto, PROFUNDIDAD)
                puntajes, posiciones = rec.indice.search(rec.encoder.codificar(texto), PROFUNDIDAD)
                denso = [(int(i) + 1, float(s)) for i, s in zip(posiciones[0], puntajes[0]) if i >= 0]
                final = rec.ranking(entrada)
                pasajes = rec.buscar(entrada)
                doc_de = {f["id"]: f["doc_id"] for f in rec.fragmentos.filas(
                    list({f for f, _ in bm25 + denso + final}))}
                for cuerpo in cuerpos:
                    # Documentos del corpus que el extractor oficial lee como esta norma (mismo índice que el sistema).
                    docs = set(rec.evidencia._por_cuerpo.get(cuerpo, []))
                    fila = {"id": qid, "formato": ref["formato"], "norma": " ".join(str(x) for x in cuerpo if x),
                            "legal_basis": (ref.get("legal_basis") or "").strip()[:80], "docs": []}
                    if not docs:
                        fila["etapa"] = "corpus: la norma no está en el inventario (o no se reconoce por su nombre)"
                        resultados.append(fila)
                        continue
                    fila["docs"] = [salud_documento(con, rec.evidencia.documentos[d]) for d in sorted(docs)]
                    sanos = [d for d in fila["docs"] if d["fragmentos"] and (d["caracteres"] or 0) >= 1500]
                    fila.update(bm25=puesto(bm25, docs, doc_de), denso=puesto(denso, docs, doc_de),
                                rerank=puesto(final, docs, doc_de),
                                en_pasajes=any(p["doc_id"] in docs for p in pasajes))
                    # Mejor fragmento de la norma para esta pregunta y su puntaje del reranker frente al último pasaje.
                    propios = rec.fragmentos.bm25(texto, 3, doc_ids=sorted(docs))
                    if propios and rec.reordenador is not None:
                        filas = rec.fragmentos.filas([f for f, _ in propios])
                        mejor = max(rec.reordenador.puntuar(texto, [f["texto_busqueda"] for f in filas]))
                        fila["reranker_mejor_propio"] = round(mejor, 2)
                        fila["reranker_ultimo_pasaje"] = round(min(p["score"] for p in pasajes), 2) if pasajes else None
                        fila["mejor_fragmento"] = filas[0]["texto"][:160].replace("\n", " ")
                    if not sanos:
                        fila["etapa"] = "corpus: el documento está pero sin fragmentos o con texto corto (reprocesar)"
                    elif fila["en_pasajes"]:
                        fila["etapa"] = "pasajes: sí llega con el sistema actual"
                    elif fila["rerank"] is not None:
                        fila["etapa"] = f"reranker: entra a candidatos y queda en el puesto {fila['rerank']}"
                    elif min(x for x in (fila["bm25"], fila["denso"], PROFUNDIDAD + 1) if x) > 100:
                        fila["etapa"] = "candidatos: ni BM25 ni el denso la traen entre los 100 primeros"
                    else:
                        fila["etapa"] = "candidatos: aparece en BM25/denso pero no sobrevive a la fusión"
                    resultados.append(fila)
    finally:
        rec.cerrar()

    for f in resultados:
        print(f"\n{f['id']} ({f['formato']}) · norma {f['norma']} · fundamento «{f['legal_basis']}»")
        print(f"  ETAPA: {f['etapa']}")
        if "bm25" in f:
            print(f"  puesto en BM25: {f['bm25']} · denso: {f['denso']} · tras reranker: {f['rerank']} (de {PROFUNDIDAD})")
        if "reranker_mejor_propio" in f:
            print(f"  reranker: su mejor fragmento {f['reranker_mejor_propio']} · último pasaje entregado "
                  f"{f['reranker_ultimo_pasaje']}\n  mejor fragmento: «{f['mejor_fragmento']}»")
        for d in f["docs"]:
            print(f"  doc {d['doc_id']}: {d['caracteres']} caracteres, {d['fragmentos']} fragmentos, {d['articulos']} "
                  f"artículos distintos, estado {d['estado']}, fuente {d['fuente']}"
                  + (f", avisos {d['avisos']}" if d["avisos"] else ""))
    salida = args.entrega.with_name(args.entrega.stem + "_auditoria_fallas.json")
    salida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalle: {salida}")


if __name__ == "__main__":
    main()
