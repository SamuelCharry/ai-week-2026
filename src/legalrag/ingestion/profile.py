from collections import Counter
from pathlib import Path
import json
from legalrag.io import corpus_records, now, partition, write_json, atomic_text


def profile(config):
    import pandas as pd
    from ydata_profiling import ProfileReport
    rows = []
    warnings = Counter()
    config.reports.mkdir(parents=True, exist_ok=True)
    # El manifiesto de este paso es un inventario, no una auditoría de hashes.
    with atomic_text(config.root / "corpus_manifest.json") as output:
        output.write(json.dumps({"equipo": "P34K", "licencia": "CC-BY-4.0",
                                "fecha_generacion": now(), "estado": "inventario_preparado"},
                               ensure_ascii=False)[:-1] + ',"documentos":[')
        for i, doc in enumerate(corpus_records(config), 1):
            row = {
                "doc_id": doc["doc_id"], "tipo": doc.get("tipo"), "anio": doc.get("anio"),
                "nivel": partition(doc), "caracteres": doc.get("caracteres", 0),
                "apta_para_busqueda": doc.get("apta_para_busqueda", False),
                "n_avisos": len(doc.get("avisos", [])),
                "n_pendientes": len(doc.get("pendientes_preparacion", [])),
                "n_areas": len(doc.get("areas", [])),
                "anotaciones": bool(doc.get("edicion_con_anotaciones")),
                "dominio": doc.get("dominio_publicador"),
                "tiene_url": bool(doc.get("url")), "tiene_hash": bool(doc.get("sha256_texto")),
                "vigencia": doc.get("vigencia"),
            }
            rows.append(row)
            warnings.update(doc.get("avisos", []))
            manifest = {k: doc.get(k) for k in ("doc_id", "titulo", "fuente", "url", "fecha_consulta",
                       "areas", "tipo", "numero", "anio", "vigencia", "avisos", "apta_para_busqueda",
                       "numero_catalogo", "tipo_catalogo", "organo_emisor", "organo_emisor_catalogo",
                       "raiz_originales", "integridad_originales",
                       "caracteres", "licencia_fuente", "redistribuir_raw", "edicion_con_anotaciones",
                       "restricciones", "revision_juridica", "url_final", "archivo_raw", "sha256_raw",
                       "solicitud_api", "source_id", "pagina_catalogo", "fecha_descarga", "origen_ampliacion",
                       "archivos_raw", "partes_html", "metodo_extraccion", "ocr", "areas_metodo", "pendientes_preparacion", "revision_preparacion")}
            relative = doc.get("texto_archivo") or "corpus/" + doc["doc_id"] + ".txt"
            if relative.startswith("textos/"):
                relative = "data/processed/corpus_preparado/" + relative
            manifest.update(sha256=doc.get("sha256_texto") or doc.get("sha256"), nivel=partition(doc),
                            texto_archivo=relative,
                            metodo_ingesta="Inventario de extracción preparada, sin nueva auditoría")
            if i > 1:
                output.write(",")
            json.dump(manifest, output, ensure_ascii=False)
            if i % 1000 == 0:
                print(f"[profiling] {i} documentos — {sum(warnings.values())} avisos", flush=True)
        output.write("]}")
    frame = pd.DataFrame(rows)
    summary = {
        "fecha": now(), "documentos": len(frame),
        "aptos": int(frame.apta_para_busqueda.sum()),
        "caracteres": int(frame.caracteres.sum()),
        "niveles": frame.nivel.value_counts().to_dict(),
        "aptos_por_nivel": frame.loc[frame.apta_para_busqueda, "nivel"].value_counts().to_dict(),
        "tipos": frame.tipo.value_counts().to_dict(),
        "documentos_con_avisos": int((frame.n_avisos > 0).sum()),
        "documentos_con_pendientes": int((frame.n_pendientes > 0).sum()),
        "avisos": dict(warnings),
        "longitud": {str(k): float(v) for k, v in frame.caracteres.quantile([0, .5, .9, .95, .99, 1]).items()},
        "alcance": "Metadatos. No verifica vigencia, hashes del texto ni cobertura jurídica.",
    }
    write_json(config.reports / "perfil_resumen.json", summary)
    frame.to_csv(config.reports / "perfil_metadata.csv", index=False)
    report = ProfileReport(frame.drop(columns=["doc_id"]), minimal=True,
                           title="Corpus jurídico — metadatos", progress_bar=True,
                           pool_size=1)
    report.to_file(config.reports / "perfil_corpus.html")
    print(f"[profiling] {len(frame)}/{len(frame)} — aptos: {summary['aptos']} — perfil guardado", flush=True)
    return summary
