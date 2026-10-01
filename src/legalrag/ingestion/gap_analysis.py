from collections import Counter
import json
from legalrag.citations.extract import norm_mentions, norm_identity
from legalrag.chunking.hierarchical import parents
from legalrag.io import corpus_records, source_path, sha256, atomic_text, dump_line, write_json, now, partition, inventory_path


def audit(config):
    inventory_hash = sha256(inventory_path(config))
    identities = {norm_identity(d) for d in corpus_records(config) if d.get("apta_para_busqueda") is True}
    missing = Counter()
    cited_by = Counter()
    area_counts = Counter()
    total_citations = 0
    integrity_errors = []
    seen_ids = set()
    config.reports.mkdir(parents=True, exist_ok=True)
    with atomic_text(config.reports / "grafo_normativo.jsonl") as graph, \
         atomic_text(config.root / "corpus_manifest.json") as manifest:
        manifest.write(json.dumps({"licencia": "CC-BY-4.0", "fecha_generacion": now(),
                                   "estado": "auditado"}, ensure_ascii=False)[:-1] + ', "documentos":[')
        for count, doc in enumerate(corpus_records(config), 1):
            if doc["doc_id"] in seen_ids:
                raise ValueError(f"doc_id duplicado: {doc['doc_id']}")
            seen_ids.add(doc["doc_id"])
            entry = {key: doc.get(key) for key in ("doc_id", "titulo", "fuente", "url", "fecha_consulta",
                     "areas", "tipo", "numero", "anio", "vigencia", "avisos", "apta_para_busqueda",
                     "numero_catalogo", "tipo_catalogo", "organo_emisor", "organo_emisor_catalogo",
                     "raiz_originales", "integridad_originales",
                     "caracteres", "licencia_fuente", "redistribuir_raw", "edicion_con_anotaciones",
                     "restricciones", "revision_juridica", "url_final", "archivo_raw", "sha256_raw",
                     "solicitud_api", "source_id", "pagina_catalogo", "fecha_descarga", "origen_ampliacion",
                     "archivos_raw", "partes_html", "metodo_extraccion", "ocr", "areas_metodo", "pendientes_preparacion", "revision_preparacion")}
            try:
                path = source_path(config, doc)
                digest = sha256(path)
                text = path.read_text(encoding="utf-8")
                own = norm_identity(doc)
                citations = Counter(target for target in norm_mentions(text) if target != own)
                area_counts.update(doc.get("areas", []))
                for target, frequency in citations.items():
                    dump_line(graph, {"origen": doc["doc_id"], "destino": target, "frecuencia": frequency,
                                      "presente": target in identities, "areas": doc.get("areas", [])})
                    if target not in identities:
                        missing[target] += frequency
                        cited_by[target] += 1
                total_citations += sum(citations.values())
                expected_hash = doc.get("sha256_texto") or doc.get("sha256")
                okay = not expected_hash or digest == expected_hash
                entry.update(sha256=digest, integridad=okay,
                             n_articulos=sum(p["numero_articulo"] is not None for p in parents(doc, text)),
                             texto_archivo=str(path.relative_to(config.root)).replace("\\", "/"),
                             nivel=partition(doc), metodo_ingesta="Texto preparado, auditoría de hash y estructura")
                if not okay:
                    integrity_errors.append(doc["doc_id"])
                    entry["apta_para_busqueda"] = False
            except (OSError, UnicodeError) as exc:
                entry.update(integridad=False, apta_para_busqueda=False, error=str(exc))
                integrity_errors.append(doc["doc_id"])
            if count > 1:
                manifest.write(",")
            json.dump(entry, manifest, ensure_ascii=False)
            if count % 100 == 0:
                print(f"[ingestion] {count} normas escaneadas — {total_citations} citas externas", flush=True)
        manifest.write("]}")
    summary = {"documentos": len(seen_ids), "citas": total_citations,
               "fecha": now(), "inventario_sha256": inventory_hash,
               "fallos_integridad": integrity_errors,
               "documentos_por_area": dict(area_counts),
               "normas_ausentes": [{"norma": k, "frecuencia": missing[k], "documentos_citantes": v}
                                   for k, v in cited_by.most_common()]}
    write_json(config.reports / "brechas_corpus.json", summary)
    version_path = config.root / "data/processed/corpus_final/version.json"
    if version_path.exists():
        version = json.loads(version_path.read_text(encoding="utf-8"))
        if version["sha256"] != inventory_hash:
            raise ValueError("La versión del corpus no coincide con el inventario auditado")
        version["auditoria"] = {
            "fecha": summary["fecha"], "estado": "correcta" if not integrity_errors else "con_errores",
            "inventario_sha256": inventory_hash,
            "manifiesto_sha256": sha256(config.root / "corpus_manifest.json"),
            "fallos_integridad": len(integrity_errors),
        }
        write_json(version_path, version)
    with atomic_text(config.reports / "brechas_corpus.md") as output:
        output.write("# Brechas del corpus\n\n| Norma ausente | Citas |\n|---|---:|\n")
        for identity, frequency in missing.most_common():
            output.write(f"| {identity} | {frequency} |\n")
    print(f"[ingestion] {len(seen_ids)} documentos — {total_citations} citas — "
          f"{len(integrity_errors)} fallos de integridad\nTop 20 ausentes: {missing.most_common(20)}", flush=True)
    return summary
