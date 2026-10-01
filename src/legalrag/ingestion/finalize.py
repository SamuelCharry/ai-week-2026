"""Cierra una versión auditable del corpus antes de indexar."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import islice, repeat
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import zipfile
from xml.etree import ElementTree
from urllib.parse import urlsplit
from pypdf.errors import PyPdfError

from legalrag.citations.extract import norm_identity
from legalrag.io import records, atomic_text, dump_line, write_json, source_path, sha256, now, partition, safe_path


def catalog_metadata(doc):
    doc = dict(doc)
    if doc.get("origen_ampliacion") == "catalogo_tributario_2026":
        for kind in ("circular", "memorando"):
            if doc["doc_id"].startswith("dian_" + kind + "_"):
                doc["tipo_catalogo"] = doc["tipo"]
                doc["tipo"] = kind
                doc["nivel"] = "complementario"
    if doc.get("tipo") == "resolucion":
        match = re.fullmatch(r"resolucion_([a-z0-9_]+)_(\d+)_(\d{4})", str(doc.get("numero", "")))
        if match and int(match[3]) == doc.get("anio"):
            doc["numero_catalogo"] = doc["numero"]
            doc["numero"] = str(int(match[2]))
            doc["organo_emisor_catalogo"] = match[1].upper()
    return doc


def dedup_identity(doc):
    if doc.get("tipo") == "resolucion":
        # El número y año no distinguen resoluciones de entidades diferentes.
        issuer = doc.get("organo_emisor_catalogo") or doc.get("organo_emisor")
        return (norm_identity(doc), str(issuer).casefold()) if issuer else (doc["doc_id"], "emisor_no_identificado")
    return norm_identity(doc)


def extracted_original(path):
    from legalrag.ingestion.acquire import html_text
    if path.suffix.lower() in {".html", ".htm"}:
        return html_text(path.read_bytes())[1].strip()
    if path.suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            xml = ElementTree.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        return "\n\n".join("".join(p.itertext()).strip() for p in xml.findall(".//w:p", ns)
                          if "".join(p.itertext()).strip())
    if path.suffix == ".pdf":
        from legalrag.preprocessing.pdf import extract_pdf
        return extract_pdf(path.read_bytes()).strip()
    raise ValueError(f"Formato de fuente no reconocido: {path}")


def extract_reviewed_ocr(config, doc):
    import pymupdf

    details = doc["ocr"]
    tessdata = os.environ.get("TESSDATA_PREFIX") or str(config.root / "data/raw/ocr_model")
    language = safe_path(tessdata, "spa.traineddata")
    if not language.is_file() or sha256(language) != details["idioma_sha256"]:
        raise ValueError("spa.traineddata no coincide con el OCR documentado")
    corrections_path = safe_path(config.root, details["correcciones"])
    if sha256(corrections_path) != details["correcciones_sha256"]:
        raise ValueError("Cambió el registro de correcciones de OCR")
    corrections = json.loads(corrections_path.read_text(encoding="utf-8"))
    pages = []
    with pymupdf.open(config.root / doc["archivo_raw"]) as pdf:
        for page in pdf:
            layer = page.get_textpage_ocr(language="spa", dpi=details["dpi"], full=True,
                                          tessdata=str(Path(tessdata).resolve()))
            pages.append(page.get_text(textpage=layer).strip())
    original = "\n\f\n".join(pages)
    intermediate_ok = hashlib.sha256(original.encode("utf-8")).hexdigest() == details["original_texto_sha256"]
    # El OCR puede no ser reproducible bit a bit entre entornos (no determinismo de Tesseract/PyMuPDF).
    # Se intenta aplicar las correcciones de todos modos y se verifica el texto final.
    for change in corrections["cambios"]:
        index = change["pagina"] - 1
        if pages[index].count(change["antes"]) != 1:
            if not intermediate_ok:
                raise ValueError(
                    f"El OCR reproducido difiere del documentado y la corrección de la página "
                    f"{change['pagina']} no es aplicable al texto actual"
                )
            raise ValueError(f"Corrección de OCR ambigua en página {change['pagina']}")
        pages[index] = pages[index].replace(change["antes"], change["despues"], 1)
    reviewed = "\n\f\n".join(pages)
    if hashlib.sha256(reviewed.encode("utf-8")).hexdigest() != corrections["revisado_sha256"]:
        raise ValueError(
            "El OCR corregido no coincide con la revisión documentada"
            + ("" if intermediate_ok else " (el OCR intermedio también difirió del original)")
        )
    return reviewed


def extract_new_document(config, doc):
    try:
        if doc.get("partes_html"):
            text = "\n\n".join(extracted_original(config.root / p["archivo"]) for p in doc["partes_html"])
        elif doc.get("metodo_extraccion") == "ocr_tesseract":
            text = extract_reviewed_ocr(config, doc)
        else:
            text = extracted_original(config.root / doc["archivo_raw"])
        return text, None
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, ElementTree.ParseError, PyPdfError) as exc:
        return None, str(exc)


def extracted_records(config, inventory):
    iterator = iter(records(inventory))
    workers = max(1, min(6, (os.cpu_count() or 2) // 2))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while batch := list(islice(iterator, 24)):
            # Lotes pequeños y resultados en orden para limitar RAM y conservar reproducibilidad.
            for doc, (text, error) in zip(batch, pool.map(extract_new_document, repeat(config), batch)):
                yield doc, text, error


def finalize(config):
    folder = config.root / "data/raw/ampliacion"
    output = config.root / "data/processed/corpus_final/documentos.jsonl"
    report_path = config.reports / "ampliacion_corpus.json"
    area_before, area_after, origins, seen_hashes = Counter(), Counter(), Counter(), set()
    seen_ids, seen_norms = set(), set()
    exclusions, additions, sources = [], [], []
    acquisition = []
    levels, eligible_levels, document_types, publishers = Counter(), Counter(), Counter(), Counter()
    characters = 0
    original_hashes = {}
    review_path = config.root / "data/raw/revision_corpus.json"
    reviews = {d["doc_id"]: d for d in json.loads(review_path.read_text(encoding="utf-8"))["documentos"]} if review_path.exists() else {}
    base = config.prepared / "documentos.jsonl"
    if not base.exists():
        raise FileNotFoundError(base)
    if any(folder.glob("candidatos_*.jsonl.tmp")):
        raise RuntimeError("Hay un catálogo sin terminar. Retoma su descubrimiento antes de cerrar el corpus.")
    for state in folder.glob("estado_*.json"):
        if json.loads(state.read_text(encoding="utf-8"))["estado"] != "terminado":
            raise RuntimeError(f"La adquisición {state.stem} no está cerrada")
    for catalog in folder.glob("candidatos_*.jsonl"):
        source = catalog.stem.removeprefix("candidatos_")
        result = folder / f"resultado_{source}.json"
        if not result.exists():
            raise RuntimeError(f"La descarga {source} no ha cerrado. Finaliza la adquisición antes de congelar el corpus.")
        acquisition.append({"fuente": source, "catalogo": catalog.relative_to(config.root).as_posix(),
                            "catalogo_sha256": sha256(catalog),
                            "candidatos_catalogo": sum(1 for _ in records(catalog)),
                            "resultado_ultima_ejecucion": json.loads(result.read_text(encoding="utf-8"))})
    with atomic_text(output) as stream:
        for i, doc in enumerate(records(base), 1):
            identity = dedup_identity(doc)
            path = source_path(config, doc)
            digest = sha256(path)
            if digest != doc["sha256_texto"]:
                raise ValueError(f"El texto original cambió: {doc['doc_id']}")
            for original in doc.get("archivos_raw", []):
                original_path = safe_path(config.root / "data/raw/base", original["archivo"])
                if original_path not in original_hashes:
                    original_hashes[original_path] = sha256(original_path)
                if original_hashes[original_path] != original["sha256"]:
                    raise ValueError(f"El original inicial cambió: {original['archivo']}")
            if doc.get("archivos_raw"):
                doc["raiz_originales"] = "data/raw/base"
                doc["integridad_originales"] = "verificada_en_cierre"
            if doc["doc_id"] in reviews:
                review = reviews[doc["doc_id"]]
                doc["apta_para_busqueda"] = review["apta_para_busqueda"]
                doc["revision_preparacion"] = review
                if review.get("sha256_texto") != digest:
                    doc["avisos"] = list(dict.fromkeys(doc.get("avisos", []) +
                                                      ["texto_reextraido_tras_revision_previa"]))
            seen_ids.add(doc["doc_id"])
            seen_norms.add(identity)
            seen_hashes.add(digest)
            area_before.update(doc.get("areas", []))
            doc["texto_archivo"] = path.relative_to(config.root).as_posix()
            dump_line(stream, doc)
            levels[partition(doc)] += 1
            eligible_levels[partition(doc)] += int(doc.get("apta_para_busqueda") is True)
            document_types[doc.get("tipo", "sin_tipo")] += 1
            publishers[doc.get("dominio_publicador") or urlsplit(doc.get("url", "")).hostname or "sin_dominio"] += 1
            characters += doc.get("caracteres", 0)
            if i % 1000 == 0:
                print(f"[finalize] {i} documentos originales verificados", flush=True)
        area_after.update(area_before)
        for inventory in sorted(folder.glob("documentos*.jsonl")):
            sources.append({"archivo": inventory.relative_to(config.root).as_posix(), "sha256": sha256(inventory)})
            for doc, text, extraction_error in extracted_records(config, inventory):
                doc = catalog_metadata(doc)
                identity = dedup_identity(doc)
                if doc["doc_id"] in seen_ids or identity in seen_norms:
                    exclusions.append({"doc_id": doc["doc_id"], "motivo": "identidad_ya_presente", "archivo": doc["archivo_raw"]})
                    continue
                try:
                    raw = config.root / doc["archivo_raw"]
                    if sha256(raw) != doc["sha256_raw"]:
                        raise ValueError("El original descargado cambió")
                    if raw.suffix == ".html" and not doc.get("partes_html"):
                        from legalrag.ingestion.complete_parts import part_links
                        if part_links(raw.read_bytes(), doc["url"]):
                            raise ValueError("Norma HTML dividida en páginas sin completar")
                    # La extracción se hace en procesos separados. La aceptación se hace en orden.
                    if doc.get("partes_html"):
                        for part in doc["partes_html"]:
                            if sha256(config.root / part["archivo"]) != part["sha256"]:
                                raise ValueError(f"Cambió la página HTML: {part['archivo']}")
                    elif doc.get("metodo_extraccion") == "ocr_tesseract":
                        ocr = doc["ocr"]
                        if sha256(config.root / ocr["correcciones"]) != ocr["correcciones_sha256"]:
                            raise ValueError("Cambió la revisión de OCR")
                    if extraction_error is not None:
                        # OCR no reproducible se trata como exclusión (no detiene el corpus).
                        raise ValueError(extraction_error)
                    text = text.replace("\r\n", "\n").replace("\r", "\n")
                    normative = doc["tipo"] in {"ley", "decreto", "acto_legislativo", "resolucion"}
                    if len(text) < (300 if normative else 1500):
                        raise ValueError("Texto insuficiente")
                    if text.count("\ufffd") / max(1, len(text)) > 0.002:
                        raise ValueError("Codificación defectuosa tras extraer el original")
                    if doc["tipo"] in {"circular", "memorando"}:
                        heading = re.search(r"\b" + doc["tipo"] + r"\s+\d+\s+de\s+(\d{4})", text[:1000], re.I)
                        if heading and int(heading[1]) != int(doc["anio"]):
                            doc["apta_para_busqueda"] = False
                            doc["avisos"] = list(dict.fromkeys(doc.get("avisos", []) + ["anio_catalogo_no_coincide_con_encabezado"]))
                            doc["revision_preparacion"] = {"motivo": "El año del catálogo no coincide con el encabezado del documento.",
                                                          "anio_catalogo": doc["anio"], "anio_encabezado": int(heading[1])}
                    if normative and not re.search(r"\bart[ií]culo\s+(?:\d+|[uú]nico|primero)", text, re.I):
                        raise ValueError("Norma sin articulado reconocible")
                    if doc["tipo"] == "sentencia" and not re.search(r"resuelve|decisi[oó]n|fall[ao]|decide", text, re.I):
                        raise ValueError("No se reconoce una decisión")
                    encoded = text.encode("utf-8")
                    digest = hashlib.sha256(encoded).hexdigest()
                    if digest in seen_hashes:
                        raise ValueError("Contenido duplicado")
                    # La nueva versión canónica se guarda aparte. La descarga permanece intacta.
                    final_text = output.parent / "textos" / (doc["doc_id"] + ".txt")
                    final_text.parent.mkdir(parents=True, exist_ok=True)
                    final_text.write_bytes(encoded)
                    doc.update(texto_archivo=final_text.relative_to(config.root).as_posix(),
                               sha256_texto=digest, caracteres=len(text), version_ingesta="ampliacion-1",
                               dominio_publicador=urlsplit(doc.get("url_final") or doc["url"]).hostname)
                    annotated_publishers = {
                        "normograma.dian.gov.co": "DIAN - Normograma",
                        "www.secretariasenado.gov.co": "Secretaría del Senado",
                        "normativa.colpensiones.gov.co": "Colpensiones - Normativa",
                        "normograma.superservicios.gov.co": "Superservicios - Normograma",
                    }
                    if doc["dominio_publicador"] in annotated_publishers:
                        doc["fuente"] = annotated_publishers[doc["dominio_publicador"]]
                        doc["edicion_con_anotaciones"] = True
                        doc["avisos"] = list(dict.fromkeys(doc.get("avisos", []) + ["edicion_con_anotaciones"]))
                    if doc["origen_ampliacion"] == "fallos_consumidor_financiero_2026":
                        doc["nivel"] = "complementario"
                    if doc.get("edicion_con_anotaciones"):
                        doc["redistribuir_raw"] = False
                    dump_line(stream, doc)
                    levels[partition(doc)] += 1
                    eligible_levels[partition(doc)] += int(doc.get("apta_para_busqueda") is True)
                    document_types[doc.get("tipo", "sin_tipo")] += 1
                    publishers[doc["dominio_publicador"] or "sin_dominio"] += 1
                    characters += len(text)
                    seen_ids.add(doc["doc_id"])
                    seen_norms.add(identity)
                    seen_hashes.add(digest)
                    area_after.update(doc.get("areas", []))
                    origins[doc["origen_ampliacion"]] += 1
                    additions.append(doc["doc_id"])
                    if len(additions) % 100 == 0:
                        print(f"[finalize] {len(additions)} fuentes nuevas verificadas — {len(exclusions)} exclusiones", flush=True)
                except (OSError, ValueError, zipfile.BadZipFile, ElementTree.ParseError, PyPdfError) as exc:
                    exclusions.append({"doc_id": doc["doc_id"], "motivo": str(exc), "archivo": doc["archivo_raw"]})
    report = {"fecha": now(), "documentos_base": sum(1 for _ in records(base)), "incorporados": len(additions),
              "total": len(seen_ids), "por_origen": dict(origins), "areas_antes": dict(area_before),
              "niveles": dict(levels), "aptos_por_nivel": dict(eligible_levels),
              "tipos": dict(document_types), "publicadores": dict(publishers), "caracteres": characters,
              "inventario_base_sha256": sha256(base),
              "originales_base_verificados": len(original_hashes),
              "revision_corpus_sha256": sha256(review_path) if review_path.exists() else None,
              "areas_despues": dict(area_after), "documentos_nuevos": additions, "exclusiones": exclusions,
              "inventarios_descargados": sources, "inventario_final_sha256": sha256(output),
              "adquisicion": acquisition,
              "etiquetas_areas": "Heurísticas y catálogo de procedencia. No equivalen a cobertura jurídica.",
              "vigencia": "Pendiente de revisión jurídica. Fuentes complementarias separadas del núcleo."}
    write_json(report_path, report)
    write_json(output.parent / "version.json", {"fecha": report["fecha"], "sha256": sha256(output),
                                               "documentos": report["total"], "estado": "congelado"})
    print(f"[finalize] {report['total']} documentos — {len(additions)} nuevos — corpus congelado", flush=True)
    return report


if __name__ == "__main__":
    from legalrag.config import CONFIG
    finalize(CONFIG)
