"""Perfil descriptivo reproducible del manifiesto y una muestra del texto.

No modifica raw, no descarga documentos, no ejecuta OCR ni modelos. Verifica
todos los originales/derivados. Los indicadores de texto NO certifican
completitud juridica; la muestra estratificada no es una estimacion poblacional.

    .venv-profiling/Scripts/python.exe -m legalrag.preprocessing.perfil_raw \
        --raw data/data/raw --output reports/perfil_raw --sample-size 320
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import io
import importlib.util
import json
import os
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile
from xml.etree import ElementTree


VERSION = "1.0.0"
SEED = "perfil-raw-2026-09-29-v1"
ROOT = Path(__file__).resolve().parents[3]


def dump_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_raw(raw, name):
    path = (raw / name).resolve()
    if not path.is_relative_to(raw.resolve()):
        raise ValueError("ruta_fuera_de_raw")
    return path


def missing(value):
    return value is None or value == "" or value == [] or value == {}


def field_missing(records, level):
    for key in sorted(set().union(*(r.keys() for r in records))):
        values = [r.get(key) for r in records]
        count = sum(missing(v) for v in values)
        yield {"nivel": level, "campo": key, "total": len(records), "faltantes": count,
               "porcentaje": round(100 * count / len(records), 4),
               "ausente": sum(key not in r for r in records),
               "null": sum(key in r and r[key] is None for r in records),
               "lista_vacia": sum(isinstance(v, list) and len(v) == 0 for v in values),
               "string_vacio": sum(isinstance(v, str) and not v for v in values)}


def flatten(doc):
    row = {key: value for key, value in doc.items() if not isinstance(value, (dict, list))}
    files = doc.get("archivos_raw", [])
    row.update(n_archivos=len(files), n_derivados=sum(bool(f.get("texto_derivado")) for f in files),
               bytes_raw=sum(f.get("bytes", 0) or 0 for f in files),
               formatos="|".join(sorted({Path(f["archivo"]).suffix.lower() for f in files})))
    for key in ["areas", "temas", "advertencias_preliminares_fuente", "actualizacion_declarada_fuente", "areas_por_epigrafe"]:
        value = doc.get(key)
        row[key + "_n"] = len(value) if isinstance(value, (list, dict)) else 0
        row[key] = "|".join(map(str, value)) if isinstance(value, list) else None
    return row


def check_file(task):
    raw, doc_id, rol, entry = task
    row = {"doc_id": doc_id, "rol": rol, "archivo": entry.get("archivo"),
           "extension": Path(entry.get("archivo", "")).suffix.lower(),
           "bytes_declarados": entry.get("bytes"), "sha256_declarado": entry.get("sha256"),
           "metodo_derivado": entry.get("metodo"), "motivo_derivado": entry.get("motivo"),
           "content_type": entry.get("content_type"), "http_status": entry.get("http_status"),
           "existe": False, "size_ok": False, "hash_ok": False, "error": ""}
    try:
        path = resolve_raw(raw, entry["archivo"])
        row["existe"] = path.is_file()
        if row["existe"]:
            row["bytes_reales"] = path.stat().st_size
            row["sha256_real"] = digest(path)
            row["size_ok"] = row["bytes_reales"] == entry.get("bytes")
            row["hash_ok"] = row["sha256_real"] == entry.get("sha256")
        else:
            row["error"] = "archivo_ausente"
    except Exception as exc:
        row["error"] = str(exc)[:240]
    return row


def quantiles(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return {}
    def q(p):
        x = (len(values) - 1) * p
        lo, hi = math.floor(x), math.ceil(x)
        return round(values[lo] + (values[hi] - values[lo]) * (x - lo), 3)
    return {"n": len(values), "min": values[0], "p25": q(.25), "p50": q(.5),
            "p75": q(.75), "p95": q(.95), "p99": q(.99), "max": values[-1]}


def stratified_sample(docs, size):
    groups = defaultdict(list)
    for doc in docs:
        row = flatten(doc)
        key = " / ".join(str(row.get(k, "")) for k in ["fuente", "tipo", "nivel", "formatos"])
        key += " / derivado=" + str(row["n_derivados"] > 0)
        groups[key].append(doc)
    target = min(len(docs), max(size, len(groups)))
    counts = {key: 1 for key in groups}
    # Uno por estrato y resto proporcional al tamano (sin reemplazo).
    while sum(counts.values()) < target:
        key = max((k for k in groups if counts[k] < len(groups[k])),
                  key=lambda k: (len(groups[k]) / (counts[k] + 1), k))
        counts[key] += 1
    sample, coverage = [], []
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda d: hashlib.sha256((SEED + d["doc_id"]).encode()).hexdigest())
        for doc in ordered[:counts[key]]:
            sample.append((doc, key))
        coverage.append({"estrato": key, "poblacion": len(groups[key]), "muestra": counts[key],
                         "fraccion": counts[key] / len(groups[key])})
    return sample, coverage


def extract_doc(doc, raw, page_limit, extractor_path):
    # Se reutiliza normalizacion y extraccion HTML de la version local.
    # Este adaptador agrega derivados/DOCX; no ejecuta la ingesta antigua.
    spec = importlib.util.spec_from_file_location("perfil_ingesta_snapshot", extractor_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extraer_html, normalizar, VERSION_INGESTA = module.extraer_html, module.normalizar, module.VERSION_INGESTA
    import pdfplumber
    parts, methods, errors = [], [], []
    pages, low_pages, pdf_total, limited, derived = 0, 0, 0, False, 0
    for entry in doc["archivos_raw"]:
        source = entry.get("texto_derivado") or entry
        data = resolve_raw(raw, source["archivo"]).read_bytes()
        if len(data) != source.get("bytes") or hashlib.sha256(data).hexdigest() != source.get("sha256"):
            errors.append("integridad_invalida:" + source["archivo"])
            continue
        ext = Path(source["archivo"]).suffix.lower()
        try:
            if entry.get("texto_derivado"):
                parts.append(normalizar(data.decode("utf-8-sig")))
                derived += 1
                methods.append("texto_derivado:" + source.get("metodo", "sin_metodo"))
            elif ext in {".html", ".htm"}:
                blocks, info = extraer_html(data, entry)
                parts.extend(block["texto"] for block in blocks)
                methods.append("ingesta_" + VERSION_INGESTA + ":" + info["selector"])
            elif ext == ".docx":
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    xml = ElementTree.fromstring(archive.read("word/document.xml"))
                ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                parts.append(normalizar("\n".join("".join(p.itertext()) for p in xml.findall(".//w:p", ns))))
                methods.append("docx_xml_stdlib")
            elif ext == ".pdf":
                start = data[:1024].find(b"%PDF-")
                if start < 0:
                    raise ValueError("cabecera_pdf_invalida")
                with pdfplumber.open(io.BytesIO(data[start:])) as pdf:
                    pdf_total += len(pdf.pages)
                    selected = pdf.pages[:page_limit]
                    limited |= len(selected) < len(pdf.pages)
                    for page in selected:
                        text = normalizar(page.extract_text(use_text_flow=True) or "")
                        parts.append(text)
                        pages += 1
                        low_pages += len(text.strip()) < 40
                methods.append("pdfplumber_primeras_paginas")
            else:
                errors.append("sin_extractor:" + ext)
        except Exception as exc:
            errors.append(type(exc).__name__ + ":" + str(exc)[:120])
    text = "\n\n".join(parts)
    words = re.findall(r"\b\w+\b", text)
    articles = list(re.finditer(r"(?im)^\s*(?:ART[IÍ]CULO|ART\.)\s+\d+(?:[.\-]\d+)*[A-Z]?", text))
    article_words = [len(re.findall(r"\b\w+\b", text[m.start():articles[i + 1].start() if i + 1 < len(articles) else len(text)]))
                     for i, m in enumerate(articles)]
    aq = quantiles(article_words)
    return {"doc_id": doc["doc_id"], "tipo": doc.get("tipo"), "fuente": doc.get("fuente"),
            "nivel": doc.get("nivel"), "estado": "parcial" if errors or limited else "ok" if text else "sin_texto",
            "caracteres": len(text), "palabras": len(words), "lineas": text.count("\n") + bool(text),
            "articulos_detectados": len(articles), "articulo_palabras_p50": aq.get("p50"),
            "articulo_palabras_p95": aq.get("p95"), "articulo_palabras_max": aq.get("max"),
            "articulos_mas_512_palabras": sum(n > 512 for n in article_words),
            "articulos_mas_2048_palabras": sum(n > 2048 for n in article_words),
            "paginas_pdf": pdf_total, "paginas_extraidas": pages, "paginas_poco_texto": low_pages,
            "pdf_truncado": limited, "derivados_usados": derived,
            "mojibake": len(re.findall(r"\ufffd|Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|â[€\u0080-\u00bf]", text)),
            "reemplazos_unicode": text.count("\ufffd"),
            "proporcion_alfabetica": round(sum(c.isalpha() for c in text) / max(len(text), 1), 5),
            "pie_editorial": bool(re.search(r"Avance Jur[ií]dico|disposiciones analizadas por", text, re.I)),
            "senales_derogacion": len(re.findall(r"derogad[oa]|inexequible", text, re.I)),
            "senales_jerarquia": len(re.findall(r"(?im)^\s*(?:LIBRO|T[IÍ]TULO|CAP[IÍ]TULO|SECCI[OÓ]N)\s+", text)),
            "sha256_texto": hashlib.sha256(text.encode()).hexdigest(),
            "metodos": " | ".join(sorted(set(methods))), "errores": " | ".join(errors)}


def run_extract(task):
    doc, stratum, raw, timeout, page_limit, extractor_path = task
    start = time.monotonic()
    try:
        result = subprocess.run([sys.executable, "-m", "legalrag.preprocessing.perfil_raw", "--worker", "--raw", str(raw),
                                 "--pdf-page-limit", str(page_limit), "--extractor", str(extractor_path)],
                                input=json.dumps(doc), capture_output=True, text=True, encoding="utf-8",
                                timeout=timeout, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
        if result.returncode:
            row = {"doc_id": doc["doc_id"], "estado": "error", "errores": result.stderr[-400:]}
        else:
            row = json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        row = {"doc_id": doc["doc_id"], "estado": "timeout", "errores": f"limite_{timeout}s_documento"}
    row.update(estrato=stratum, segundos=round(time.monotonic() - start, 3))
    return row


def make_ydata(rows, output, name, title):
    import pandas as pd
    from ydata_profiling import ProfileReport
    frame = pd.DataFrame(rows).convert_dtypes()
    # Perfil minimo: univariados, nulos, cardinalidad/duplicados. Evita correlaciones
    # costosas de identificadores/URLs; el HTML es un ProfileReport real.
    report = ProfileReport(frame, title=title, minimal=True, explorative=False,
                           progress_bar=False, pool_size=2,
                           samples={"head": 5, "tail": 5},
                           html={"style": {"full_width": True}})
    report.to_file(output / (name + ".html"))
    report.to_file(output / (name + ".json"))


def compare_profiles(baseline, current):
    """Compara tablas ya terminadas, sin volver a leer ni modificar el corpus."""
    baseline, current = Path(baseline), Path(current)
    def read_rows(directory, name):
        with (directory / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    before = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    after = json.loads((current / "metrics.json").read_text(encoding="utf-8"))
    if after.get("muestra", {}).get("ids_reutilizados_desde"):
        after["muestra"]["metodo"] = "IDs fijos de muestra estratificada previa; comparacion pareada sin incluir nuevas incorporaciones"
        dump_json(current / "metrics.json", after)
        readme = current / "LEEME.md"
        if readme.exists():
            note = "\n\nComparacion pareada: en este reporte se reutilizan los IDs del perfil previo. "
            note += "No todos los estratos del corpus ampliado tienen muestra y las nuevas incorporaciones no se muestrean.\n"
            if "Comparacion pareada:" not in readme.read_text(encoding="utf-8"):
                readme.write_text(readme.read_text(encoding="utf-8") + note, encoding="utf-8")
    before_docs = {r["doc_id"]: r for r in read_rows(baseline, "documentos.csv")}
    after_docs = {r["doc_id"]: r for r in read_rows(current, "documentos.csv")}
    shared = before_docs.keys() & after_docs.keys()
    changes = []
    for doc_id in sorted(shared):
        old, new = before_docs[doc_id], after_docs[doc_id]
        for field in sorted(old.keys() | new.keys()):
            if (old.get(field) or "") != (new.get(field) or ""):
                changes.append({"doc_id": doc_id, "campo": field, "antes": old.get(field), "despues": new.get(field)})
    dump_csv(current / "comparacion_metadatos.csv", changes, ["doc_id", "campo", "antes", "despues"])
    new_ids = sorted(after_docs.keys() - before_docs.keys())
    dump_csv(current / "documentos_agregados.csv", [after_docs[doc_id] for doc_id in new_ids], list(next(iter(after_docs.values()))))
    old_files = {(r["rol"], r["archivo"]): r for r in read_rows(baseline, "archivos.csv")}
    new_files = {(r["rol"], r["archivo"]): r for r in read_rows(current, "archivos.csv")}
    file_changes = [{"rol": key[0], "archivo": key[1], "sha256_antes": old_files[key].get("sha256_real"),
                     "sha256_despues": new_files[key].get("sha256_real")}
                    for key in sorted(old_files.keys() & new_files.keys())
                    if old_files[key].get("sha256_real") != new_files[key].get("sha256_real")]
    dump_csv(current / "comparacion_hashes_modificados.csv", file_changes,
             ["rol", "archivo", "sha256_antes", "sha256_despues"])
    old_text = {r["doc_id"]: r for r in read_rows(baseline, "muestra_texto.csv")}
    new_text = {r["doc_id"]: r for r in read_rows(current, "muestra_texto.csv")}
    paired = []
    for doc_id in sorted(old_text.keys() & new_text.keys()):
        old, new = old_text[doc_id], new_text[doc_id]
        row = {"doc_id": doc_id}
        for field in ["estado", "caracteres", "palabras", "pie_editorial", "mojibake", "reemplazos_unicode", "sha256_texto", "errores"]:
            row[field + "_antes"] = old.get(field)
            row[field + "_despues"] = new.get(field)
        if old.get("caracteres") and new.get("caracteres"):
            row["delta_caracteres"] = int(new["caracteres"]) - int(old["caracteres"])
        paired.append(row)
    dump_csv(current / "comparacion_muestra_texto.csv", paired)
    report = {"baseline": str(baseline.resolve()), "despues": str(current.resolve()),
              "manifest_sha256_antes": before["manifest_sha256"], "manifest_sha256_despues": after["manifest_sha256"],
              "documentos_antes": before["documentos"], "documentos_despues": after["documentos"],
              "documentos_agregados": len(new_ids), "documentos_eliminados": sorted(before_docs.keys() - after_docs.keys()),
              "campos_csv_modificados": len(changes), "documentos_csv_modificados": len({r["doc_id"] for r in changes}),
              "cambios_por_campo": dict(Counter(r["campo"] for r in changes)),
              "integridad_antes": before["integridad"], "integridad_despues": after["integridad"],
              "referencias_previas_con_hash_modificado": len(file_changes),
              "referencias_previas_eliminadas": len(old_files.keys() - new_files.keys()),
              "referencias_nuevas": len(new_files.keys() - old_files.keys()),
              "mismos_ids_muestra": old_text.keys() == new_text.keys(), "documentos_muestra_pareados": len(paired),
              "muestra_estados_antes": before["resumen_texto"]["estados"], "muestra_estados_despues": after["resumen_texto"]["estados"],
              "muestra_pie_editorial_antes": before["resumen_texto"]["documentos_pie_editorial"],
              "muestra_pie_editorial_despues": after["resumen_texto"]["documentos_pie_editorial"],
              "muestra_pie_retirado_pareado": sum(r["pie_editorial_antes"] == "True" and r["pie_editorial_despues"] == "False" for r in paired),
              "muestra_textos_cambiados_ambos_extraidos": sum(bool(r["sha256_texto_antes"]) and bool(r["sha256_texto_despues"])
                                                              and r["sha256_texto_antes"] != r["sha256_texto_despues"] for r in paired),
              "limitaciones": ["La comparacion de texto usa el adaptador diagnostico del perfil: no certifica la ingesta completa.",
                                "Tiempos y timeouts pueden variar por contencion de CPU; no prueban defectos del original.",
                                "La muestra pareada conserva los IDs previos y no representa las nuevas incorporaciones.",
                                "La auditoria heredada no se ha reejecutado; sus 267 alertas altas no son un conteo posterior a las correcciones."]}
    residual_ids = sorted(r["doc_id"] for r in paired if r["pie_editorial_despues"] == "True")
    if residual_ids == ["ley_160_1994", "ley_446_1998"]:
        report["interpretacion_residuos_editoriales"] = {
            "doc_ids": residual_ids, "verificacion": "revision_manual_de_textos_completos",
            "descripcion": "Notas de Pie de Pagina incluidas por Avance Juridico: anotaciones sobre reformas conservadas intencionalmente. No son navegacion ni el aviso general de copyright.",
            "decision": "Conservar anotaciones y su procedencia; el detector identifica correctamente contenido editorial."}
        readme = current / "LEEME.md"
        if readme.exists():
            note = "\n\nLas dos senales editoriales residuales (ley_160_1994 y ley_446_1998) corresponden a "
            note += "notas de pie incluidas por Avance Juridico sobre reformas, conservadas intencionalmente. "
            note += "La revision manual confirma anotaciones editoriales, no navegacion ni el aviso general de copyright. "
            note += "No se deben borrar por el solo hecho de activar el detector.\n"
            if "Las dos senales editoriales residuales" not in readme.read_text(encoding="utf-8"):
                readme.write_text(readme.read_text(encoding="utf-8") + note, encoding="utf-8")
    dump_json(current / "comparacion.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/perfil_raw")
    parser.add_argument("--sample-size", type=int, default=320)
    parser.add_argument("--sample-ids", type=Path, help="Reutiliza IDs de una muestra previa para comparacion pareada.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--pdf-page-limit", type=int, default=30)
    parser.add_argument("--extractor", type=Path, default=ROOT / "src/legalrag/preprocessing/ingesta.py")
    parser.add_argument("--skip-ydata", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    raw = args.raw.resolve()
    if args.worker:
        sys.stdout.reconfigure(encoding="utf-8")
        print(json.dumps(extract_doc(json.loads(sys.stdin.read()), raw, args.pdf_page_limit, args.extractor), ensure_ascii=False))
        return
    if args.sample_size < 0 or args.workers < 1 or args.timeout < 1 or args.pdf_page_limit < 1:
        parser.error("limites invalidos")
    output = args.output.resolve()
    if output == raw or output.is_relative_to(raw):
        parser.error("output debe quedar fuera de raw")
    output.mkdir(parents=True, exist_ok=True)
    extractor_path = output / "ingesta_snapshot.py"
    extractor_path.write_bytes(args.extractor.read_bytes())
    manifest_path = raw / "manifest.json"
    manifest_sha = digest(manifest_path)
    docs = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    document_rows = [flatten(doc) for doc in docs]
    dump_csv(output / "documentos.csv", document_rows)
    raw_entries = [f for doc in docs for f in doc.get("archivos_raw", [])]
    derived_entries = [f["texto_derivado"] for f in raw_entries if f.get("texto_derivado")]
    missing_rows = list(field_missing(docs, "documento")) + list(field_missing(raw_entries, "archivo_raw"))
    if derived_entries:
        missing_rows += list(field_missing(derived_entries, "texto_derivado"))
    dump_csv(output / "faltantes_campos.csv", missing_rows)
    tasks = []
    for doc in docs:
        for entry in doc.get("archivos_raw", []):
            tasks.append((raw, doc["doc_id"], "raw", entry))
            if entry.get("texto_derivado"):
                tasks.append((raw, doc["doc_id"], "derivado", entry["texto_derivado"]))
    print(f"Verificando {len(tasks)} referencias originales/derivadas...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        file_rows = list(pool.map(check_file, tasks))
    dump_csv(output / "archivos.csv", file_rows)
    counts = Counter(doc.get("doc_id") for doc in docs)
    dump_csv(output / "ids_duplicados.csv", [{"doc_id": k, "n": v} for k, v in counts.items() if v > 1], ["doc_id", "n"])
    hashes = defaultdict(list)
    for row in file_rows:
        if row.get("sha256_real"):
            hashes[row["sha256_real"]].append(row)
    duplicate_rows = [{"sha256": sha, "n_referencias": len(group), **row}
                      for sha, group in hashes.items() if len(group) > 1 for row in group]
    dump_csv(output / "hashes_duplicados.csv", duplicate_rows, ["sha256", "n_referencias", "doc_id", "rol", "archivo"])
    paths = {str(resolve_raw(raw, row["archivo"])) for row in file_rows if row["archivo"] and row["error"] != "ruta_fuera_de_raw"}
    untracked = [{"archivo": str(path.relative_to(raw)), "bytes": path.stat().st_size}
                 for path in raw.rglob("*") if path.is_file() and str(path.resolve()) not in paths]
    dump_csv(output / "archivos_no_referenciados.csv", untracked, ["archivo", "bytes"])
    distributions = {key: dict(Counter(str(doc.get(key) if not missing(doc.get(key)) else "(sin dato)") for doc in docs).most_common())
                     for key in ["tipo", "fuente", "anio", "nivel", "vigencia", "vigencia_fuente", "organo_emisor", "origen_ampliacion"]}
    distributions["formato_raw"] = dict(Counter(Path(f["archivo"]).suffix.lower() for f in raw_entries))
    distributions["metodo_derivado"] = dict(Counter(f.get("metodo") for f in derived_entries))
    distributions["areas"] = dict(Counter(area for doc in docs for area in doc.get("areas", [])))
    dump_csv(output / "distribuciones.csv", [{"variable": k, "valor": value, "n": count}
                                            for k, values in distributions.items() for value, count in values.items()])
    inherited_path = raw.parents[1] / "reports/auditoria_corpus.json"
    inherited = json.loads(inherited_path.read_text(encoding="utf-8-sig")) if inherited_path.exists() else {}
    inherited_summary = {k: inherited[k] for k in ["fecha", "hallazgos", "metricas", "por_comprobacion"] if k in inherited}
    inherited_summary.update(origen=str(inherited_path), reejecutada=False)
    dump_json(output / "auditoria_heredada.json", inherited_summary)
    dump_csv(output / "auditoria_heredada_por_comprobacion.csv", inherited.get("por_comprobacion", []))
    graph_path = raw / "grafo_faltantes.csv"
    graph = list(csv.DictReader(graph_path.open(encoding="utf-8-sig", newline=""))) if graph_path.exists() else []
    graph_summary = {"filas": len(graph), "claves_unicas": len({r["clave"] for r in graph}),
                     "top_15": sorted(graph, key=lambda r: int(r["documentos_que_citan"]), reverse=True)[:15],
                     "filas_4_mas": sum(int(r["documentos_que_citan"]) >= 4 for r in graph),
                     "nota": "Grafo generado por la entrega; no equivale al universo juridico faltante."}
    sample, strata = stratified_sample(docs, args.sample_size) if args.sample_size else ([], [])
    if args.sample_ids:
        requested_ids = json.loads(args.sample_ids.read_text(encoding="utf-8-sig"))
        if len(set(requested_ids)) != len(requested_ids):
            raise ValueError("sample-ids contiene duplicados")
        all_sample, _ = stratified_sample(docs, len(docs))
        by_id = {doc["doc_id"]: (doc, key) for doc, key in all_sample}
        absent = set(requested_ids) - by_id.keys()
        if absent:
            raise ValueError(f"IDs de muestra ausentes: {sorted(absent)}")
        sample = [by_id[doc_id] for doc_id in requested_ids]
        chosen_counts = Counter(key for _, key in sample)
        strata = [{**s, "muestra": chosen_counts[s["estrato"]],
                   "fraccion": chosen_counts[s["estrato"]] / s["poblacion"]} for s in strata]
    dump_csv(output / "muestra_estratos.csv", strata)
    dump_json(output / "muestra_ids.json", [doc["doc_id"] for doc, _ in sample])
    integrity = {"referencias": len(file_rows), "archivos_raw": len(raw_entries), "archivos_derivados": len(derived_entries),
                 "ausentes": sum(not row["existe"] for row in file_rows),
                 "tamano_discrepante": sum(row["existe"] and not row["size_ok"] for row in file_rows),
                 "hash_discrepante": sum(row["existe"] and not row["hash_ok"] for row in file_rows),
                 "bytes_raw_declarados": sum(f.get("bytes", 0) for f in raw_entries),
                 "bytes_derivados_declarados": sum(f.get("bytes", 0) for f in derived_entries),
                 "ids_duplicados": sum(v > 1 for v in counts.values()),
                 "grupos_hash_duplicados": sum(len(v) > 1 for v in hashes.values()),
                 "referencias_en_grupos_hash_duplicados": len(duplicate_rows),
                 "archivos_no_referenciados": len(untracked)}
    metrics = {"version": VERSION, "fecha_utc": datetime.now(timezone.utc).isoformat(),
               "raw": str(raw), "manifest_sha256": manifest_sha, "documentos": len(docs),
               "extractor_sha256": digest(extractor_path), "extractor_snapshot": str(extractor_path),
               "integridad": integrity, "distribuciones": distributions,
               "faltantes": missing_rows, "bytes_por_documento": quantiles(r["bytes_raw"] for r in document_rows),
               "archivos_por_documento": quantiles(r["n_archivos"] for r in document_rows),
               "auditoria_heredada": inherited_summary, "grafo_faltantes": graph_summary,
               "muestra": {"solicitada": args.sample_size, "seleccionada": len(sample), "estratos": len(strata),
                           "semilla": SEED, "metodo": ("IDs fijos de muestra estratificada previa; comparacion pareada sin incluir nuevas incorporaciones"
                                                          if args.sample_ids else "1 por estrato fuente/tipo/nivel/formatos/derivado + asignacion proporcional determinista"),
                           "ids_reutilizados_desde": str(args.sample_ids) if args.sample_ids else None,
                           "limite_segundos_documento": args.timeout, "limite_paginas_por_pdf": args.pdf_page_limit,
                           "advertencia": "Sobremuestrea estratos raros. No extrapolar porcentajes de texto a todo el corpus."},
               "limitaciones": ["Descargas completas y hashes correctos no certifican texto integro, vigencia ni cobertura legal.",
                                 "Auditoria heredada y grafo no se regeneran; se identifican como evidencia heredada.",
                                 "Extraction HTML reutiliza la version local de ingesta; puede conservar pie editorial.",
                                 "Articulos por regex son candidatos, incluyen citas/anotaciones y no prueban unidades propias.",
                                 "Palabras son regex Unicode; no son tokens de ningun encoder/decoder.",
                                 "No se ejecuta OCR: derivados existentes se validan y reutilizan."]}
    dump_json(output / "metrics.json", metrics)
    print("Integridad completa: " + json.dumps(integrity), flush=True)
    print(f"Extrayendo muestra de {len(sample)} documentos / {len(strata)} estratos...", flush=True)
    text_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        extraction_tasks = [(doc, key, raw, args.timeout, args.pdf_page_limit, extractor_path) for doc, key in sample]
        for i, row in enumerate(pool.map(run_extract, extraction_tasks), 1):
            text_rows.append(row)
            if i % 40 == 0:
                print(f"Muestra {i}/{len(sample)}", flush=True)
    dump_csv(output / "muestra_texto.csv", text_rows)
    metrics["resumen_texto"] = {"estados": dict(Counter(r["estado"] for r in text_rows)),
                                "caracteres": quantiles(r.get("caracteres") for r in text_rows),
                                "palabras": quantiles(r.get("palabras") for r in text_rows),
                                "articulo_palabras_p95": quantiles(r.get("articulo_palabras_p95") for r in text_rows),
                                "documentos_sin_articulos_regex": sum(r.get("articulos_detectados") == 0 for r in text_rows),
                                "documentos_pie_editorial": sum(bool(r.get("pie_editorial")) for r in text_rows),
                                "documentos_mojibake": sum((r.get("mojibake") or 0) > 0 for r in text_rows),
                                "documentos_pdf_truncado": sum(bool(r.get("pdf_truncado")) for r in text_rows),
                                "documentos_con_derivado_usado": sum((r.get("derivados_usados") or 0) > 0 for r in text_rows)}
    metrics["manifest_sin_cambios_durante_perfil"] = digest(manifest_path) == manifest_sha
    metrics["ydata_generado"] = False
    dump_json(output / "metrics.json", metrics)
    if not args.skip_ydata:
        from importlib.metadata import version
        print("Generando ydata-profiling real (documentos, archivos y muestra)...", flush=True)
        for rows, name, title in [(document_rows, "ydata_documentos", "Corpus: manifiesto completo por documento"),
                                  (file_rows, "ydata_archivos", "Corpus: integridad completa de originales y derivados"),
                                  (text_rows, "ydata_muestra_texto", "Corpus: muestra estratificada de texto (no extrapolar)")]:
            if rows:
                make_ydata(rows, output, name, title)
        metrics["ydata_generado"] = True
        metrics["ydata_version"] = version("ydata-profiling")
        dump_json(output / "metrics.json", metrics)
    dump_json(output / "schema.json", {"documentos.csv": list(document_rows[0]), "archivos.csv": list(file_rows[0]),
                                       "muestra_texto.csv": list(dict.fromkeys(k for r in text_rows for k in r)),
                                       "faltantes_campos.csv": list(missing_rows[0])})
    resolved = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (output / "requirements-resolved.txt").write_text(resolved.stdout, encoding="utf-8")
    (output / "LEEME.md").write_text(
        "# Perfil de corpus raw\n\n"
        f"Manifest SHA-256: `{manifest_sha}`. Documentos: {len(docs)}.\n\n"
        "`ydata_documentos.html`, `ydata_archivos.html` y `ydata_muestra_texto.html` son reportes ydata-profiling reales "
        "(modo minimal: univariados, faltantes y cardinalidad). Sus JSON completos tienen el mismo nombre. "
        "`metrics.json` resume resultados; los CSV permiten inspeccionarlos.\n\n"
        "La integridad verifica todas las referencias originales y derivadas, incluyendo hashes SHA-256. "
        "La muestra de texto es determinista y estratificada; cada estrato recibe al menos un documento, por lo que "
        "sobrerrepresenta grupos raros. No extrapolar sus porcentajes. PDF: primeras "
        f"{args.pdf_page_limit} paginas por archivo; tiempo maximo {args.timeout}s por documento. "
        "La extraccion no ejecuta OCR, usa derivados verificados existentes y el extractor HTML del repositorio. "
        "Palabras no son tokens; articulos por regex no son articulos juridicos validados.\n\n"
        "`auditoria_heredada.json` conserva fecha y procedencia: sus comprobaciones no fueron reejecutadas. "
        "Integridad de descarga no es completitud normativa ni certificacion de vigencia. "
        "Los archivos no referenciados pueden ser metadatos auxiliares legitimos: no borrarlos automaticamente.\n",
        encoding="utf-8")
    print(f"Terminado: {output}", flush=True)


if __name__ == "__main__":
    main()
