"""Ydata de toda la extraccion terminada, sin incorporar los textos al reporte.

    .venv-profiling/Scripts/python.exe -m legalrag.preprocessing.perfil_preparado \
        --input data/processed/corpus_preparado --output reports/perfil_corpus_preparado

Lee cada texto canonico para verificar SHA-256/caracteres y contar palabras,
lineas y marcadores [NOTA]. Palabras no son tokens; marcadores no son notas
juridicas verificadas. No realiza inferencia, ingesta, segmentacion ni OCR.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

from legalrag.preprocessing.perfil_raw import digest, dump_csv, dump_json, make_ydata, quantiles


ROOT = Path(__file__).resolve().parents[3]
VERSION = "1.0.0"


def summarize_record(record):
    extraction = record.get("extraccion", [])
    originals = record.get("archivos_raw", [])
    return {
        "doc_id": record["doc_id"],
        "fuente": record.get("fuente"), "tipo": record.get("tipo"), "anio": record.get("anio"),
        "nivel": record.get("nivel"), "vigencia": record.get("vigencia"),
        "vigencia_fuente": record.get("vigencia_fuente"),
        "version_ingesta": record.get("version_ingesta"),
        "estado_extraccion": record.get("estado_extraccion"),
        "apta_para_busqueda_segun_preparador": record.get("apta_para_busqueda"),
        "caracteres_registrados": record.get("caracteres", 0),
        "n_archivos_originales": len(originals),
        "formatos_originales": "|".join(sorted({Path(f["archivo"]).suffix.lower() for f in originals})),
        "n_derivados_declarados": sum(bool(f.get("texto_derivado")) for f in originals),
        "n_partes": len(record.get("partes", [])),
        "n_avisos": len(record.get("avisos", [])),
        "avisos": "|".join(sorted(record.get("avisos", []))),
        "n_pendientes_preparacion": len(record.get("pendientes_preparacion", [])),
        "pendientes_preparacion": "|".join(sorted(record.get("pendientes_preparacion", []))),
        "paginas_pdf_registradas": sum(e.get("paginas_pdf") or 0 for e in extraction),
        "paginas_con_poco_texto_registradas": sum(len(e.get("paginas_con_poco_texto", [])) for e in extraction),
        "n_anexos_xml_docx": sum(len(e.get("anexos_docx", [])) for e in extraction),
        "n_tablas_html_estructuradas": sum(e.get("tablas_estructuradas", 0) for e in extraction),
        "n_elementos_html_retirados": sum(len(e.get("elementos_html_retirados", [])) for e in extraction),
        "n_archivos_anotaciones_detectadas": sum(bool(e.get("anotaciones_detectadas")) for e in extraction),
        "coincidencias_mojibake_registradas": record.get("coincidencias_mojibake", 0),
        "sha256_texto_registrado": record.get("sha256_texto"),
        "texto_archivo": record.get("texto_archivo"),
        "error_extraccion": record.get("error") or "",
    }


def inspect_text(task):
    row, directory = task
    row = dict(row)
    row.update(texto_existe=False, sha256_texto_ok=False, caracteres_coinciden=False,
               sha256_texto_calculado=None, error_verificacion="")
    if not row.get("texto_archivo"):
        row["error_verificacion"] = "sin_texto_archivo_en_registro"
        return row
    try:
        path = (directory / row["texto_archivo"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("texto_archivo_fuera_de_directorio")
        row["texto_existe"] = path.is_file()
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8")
        row.update(
            sha256_texto_calculado=sha,
            sha256_texto_ok=sha == row.get("sha256_texto_registrado"),
            caracteres_recontados=len(text),
            caracteres_coinciden=len(text) == row.get("caracteres_registrados"),
            bytes_texto_utf8=len(data),
            palabras_regex=sum(1 for _ in re.finditer(r"\b\w+\b", text)),
            lineas=text.count("\n") + bool(text),
            marcadores_NOTA_referencias_y_bloques=len(re.findall(r"\[NOTA\s+[^\]\r\n]+\]", text)),
            marcadores_NOTA_inicio_linea=len(re.findall(r"(?m)^\[NOTA\s+[^\]\r\n]+\]", text)),
            reemplazos_unicode_recontados=text.count("\ufffd"),
            proporcion_alfabetica=round(sum(c.isalpha() for c in text) / max(len(text), 1), 6),
        )
    except Exception as exc:
        row["error_verificacion"] = type(exc).__name__ + ":" + str(exc)[:220]
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/corpus_preparado")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/perfil_corpus_preparado")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/data/raw/manifest.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-ydata", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers debe ser positivo")
    directory, output = args.input.resolve(), args.output.resolve()
    if output == directory or output.is_relative_to(directory):
        parser.error("La salida debe quedar fuera del corpus preparado")
    summary_path = directory / "resumen.json"
    source = directory / "documentos.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("modo") != "completo":
        raise ValueError("La preparacion no declara modo completo; espere el snapshot final")
    source_sha, summary_sha = digest(source), digest(summary_path)
    if digest(args.manifest.resolve()) != summary.get("sha256_manifest"):
        raise ValueError("Manifest actual no coincide con snapshot preparado")
    rows, warnings, pending = [], Counter(), Counter()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(summarize_record(record))
            warnings.update(record.get("avisos", []))
            pending.update(record.get("pendientes_preparacion", []))
    if len(rows) != summary.get("documentos_manifest") or len(rows) != summary.get("documentos_procesados"):
        raise ValueError("Cantidad JSONL no coincide con resumen de preparacion completa")
    ids = Counter(row["doc_id"] for row in rows)
    if any(n != 1 for n in ids.values()):
        raise ValueError("IDs repetidos en documentos.jsonl")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if set(ids) != {doc["doc_id"] for doc in manifest}:
        raise ValueError("Los IDs preparados no cubren exactamente el manifest")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Verificando textos y contando palabras/lineas de {len(rows)} documentos...", flush=True)
    verified = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(inspect_text, ((row, directory) for row in rows)), 1):
            verified.append(row)
            if i % 1000 == 0 or i == len(rows):
                print(f"Textos perfilados {i}/{len(rows)}", flush=True)
    if digest(source) != source_sha or digest(summary_path) != summary_sha:
        raise RuntimeError("Snapshot preparado cambio durante perfil; repetir al terminar correcciones")
    if digest(args.manifest.resolve()) != summary["sha256_manifest"]:
        raise RuntimeError("Manifest cambio durante perfil")
    dump_csv(output / "textos_preparados.csv", verified)
    dump_csv(output / "avisos.csv", [{"aviso": key, "documentos": value} for key, value in warnings.most_common()])
    dump_csv(output / "pendientes.csv", [{"pendiente": key, "documentos": value} for key, value in pending.most_common()])
    invalid = [r for r in verified if not r["texto_existe"] or not r["sha256_texto_ok"] or not r["caracteres_coinciden"]]
    dump_csv(output / "fallos_verificacion_textos.csv", invalid,
             ["doc_id", "texto_archivo", "texto_existe", "sha256_texto_ok", "caracteres_coinciden", "error_verificacion"])
    profiles = [dict((k, v) for k, v in row.items() if k not in {
        "doc_id", "sha256_texto_registrado", "sha256_texto_calculado", "texto_archivo", "error_extraccion", "error_verificacion"
    }) for row in verified]
    metrics = {
        "version": VERSION, "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "modo": "censo_de_todos_los_documentos_preparados", "documentos": len(verified),
        "manifest_sha256": summary["sha256_manifest"], "documentos_jsonl_sha256": source_sha,
        "resumen_preparador_sha256": summary_sha, "parser_y_preparador_sha256": summary.get("sha256_parser_y_preparador"),
        "version_ingesta": summary.get("version_ingesta"), "fuente_preparada": str(directory),
        "validacion": {"ids_coinciden_con_manifest": True, "snapshot_sin_cambios_durante_perfil": True,
                       "textos_ausentes": sum(not row["texto_existe"] for row in verified),
                       "textos_hash_invalido": sum(row["texto_existe"] and not row["sha256_texto_ok"] for row in verified),
                       "textos_caracteres_no_coinciden": sum(row["texto_existe"] and not row["caracteres_coinciden"] for row in verified),
                       "fallos_verificacion": len(invalid)},
        "caracteres_totales": sum(row.get("caracteres_recontados", 0) for row in verified),
        "palabras_regex_totales": sum(row.get("palabras_regex", 0) for row in verified),
        "distribuciones_numericas": {key: quantiles(row.get(key) for row in verified) for key in [
            "caracteres_recontados", "palabras_regex", "lineas", "n_partes", "marcadores_NOTA_inicio_linea",
            "n_anexos_xml_docx", "n_elementos_html_retirados", "n_avisos", "n_pendientes_preparacion"]},
        "estados_extraccion": dict(Counter(row["estado_extraccion"] for row in verified)),
        "documentos_no_aptos_busqueda_segun_preparador": sum(row["apta_para_busqueda_segun_preparador"] is False for row in verified),
        "documentos_con_marcadores_NOTA": sum((row.get("marcadores_NOTA_referencias_y_bloques") or 0) > 0 for row in verified),
        "documentos_con_anexos_xml_docx": sum(row["n_anexos_xml_docx"] > 0 for row in verified),
        "avisos_documentos": dict(warnings), "pendientes_documentos": dict(pending),
        "ydata_generado": False,
        "limitaciones": ["Censo del corpus entregado/preparado; no prueba completitud normativa ni vigencia.",
                         "Palabras_regex cuenta grupos Unicode de letras/digitos; no equivale a tokens de un modelo.",
                         "Marcadores NOTA incluyen referencias y bloques sinteticos del extractor; no son notas juridicas validadas.",
                         "n_anexos_xml_docx cuenta archivos footnotes/endnotes XML, no numero de notas.",
                         "La marca apta_para_busqueda proviene de reglas del preparador, no de certificacion juridica.",
                         "Los hashes verifican fidelidad al texto canonico preparado; no validan exactitud de OCR ni texto faltante en la fuente."]
    }
    dump_json(output / "metrics.json", metrics)
    if not args.skip_ydata:
        from importlib.metadata import version
        print("Generando ydata completo de textos preparados (sin contenido textual)...", flush=True)
        make_ydata(profiles, output, "ydata_textos_preparados", "Corpus preparado completo: metricas de todos los textos")
        metrics.update(ydata_generado=True, ydata_version=version("ydata-profiling"))
        dump_json(output / "metrics.json", metrics)
    dump_json(output / "schema.json", {"textos_preparados.csv": list(verified[0]), "variables_ydata": list(profiles[0])})
    (output / "LEEME.md").write_text(
        "# Perfil de toda la extraccion preparada\n\n"
        f"Censo de {len(verified)} documentos; no es una muestra. Los IDs coinciden con el manifiesto "
        f"SHA-256 `{summary['sha256_manifest']}`.\n\n"
        "Cada texto canonico se leyo localmente para verificar SHA-256 y longitud y contar palabras regex, lineas y "
        "marcadores [NOTA]. Ningun texto completo se incluye en el HTML ni el JSON de ydata. IDs, rutas y hashes "
        "se conservan en CSV para trazabilidad y se excluyen de las variables de ydata. Correlaciones desactivadas "
        "(perfil minimo univariado).\n\n"
        "Palabras no son tokens del encoder/decoder. Marcadores NOTA no son un conteo de notas juridicas verificadas. "
        "Las alertas de OCR, paginacion y omisiones declaradas se mantienen como pendientes. Este censo describe "
        "lo extraido; no certifica vigencia ni cobertura de todo el derecho colombiano.\n",
        encoding="utf-8")
    print(json.dumps({"documentos": len(verified), "validacion": metrics["validacion"],
                      "caracteres": metrics["caracteres_totales"], "palabras_regex": metrics["palabras_regex_totales"],
                      "ydata_generado": metrics["ydata_generado"], "output": str(output)}, ensure_ascii=False), flush=True)
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
