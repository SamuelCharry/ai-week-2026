"""Censo estructural descriptivo; no segmenta ni altera el corpus canonico.

    .venv-profiling/Scripts/python.exe -X utf8 -m legalrag.preprocessing.perfil_estructura

Todos los encabezados/unidades inferidos por expresiones regulares se llaman
candidatos. Una cita de un articulo en una sentencia NO es un articulo propio.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import re
from urllib.parse import urlparse

from legalrag.preprocessing.perfil_raw import digest, dump_csv, dump_json, field_missing, make_ydata


ROOT = Path(__file__).resolve().parents[3]
VERSION = "1.1.0"
WORD = re.compile(r"\b\w+\b")
LINE = re.compile(r"[^\r\n]+")
BLANK = re.compile(r"\n[ \t]*\n(?:[ \t]*\n)*")
ARTICLE = re.compile(
    r'(?im)^[ \t]*(?P<comilla>["“«]?[ \t]*)'
    r'(?:ART[ÍI]CULO|ART\.)[ \t]+(?:TRANSITORIO[ \t]+)?'
    r'(?P<numero>\d+(?:\.\d+)*[A-Z]?[º°ªo]?|[ÚU]NICO|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO)'
    r'(?=[\s.°ºª:;–—-]|$)'
)
PREFIX = r"^[ \t]*(?:(?:[IVXLCDM]{1,10}|\d+(?:\.\d+)*)[.)\-:]?[ \t]+)?"
JUDICIAL = {
    "antecedentes": re.compile(PREFIX + r"ANTECEDENTES\b", re.I),
    "consideraciones": re.compile(PREFIX + r"CONSIDERACIONES\b", re.I),
    "fundamentos": re.compile(PREFIX + r"FUNDAMENTOS(?:[ \t]+JUR[ÍI]DICOS)?\b", re.I),
    "problema_juridico": re.compile(PREFIX + r"PROBLEMAS?[ \t]+JUR[ÍI]DICOS?\b", re.I),
    "competencia": re.compile(PREFIX + r"COMPETENCIA\b", re.I),
    "resuelve": re.compile(PREFIX + r"RESUELVE\b", re.I),
    "falla": re.compile(PREFIX + r"(?:FALLA|FALLO)\b", re.I),
    "decision": re.compile(PREFIX + r"DECISI[ÓO]N\b", re.I),
}
HIERARCHY = re.compile(r"^[ \t]*(LIBRO|T[ÍI]TULO|CAP[ÍI]TULO|SECCI[ÓO]N|PARTE)[ \t]+(?:\d|[IVXLCDM]+\b|[ÚU]NICO\b|PRELIMINAR\b|PRIMER[OA]?\b|SEGUND[OA]\b|TERCER[OA]?\b|CUART[OA]\b|QUINT[OA]\b|SEXT[OA]\b|S[ÉE]PTIM[OA]\b|OCTAV[OA]\b|NOVEN[OA]\b|D[ÉE]CIM[OA]\b)", re.I)
STATUS = re.compile(r"(?P<vigencia_anterior>vigencia[ \t]+anterior|legislaci[oó]n[ \t]+anterior|texto[ \t]+anterior)|(?P<derogacion>\bderogad[oa]s?\b)|(?P<inexequibilidad>\binexequible(?:s)?\b)|(?P<modificacion>\bmodificad[oa]s?\b)|(?P<notas_vigencia>notas?[ \t]+de[ \t]+vigencia)|(?P<concordancias>\bconcordancias\b)", re.I)


def article_candidates(text):
    accepted, rejected = [], 0
    for match in ARTICLE.finditer(text):
        end = text.find("\n", match.end())
        remainder = text[match.end():end if end >= 0 else len(text)].lstrip(" \t")
        punctuation = bool(re.match(r"^[.°ºª:;–—-]", remainder))
        ordinal = bool(re.search(r"[º°ªo]$", match.group("numero")))
        heading = bool(remainder[:1].isupper()) and not re.match(r"^(?:DE|DEL|EN|QUE|SE|Y|O|A|POR)\b", remainder, re.I)
        if not remainder or punctuation or ordinal or heading:
            accepted.append(match)
        else:
            rejected += 1
    return accepted, rejected


def qs(values):
    values = sorted(values)
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    result = {"n": len(values), "max": values[-1]}
    for name, p in [("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)]:
        point = (len(values) - 1) * p
        lo, hi = int(point), min(int(point) + 1, len(values) - 1)
        result[name] = round(values[lo] + (values[hi] - values[lo]) * (point - lo), 3)
    return result


def structural_group(kind):
    if kind in {"sentencia", "auto", "providencia", "fallo"}:
        return "jurisprudencia"
    if kind == "compendio":
        return "compendios"
    if kind == "concepto":
        return "doctrina"
    return "normativa_y_otros_actos"


def domain(url):
    return (urlparse(url or "").hostname or "").lower().removeprefix("www.")


def source_family(label):
    label = (label or "").lower()
    for fragment, sites in [
        ("senado", ["secretariasenado.gov.co"]),
        ("corte constitucional", ["corteconstitucional.gov.co"]),
        ("corte suprema", ["cortesuprema.gov.co"]),
        ("función pública", ["funcionpublica.gov.co"]),
        ("consejo de estado", ["consejodeestado.gov.co"]),
        ("colpensiones", ["colpensiones.gov.co"]),
        ("dian", ["dian.gov.co"]),
        ("relaciones exteriores", ["cancilleria.gov.co"]),
        ("cancillería", ["cancilleria.gov.co"]),
        ("industria y comercio", ["sic.gov.co"]),
        ("rama judicial", ["ramajudicial.gov.co"]),
        ("regulación de agua", ["cra.gov.co"]),
        ("ministerio de salud", ["minsalud.gov.co"]),
        ("icbf", ["icbf.gov.co"]),
    ]:
        if fragment in label:
            return sites
    return []


def analyze_text(task):
    directory, row = task
    row = dict(row)
    path = (Path(directory) / row.pop("texto_archivo")).resolve()
    if not path.is_relative_to(Path(directory)):
        raise ValueError("Texto fuera del directorio canonico")
    text = path.read_text(encoding="utf-8")
    words = [match.start() for match in WORD.finditer(text)]
    if len(text) != row["caracteres"] or len(words) != row["palabras_regex"]:
        raise ValueError(f"Texto/contadores no coinciden con censo previo: {row['doc_id']}")

    def word_count(start, end):
        return bisect_left(words, end) - bisect_left(words, start)

    line_words, line_chars, sections, hierarchy = [], [], Counter(), Counter()
    table_lines, table_blocks, was_table = 0, 0, False
    for match in LINE.finditer(text):
        line = match.group()
        if not line.strip():
            was_table = False
            continue
        line_words.append(word_count(match.start(), match.end()))
        line_chars.append(len(line))
        is_table = line.count("|") >= 2
        table_lines += is_table
        table_blocks += is_table and not was_table
        was_table = is_table
        if len(line) <= 240:
            for name, pattern in JUDICIAL.items():
                if pattern.match(line):
                    sections[name] += 1
        if len(line) <= 180:
            hit = HIERARCHY.match(line)
            if hit:
                key = hit.group(1).upper().replace("Í", "I").replace("Ó", "O")
                hierarchy[key] += 1
    starts = [0] + [m.end() for m in BLANK.finditer(text)]
    ends = [m.start() for m in BLANK.finditer(text)] + [len(text)]
    blank_blocks = [word_count(start, end) for start, end in zip(starts, ends) if text[start:end].strip()]
    articles, rejected_articles = article_candidates(text)
    article_intervals = [word_count(m.start(), articles[i + 1].start() if i + 1 < len(articles) else len(text))
                         for i, m in enumerate(articles)]
    numbers = Counter(m.group("numero").upper() for m in articles)
    status = Counter(m.lastgroup for m in STATUS.finditer(text))
    row.update(
        lineas_no_vacias=len(line_words),
        lineas_mas_512_palabras=sum(n > 512 for n in line_words),
        bloques_candidatos_por_linea_vacia=len(blank_blocks),
        candidatos_encabezado_articular=len(articles),
        candidatos_articular_descartados_por_contexto=rejected_articles,
        candidatos_articular_rotulos_distintos=len(numbers),
        candidatos_articular_rotulos_repetidos=sum(n - 1 for n in numbers.values()),
        candidatos_articular_entrecomillados=sum(bool(m.group("comilla").strip()) for m in articles),
        candidatos_articular_rotulo_jerarquico=sum("." in m.group("numero") for m in articles),
        palabras_antes_primer_candidato_articular=word_count(0, articles[0].start()) if articles else None,
        intervalos_candidatos_articular_mas_512_palabras=sum(n > 512 for n in article_intervals),
        intervalos_candidatos_articular_mas_2048_palabras=sum(n > 2048 for n in article_intervals),
        lineas_candidatas_tabla_por_barras=table_lines,
        bloques_candidatos_tabla_por_barras=table_blocks,
        marcas_texto_tachado=text.count("[TEXTO TACHADO EN LA FUENTE:"),
        familias_candidatas_seccion_judicial=len(sections),
        familias_candidatas_jerarquia=len(hierarchy),
        candidatos_encabezados_jerarquicos=sum(hierarchy.values()),
        candidatos_antecedentes_consideraciones_decision=bool(sections["antecedentes"] and sections["consideraciones"]
                                                             and (sections["resuelve"] or sections["falla"] or sections["decision"])),
    )
    for prefix, values in [("linea_palabras", line_words), ("linea_caracteres", line_chars),
                           ("bloque_linea_vacia_palabras", blank_blocks),
                           ("intervalo_candidatos_articular_palabras", article_intervals)]:
        for name, value in qs(values).items():
            row[prefix + "_" + name] = value
    for key in JUDICIAL:
        row["candidatos_seccion_" + key] = sections[key]
    for key in ["LIBRO", "TITULO", "CAPITULO", "SECCION", "PARTE"]:
        row["candidatos_jerarquia_" + key.lower()] = hierarchy[key]
    for key in ["vigencia_anterior", "derogacion", "inexequibilidad", "modificacion", "notas_vigencia", "concordancias"]:
        row["senales_textuales_" + key] = status[key]
    return row


def refine_articular(task):
    """Reutiliza metricas no afectadas y vuelve a leer el texto completo una vez."""
    directory, row = task
    row = dict(row)
    path = (Path(directory) / row.pop("texto_archivo")).resolve()
    if not path.is_relative_to(Path(directory)):
        raise ValueError("Texto fuera del corpus")
    text = path.read_text(encoding="utf-8")
    words = [m.start() for m in WORD.finditer(text)]
    if len(text) != row["caracteres"] or len(words) != row["palabras_regex"]:
        raise ValueError("Texto no coincide con censo estructural previo")
    articles, rejected = article_candidates(text)
    numbers = Counter(m.group("numero").upper() for m in articles)
    intervals = [bisect_left(words, articles[i + 1].start() if i + 1 < len(articles) else len(text)) - bisect_left(words, m.start())
                 for i, m in enumerate(articles)]
    row.update(candidatos_encabezado_articular=len(articles), candidatos_articular_descartados_por_contexto=rejected,
               candidatos_articular_rotulos_distintos=len(numbers),
               candidatos_articular_rotulos_repetidos=sum(n-1 for n in numbers.values()),
               candidatos_articular_entrecomillados=sum(bool(m.group("comilla").strip()) for m in articles),
               candidatos_articular_rotulo_jerarquico=sum("." in m.group("numero") for m in articles),
               palabras_antes_primer_candidato_articular=bisect_left(words, articles[0].start()) if articles else None,
               intervalos_candidatos_articular_mas_512_palabras=sum(n > 512 for n in intervals),
               intervalos_candidatos_articular_mas_2048_palabras=sum(n > 2048 for n in intervals))
    for name, value in qs(intervals).items():
        row["intervalo_candidatos_articular_palabras_" + name] = value
    return row


def restore_csv_value(value):
    if value == "":
        return None
    if value in {"True", "False"}:
        return value == "True"
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def add_orientation(output):
    """Tablas descriptivas para plantear experimentos, sin elegir modelos/chunks."""
    output = Path(output)
    def read(name):
        with (output / name).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    coverage = {r["campo"]: r for r in read("cobertura_metadatos.csv")}
    provenance_alerts = sum(int(r["documentos"]) for r in read("fuente_dominio.csv")
                            if r["compatibilidad_diccionario"] == "dominio_distinto_revisar_procedencia")
    guidance = [
        ("doc_id", "clave_de_trazabilidad", "Localizar documento/citas; no confundir identificacion con relevancia."),
        ("tipo", "routing_estructural", "Comparar tratamiento de normas, sentencias y compendios; comprobar casos fronterizos."),
        ("nivel", "ablacion_de_cobertura", "Comparar nucleo frente a nucleo+complementario; no equivale a vigencia o calidad certificada."),
        ("areas", "filtro_blando_multivaluado", "Usar como senal o expansion; probar recall antes de excluir otras areas. Cobertura no prueba exactitud ni procedencia por etiqueta."),
        ("temas", "no_utilizable_vacio", "No filtrar usando este campo mientras siga vacio; no confundirlo con areas."),
        ("areas_por_epigrafe", "traza_parcial_del_proceso", "Booleano de metodo; ausente no significa False ni que la materia sea incorrecta."),
        ("origen_ampliacion", "analisis_de_ingreso", "Estratificar pruebas por origen; no demuestra de donde viene cada area ni debe sustituir relevancia."),
        ("anio", "filtro_explicito_con_desconocidos", "Si la pregunta exige ano, contemplar registros sin ano y compendios de varios anos."),
        ("numero", "requiere_clave_compuesta_y_alias", "Combinar tipo/organo/ano y numeros alternativos; un numero solo no identifica de forma unica una norma o sentencia."),
        ("organo_emisor", "normalizar_alias", "Entidad emisora distinta del publicador; verificar normalizacion de nombres antes de filtros estrictos."),
        ("fuente", "requiere_procedencia_y_alias", f"{provenance_alerts} registros presentan diferencia etiqueta/dominio segun el diccionario; revisar dominios/fuentes finales, no declarar identidad erronea."),
        ("vigencia", "no_utilizable_para_vigencia_actual", "Todos por_verificar: no filtrar vigentes usando este campo."),
        ("vigencia_fuente", "senal_editorial_no_certificacion", "sin_marca no significa vigente; con_notas y cambios parciales requieren interpretar unidades/versiones."),
        ("fecha_descarga", "fecha_de_adquisicion", "Fecha del corpus, no fecha de entrada en vigor ni periodo de aplicabilidad."),
        ("sala", "cobertura_parcial", "No confundir ausencia con inexistencia de sala; medir cobertura antes de filtrar jurisprudencia."),
    ]
    filtered = []
    for field, use, proposal in guidance:
        cov = coverage.get(field)
        if cov:
            filtered.append({"campo": field, "documentos": int(cov["total"]), "sin_dato_o_vacio": int(cov["faltantes"]),
                             "cobertura_porcentaje": round(100-float(cov["porcentaje"]), 4),
                             "uso_propuesto_para_experimento": use, "condiciones": proposal})
    dump_csv(output / "filtrabilidad_campos.csv", filtered)
    docs = read("estructura_documentos.csv")
    formats = defaultdict(list)
    for row in docs:
        formats[row["formatos_originales"]].append(row)
    format_rows = []
    for label, rows in sorted(formats.items(), key=lambda item: len(item[1]), reverse=True):
        format_rows.append({"formatos_originales": label, "documentos": len(rows),
                            "p50_mediana_palabras_por_linea": qs([float(r["linea_palabras_p50"]) for r in rows])["p50"],
                            "p95_mediana_palabras_por_linea": qs([float(r["linea_palabras_p50"]) for r in rows])["p95"],
                            "documentos_un_solo_bloque_linea_vacia": sum(int(r["bloques_candidatos_por_linea_vacia"]) == 1 for r in rows),
                            "documentos_con_candidatos_articulares": sum(int(r["candidatos_encabezado_articular"]) > 0 for r in rows)})
    dump_csv(output / "estructura_por_formato.csv", format_rows)
    missing_candidates = [r for r in docs if r["grupo_estructural"] == "normativa_y_otros_actos" and int(r["candidatos_encabezado_articular"]) == 0]
    dump_csv(output / "normativa_sin_candidato_articular.csv", missing_candidates,
             ["doc_id", "tipo", "palabras_regex", "apta_para_busqueda_segun_preparador", "formatos_originales", "estado_extraccion"])
    longest_lines = sorted(docs, key=lambda r: int(r["linea_palabras_max"]), reverse=True)[:20]
    dump_csv(output / "lineas_mas_extensas.csv", longest_lines,
             ["doc_id", "tipo", "formatos_originales", "palabras_regex", "lineas_no_vacias", "linea_palabras_p50", "linea_palabras_max"])
    by_group = defaultdict(list)
    for row in docs:
        by_group[row["grupo_estructural"]].append(row)
    signals = []
    for label, rows in by_group.items():
        signals.append({"grupo": label, "documentos": len(rows),
                        "con_rotulos_articulares_repetidos": sum(int(r["candidatos_articular_rotulos_repetidos"]) > 0 for r in rows),
                        "con_intervalo_candidato_mayor_2048_palabras": sum(int(r["intervalos_candidatos_articular_mas_2048_palabras"]) > 0 for r in rows),
                        "con_lineas_mayores_512_palabras": sum(int(r["lineas_mas_512_palabras"]) > 0 for r in rows),
                        "con_senales_derogacion": sum(int(r["senales_textuales_derogacion"]) > 0 for r in rows),
                        "con_senales_inexequibilidad": sum(int(r["senales_textuales_inexequibilidad"]) > 0 for r in rows)})
    dump_csv(output / "senales_para_experimentos_por_grupo.csv", signals)
    report = output / "LEEME.md"
    note = report.read_text(encoding="utf-8").split("\n## Comparacion estructural por tipo\n")[0]
    note += "\n## Comparacion estructural por tipo\n\n"
    note += "Conteos de documentos con al menos una senal/candidato; no conteos de unidades juridicas verdaderas.\n\n"
    note += "| Tipo | Documentos | Con candidato articular | Con seccion judicial candidata | Con jerarquia candidata | Un bloque por linea vacia |\n|---|---:|---:|---:|---:|---:|\n"
    for row in read("cobertura_estructura_por_tipo.csv"):
        if row["dimension"] == "tipo":
            keys = ["valor", "documentos", "con_candidatos_articulares", "con_candidatos_seccion_judicial", "con_candidatos_jerarquia", "un_solo_bloque_por_linea_vacia"]
            note += "| " + " | ".join(row[k] for k in keys) + " |\n"
    note += "\n## Campos para filtros experimentales\n\n"
    note += "| Campo | Cobertura | Uso que conviene probar | Condiciones |\n|---|---:|---|---|\n"
    for row in filtered:
        note += f"| {row['campo']} | {row['cobertura_porcentaje']:.2f}% | {row['uso_propuesto_para_experimento']} | {row['condiciones']} |\n"
    note += "\n## Consecuencias para las pruebas\n\n"
    note += "- Normas: contrastar candidatos articulares con pertenencia a la norma, citas, jerarquia y versiones antes de convertirlos en unidades de recuperacion. No fusionar rotulos repetidos automaticamente.\n"
    note += "- Jurisprudencia: contrastar secciones/rangos y contexto completo de la providencia; un encabezado ARTICULO suele poder ser una norma citada y no una unidad propia del fallo.\n"
    note += "- Compendios: conservar documento padre y referencia a la providencia resumida; comparar unidades editoriales con ventanas, sin asumir que todo compendio es la sentencia completa.\n"
    note += "- Lineas: el formato original modifica su longitud. Dividir solo por lineas vacias puede dejar documentos enteros; usar el CSV por formato para plantear una ablacion sin afirmar parrafos originales.\n"
    note += "- Tablas/notas: preservar relacion entre encabezado, filas y notas; las barras o marcadores son senales de estructura y no justifican borrar contenido.\n"
    note += "- Versiones/vigencia: separar texto observado y etiqueta juridica; las menciones a derogacion o versiones anteriores no certifican la regla vigente.\n"
    note += f"\n`normativa_sin_candidato_articular.csv` contiene {len(missing_candidates)} normas/actos sin encabezado reconocido. No son defectos demostrados: pueden reflejar omisiones en la fuente, estructura ausente o variantes no cubiertas. `decreto_1147_1999` mantiene el pendiente de unidades normativas; su referencia cortada a otro articulo se excluye por contexto.\n"
    note += "\n`lineas_mas_extensas.csv` documenta casos donde una linea no sirve como unidad universal. `senales_para_experimentos_por_grupo.csv` permite distinguir rotulos repetidos, intervalos largos y menciones juridicas. Los rotulos repetidos pueden provenir de citas o versiones; no se deben eliminar automaticamente.\n"
    report.write_text(note, encoding="utf-8")
    return {"filtrabilidad": filtered, "por_formato": format_rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/corpus_preparado")
    parser.add_argument("--previous", type=Path, default=ROOT / "reports/perfil_corpus_preparado")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/data/raw/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/perfil_estructura")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-structure", type=Path, help="Refina solo candidatos articulares; reutiliza el resto de un censo del mismo snapshot")
    parser.add_argument("--skip-ydata", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("workers debe ser positivo")
    directory, previous, output = args.input.resolve(), args.previous.resolve(), args.output.resolve()
    if output == directory or output.is_relative_to(directory) or output == args.manifest.resolve().parent or output.is_relative_to(args.manifest.resolve().parent):
        parser.error("output debe quedar fuera del corpus preparado y raw")
    source, summary_path = directory / "documentos.jsonl", directory / "resumen.json"
    source_sha, summary_sha, manifest_sha = digest(source), digest(summary_path), digest(args.manifest.resolve())
    previous_metrics = json.loads((previous / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (previous_metrics["documentos_jsonl_sha256"] != source_sha or previous_metrics["resumen_preparador_sha256"] != summary_sha
            or previous_metrics["manifest_sha256"] != manifest_sha or summary["modo"] != "completo"):
        raise ValueError("El snapshot no coincide con el censo preparado previo")
    with (previous / "textos_preparados.csv").open(encoding="utf-8-sig", newline="") as stream:
        prior = {r["doc_id"]: r for r in csv.DictReader(stream)}
    documents = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if set(prior) != {doc["doc_id"] for doc in documents}:
        raise ValueError("IDs manifiesto/censo previo difieren")
    prepared_meta = {}
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            prepared_meta[record["doc_id"]] = {
                "n_anotaciones_editoriales_archivos": sum(bool(e.get("anotaciones_detectadas")) for e in record.get("extraccion", [])),
                "n_tablas_html_estructuradas": sum(e.get("tablas_estructuradas", 0) for e in record.get("extraccion", [])),
                "n_tablas_html_no_estructuradas": sum(e.get("tablas_en_texto", 0) for e in record.get("extraccion", [])),
                "n_partes_extraccion": len(record.get("partes", [])),
                "estado_extraccion": record["estado_extraccion"],
                "apta_para_busqueda_segun_preparador": record.get("apta_para_busqueda"),
            }
    output.mkdir(parents=True, exist_ok=True)
    dump_csv(output / "cobertura_metadatos.csv", list(field_missing(documents, "documento")))
    area_count, area_pairs, area_by_type, area_by_origin, source_domains = Counter(), Counter(), Counter(), Counter(), Counter()
    provenance, tasks = [], []
    for doc in documents:
        p = prior[doc["doc_id"]]
        areas = sorted(set(doc.get("areas", [])))
        area_count.update(areas)
        area_pairs.update(combinations(areas, 2))
        area_by_type.update((doc["tipo"], area) for area in areas)
        area_by_origin.update((doc.get("origen_ampliacion"), area) for area in areas)
        url_domain = domain(doc.get("url"))
        final_domains = sorted({domain(f.get("url_final") or f.get("url")) for f in doc.get("archivos_raw", [])})
        expected = source_family(doc.get("fuente"))
        compatible = bool(expected and any(url_domain == site or url_domain.endswith("." + site) for site in expected))
        relation = "compatible_con_diccionario" if compatible else "dominio_distinto_revisar_procedencia" if expected and url_domain else "etiqueta_sin_mapeo" if not expected else "sin_dominio"
        source_domains[(doc.get("fuente"), url_domain, relation)] += 1
        provenance.append({"doc_id": doc["doc_id"], "fuente_etiqueta": doc.get("fuente"), "dominio_url_documento": url_domain,
                           "dominios_url_final_archivos": "|".join(final_domains), "compatibilidad_diccionario": relation,
                           "url_documento": doc.get("url"), "origen_ampliacion": doc.get("origen_ampliacion")})
        row = {"doc_id": doc["doc_id"], "tipo": doc.get("tipo"), "grupo_estructural": structural_group(doc.get("tipo")),
               "fuente": doc.get("fuente"), "dominio_url_documento": url_domain, "fuente_dominio_compatibilidad": relation,
               "anio": doc.get("anio"), "nivel": doc.get("nivel"), "origen_ampliacion": doc.get("origen_ampliacion"),
               "vigencia": doc.get("vigencia"), "vigencia_fuente": doc.get("vigencia_fuente"),
               "areas": "|".join(areas), "areas_n": len(areas), "temas_n": len(doc.get("temas", [])),
               "areas_por_epigrafe": doc.get("areas_por_epigrafe"),
               "numero_presente": doc.get("numero") not in {None, ""},
               "formatos_originales": p["formatos_originales"], "texto_archivo": p["texto_archivo"],
               "caracteres": int(p["caracteres_recontados"]), "palabras_regex": int(p["palabras_regex"]),
               "marcadores_NOTA_referencias_y_bloques": int(p["marcadores_NOTA_referencias_y_bloques"]),
               "marcadores_NOTA_inicio_linea": int(p["marcadores_NOTA_inicio_linea"]),
               "n_anexos_xml_docx": int(p["n_anexos_xml_docx"]), **prepared_meta[doc["doc_id"]]}
        tasks.append((str(directory), row))
    dump_csv(output / "areas_valores.csv", [{"area": k, "documentos": v, "porcentaje_documentos": round(100*v/len(documents), 3)} for k, v in area_count.most_common()])
    dump_csv(output / "areas_solapamientos.csv", [{"area_a": k[0], "area_b": k[1], "documentos": v} for k, v in area_pairs.most_common()])
    dump_csv(output / "areas_por_tipo.csv", [{"tipo": k[0], "area": k[1], "documentos": v} for k, v in area_by_type.most_common()])
    dump_csv(output / "areas_por_origen.csv", [{"origen_ampliacion": k[0], "area": k[1], "documentos": v} for k, v in area_by_origin.most_common()])
    dump_csv(output / "procedencia_documentos.csv", provenance)
    dump_csv(output / "fuente_dominio.csv", [{"fuente_etiqueta": k[0], "dominio_url_documento": k[1], "compatibilidad_diccionario": k[2], "documentos": v} for k, v in source_domains.most_common()])
    analyze = analyze_text
    reuse_info = None
    if args.reuse_structure:
        old_metrics_path = args.reuse_structure / "metrics.json"
        old_table_path = args.reuse_structure / "estructura_documentos.csv"
        old_metrics = json.loads(old_metrics_path.read_text(encoding="utf-8"))
        if old_metrics["manifest_sha256"] != manifest_sha or old_metrics["documentos_jsonl_sha256"] != source_sha or old_metrics["resumen_preparador_sha256"] != summary_sha:
            raise ValueError("El censo reutilizado no corresponde al snapshot actual")
        with old_table_path.open(encoding="utf-8-sig", newline="") as stream:
            old_rows = [{k: restore_csv_value(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        if {r["doc_id"] for r in old_rows} != set(prior):
            raise ValueError("IDs del censo reutilizado difieren")
        tasks = [(str(directory), {**r, "texto_archivo": prior[r["doc_id"]]["texto_archivo"]}) for r in old_rows]
        analyze = refine_articular
        reuse_info = {"origen": str(args.reuse_structure.resolve()), "version_anterior": old_metrics["version"],
                      "tabla_anterior_sha256": digest(old_table_path), "metricas_anteriores_sha256": digest(old_metrics_path),
                      "alcance": "Solo candidatos articulares e intervalos recalculados; demas metricas reutilizadas del mismo censo"}
    print(f"Analizando estructura completa de {len(tasks)} textos; {args.workers} procesos, sin truncar...", flush=True)
    result = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(analyze, tasks, chunksize=8), 1):
            result.append(row)
            if i % 1000 == 0 or i == len(tasks):
                print(f"Estructura {i}/{len(tasks)}", flush=True)
    if digest(source) != source_sha or digest(summary_path) != summary_sha or digest(args.manifest.resolve()) != manifest_sha:
        raise RuntimeError("Snapshot cambio durante el censo estructural")
    dump_csv(output / "estructura_documentos.csv", result)
    group_rows, quantitative_rows = [], []
    for dimension in ["tipo", "grupo_estructural"]:
        groups = defaultdict(list)
        for row in result:
            groups[row[dimension]].append(row)
        for label, rows in sorted(groups.items(), key=lambda item: len(item[1]), reverse=True):
            n = len(rows)
            group_rows.append({
                "dimension": dimension, "valor": label, "documentos": n,
                "con_candidatos_articulares": sum(r["candidatos_encabezado_articular"] > 0 for r in rows),
                "candidatos_articulares_total": sum(r["candidatos_encabezado_articular"] for r in rows),
                "con_candidatos_jerarquia": sum(r["candidatos_encabezados_jerarquicos"] > 0 for r in rows),
                "con_candidatos_seccion_judicial": sum(r["familias_candidatas_seccion_judicial"] > 0 for r in rows),
                "con_candidatos_antecedentes_consideraciones_decision": sum(r["candidatos_antecedentes_consideraciones_decision"] for r in rows),
                "con_senales_version_anterior": sum(r["senales_textuales_vigencia_anterior"] > 0 for r in rows),
                "con_marcas_tachado": sum(r["marcas_texto_tachado"] > 0 for r in rows),
                "con_tablas_html_estructuradas": sum(r["n_tablas_html_estructuradas"] > 0 for r in rows),
                "con_lineas_candidatas_tabla": sum(r["lineas_candidatas_tabla_por_barras"] > 0 for r in rows),
                "con_marcadores_NOTA": sum(r["marcadores_NOTA_referencias_y_bloques"] > 0 for r in rows),
                "un_solo_bloque_por_linea_vacia": sum(r["bloques_candidatos_por_linea_vacia"] == 1 for r in rows),
                "areas_multiples": sum(r["areas_n"] > 1 for r in rows),
                "p50_palabras_documento": qs([r["palabras_regex"] for r in rows])["p50"],
                "p95_palabras_documento": qs([r["palabras_regex"] for r in rows])["p95"],
            })
            for variable in ["palabras_regex", "lineas_no_vacias", "linea_palabras_p50", "linea_palabras_p95", "linea_palabras_max",
                             "bloques_candidatos_por_linea_vacia", "candidatos_encabezado_articular", "candidatos_articular_rotulos_repetidos",
                             "intervalo_candidatos_articular_palabras_p50", "intervalo_candidatos_articular_palabras_p95", "intervalo_candidatos_articular_palabras_max",
                             "familias_candidatas_seccion_judicial", "candidatos_encabezados_jerarquicos"]:
                quantitative_rows.append({"dimension": dimension, "valor": label, "variable": variable,
                                          **qs([r[variable] for r in rows if r[variable] is not None])})
    dump_csv(output / "cobertura_estructura_por_tipo.csv", group_rows)
    dump_csv(output / "cuantiles_estructura_por_tipo.csv", quantitative_rows)
    metadata_flags = {
        "areas_no_vacias": sum(bool(d.get("areas")) for d in documents), "temas_no_vacios": sum(bool(d.get("temas")) for d in documents),
        "areas_multiples": sum(len(d.get("areas", [])) > 1 for d in documents),
        "areas_por_epigrafe_true": sum(d.get("areas_por_epigrafe") is True for d in documents),
        "areas_por_epigrafe_false": sum(d.get("areas_por_epigrafe") is False for d in documents),
        "areas_por_epigrafe_ausente": sum("areas_por_epigrafe" not in d for d in documents),
        "origen_grafo_normativo": sum(d.get("origen_ampliacion") == "grafo_normativo" for d in documents),
        "fuente_dominio_distinto_segun_diccionario": sum(p["compatibilidad_diccionario"] == "dominio_distinto_revisar_procedencia" for p in provenance),
    }
    metrics = {"version": VERSION, "fecha_utc": datetime.now(timezone.utc).isoformat(), "documentos": len(result),
               "modo": "censo_estructural_completo_sin_truncamiento", "manifest_sha256": manifest_sha,
               "documentos_jsonl_sha256": source_sha, "resumen_preparador_sha256": summary_sha,
               "parser_y_preparador_sha256": summary["sha256_parser_y_preparador"],
               "snapshot_sin_cambios_durante_perfil": True, "reutiliza_integridad_verificada_en": str(previous),
               "conteos_palabras_y_caracteres_coinciden_con_censo_previo": True,
               "reutilizacion_estructural": reuse_info,
               "metadatos": metadata_flags, "areas_valores": dict(area_count), "cobertura_por_grupo_y_tipo": group_rows,
               "ydata_generado": False,
               "limites": ["Candidatos articulares y sus intervalos pueden incluir citas, notas, reformas y versiones; no son articulos juridicos propios verificados.",
                          "En sentencias/autos, los encabezados articulo normalmente pueden pertenecer a normas citadas; no se asigna autoria normativa.",
                          "normalizar elimina lineas vacias; lineas/bloques son proxies de extraccion, no parrafos originales universales.",
                          "Reglas conservadoras ancladas a lineas pueden omitir encabezados pegados, OCR espaciado u otras variantes.",
                          "Ausencia de candidatos no demuestra defecto del regex ni falta de contenido; decreto_1147_1999 carece de encabezados en la fuente.",
                          "Senales de derogacion/inexequibilidad/version son menciones textuales, no estado juridico certificado y pueden aparecer negadas o citadas.",
                          "Areas es un campo multivaluado existente; temas es otro campo. No se atribuye una procedencia por etiqueta cuando falta trazabilidad.",
                          "areas_por_epigrafe es booleano; ausente no significa False. origen_ampliacion identifica ingreso, no demuestra origen de cada area.",
                          "Diferencias fuente/dominio son alertas de alias/procedencia segun diccionario, no errores automaticos de identidad.",
                          "Palabras regex no son tokens. n_partes_extraccion no son chunks definitivos."]}
    dump_json(output / "metrics.json", metrics)
    if not args.skip_ydata:
        print("Generando ydata estructural completo...", flush=True)
        profile_rows = [{k: v for k, v in row.items() if k != "doc_id"} for row in result]
        make_ydata(profile_rows, output, "ydata_estructura", "Corpus completo: candidatos estructurales y metadatos (no articulos verificados)")
        metrics["ydata_generado"] = True
        from importlib.metadata import version
        metrics["ydata_version"] = version("ydata-profiling")
        dump_json(output / "metrics.json", metrics)
    dump_json(output / "metodologia.json", {"version": VERSION, "regex_articular": ARTICLE.pattern,
                                           "condiciones_articulares": "Despues del numero: fin de linea, puntuacion/ordinal o titulo con inicial mayuscula excluyendo DE/DEL/EN/QUE/SE/Y/O/A/POR. Evita contar referencias en prosa cortadas al inicio de linea; puede omitir variantes reales.",
                                           "regex_secciones_judiciales": {k: v.pattern for k, v in JUDICIAL.items()},
                                           "max_caracteres_linea_seccion_judicial": 240, "max_caracteres_linea_jerarquia": 180,
                                           "regex_jerarquia": HIERARCHY.pattern, "regex_senales_version_vigencia": STATUS.pattern,
                                           "regex_palabras": WORD.pattern, "intervalos": "desde inicio de un candidato hasta siguiente; ultimo hasta EOF; excluye preambulo previo al primero",
                                           "cuantiles_por_tipo": "cuantiles entre documentos de las metricas por documento; p95 de p95 NO es p95 de todos los intervalos juntos",
                                           "reglas_fuente_dominio": "diccionario heuristico source_family en script; comprobar URL final y etiqueta antes de filtrar estrictamente",
                                           "comando": ".venv-profiling/Scripts/python.exe -X utf8 -m legalrag.preprocessing.perfil_estructura --workers 4",
                                           "limites": metrics["limites"]})
    dump_json(output / "schema.json", {"estructura_documentos.csv": list(result[0]), "cuantiles_estructura_por_tipo.csv": list(quantitative_rows[0])})
    note = "# Censo estructural completo\n\n"
    note += f"{len(result)} documentos completos, sin truncar y sin modificar el texto canonico. Manifest `{manifest_sha}`.\n\n"
    note += "`ydata_estructura.html` y su JSON son ydata-profiling real, modo minimal (sin correlaciones). Los CSV conservan IDs para trazabilidad; el HTML no incluye textos ni IDs.\n\n"
    note += "## Como leer las medidas\n\n"
    note += "\n".join("- " + limitation for limitation in metrics["limites"]) + "\n\n"
    note += "Los intervalos entre encabezados candidatos se miden en palabras regex, incluyendo el ultimo candidato hasta EOF. Los cuantiles por tipo resumen metricas de documentos; no deben interpretarse como cuantiles globales de todos los articulos.\n\n"
    note += "## Metadatos y filtros\n\n"
    note += "`areas_valores.csv`, `areas_solapamientos.csv`, `areas_por_tipo.csv` y `areas_por_origen.csv` describen areas multivaluadas. `cobertura_metadatos.csv` distingue campo ausente/null/lista vacia. `areas_por_epigrafe` es booleano y su ausencia no significa False.\n\n"
    note += "`procedencia_documentos.csv` y `fuente_dominio.csv` separan etiqueta de fuente, dominio URL y dominios finales de los archivos. Las diferencias requieren comprobar procedencia y alias, no corregir identidades automaticamente.\n\n"
    note += "No se ha decidido chunking, routing definitivo ni modelos. Estructura y cobertura sirven para disenar experimentos por tipo conservando evidencia, citas, notas y versiones.\n"
    (output / "LEEME.md").write_text(note, encoding="utf-8")
    add_orientation(output)
    print(json.dumps({"documentos": len(result), "snapshot_estable": True, "metadatos": metadata_flags,
                      "ydata_generado": metrics["ydata_generado"], "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
