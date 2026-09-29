"""Reconcilia alertas heredadas con una preparación completa, sin modificar el corpus.

Valida en streaming el inventario y los textos canónicos. No reconstruye el grafo
de citas, no certifica vigencia y no convierte una referencia ausente en una
norma cuya existencia o identidad haya sido comprobada.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


OMISION = re.compile(
    r'esta compilaci[oó]n no incluye el texto|texto de esta norma no se sistematizar[aá]', re.I)
RELACIONADOS = {
    'decreto_1497_1993': 'sentencia_cc_c416_1993',
    'decreto_1940_1992': 'sentencia_cc_c069_1993',
    'decreto_543_1993': 'sentencia_cc_c261_1993',
}
IDENTIDAD_CITA_PENDIENTE = {
    'decreto_2963_2010': 'Indicio de número citado erróneamente; contrastar con Decreto 2693/2010. No se reemplazó la referencia.',
    'decreto_2067_2000': 'Indicio de año citado erróneamente; contrastar con Decreto 2067/1991. No se reemplazó la referencia.',
}
PRIORIDAD_PREPARACION = {
    'error_extraccion': 0, 'fuente_declara_omision_del_texto': 0,
    'fuente_restringida_por_defecto_confirmado': 0,
    'revisar_calidad_OCR': 1, 'bytes_no_decodificables_en_charset_declarado': 1,
    'codificacion_residual_por_revisar': 1,
    'paginacion_derivado_no_verificada': 2, 'paginas_sin_texto_suficiente': 2,
    'texto_corto_por_revisar': 2,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        yield from csv.DictReader(stream)


def key(value):
    """Normaliza únicamente alias de inventario; no corrige identidades jurídicas."""
    value = str(value).lower().removeprefix('co_')
    match = re.fullmatch(r'(sentencia_(?:cc|csj)_[a-z]+)0*(\d+)_(\d{4})', value)
    if match:
        return f'{match[1]}{int(match[2])}_{match[3]}'
    return value


def inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Ruta fuera de la preparación: {relative}')
    return path


def atomic_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def seed_key(detail):
    match = re.search(r'Sentencia\s+(SC|SL|T)[- ]?(\d+)\s+de\s+(\d{4})', detail, re.I)
    if not match:
        return None
    family, number, year = match.groups()
    court = 'cc' if family.upper() == 'T' else 'csj'
    return key(f'sentencia_{court}_{family}{number}_{year}')


def source_evidence(record):
    return {k: record.get(k) for k in ('url', 'fecha_consulta', 'texto_archivo', 'sha256_texto',
                                       'integridad_texto_verificada', 'caracteres')}


def verify_prepared(manifest, prepared, summary):
    records, failures, selected_texts = {}, [], {}
    counters = Counter()
    pending = Counter()
    states = Counter()
    text_hash = hashlib.sha256()
    manifest_by_id = {d['doc_id']: d for d in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError('El manifiesto contiene doc_id duplicados.')
    targets = set(RELACIONADOS) | set(RELACIONADOS.values())
    with (prepared / 'documentos.jsonl').open('rb') as stream:
        for line_number, line in enumerate(stream, 1):
            text_hash.update(line)
            record = json.loads(line)
            doc_id = record['doc_id']
            if doc_id in records:
                failures.append({'doc_id': doc_id, 'error': 'doc_id_duplicado', 'linea': line_number})
            original = manifest_by_id.get(doc_id)
            if original is None:
                failures.append({'doc_id': doc_id, 'error': 'registro_no_incluido_en_manifest'})
            elif record.get('archivos_raw') != original.get('archivos_raw'):
                failures.append({'doc_id': doc_id, 'error': 'procedencia_difiere_del_manifest'})
            states[record.get('estado_extraccion', 'sin_estado')] += 1
            counters['registros'] += 1
            counters['errores_extraccion'] += record.get('estado_extraccion') == 'error'
            counters['caracteres_declarados'] += record.get('caracteres', 0)
            counters['documentos_no_aptos'] += record.get('apta_para_busqueda') is False
            counters['registros_integridad_raw_declarada'] += record.get('integridad_archivos_verificada') is True
            pending.update(record.get('pendientes_preparacion', []))
            verified, text = False, ''
            if record.get('texto_archivo'):
                try:
                    data = inside(prepared, record['texto_archivo']).read_bytes()
                    text = data.decode('utf-8')
                    if sha(data) != record.get('sha256_texto'):
                        raise ValueError('sha256_texto_no_coincide')
                    if len(text) != record.get('caracteres'):
                        raise ValueError('longitud_texto_no_coincide')
                    for part in record.get('partes', []):
                        if not (0 <= part['inicio'] <= part['fin'] <= len(text)):
                            raise ValueError('offset_fuera_del_texto')
                    verified = True
                    counters['textos_verificados'] += 1
                except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
                    failures.append({'doc_id': doc_id, 'error': str(exc)})
            elif record.get('estado_extraccion') != 'error':
                failures.append({'doc_id': doc_id, 'error': 'registro_sin_texto_sin_error_de_extraccion'})
            if doc_id in targets or doc_id.startswith('sentencia_csj_rad'):
                selected_texts[doc_id] = text
            compact_fields = ('doc_id', 'numero', 'anio', 'url', 'fecha_consulta', 'texto_archivo',
                              'sha256_texto', 'caracteres', 'estado_extraccion', 'apta_para_busqueda',
                              'pendientes_preparacion', 'evidencia_omision', 'error', 'rechazos_fuente',
                              'numero_original_manifest', 'correcciones_metadatos')
            records[doc_id] = {k: record.get(k) for k in compact_fields}
            records[doc_id]['integridad_texto_verificada'] = verified
    missing = sorted(set(manifest_by_id) - set(records))
    for doc_id in missing:
        failures.append({'doc_id': doc_id, 'error': 'documento_manifest_no_preparado'})
    comparisons = {'documentos_procesados': counters['registros'],
                   'documentos_manifest': len(manifest), 'errores': counters['errores_extraccion'],
                   'textos_guardados': counters['textos_verificados'],
                   'caracteres_totales': counters['caracteres_declarados'],
                   'documentos_no_aptos_busqueda': counters['documentos_no_aptos']}
    for field, actual in comparisons.items():
        if summary.get(field) != actual:
            failures.append({'error': 'resumen_no_coincide', 'campo': field,
                             'declarado': summary.get(field), 'observado': actual})
    if summary.get('pendientes_por_tipo') != dict(pending):
        failures.append({'error': 'resumen_pendientes_no_coincide'})
    return records, selected_texts, {
        'contadores': dict(counters), 'estados_extraccion': dict(states),
        'pendientes_por_tipo': dict(pending), 'fallos_integridad': failures,
        'sha256_documentos_jsonl': text_hash.hexdigest(),
        'alcance': 'SHA-256 y longitud de cada texto, offsets de partes, IDs y procedencia. '
                   'Los originales no se releen: su comprobación corresponde al preparador.'}


def reconcile(legacy, graph, exclusions, records, texts, prepared):
    aliases = defaultdict(list)
    for doc_id in records:
        aliases[key(doc_id)].append(doc_id)
    graph_by_key = {key(row['clave']): row for row in graph}
    excluded = {key(row['clave']): row for row in exclusions}
    findings = []
    for index, old in enumerate(legacy, 1):
        check = old['comprobacion']
        target = key(old['doc_id']) if old['doc_id'] else seed_key(old['detalle'])
        matches = aliases.get(target, [])
        record = records[matches[0]] if len(matches) == 1 else None
        row = {'id': f'heredada_{index:05d}', 'origen': 'auditoria_heredada',
               'comprobacion': check, 'doc_id': old['doc_id'], 'clave': target,
               'severidad_heredada': old['severidad'], 'estado': 'heredada_no_reevaluada',
               'prioridad': 3, 'detalle_heredado': old['detalle'],
               'accion': 'Reevaluar sobre el texto actual antes de afirmar que persiste.', 'evidencia': {}}
        if check in {'grafo_abierto', 'citada_por_el_corpus_y_ausente', 'seed_target_ausente'}:
            edge = graph_by_key.get(target, {})
            row['documentos_que_citan_heredado'] = int(edge.get('documentos_que_citan') or 0)
            row['prioridad'] = 1 if check in {'grafo_abierto', 'seed_target_ausente'} else 2
            if record:
                row['doc_id'] = record['doc_id']
                row['evidencia'] = source_evidence(record)
                if record['integridad_texto_verificada'] and record.get('estado_extraccion') != 'error':
                    row['estado'] = 'resuelta_presencia_en_inventario_y_texto'
                    row['prioridad'] = 4
                    row['accion'] = 'Referencia incorporada. La presencia no certifica vigencia ni completitud jurídica.'
                else:
                    row['estado'] = 'presente_con_texto_no_verificado'
                    row['prioridad'] = 0
                    row['accion'] = 'Corregir la extracción/integridad del documento ya inventariado.'
            elif len(matches) > 1:
                row['estado'] = 'alias_ambiguo_requiere_revision'
                row['evidencia'] = {'doc_ids': matches}
            elif target in excluded:
                row['estado'] = 'ausente_con_exclusion_documentada_heredada'
                row['evidencia'] = excluded[target]
                row['accion'] = 'Mantener trazabilidad de la exclusión; no implica que la norma no exista.'
                row['prioridad'] = 2
            else:
                row['estado'] = 'referencia_aun_ausente_en_inventario'
                row['evidencia'] = {'grafo_heredado': edge}
                row['accion'] = 'Verificar identidad, alcance y texto completo oficial antes de incorporar.'
            if target in IDENTIDAD_CITA_PENDIENTE and not record:
                row['accion'] = IDENTIDAD_CITA_PENDIENTE[target]
            if check == 'seed_target_ausente' and target == 'sentencia_csj_sl1972_2025' and not record:
                row['accion'] = 'Pendiente sentencia íntegra; un edicto no sustituye el texto judicial.'
        elif check == 'no_nombra_la_norma' and record:
            number = str(record.get('numero', ''))
            text = texts.get(record['doc_id'], '')
            heading = re.search(rf'(?im)^\s*Proceso\s+(?:No\.?|n[°ºo])\s*{re.escape(number)}\b', text[:2500])
            corrected = any(c.get('campo') == 'numero' and str(c.get('despues')) == number
                            for c in record.get('correcciones_metadatos') or [])
            row['evidencia'] = {**source_evidence(record), 'numero_actual': number,
                                'numero_anterior': record.get('numero_original_manifest'),
                                'encabezado': heading[0].strip() if heading else None,
                                'correccion_registrada': corrected}
            if heading and corrected and record['integridad_texto_verificada']:
                row.update(estado='resuelta_identidad_verificada_en_encabezado', prioridad=4,
                           accion='El número de proceso coincide; se corrigió la fecha del nombre de archivo incorporada al número.')
            else:
                row.update(estado='identidad_judicial_pendiente', prioridad=1,
                           accion='Verificar número contra el encabezado de la sentencia.')
        elif check == 'norma_sin_articulos' and record:
            from scripts.corpus.ingesta import candidatos_articulo
            text = texts.get(record['doc_id'])
            if text is None and record.get('texto_archivo'):
                text = inside(prepared, record['texto_archivo']).read_text(encoding='utf-8')
            text = text or ''
            omission = OMISION.search(text)
            candidates = candidatos_articulo(text)
            row['evidencia'] = {**source_evidence(record), 'encabezados_candidatos': len(candidates),
                                'muestra_encabezados': [c['encabezado'] for c in candidates[:12]]}
            if omission:
                row.update(estado='confirmada_fuente_declara_omision', prioridad=0,
                           accion='Localizar texto normativo íntegro; conservar la ficha pero excluirla como evidencia del articulado.')
                row['evidencia']['literal_omision'] = text[max(0, omission.start()-60):omission.end()+140]
                related_id = RELACIONADOS.get(record['doc_id'])
                if related_id in records:
                    related_text = texts.get(related_id, '')
                    number = re.escape(str(record['numero']))
                    match = re.search(rf'decreto[^\n]{{0,50}}\b{number}\b[^\n]{{0,60}}', related_text, re.I)
                    compact_text = re.sub(r'\s+', ' ', related_text)
                    transcription = re.search(
                        rf'(?:TEXTO DE LAS NORMAS BAJO REVISION|NORMA QUE SE REVISA\.|II\. TEXTO)'
                        rf'.{{0,450}}?DECRETO\s+NUMERO\s+0*{number}\b.{{0,180}}', compact_text, re.I)
                    row['evidencia']['fuente_judicial_relacionada'] = {
                        'doc_id': related_id, **source_evidence(records[related_id]),
                        'mencion_verificada': bool(match),
                        'muestra_mencion': match[0] if match else None,
                        'encabezado_transcripcion_verificado': bool(transcription),
                        'muestra_encabezado_transcripcion': transcription[0] if transcription else None,
                        'alcance': 'Sentencia relacionada con sección de transcripción identificada cuando consta en la evidencia; '
                                   'no se certifica aquí integridad ni se sustituye la norma por fragmentos.'}
            elif candidates and record['integridad_texto_verificada']:
                row.update(estado='resuelta_alerta_cero_encabezados', prioridad=4,
                           accion='Hay encabezados detectables con el parser actual. Validar atribución y completitud al segmentar; no son artículos certificados.')
            else:
                row.update(estado='estructura_articular_pendiente', prioridad=1,
                           accion='Hay texto pero no encabezados detectables. Comparar estructura con original sin inferir pérdida total de contenido.')
        findings.append(row)
    for doc_id, record in records.items():
        problems = list(record.get('pendientes_preparacion') or [])
        if record.get('rechazos_fuente'):
            problems.append('fuente_restringida_por_defecto_confirmado')
        for problem in problems:
            findings.append({
                'id': f'actual_{doc_id}_{problem}', 'origen': 'preparacion_actual',
                'comprobacion': problem, 'doc_id': doc_id, 'clave': key(doc_id),
                'severidad_heredada': None, 'estado': 'pendiente_preparacion_actual',
                'prioridad': PRIORIDAD_PREPARACION.get(problem, 2),
                'accion': 'Revisar fuente y extracción. Una alerta heurística no prueba por sí sola contenido ausente.',
                'evidencia': {**source_evidence(record), 'error': record.get('error'),
                              'apta_para_busqueda': record.get('apta_para_busqueda'),
                              'omision': record.get('evidencia_omision'),
                              'rechazos_fuente': record.get('rechazos_fuente')}})
    return findings


def aggregate(findings):
    by_check = defaultdict(Counter)
    for row in findings:
        by_check[row['comprobacion']][row['estado']] += 1
    return {check: dict(counts) for check, counts in sorted(by_check.items())}


def markdown(report):
    integrity = report['validacion']
    counts = integrity['contadores']
    rows = report['hallazgos']
    grouped = report['reconciliacion_por_comprobacion']
    lines = ['# Auditoría reconciliada del corpus preparado', '',
             f"Fecha UTC: {report['fecha_utc']}. Inventario: **{counts['registros']:,} documentos**; "
             f"textos verificados: **{counts['textos_verificados']:,}**; errores de extracción: "
             f"**{counts['errores_extraccion']}**; fallos de integridad: **{len(integrity['fallos_integridad'])}**.", '',
             'La extracción está completa cuando el resumen y los registros coinciden. Esto no certifica '
             'que el universo jurídico esté completo, que cada texto sea íntegro, ni su vigencia.', '',
             '## Alertas heredadas reevaluadas', '',
             '| Comprobación | Estado actual | Filas |', '|---|---|---:|']
    for check in ['grafo_abierto', 'seed_target_ausente', 'no_nombra_la_norma', 'norma_sin_articulos']:
        for state, count in grouped.get(check, {}).items():
            lines.append(f'| {check} | {state} | {count} |')
    lines += ['', 'Las cantidades anteriores son filas de alerta: un documento puede tener varias. '
              'Se cotejó el grafo heredado contra el inventario actual; no se reconstruyeron aristas ni '
              'se incorporaron las citas nuevas de las adiciones. Referencia ausente significa que '
              'falta esa clave en el inventario, no que su identidad jurídica esté comprobada.', '',
              '## Pendientes con prioridad', '']
    for row in rows:
        if row['comprobacion'] in {'seed_target_ausente', 'norma_sin_articulos'} and row['prioridad'] < 4:
            lines.append(f"- **{row['clave']}** — {row['estado']}. {row['accion']}")
            related = row['evidencia'].get('fuente_judicial_relacionada')
            if related:
                lines.append(f"  Fuente relacionada: [{related['doc_id']}]({related['url']}); "
                             'no sustituye automáticamente el articulado íntegro.')
    lines += ['', 'Preparación actual:', '', '| Alerta | Documentos |', '|---|---:|']
    for problem, count in sorted(integrity['pendientes_por_tipo'].items()):
        lines.append(f'| {problem} | {count} |')
    restricted = grouped.get('fuente_restringida_por_defecto_confirmado', {})
    if restricted:
        lines.append(f"| fuente_restringida_por_defecto_confirmado | {sum(restricted.values())} |")
    lines += ['', f"**{counts['documentos_no_aptos']} documentos no aptos para búsqueda** según el preparador. "
              'Los avisos de páginas con poco texto, texto corto y paginación son señales para revisión; '
              'no equivalen a OCR fallido confirmado.', '',
              '## Referencias prioritarias aún ausentes', '',
              'Ordenadas por documentos citantes en el grafo heredado. Verificar identidad antes de descargar.', '',
              '| Clave | Citantes heredados | Acción |', '|---|---:|---|']
    gaps = [r for r in rows if r['comprobacion'] == 'grafo_abierto'
            and r['estado'] == 'referencia_aun_ausente_en_inventario']
    for row in sorted(gaps, key=lambda r: (-r.get('documentos_que_citan_heredado', 0), r['clave']))[:20]:
        lines.append(f"| {row['clave']} | {row.get('documentos_que_citan_heredado', 0)} | {row['accion']} |")
    unreviewed = sum(r['estado'] == 'heredada_no_reevaluada' for r in rows)
    lines += ['', '## Límites y reproducción', '',
              f'Quedan **{unreviewed:,} filas heredadas no reevaluadas**. Se conservan explícitamente: '
              'no deben contarse como defectos actuales confirmados ni como correcciones realizadas. '
              'Incluyen cierre judicial, huecos numéricos y las antiguas heurísticas de longitud/OCR. '
              'Los avisos actuales de preparación se registran por separado.', '',
              'La etiqueta heredada de pie editorial no distingue navegación de notas de reforma '
              'o concordancia conservadas deliberadamente. Las menciones a notas de Avance Jurídico '
              'en Ley 446/1998 y Ley 160/1994 no se declaran aquí defectos actuales confirmados.', '',
              'Se verificaron SHA-256, longitudes y rangos de offsets de cada texto; la validación de '
              'originales corresponde al preparador. No se modificaron originales, manifiesto, '
              'textos, índices ni modelos. No se consultaron respuestas del banco.', '',
              '```powershell', 'python -m scripts.corpus.auditar_preparacion', '```', '',
              'El JSON contiene evidencia, procedencia y cada alerta. El CSV permite filtrar por '
              'prioridad, estado, origen y clave. Una exclusión documentada conserva el resultado '
              'de una búsqueda anterior; no prueba inexistencia permanente de una fuente.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/data/raw'))
    parser.add_argument('--prepared', type=Path, default=Path('data/processed/corpus_preparado'))
    parser.add_argument('--legacy', type=Path, default=Path('data/reports/auditoria_corpus.csv'))
    parser.add_argument('--exclusions', type=Path, default=Path('data/configs/corpus_exclusiones.json'))
    parser.add_argument('--output-prefix', type=Path,
                        default=Path('reports/correcciones_corpus/auditoria_actualizada'))
    args = parser.parse_args(argv)
    raw, prepared = args.raw.resolve(), args.prepared.resolve()
    prefix = args.output_prefix.resolve()
    if prefix.is_relative_to(raw) or prefix.is_relative_to(prepared):
        parser.error('El informe debe escribirse fuera de raw y de la preparación.')
    summary_path = prepared / 'resumen.json'
    if not summary_path.exists() or not (prepared / 'documentos.jsonl').exists():
        parser.error('Aún no existe una preparación terminada; esperar resumen.json y documentos.jsonl.')
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    manifest_bytes = (raw / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if summary.get('modo') != 'completo':
        parser.error('La preparación es una selección: no permite auditoría de corpus completo.')
    if summary.get('sha256_manifest') != sha(manifest_bytes):
        parser.error('La preparación no corresponde al manifiesto actual; regenerarla primero.')
    if summary.get('documentos_procesados') != len(manifest):
        parser.error('La preparación no ha procesado todo el manifiesto.')
    parser_sha = sha(Path(__file__).with_name('ingesta.py').read_bytes()
                     + Path(__file__).with_name('preparar_corpus.py').read_bytes())
    if summary.get('sha256_parser_y_preparador') != parser_sha:
        parser.error('El extractor/preparador cambió desde el snapshot; completar de nuevo la preparación.')
    graph_path = raw / 'grafo_faltantes.csv'
    records, texts, validation = verify_prepared(manifest, prepared, summary)
    findings = reconcile(list(csv_rows(args.legacy)), list(csv_rows(graph_path)),
                         read_json(args.exclusions)['exclusiones'], records, texts, prepared)
    if summary_path.read_bytes() != summary_bytes or (raw / 'manifest.json').read_bytes() != manifest_bytes:
        raise RuntimeError('El snapshot cambió durante la auditoría; no se publicará un informe inconsistente.')
    findings.sort(key=lambda r: (r['prioridad'], r['comprobacion'],
                                -r.get('documentos_que_citan_heredado', 0), r.get('clave') or ''))
    report = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'estado': ('fallos_integridad' if validation['fallos_integridad'] else
                   'errores_extraccion' if validation['contadores'].get('errores_extraccion') else
                   'integridad_verificada'),
        'snapshot': {'sha256_manifest': sha(manifest_bytes), 'sha256_resumen': sha(summary_bytes),
                     'sha256_script_auditoria': sha(Path(__file__).read_bytes()),
                     'sha256_auditoria_heredada': sha(args.legacy.read_bytes()),
                     'sha256_grafo_heredado': sha(graph_path.read_bytes()),
                     'sha256_exclusiones': sha(args.exclusions.read_bytes())},
        'resumen_preparacion': summary, 'validacion': validation,
        'reconciliacion_por_comprobacion': aggregate(findings),
        'limites': ['No reconstruye grafo ni certifica completitud o vigencia jurídica.',
                   'No descarga nuevas fuentes ni utiliza respuestas del banco.',
                   'Ausencia en inventario no confirma identidad/existencia de una norma.',
                   'Las alertas heredadas no reevaluadas no son defectos actuales confirmados.'],
        'hallazgos': findings,
    }
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(prefix.with_suffix('.json'), json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    fields = ['id', 'origen', 'comprobacion', 'doc_id', 'clave', 'severidad_heredada', 'estado',
              'prioridad', 'documentos_que_citan_heredado', 'detalle_heredado', 'accion', 'evidencia']
    temporary = prefix.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in findings:
            writer.writerow({**row, 'evidencia': json.dumps(row['evidencia'], ensure_ascii=False)})
    temporary.replace(prefix.with_suffix('.csv'))
    atomic_text(prefix.with_suffix('.md'), markdown(report))
    print(json.dumps({'informe': str(prefix), 'validacion': validation,
                      'reconciliacion': {k: v for k, v in report['reconciliacion_por_comprobacion'].items()
                                        if k in {'grafo_abierto', 'seed_target_ausente',
                                                 'no_nombra_la_norma', 'norma_sin_articulos'}}},
                     ensure_ascii=False, indent=2))
    return int(bool(validation['fallos_integridad'] or validation['contadores'].get('errores_extraccion')))


if __name__ == '__main__':
    sys.exit(main())
