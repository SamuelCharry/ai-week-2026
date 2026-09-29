"""Prepara texto canónico verificable sin elegir chunks, encoder ni decoder.

Conserva raw; usa caché por documento y escribe un inventario explícito de pendientes.
Los offsets futuros se referirán al texto canónico, nunca al HTML binario.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from legalrag.preprocessing import ingesta


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def prepare_one(document, raw, output, parser_sha):
    doc_id = document['doc_id']
    output = Path(output)
    fingerprint = digest(json.dumps(document, sort_keys=True).encode() + parser_sha.encode())
    cached = output / 'registros' / f'{doc_id}.json'
    if cached.exists():
        try:
            previous = json.loads(cached.read_text(encoding='utf-8'))
            text_path = output / 'textos' / f'{doc_id}.txt'
            if (previous.get('huella_preparacion') == fingerprint
                    and previous.get('estado_extraccion') != 'error' and text_path.exists()
                    and digest(text_path.read_bytes()) == previous.get('sha256_texto')):
                # Raw is revalidated even when reusing extracted text.
                for entry in document.get('archivos_raw', []):
                    ingesta.leer_original(entry, raw)
                    if entry.get('texto_derivado'):
                        ingesta.leer_original(entry['texto_derivado'], raw)
                return previous
        except (OSError, ValueError, KeyError, TypeError):
            # Re-extract invalid cache; genuine raw errors become a document error below.
            pass
    try:
        record, text = ingesta.procesar_documento(document, raw)
        problems = []
        incomplete = re.search(
            r'esta compilaci[oó]n no incluye el texto|texto de esta norma no se sistematizar[aá]',
            text, re.I)
        if incomplete:
            problems.append('fuente_declara_omision_del_texto')
            record['evidencia_omision'] = text[max(0, incomplete.start() - 80):incomplete.end() + 180]
            record['apta_para_busqueda'] = False
        if record.get('coincidencias_mojibake', 0):
            problems.append('codificacion_residual_por_revisar')
        if len(text.strip()) < 200:
            problems.append('texto_corto_por_revisar')
        if any(x.get('paginas_con_poco_texto') for x in record.get('extraccion', [])):
            problems.append('paginas_sin_texto_suficiente')
        for warning in ['paginacion_derivado_no_verificada', 'revisar_calidad_OCR',
                        'bytes_no_decodificables_en_charset_declarado']:
            if warning in record.get('avisos', []):
                problems.append(warning)
        record['pendientes_preparacion'] = problems
        record['integridad_archivos_verificada'] = True
        record['calidad_juridica_certificada'] = False
        record['dominio_publicador'] = urlparse(document.get('url', '')).hostname
        record['texto_busqueda_recomendado'] = 'derivar_del_texto_canonico_sin_modificar_evidencia'
        target = output / record['texto_archivo']
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.txt.tmp')
        temporary.write_bytes(text.encode('utf-8'))
        temporary.replace(target)
    except Exception as error:
        record = {**document, 'version_ingesta': ingesta.VERSION_INGESTA,
                  'estado_extraccion': 'error', 'texto_archivo': None,
                  'sha256_texto': None, 'caracteres': 0, 'apta_para_busqueda': False,
                  'pendientes_preparacion': ['error_extraccion'],
                  'error': f'{type(error).__name__}: {error}'}
    record['huella_preparacion'] = fingerprint
    write_json(cached, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/data/raw'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/corpus_preparado'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--ids', nargs='*', help='Solo para pruebas; queda indicado en resumen.')
    args = parser.parse_args(argv)
    raw, output = args.raw.resolve(), args.output.resolve()
    if output == raw or output.is_relative_to(raw):
        parser.error('La salida debe estar fuera de raw para conservar los originales.')
    if args.workers < 1:
        parser.error('--workers debe ser positivo')
    manifest_bytes = (raw / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    ids = [d['doc_id'] for d in manifest]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids):
        raise ValueError('doc_id repetidos o inseguros; corregir inventario antes de preparar')
    if args.ids and set(args.ids) - set(ids):
        raise ValueError('La selección contiene doc_id desconocidos')
    selected = [d for d in manifest if not args.ids or d['doc_id'] in args.ids]
    output.mkdir(parents=True, exist_ok=True)
    (output / 'registros').mkdir(exist_ok=True)
    parser_sha = digest(Path(ingesta.__file__).read_bytes() + Path(__file__).read_bytes())
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepare_one, d, str(raw), str(output), parser_sha): d['doc_id']
                   for d in selected}
        for future in as_completed(futures):
            records.append(future.result())
            if len(records) % 250 == 0 or len(records) == len(selected):
                errors = sum(r['estado_extraccion'] == 'error' for r in records)
                print(f'Preparados {len(records)}/{len(selected)}; errores {errors}', flush=True)
    if digest((raw / 'manifest.json').read_bytes()) != digest(manifest_bytes):
        raise RuntimeError('El manifiesto cambió durante la preparación; repetir para snapshot consistente')
    if digest(Path(ingesta.__file__).read_bytes() + Path(__file__).read_bytes()) != parser_sha:
        raise RuntimeError('El extractor cambió durante la preparación; repetir con una versión estable')
    records.sort(key=lambda r: r['doc_id'])
    tmp = output / 'documentos.jsonl.tmp'
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    tmp.replace(output / 'documentos.jsonl')
    write_json(output / 'corpus_manifest.json', selected)
    pending = [{'doc_id': r['doc_id'], 'problema': p, 'error': r.get('error') or '',
                'url': r.get('url', '')}
               for r in records for p in r.get('pendientes_preparacion', [])]
    write_csv(output / 'pendientes.csv', pending, ['doc_id', 'problema', 'error', 'url'])
    duplicate_groups = defaultdict(list)
    for record in records:
        if record.get('sha256_texto'):
            duplicate_groups[record['sha256_texto']].append(record['doc_id'])
    duplicates = [{'sha256_texto': key, 'doc_ids': value} for key, value in duplicate_groups.items()
                  if len(value) > 1]
    write_json(output / 'textos_identicos.json', duplicates)
    summary = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(), 'raw': str(raw),
        'salida': str(output), 'sha256_manifest': digest(manifest_bytes),
        'sha256_parser_y_preparador': parser_sha, 'version_ingesta': ingesta.VERSION_INGESTA,
        'modo': 'seleccion' if args.ids else 'completo', 'documentos_manifest': len(manifest),
        'documentos_procesados': len(records),
        'textos_guardados': sum(bool(r.get('texto_archivo')) for r in records),
        'errores': sum(r['estado_extraccion'] == 'error' for r in records),
        'caracteres_totales': sum(r.get('caracteres', 0) for r in records),
        'documentos_con_derivado': sum(any(a.get('texto_derivado') for a in r['archivos_raw']) for r in records),
        'pendientes_por_tipo': dict(Counter(p['problema'] for p in pending)),
        'documentos_no_aptos_busqueda': sum(r.get('apta_para_busqueda') is False for r in records),
        'grupos_texto_identico': len(duplicates),
        'avisos': dict(Counter(a for r in records for a in r.get('avisos', []))),
        'alcance': 'Extracción completa; no certifica vigencia, completitud jurídica ni OCR perfecto. Sin índices ni modelos.',
        'originales_modificados': False,
    }
    write_json(output / 'resumen.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 1 if summary['errores'] else 0


if __name__ == '__main__':
    sys.exit(main())
