"""Correcciones verificadas del inventario, con copia previa y bitácora.

Separa número de proceso de fecha del nombre de archivo únicamente cuando el
encabezado del texto derivado confirma el mismo proceso. Las adiciones deben
haber sido descargadas y verificadas previamente en un área de staging.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


def sha(data):
    return hashlib.sha256(data).hexdigest()


def resolve_inside(root, relative):
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f'Ruta fuera del corpus: {relative}')
    return target


def validate_file(root, entry):
    path = resolve_inside(root, entry['archivo'])
    data = path.read_bytes()
    if len(data) != entry['bytes'] or sha(data) != entry['sha256']:
        raise ValueError(f'Integridad inválida: {path}')
    return data


def correct_rad_numbers(manifest, raw):
    changes = []
    for document in manifest:
        match = re.fullmatch(r'(\d+)\((\d{2}-\d{2}-\d{2})\)', str(document.get('numero', '')))
        if not match or not re.fullmatch(rf'sentencia_csj_rad{match[1]}_\d{{4}}', document['doc_id']):
            continue
        confirmation = None
        for entry in document['archivos_raw']:
            derived = entry.get('texto_derivado')
            if not derived:
                continue
            validate_file(raw, entry)
            text = validate_file(raw, derived).decode('utf-8-sig')
            header = re.search(rf'(?im)^\s*Proceso\s+(?:No\.?|n[°ºo])\s*{match[1]}\b', text[:700])
            if header:
                confirmation = {'archivo': derived['archivo'], 'sha256': derived['sha256'],
                                'encabezado': header[0].strip()}
                break
        if not confirmation:
            continue
        change = {'doc_id': document['doc_id'], 'campo': 'numero', 'antes': document['numero'],
                  'despues': match[1], 'motivo': 'fecha_del_archivo_no_es_numero_de_proceso',
                  'evidencia': confirmation}
        document['numero_original_manifest'] = document['numero']
        document['numero'] = match[1]
        document.setdefault('correcciones_metadatos', []).append(change)
        changes.append(change)
    return changes


def apply_verified_patches(manifest, raw, patches):
    from legalrag.preprocessing.ingesta import procesar_documento
    by_id = {d['doc_id']: d for d in manifest}
    changes, texts = [], {}
    for patch in patches:
        document = by_id[patch['doc_id']]
        if document.get(patch['campo']) == patch['despues']:
            continue
        if document.get(patch['campo']) != patch['antes']:
            raise ValueError('Precondición de corrección incumplida: ' + patch['doc_id'])
        if patch['doc_id'] not in texts:
            record, text = procesar_documento(document, raw)
            texts[patch['doc_id']] = re.sub(r'\s+', ' ', text)
        literal = re.sub(r'\s+', ' ', patch['evidencia_literal'])
        if literal not in texts[patch['doc_id']]:
            raise ValueError('Evidencia literal no encontrada: ' + patch['doc_id'])
        change = {**patch, 'motivo': 'metadato_verificado_en_encabezado_oficial',
                  'sha256_originales': [a['sha256'] for a in document['archivos_raw']]}
        document[patch['campo']] = patch['despues']
        document.setdefault('correcciones_metadatos', []).append(change)
        changes.append(change)
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/data/raw'))
    parser.add_argument('--report', type=Path, default=Path('reports/correcciones_corpus'))
    parser.add_argument('--additions', type=Path)
    parser.add_argument('--staging', type=Path)
    parser.add_argument('--patches', type=Path, default=Path('configs/correcciones_corpus_verificadas.json'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    raw, report = args.raw.resolve(), args.report.resolve()
    report.mkdir(parents=True, exist_ok=True)
    path = raw / 'manifest.json'
    before = path.read_bytes()
    manifest = json.loads(before)
    changes = correct_rad_numbers(manifest, raw)
    if args.patches.exists():
        changes.extend(apply_verified_patches(manifest, raw, json.loads(args.patches.read_text(encoding='utf-8'))))
    additions = []
    if args.additions:
        if not args.staging:
            parser.error('--additions requiere --staging')
        additions = json.loads(args.additions.read_text(encoding='utf-8'))
        if not isinstance(additions, list):
            raise ValueError('additions.json debe ser una lista de documentos')
        existing = {d['doc_id'] for d in manifest}
        for document in additions:
            if document['doc_id'] in existing:
                raise ValueError('Adición duplicada: ' + document['doc_id'])
            if not re.fullmatch(r'[A-Za-z0-9_-]+', document['doc_id']):
                raise ValueError('doc_id inválido')
            for entry in document['archivos_raw']:
                for asset in [entry] + ([entry['texto_derivado']] if entry.get('texto_derivado') else []):
                    validate_file(args.staging, asset)
                    destination = resolve_inside(raw, asset['archivo'])
                    if destination.exists() and sha(destination.read_bytes()) != asset['sha256']:
                        raise ValueError('Conflicto con original existente: ' + str(destination))
            existing.add(document['doc_id'])
        manifest.extend(additions)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    result = {'fecha_utc': stamp, 'aplicado': args.apply, 'sha256_antes': sha(before),
              'cambios': changes, 'adiciones': [d['doc_id'] for d in additions],
              'total_antes': len(json.loads(before)), 'total_despues': len(manifest)}
    if args.apply and (changes or additions):
        backup = raw.parent / 'respaldo_manifest' / f'manifest.{sha(before)[:16]}.json'
        backup.parent.mkdir(exist_ok=True)
        if not backup.exists():
            backup.write_bytes(before)
        result['respaldo'] = str(backup)
        for document in additions:
            for entry in document['archivos_raw']:
                for asset in [entry] + ([entry['texto_derivado']] if entry.get('texto_derivado') else []):
                    destination = resolve_inside(raw, asset['archivo'])
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copyfile(resolve_inside(args.staging, asset['archivo']), destination)
        after = (json.dumps(sorted(manifest, key=lambda d: d['doc_id']), ensure_ascii=False, indent=1) + '\n').encode('utf-8')
        if path.read_bytes() != before:
            raise RuntimeError('El manifiesto cambió durante la operación')
        temp = path.with_suffix('.json.tmp')
        temp.write_bytes(after)
        temp.replace(path)
        result['sha256_despues'] = sha(after)
        # El checksum de entrega se conserva: publicar un suplemento de cambios.
        supplemental = raw.parent / 'SHA256SUMS.correcciones'
        lines = [f'{sha(after)}  raw/manifest.json']
        for document in additions:
            for entry in document['archivos_raw']:
                for asset in [entry] + ([entry['texto_derivado']] if entry.get('texto_derivado') else []):
                    lines.append(f"{asset['sha256']}  raw/{asset['archivo']}")
        prior_lines = supplemental.read_text(encoding='utf-8').splitlines() if supplemental.exists() else []
        by_path = {line.split('  ', 1)[1]: line for line in prior_lines + lines if '  ' in line}
        supplemental.write_text('\n'.join(by_path[k] for k in sorted(by_path)) + '\n', encoding='utf-8')
    (report / f'cambios_{stamp}.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({**{k: v for k, v in result.items() if k not in {'cambios', 'adiciones'}},
                      'cambios': len(changes), 'adiciones': len(additions)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
