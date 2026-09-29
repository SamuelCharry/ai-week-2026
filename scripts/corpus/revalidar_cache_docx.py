"""Revalida cachés no DOCX tras cambiar exclusivamente extraer_docx.

La igualdad del AST restante se exige antes de escribir. No migra documentos
DOCX, errores, hashes de origen distintos ni textos alterados. La preparación
normal vuelve a verificar todos los originales antes de reutilizar una caché.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def unchanged_except_docx(before, after):
    def rest(source):
        tree = ast.parse(source)
        count = sum(isinstance(n, ast.FunctionDef) and n.name == 'extraer_docx' for n in tree.body)
        if count != 1:
            raise ValueError('Se requiere exactamente una función extraer_docx')
        tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == 'extraer_docx')]
        return ast.dump(tree, include_attributes=False)
    return rest(before) == rest(after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, default=Path('scripts/corpus/ingesta.py'))
    parser.add_argument('--raw', type=Path, default=Path('data/data/raw'))
    parser.add_argument('--output', type=Path, default=Path('data/processed/corpus_preparado'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    before, after = args.before.read_bytes(), args.after.read_bytes()
    if not unchanged_except_docx(before, after):
        raise ValueError('Hay cambios fuera de extraer_docx: la caché no puede migrarse')
    preparer = Path(__file__).with_name('preparar_corpus.py').read_bytes()
    old_sha, new_sha = digest(before + preparer), digest(after + preparer)
    manifest = json.loads((args.raw / 'manifest.json').read_bytes())
    migrated, skipped, docx = [], [], []
    for document in manifest:
        doc_id = document['doc_id']
        if any(Path(a['archivo']).suffix.lower() == '.docx' for a in document['archivos_raw']):
            docx.append(doc_id)
            continue
        path = args.output / 'registros' / f'{doc_id}.json'
        if not path.exists():
            skipped.append(doc_id)
            continue
        record = json.loads(path.read_bytes())
        payload = json.dumps(document, sort_keys=True).encode()
        text_path = args.output / 'textos' / f'{doc_id}.txt'
        if (record.get('huella_preparacion') != digest(payload + old_sha.encode())
                or record.get('estado_extraccion') == 'error' or not text_path.exists()
                or digest(text_path.read_bytes()) != record.get('sha256_texto')):
            skipped.append(doc_id)
            continue
        if args.apply:
            record['huella_preparacion'] = digest(payload + new_sha.encode())
            record['revalidacion_cache'] = {
                'motivo': 'AST idéntico salvo extraer_docx; documento sin originales DOCX',
                'sha256_parser_preparador_antes': old_sha,
                'sha256_parser_preparador_despues': new_sha,
                'sha256_texto_sin_cambio': record['sha256_texto'],
            }
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            tmp.replace(path)
        migrated.append(doc_id)
    report = {'fecha_utc': datetime.now(timezone.utc).isoformat(), 'aplicado': args.apply,
              'ast_no_docx_identico': True, 'sha256_parser_preparador_antes': old_sha,
              'sha256_parser_preparador_despues': new_sha, 'revalidados': migrated,
              'requieren_reextraccion_docx': docx, 'omitidos': skipped}
    destination = Path('reports/correcciones_corpus/revalidacion_cache_docx.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({**{k: v for k, v in report.items() if not isinstance(v, list)},
                      'revalidados': len(migrated), 'requieren_reextraccion_docx': len(docx),
                      'omitidos': len(skipped)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
