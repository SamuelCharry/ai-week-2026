"""Fija un inventario para evaluar, sin copiar ni modificar los textos canónicos.

La versión referencia textos locales por ruta relativa y SHA-256. --verificar
rechaza cambios del inventario o de cualquiera de los textos seleccionados.
No genera chunks, índices, áreas inferidas ni respuestas de evaluación.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def verify(destination):
    lock = json.loads((destination / 'snapshot.json').read_bytes())
    expected = ['corpus_manifest.json', 'excluidos.json', 'restricciones.json']
    if [item.get('archivo') for item in lock['archivos_version']] != expected:
        raise ValueError('La lista de archivos de la versión no coincide con el contrato')
    identity = hashlib.sha256(json.dumps(lock['archivos_version'], sort_keys=True).encode()).hexdigest()
    if identity != lock['snapshot_id']:
        raise ValueError('snapshot_id no corresponde a los hashes del inventario')
    # Las partes/offsets/páginas y demás procedencia se conservan en este origen.
    source = ROOT / 'data/processed/corpus_preparado/documentos.jsonl'
    if sha(source) != lock['sha256_documentos_jsonl_origen']:
        raise ValueError('Cambió la procedencia/estructura de origen; crear otra versión')
    for entry in lock['archivos_version']:
        path = destination / entry['archivo']
        if sha(path) != entry['sha256']:
            raise ValueError(f"Inventario alterado: {path}")
    manifest = json.loads((destination / 'corpus_manifest.json').read_bytes())
    if len(manifest) != lock['documentos_evaluables']:
        raise ValueError('El conteo de documentos seleccionados no coincide')
    for doc in manifest:
        path = (ROOT / doc['texto_archivo']).resolve()
        if not path.is_relative_to(ROOT / 'data/processed/corpus_preparado'):
            raise ValueError('Texto fuera del corpus preparado')
        if sha(path) != doc['sha256_texto']:
            raise ValueError(f"Texto alterado: {doc['doc_id']}; crear otra versión antes de evaluar")
    result = {'version': lock['version'], 'documentos_verificados': len(manifest),
              'hashes_correctos': True, 'snapshot_id': lock['snapshot_id']}
    print(json.dumps(result, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='corpus_eval_v1')
    parser.add_argument('--verificar', action='store_true')
    args = parser.parse_args()
    if not args.version.replace('_', '').replace('-', '').isalnum():
        parser.error('Nombre de versión inválido')
    destination = ROOT / 'data/releases' / args.version
    if args.verificar:
        verify(destination)
        return
    if destination.exists():
        parser.error('La versión ya existe; verificarla o usar un nombre nuevo, no sobrescribir')
    prepared = ROOT / 'data/processed/corpus_preparado'
    summary = json.loads((prepared / 'resumen.json').read_bytes())
    profile = json.loads((ROOT / 'reports/perfil_corpus_preparado/metrics.json').read_bytes())
    source_sha = sha(prepared / 'documentos.jsonl')
    if summary['modo'] != 'completo' or summary['errores'] or profile['validacion']['fallos_verificacion']:
        raise ValueError('Preparación o perfil incompletos')
    if source_sha != profile['documentos_jsonl_sha256']:
        raise ValueError('El perfil ya no corresponde al corpus')
    word_csv = ROOT / 'reports/correcciones_corpus/docx_numeracion_automatica_pendiente.csv'
    with word_csv.open(encoding='utf-8-sig', newline='') as stream:
        numbering = {row['doc_id'] for row in csv.DictReader(stream)}
    selected, excluded, restrictions = [], [], []
    with (prepared / 'documentos.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            record = json.loads(line)
            specific = list(record.get('pendientes_preparacion', []))
            if record['doc_id'] in numbering:
                specific.append('numeracion_automatica_word_no_reconstruida')
            if record['doc_id'] == 'decreto_1147_1999':
                specific.append('sin_unidad_articular_verificada')
            row = {key: record.get(key) for key in (
                'doc_id', 'titulo', 'tipo', 'numero', 'anio', 'fuente', 'url',
                'fecha_consulta', 'organo_emisor', 'areas', 'temas', 'nivel',
                'vigencia', 'vigencia_fuente', 'licencia_fuente', 'redistribuir_raw',
                'edicion_con_anotaciones', 'origen_ampliacion', 'areas_por_epigrafe',
                'sha256_texto', 'caracteres', 'estado_extraccion', 'avisos')}
            row['texto_archivo'] = (Path('data/processed/corpus_preparado') / record['texto_archivo']).as_posix()
            row['restricciones_especificas'] = sorted(set(specific))
            row['calidad_juridica_certificada'] = False
            row['areas_estado'] = 'etiquetas_heredadas_no_validadas_individualmente'
            row['archivos_raw'] = record['archivos_raw']
            if record.get('apta_para_busqueda') is False:
                excluded.append({**row, 'rechazos_fuente': record.get('rechazos_fuente', [])})
            else:
                selected.append(row)
                if specific:
                    restrictions.append({'doc_id': row['doc_id'], 'restricciones': row['restricciones_especificas']})
    if len(selected) + len(excluded) != summary['documentos_procesados']:
        raise ValueError('Inventario incompleto')
    selected.sort(key=lambda r: r['doc_id'])
    excluded.sort(key=lambda r: r['doc_id'])
    destination.mkdir(parents=True)
    write_json(destination / 'corpus_manifest.json', selected)
    write_json(destination / 'excluidos.json', excluded)
    write_json(destination / 'restricciones.json', restrictions)
    files = [{'archivo': name, 'sha256': sha(destination / name)} for name in
             ['corpus_manifest.json', 'excluidos.json', 'restricciones.json']]
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    lock = {
        'version': args.version, 'snapshot_id': identity,
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'tipo_snapshot': 'inventario_versionado_con_textos_referenciados_y_verificacion_SHA256',
        'sha256_manifest_origen': summary['sha256_manifest'], 'sha256_documentos_jsonl_origen': source_sha,
        'sha256_parser_preparador': summary['sha256_parser_y_preparador'],
        'archivos_version': files, 'documentos_origen': len(selected) + len(excluded),
        'documentos_evaluables': len(selected), 'documentos_excluidos': len(excluded),
        'evaluables_con_restricciones_especificas': len(restrictions),
        'restricciones_por_tipo': dict(Counter(x for d in restrictions for x in d['restricciones'])),
        'politica': {
            'originales': 'preservados; ningún texto se rellena con respuestas ni con generación',
            'exclusion': 'apta_para_busqueda false por defecto confirmado u omisión declarada',
            'avisos_generales': 'no excluir automáticamente por notas editoriales o formato',
            'restricciones': 'conservar evidencia; no citar numerales reconstruidos ni partes sin respaldo',
            'areas': 'sólo señal blanda hasta validar etiquetas; no confundir con temas',
            'vigencia': 'no interpretar por_verificar o sin_marca como vigente',
            'unidades_citables': 'por construir y verificar; este snapshot es documental, no un índice',
            'actualizacion': 'nueva versión y reconstrucción de índices; nunca mezclar resultados entre snapshots',
            'publicacion': 'no publicado; licencias de fuentes y anotaciones deben conservarse por separado',
        },
    }
    write_json(destination / 'snapshot.json', lock)
    verify(destination)


if __name__ == '__main__':
    main()
