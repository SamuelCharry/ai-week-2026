import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


raiz = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
carpeta = raiz / 'data/processed/corpus'
salida = raiz / 'reports'
salida.mkdir(exist_ok=True)
manifiesto = json.loads((raiz / 'data/raw/manifest.json').read_text(encoding='utf-8'))
documentos = [json.loads(linea) for linea in (carpeta / 'documentos.jsonl').open(encoding='utf-8')]
unidades = [json.loads(linea) for linea in (carpeta / 'unidades.jsonl').open(encoding='utf-8')]
por_doc = defaultdict(list)
ventanas = defaultdict(list)
por_tipo = defaultdict(list)
exactos = defaultdict(list)
candidatos_pares = defaultdict(list)
patron_articulo = re.compile(r'(?im)^[ \t]*(?:[«“\"\u2018\u2019])?[ \t]*art[íi]culo\s+(\d+(?:\.\d+)*(?:[ \t]*[A-Za-z])?)(?=[\s.°º:;,\-])')
patron_mojibake = re.compile(r'\ufffd|Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|â[€\u0080-\u00bf]')
patron_ruido = re.compile(r'(?i)https?://|www\.|p[áa]gina\s+\d+\s+de\s+\d+|gestor\s+normativo|funci[oó]n\s+p[úu]blica|descargar\s+pdf')
patron_corte = re.compile(r'^\d+(?:\.\d+)*[.)]?\s+(?:El|La|Los|Las|Un|Una|En|De|Por|Que|Se|Al|Del)$', re.I)
patron_entidad = re.compile(r'&(?:[aeiouAEIOU](?:acute|ACUTE)|[nN](?:tilde|TILDE)|(?:quot|amp|nbsp))(?:;|\b|(?=[A-Z]))')

for u in unidades:
    por_doc[u['doc_id']].append(u)
    por_tipo[u['tipo_unidad']].append(len(u['texto']))
    texto = u['texto']
    clave = hashlib.sha256(texto.encode()).hexdigest()
    exactos[clave].append(u)
    normal = ' '.join(texto.casefold().split())
    u['_normal'] = normal
    if 100 <= len(normal) <= 30000:
        for borde, firma in [('inicio', normal[:80]), ('fin', normal[-80:])]:
            candidatos_pares[(borde, firma)].append(u)

for linea in (carpeta / 'fragmentos.jsonl').open(encoding='utf-8'):
    f = json.loads(linea)
    ventanas[f['doc_id']].append(len(f['texto']))

limites = {}
for tipo, largos in por_tipo.items():
    q1, q3 = np.quantile(largos, [0.25, 0.75])
    limites[tipo] = (float(q1 - 1.5 * (q3 - q1)), float(q3 + 1.5 * (q3 - q1)))

duplicados_doc = Counter()
grupos_exactos = [g for g in exactos.values() if len(g) > 1]
for grupo in grupos_exactos:
    for u in grupo:
        duplicados_doc[u['doc_id']] += 1

pares = set()
bloques_omitidos = 0
for grupo in candidatos_pares.values():
    if len(grupo) > 100:
        bloques_omitidos += 1
        continue
    for a, b in combinations(grupo, 2):
        if a['_normal'] == b['_normal']:
            continue
        if min(len(a['_normal']), len(b['_normal'])) / max(len(a['_normal']), len(b['_normal'])) < 0.9:
            continue
        pares.add(tuple(sorted([a['unidad_id'], b['unidad_id']])))

por_id = {u['unidad_id']: u for u in unidades}
casi = []
tokens_cache = {}
for aid, bid in sorted(pares):
    a, b = por_id[aid], por_id[bid]
    for u in [a, b]:
        if u['unidad_id'] not in tokens_cache:
            palabras = re.findall(r'\w+', u['_normal'])
            tokens_cache[u['unidad_id']] = set(zip(palabras, palabras[1:], palabras[2:], palabras[3:], palabras[4:]))
    sa, sb = tokens_cache[aid], tokens_cache[bid]
    if not sa or not sb:
        continue
    jaccard = len(sa & sb) / len(sa | sb)
    if jaccard >= 0.9:
        casi.append((aid, bid, jaccard))
casi_doc = defaultdict(set)
for aid, bid, valor in casi:
    casi_doc[por_id[aid]['doc_id']].add(aid)
    casi_doc[por_id[bid]['doc_id']].add(bid)

esperados_revisados = {
    'co_decreto_306_1992': 10,
    'ley_1032_2006': 5,
    'ley_1581_2012': 30,
    'ley_2445_2025': 45,
}
filas = []
ejemplos = defaultdict(list)
for d in documentos:
    doc_id = d['doc_id']
    us = por_doc[doc_id]
    texto = (carpeta / d['texto_archivo']).read_text(encoding='utf-8')
    largos = [len(u['texto']) for u in us]
    articulos = [str(u['articulo']) for u in us if u['articulo'] is not None]
    simples = sorted(set(int(a) for a in articulos if re.fullmatch(r'\d+', a)))
    huecos = sorted(set(range(min(simples), max(simples) + 1)) - set(simples)) if simples else []
    repeticiones = {a: n for a, n in Counter(articulos).items() if n > 1}
    encabezados = list(patron_articulo.finditer(texto))
    candidatos = [m.group(1).strip().upper() for m in encabezados]
    candidatos_duplicados = {a: n for a, n in Counter(candidatos).items() if n > 1}
    multiples = [u for u in us if len(list(patron_articulo.finditer(u['texto']))) > 1]
    cortes = [u for u in us if patron_corte.fullmatch(u['texto'].strip())]
    outliers = [u for u in us if len(u['texto']) < limites[u['tipo_unidad']][0] or len(u['texto']) > limites[u['tipo_unidad']][1]]
    muy_largas = [u for u in us if len(u['texto']) > 20000]
    lineas = [x.strip() for x in texto.splitlines() if 5 <= len(x.strip()) <= 140]
    repetidas = [(linea, n) for linea, n in Counter(lineas).items() if n >= 3 and patron_ruido.search(linea)]
    faltan = [k for k in ['doc_id', 'titulo', 'tipo', 'numero', 'anio', 'organo_emisor', 'vigencia', 'fuente', 'url', 'fecha_consulta', 'areas'] if d.get(k) in [None, '', []]]
    norma = d['tipo'] not in ['sentencia', 'auto']
    f = {
        'doc_id': doc_id,
        'tipo': d['tipo'],
        'areas': '|'.join(d['areas']),
        'url': d['url'],
        'caracteres': len(texto),
        'unidades': len(us),
        'ventanas': len(ventanas[doc_id]),
        'articulos_asignados': len(articulos),
        'articulos_distintos': len(set(articulos)),
        'articulos_candidatos_regex': len(encabezados),
        'candidatos_distintos_regex': len(set(m.group(1).strip().upper() for m in encabezados)),
        'articulos_esperados_revision_original': esperados_revisados.get(doc_id),
        'fuente_esperados': 'Lectura del original guardado, independiente del parser' if doc_id in esperados_revisados else '',
        'huecos_numeracion_simple': json.dumps(huecos),
        'duplicados_numeracion': json.dumps(repeticiones),
        'duplicados_numeracion_candidatos': json.dumps(candidatos_duplicados),
        'unidades_varios_encabezados': len(multiples),
        'unidades_microcorte': len(cortes),
        'unidades_vacias': sum(not u['texto'].strip() for u in us),
        'unidades_menores_80': sum(n < 80 for n in largos),
        'unidades_mayores_20000': len(muy_largas),
        'unidades_atipicas_iqr_tipo': len(outliers),
        'longitud_min': min(largos),
        'longitud_p25': round(float(np.quantile(largos, .25)), 2),
        'longitud_mediana': round(float(np.median(largos)), 2),
        'longitud_p95': round(float(np.quantile(largos, .95)), 2),
        'longitud_p99': round(float(np.quantile(largos, .99)), 2),
        'longitud_max': max(largos),
        'mojibake_coincidencias': len(patron_mojibake.findall(texto)),
        'entidades_html_residuales': len(patron_entidad.findall(texto)),
        'caracteres_control': sum(ord(c) < 32 and c not in '\n\t\r' for c in texto),
        'ocr_revision': 'error_en_capa_textual_confirmado' if doc_id == 'sentencia_csj_sc5191_2020' else ('pendiente' if any(a['archivo'].endswith('.pdf') for a in d['archivos_raw']) else 'no_aplica'),
        'pdf_cierre_incompleto': sum(e.get('cierre_pdf_completo') is False for e in d['extraccion']),
        'lineas_ruido_repetidas': len(repetidas),
        'metadatos_faltantes': '|'.join(faltan),
        'articulo_nulo_en_norma': sum(u['articulo'] is None for u in us) if norma else 0,
        'unidades_identicas': duplicados_doc[doc_id],
        'unidades_casiexactas_candidatas': len(casi_doc[doc_id]),
        'estado_extraccion': d['estado_extraccion'],
        'avisos_extraccion': '|'.join(d['avisos']),
        'unidades_con_avisos': sum(bool(u['avisos']) for u in us),
        'unidades_revision': sum(u['estado_segmentacion'] == 'revisar' for u in us),
        'vigencia': d['vigencia'],
        'revision_juridica': d['revision_juridica'],
        'sha256_texto': d['sha256_texto'],
        'sha256_raw': '|'.join(a['sha256'] for a in d['archivos_raw']),
    }
    filas.append(f)
    for nombre, grupo in [('multiples', multiples), ('cortes', cortes), ('muy_largas', muy_largas)]:
        for u in grupo[:5]:
            ejemplos[nombre].append({'doc_id': doc_id, 'unidad_id': u['unidad_id'], 'inicio': u['inicio'], 'fin': u['fin'], 'texto': u['texto'][:250]})
    if repetidas:
        ejemplos['ruido'].append({'doc_id': doc_id, 'lineas': sorted(repetidas, key=lambda x: -x[1])[:3]})

with (salida / 'a_por_documento.csv').open('w', encoding='utf-8', newline='') as f:
    escritor = csv.DictWriter(f, fieldnames=list(filas[0]))
    escritor.writeheader()
    escritor.writerows(filas)

resumen = {
    'documentos': len(documentos),
    'unidades': len(unidades),
    'ventanas': sum(map(len, ventanas.values())),
    'percentiles_unidades': {str(p): float(np.quantile([len(u['texto']) for u in unidades], p)) for p in [0, .25, .5, .75, .95, .99, 1]},
    'limites_iqr_por_tipo': limites,
    'sumas': {k: sum(f[k] for f in filas) for k in ['unidades_varios_encabezados', 'unidades_microcorte', 'unidades_vacias', 'unidades_menores_80', 'unidades_mayores_20000', 'unidades_atipicas_iqr_tipo', 'mojibake_coincidencias', 'lineas_ruido_repetidas']},
    'docs_metadatos_faltantes': sum(bool(f['metadatos_faltantes']) for f in filas),
    'docs_avisos_unidades': sum(f['unidades_con_avisos'] > 0 for f in filas),
    'grupos_identicos': len(grupos_exactos),
    'unidades_identicas': sum(map(len, grupos_exactos)),
    'pares_candidatos_casiexactos': len(pares),
    'pares_jaccard_090': len(casi),
    'unidades_casiexactas': len(set(x for t in casi for x in t[:2])),
    'bloques_casiexactos_omitidos_por_mas_100': bloques_omitidos,
    'ejemplos_casiexactos': casi[:10],
    'ejemplos_identicos': [[{'doc_id': u['doc_id'], 'unidad_id': u['unidad_id'], 'inicio': u['inicio'], 'fin': u['fin'], 'texto': u['texto'][:150]} for u in g[:3]] for g in grupos_exactos[:5]],
    'ejemplos': ejemplos,
    'sha256_entradas': {n: hashlib.sha256((carpeta / n).read_bytes()).hexdigest() for n in ['documentos.jsonl', 'unidades.jsonl', 'fragmentos.jsonl']},
}
print(json.dumps(resumen, ensure_ascii=False, indent=2))
