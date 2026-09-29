"""Descarga fuentes oficiales a staging, sin cambiar raw ni su manifiesto.

Prioridad: cuatro identidades ausentes de la auditoría y referencias del grafo.
No lee preguntas ni respuestas. Guarda additions.json para integración explícita.
Uso: python -m legalrag.ingestion.completar_prioritarios --max-grafo 20
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / 'data/data/raw'
OUT = ROOT / 'reports/completar_prioritarios'
UA = 'Mozilla/5.0 (compatible; AIWeekCorpusAudit/1.0; academic source verification)'
MAX_BYTES = 30_000_000
COMPLEMENTARIOS = {
    'ley_7_1944': 'Procedimiento de vigencia de tratados: contexto internacional general.',
    'sentencia_cc_c671_2001': 'Control de enmienda al Protocolo de Montreal: ambiental/internacional.',
    'sentencia_cc_c864_2006': 'Control de acuerdo CAN-Mercosur: comercio internacional.',
    'sentencia_cc_c615_2009': 'Control de acuerdo binacional sobre población Wayuu: internacional, con contexto constitucional.',
    'sentencia_cc_c519_1994': 'Control del Convenio sobre Diversidad Biológica: ambiental/internacional.',
}


def normalizar_ficha(record):
    """Etiquetas estables y alcance temático de las fuentes nuevas."""
    host = urllib.parse.urlparse(record['url']).hostname or ''
    if 'cortesuprema.gov.co' in host:
        record['fuente'] = 'Corte Suprema de Justicia - Relatoría'
    elif 'corteconstitucional.gov.co' in host:
        record['fuente'] = 'Corte Constitucional - Relatoría'
    elif 'cancilleria.gov.co' in host:
        record['fuente'] = 'Ministerio de Relaciones Exteriores - Compilación jurídica'
    record['vigencia_fuente'] = 'sin_marca'
    record['alcance_vigencia'] = 'sin_marca indica ausencia de una marca de vigencia estructurada verificada en esta descarga. No certifica vigencia, ejecutoria ni consolidación completa.'
    reason = COMPLEMENTARIOS.get(record['doc_id'])
    record['nivel'] = 'complementario' if reason else 'nucleo'
    record['verificacion_adicion']['rationale_nivel'] = reason or 'Materia civil, procesal, constitucional, laboral o administrativa transversal al alcance del corpus.'
    if record['doc_id'] == 'sentencia_cc_c615_2009':
        record['advertencias_preliminares_fuente'] = ['El HTML oficial conserva una referencia residual a Sentencia C-241/06 antes del encabezado principal C-615/09. Se conserva el original; la identidad C-615/09 fue verificada en el cuerpo.']
    return record


SEEDS = [
    {'clave': 'sentencia_csj_sc3674_2021', 'norma': 'Sentencia SC3674-2021',
     'url_verificada': 'https://cortesuprema.gov.co/corte/wp-content/uploads/2021/09/SC3674-2021-2015-00017-01.pdf', 'areas': 'civil, comercial'},
    {'clave': 'sentencia_csj_sc425_2024', 'norma': 'Sentencia SC425-2024',
     'url_verificada': 'https://cortesuprema.gov.co/corte/wp-content/uploads/2024/05/SC425-2024-2019-00063-01.pdf', 'areas': 'civil, comercial'},
    {'clave': 'sentencia_csj_sl1972_2025', 'norma': 'Sentencia SL1972-2025',
     'pendiente': 'Solo se localizó un edicto de notificación; no sustituye la providencia completa.',
     'url_evidencia': 'https://archivodigitalapi.cortesuprema.gov.co/share/2025/10/Edictos/11001220500020229408301Edicto.pdf', 'areas': 'laboral'},
    {'clave': 'sentencia_cc_t248_2025', 'norma': 'Sentencia T-248 de 2025', 'areas': 'constitucional'},
]


def normalizar(value):
    value = unicodedata.normalize('NFD', value.casefold())
    return re.sub(r'\s+', ' ', ''.join(c for c in value if unicodedata.category(c) != 'Mn'))


def urls(fila):
    if fila.get('url_verificada'):
        return [fila['url_verificada']]
    key = fila['clave']
    m = re.fullmatch(r'sentencia_cc_([a-z]+)(\d+)_(\d{4})', key)
    if m:
        sala, number, year = m.groups()
        stem = f'{sala.upper()}{"" if sala == "su" else "-"}{int(number):03d}-{year[2:]}'
        return [f'https://www.corteconstitucional.gov.co/relatoria/{year}/{stem}.htm',
                f'http://www.secretariasenado.gov.co/senado/basedoc/{sala.lower()}-{int(number):03d}_{year}.html']
    m = re.fullmatch(r'(ley|decreto)_(\d+)_(\d{4})', key)
    if m:
        kind, number, year = m.groups()
        stem = f'{kind}_{int(number):04d}_{year}'
        return [f'http://www.secretariasenado.gov.co/senado/basedoc/{stem}.html',
                f'https://www.cancilleria.gov.co/normograma/compilacion/docs/{stem}.htm',
                f'https://normograma.sena.edu.co/compilacion/docs/{stem}.htm']
    return []


def fetch(url):
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=25) as response:
        final = response.geturl()
        host = urllib.parse.urlparse(final).hostname or ''
        if not (host.endswith('.gov.co') or host == 'normograma.sena.edu.co'):
            raise ValueError('Redirección fuera de fuente oficial admitida')
        body = response.read(MAX_BYTES + 1)
        if len(body) > MAX_BYTES:
            raise ValueError('Límite de descarga excedido')
        return body, {'url': url, 'url_final': final, 'http_status': response.status,
                      'content_type': response.headers.get('Content-Type', ''),
                      'last_modified': response.headers.get('Last-Modified')}


def text_and_format(body):
    if body.startswith(b'%PDF'):
        reader = PdfReader(io.BytesIO(body))
        text = '\n'.join(p.extract_text() or '' for p in reader.pages)
        return text, 'pdf', len(reader.pages)
    soup = BeautifulSoup(body, 'html.parser')
    for element in soup(['script', 'style', 'noscript']):
        element.decompose()
    return '\n'.join(soup.stripped_strings), 'html', None


def validate(fila, text, pages):
    value = normalizar(text)
    key = fila['clave']
    judgment = re.fullmatch(r'sentencia_(cc|csj)_([a-z]+)(\d+)_(\d{4})', key)
    if judgment:
        court, sala, number, year = judgment.groups()
        identity = rf'\b{sala}\s*-?\s*0*{int(number)}\s*(?:/|-|de)\s*(?:{year}|{year[2:]})\b'
        if not re.search(identity, value[:16000]):
            return False, 'Identidad exacta no encontrada en encabezado', {}
        if len(value) < 6000 or (pages is not None and pages < 4):
            return False, 'Texto demasiado corto para verificar sentencia completa', {}
        decision = re.search(r'\b(resuelve|decision|decide|no casar|casa la sentencia)\b', value[len(value)//2:])
        if not decision:
            return False, 'No se encontró decisión en mitad final del texto', {}
        if not re.search(r'\b(antecedentes|consideraciones|fundamentos)\b', value):
            return False, 'No se encontró cuerpo argumentativo de sentencia', {}
        return True, 'Identidad en encabezado, cuerpo argumentativo y decisión presentes', {'sala': sala.upper(), 'numero': f'{sala.upper()}{int(number)}', 'anio': int(year), 'tipo': 'sentencia', 'organo_emisor': 'Corte Constitucional' if court == 'cc' else 'Corte Suprema de Justicia'}
    m = re.fullmatch(r'(ley|decreto)_(\d+)_(\d{4})', key)
    if not m:
        return False, 'Identidad no soportada', {}
    kind, number, year = m.groups()
    identity = rf'\b{kind}\b[^\d]{{0,22}}0*{number}\s+(?:de\s+)?{year}\b'
    if not re.search(identity, value[:16000]):
        return False, 'Identidad exacta no encontrada en encabezado', {}
    if len(value) < 800 or not re.search(r'articulo\s+(?:1|primero|unico)', value):
        return False, 'No se encontró articulado mínimo', {}
    if not re.search(r'\b(publiquese|comuniquese|dada? en|rige a partir|sancionese)\b', value):
        return False, 'No se encontró cierre normativo', {}
    return True, 'Identidad en encabezado, articulado y cierre presentes', {'numero': number, 'anio': int(year), 'tipo': kind, 'organo_emisor': 'Congreso de la República' if kind == 'ley' else 'Presidencia de la República'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-grafo', type=int, default=20)
    ap.add_argument('--max-candidatos', type=int, default=35)
    args = ap.parse_args()
    if (OUT/'resumen.json').exists():
        print('Staging terminado existente: se conserva sin descargar ni sobrescribir. Revisar additions.json para integración.', flush=True)
        return
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RAW/'manifest.json').read_text(encoding='utf-8-sig'))
    existing = {row['doc_id'] for row in manifest}
    exclusions = json.loads((ROOT/'data/configs/corpus_exclusiones.json').read_text(encoding='utf-8-sig'))['exclusiones']
    excluded = {row['clave']: row for row in exclusions}
    with (RAW/'grafo_faltantes.csv').open(encoding='utf-8-sig', newline='') as handle:
        graph = sorted(csv.DictReader(handle), key=lambda x: int(x['documentos_que_citan']), reverse=True)
    graph = [row for row in graph if row['clave'] not in existing and row['clave'] not in excluded][:args.max_candidatos]
    additions, audit, accepted_graph = [], [], 0
    for row in [dict(s, prioridad='seed_identidad_auditoria') for s in SEEDS] + [dict(g, prioridad='grafo_citaciones') for g in graph]:
        key = row['clave']
        if row['prioridad'] == 'grafo_citaciones' and accepted_graph >= args.max_grafo:
            break
        event = {**row, 'fecha_verificacion': datetime.now(timezone.utc).isoformat(), 'intentos': []}
        if key in existing or (RAW/key).exists():
            event.update(estado='omitido_existente', motivo='No se sobrescriben documentos existentes')
        elif key in excluded:
            event.update(estado='exclusion_preexistente', exclusion=excluded[key])
        elif row.get('pendiente'):
            event.update(estado='pendiente_texto_completo', motivo=row['pendiente'])
        else:
            event['estado'] = 'pendiente_fuente_validada'
            for url in urls(row):
                attempt = {'url': url}
                try:
                    body, metadata = fetch(url)
                    text, ext, pages = text_and_format(body)
                    valid, reason, identity = validate(row, text, pages)
                    attempt.update(metadata, bytes=len(body), caracteres_extraidos=len(text), paginas=pages, valida=valid, motivo=reason)
                    if valid:
                        # Detectar paginación oficial para evitar declarar completo un tramo.
                        stem = Path(urllib.parse.urlparse(url).path).stem
                        if ext == 'html' and re.search(re.escape(stem)+r'_pr\d+', body.decode('latin-1'), re.I):
                            attempt.update(valida=False, motivo='Fuente paginada: requiere ingesta completa específica')
                        else:
                            dest = OUT/'raw'/key/f'000.{ext}'
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            digest = hashlib.sha256(body).hexdigest()
                            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                                raise ValueError('Staging previo diferente: no sobrescribir')
                            dest.write_bytes(body)
                            now = datetime.now(timezone.utc)
                            record = {'doc_id': key, 'titulo': row['norma'], **identity,
                                'fuente': urllib.parse.urlparse(metadata['url_final']).hostname,
                                'url': url, 'fecha_consulta': now.date().isoformat(), 'fecha_descarga': now.isoformat(),
                                'vigencia': 'por_verificar', 'alcance_vigencia': 'Texto oficial recuperado; no se certifica vigencia ni consolidación completa.',
                                'redistribuir_raw': True, 'edicion_con_anotaciones': ext == 'html' and 'corteconstitucional' not in url,
                                'licencia_fuente': 'Texto jurídico oficial: reproducción conforme al art. 41 de la Ley 23 de 1982. No se relicencian notas editoriales de terceros.',
                                'areas': [x.strip() for x in row.get('areas', '').split(',') if x.strip()],
                                'temas': [], 'advertencias_preliminares_fuente': [], 'actualizacion_declarada_fuente': [],
                                'origen_ampliacion': 'completar_prioritarios_20260929', 'nivel': 'nucleo',
                                'archivos_raw': [{**metadata, 'sha256': digest, 'bytes': len(body), 'encoding': 'auto', 'archivo': f'{key}/000.{ext}'}],
                                'verificacion_adicion': {'metodo': reason, 'caracteres_extraidos': len(text), 'paginas': pages,
                                    'documentos_que_citan': int(row.get('documentos_que_citan', 0)),
                                    'rationale': row['prioridad'], 'staging': str(dest.relative_to(ROOT))}}
                            additions.append(normalizar_ficha(record))
                            event.update(estado='listo_para_integrar', sha256=digest, staging=str(dest.relative_to(ROOT)))
                            if row['prioridad'] == 'grafo_citaciones':
                                accepted_graph += 1
                except Exception as error:
                    attempt.update(valida=False, motivo=f'{type(error).__name__}: {error}')
                event['intentos'].append(attempt)
                time.sleep(0.35)
                if event['estado'] == 'listo_para_integrar':
                    break
        audit.append(event)
        (OUT/'additions.json').write_text(json.dumps(additions, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        (OUT/'verificacion.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        print(f'{key}: {event["estado"]}', flush=True)
    summary = {'documentos_staging': len(additions), 'referencias_grafo_agregadas': accepted_graph,
               'candidatos_revisados': len(audit), 'raw_modificado': False, 'manifest_modificado': False}
    (OUT/'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
