import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.auxiliares.evaluacion import preparar_entrada, validar_esquema
from scripts.auxiliares.generacion import abstenerse, generar
from scripts.auxiliares.recuperacion import hash_json, leer_jsonl, sha256


def guardar_json(ruta, contenido):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_name(ruta.name + '.tmp')
    temporal.write_text(json.dumps(contenido, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporal.replace(ruta)


def cargar_citaciones(raiz):
    ruta = Path(raiz) / 'data/oficial/scripts/citations.py'
    spec = importlib.util.spec_from_file_location('citaciones_oficiales', ruta)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def cargar_muestra(raiz):
    preguntas = leer_jsonl(Path(raiz) / 'data/oficial/data/sample_50.jsonl')
    if len(preguntas) != 50 or len({p['id'] for p in preguntas}) != 50:
        raise ValueError('La muestra debe contener 50 identificadores únicos')
    if any(type(p['id']) is not int for p in preguntas):
        raise ValueError('El identificador oficial debe ser entero')
    return preguntas, [preparar_entrada(p) for p in preguntas]


def consulta(entrada):
    entrada = preparar_entrada(entrada)
    opciones = entrada.get('opciones', {})
    return entrada['pregunta'].strip() + ('\n' + '\n'.join(opciones.values()) if opciones else '')


def identidad(documento, citas):
    tipo = documento.get('tipo', '').replace('_', ' ')
    numero, anio = documento.get('numero'), documento.get('anio')
    if tipo == 'constitucion':
        return {('constitucion', None, None)}
    if tipo in {'ley', 'decreto', 'decreto ley', 'acto legislativo', 'resolucion', 'circular', 'acuerdo'}:
        return citas.bodies(citas.extract(f'{tipo} {numero} de {anio}'))
    if tipo == 'decision' and str(numero) == '486':
        return {('decision_andina_486', None, None)}
    if tipo in {'sentencia', 'auto'}:
        return citas.bodies(citas.extract(f'Sentencia {numero} de {anio}'))
    return citas.bodies(citas.extract(documento.get('titulo', '')))


def cobertura(raiz, documentos, unidades):
    citas = cargar_citaciones(raiz)
    objetivos = json.loads((Path(raiz) / 'data/oficial/data/seed_targets.json').read_text(encoding='utf-8'))['documentos']
    cantidades = Counter(u['doc_id'] for u in unidades)
    filas = []
    for objetivo in objetivos:
        candidatos = [d for d in documentos if tuple(objetivo['canonico']) in identidad(d, citas)]
        disponibles = [d for d in candidatos if cantidades[d['doc_id']] > 0]
        filas.append({
            'norma': objetivo['norma'], 'items_del_banco': objetivo['items_del_banco'],
            'estado': 'con unidades admitidas' if disponibles else ('documento sin unidades admitidas' if candidatos else 'sin coincidencia'),
            'doc_ids': [d['doc_id'] for d in candidatos],
            'unidades': sum(cantidades[d['doc_id']] for d in disponibles),
            'areas': objetivo['areas'], 'donde_buscar': objetivo['donde_buscar'],
        })
    return filas


def metricas_recuperacion(pregunta, pasajes, documentos, citas):
    referencia = citas.bodies(citas.extract(pregunta.get('legal_basis') or ''))
    fila = {'id': pregunta['id'], 'area': pregunta['area'], 'formato': pregunta['formato'],
            'referencias': len(referencia)}
    for k in (1, 5, 10):
        encontrados = set().union(*(identidad(documentos[p['doc_id']], citas) for p in pasajes[:k]))
        fila[f'recall_normas_{k}'] = len(referencia & encontrados) / len(referencia) if referencia else None
    return fila


class Evidencia:
    def __init__(self, raiz, corpus):
        self.citas = cargar_citaciones(raiz)
        self.corpus = Path(corpus)
        self.documentos = {d['doc_id']: d for d in leer_jsonl(self.corpus / 'documentos.jsonl')}
        self.textos = {}
        self.cabeceras = {}

    def texto(self, doc_id):
        if doc_id not in self.textos:
            doc = self.documentos[doc_id]
            ruta = (self.corpus / doc['texto_archivo']).resolve()
            if not ruta.is_relative_to(self.corpus.resolve()):
                raise ValueError('Texto fuera del corpus')
            contenido = ruta.read_text(encoding='utf-8')
            import hashlib
            if hashlib.sha256(contenido.encode()).hexdigest() != doc['sha256_texto']:
                raise ValueError('El texto no coincide con su hash')
            self.textos[doc_id] = contenido
        return self.textos[doc_id]

    def cabecera(self, doc_id):
        if doc_id not in self.cabeceras:
            texto = self.texto(doc_id)
            propia = identidad(self.documentos[doc_id], self.citas)
            elegida = None
            for linea in re.finditer(r'[^\n]+', texto[:12000]):
                if len(linea.group()) > 600:
                    continue
                if propia & self.citas.bodies(self.citas.extract(linea.group())):
                    elegida = {'doc_id': doc_id, 'inicio': linea.start(), 'fin': linea.end(),
                               'texto': linea.group(), 'tipo_evidencia': 'cabecera_fuente'}
                    break
            self.cabeceras[doc_id] = elegida
        return self.cabeceras[doc_id]

    def preparar(self, recuperados, max_unidades=5):
        pasajes, avisos = [], []
        for unidad in recuperados[:max_unidades]:
            if self.texto(unidad['doc_id'])[unidad['inicio']:unidad['fin']] != unidad['texto']:
                raise ValueError('El artículo no conserva el texto literal')
            campos = ['doc_id', 'unidad_id', 'inicio', 'fin', 'texto', 'score', 'titulo', 'articulo',
                      'tipo', 'numero', 'anio', 'unidad_inicio', 'unidad_fin', 'recuperar_unidad_completa']
            articulo = {k: unidad[k] for k in campos if k in unidad}
            articulo.update(grupo_evidencia=unidad['unidad_id'], tipo_evidencia='unidad')
            propia = identidad(self.documentos[unidad['doc_id']], self.citas)
            if not propia & self.citas.bodies(self.citas.extract(unidad['texto'])):
                cabecera = self.cabecera(unidad['doc_id'])
                if cabecera:
                    pasajes.append(dict(cabecera, grupo_evidencia=unidad['unidad_id'], score=unidad['score']))
                else:
                    avisos.append({'doc_id': unidad['doc_id'], 'motivo': 'sin_cabecera_literal_reconocible'})
            pasajes.append(articulo)
        if len(pasajes) > 10:
            raise ValueError('Se superaron los diez pasajes de evidencia')
        return pasajes, avisos

    def verificar(self, pasajes):
        for p in pasajes:
            if self.texto(p['doc_id'])[p['inicio']:p['fin']] != p['texto']:
                raise ValueError('Evidencia alterada o con offsets incorrectos')


def generar_oficial(cliente, entrada, pasajes, evidencia, carpeta_raw, contexto=8192, max_tokens=1024):
    evidencia.verificar(pasajes)
    salida, registro = generar(cliente, entrada, pasajes, contexto=contexto, max_tokens=max_tokens,
                               gramatica=True, abstencion_automatica=True, carpeta_raw=carpeta_raw)
    if salida is not None:
        guardar_json(Path(carpeta_raw) / 'salida_modelo.json', salida)
    motivo = None
    if salida is None:
        motivo = 'formato_invalido'
    elif not salida['abstencion']:
        from scripts.auxiliares.generacion import CAMPOS
        if any(salida.get(k) in (None, '', [], {}) for k in CAMPOS[entrada['formato']]):
            motivo = 'campos_vacios'
        campos = {'multiple_choice': ['justificacion'], 'semi_open': ['respuesta', 'referencia_legal'],
                  'open_ended': ['marco_normativo', 'analisis', 'jurisprudencia', 'conclusion']}
        citadas = evidencia.citas.bodies(evidencia.citas.extract(' '.join(salida[k] for k in campos[entrada['formato']])))
        respaldo = set().union(*(evidencia.citas.bodies(evidencia.citas.extract(p['texto']))
                                  for p in salida['pasajes_recuperados'][:10]))
        registro['citas_sin_respaldo_oficial'] = sorted(map(list, citadas - respaldo))
        if citadas - respaldo:
            motivo = 'citas_sin_respaldo'
    if motivo:
        usados = salida.get('pasajes_recuperados', []) if salida else []
        salida = {'id': entrada['id'], 'formato': entrada['formato'],
                  **abstenerse(entrada['formato']), 'pasajes_recuperados': usados}
        registro.update(abstencion_programatica=True, motivo=motivo)
    if salida['abstencion']:
        salida.update(abstenerse(entrada['formato']))
    evidencia.verificar(salida['pasajes_recuperados'])
    return salida, registro


def evaluar_entrega(raiz, carpeta, respuestas, ragas=False):
    raiz, carpeta = Path(raiz), Path(carpeta)
    preguntas, _ = cargar_muestra(raiz)
    if len(respuestas) != 50 or {r['id'] for r in respuestas} != {q['id'] for q in preguntas}:
        raise ValueError('Se necesitan las 50 respuestas sin duplicados para calificar')
    formatos = {q['id']: q['formato'] for q in preguntas}
    if any(r['formato'] != formatos[r['id']] for r in respuestas):
        raise ValueError('Se modificó el formato de alguna pregunta')
    carpeta.mkdir(parents=True, exist_ok=True)
    submission = carpeta / 'submissions.jsonl'
    submission.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in respuestas), encoding='utf-8')
    errores = validar_esquema(respuestas, raiz / 'data/oficial/schema/submission.schema.json')
    guardar_json(carpeta / 'validacion_esquema.json', errores)
    etiqueta = 'con_ragas' if ragas else 'sin_ragas'
    reporte = carpeta / f'evaluacion_{etiqueta}.json'
    comando = [sys.executable, str(raiz / 'data/oficial/scripts/evaluate.py'),
               '--submission', str(submission.resolve()), '--split', 'sample', '--out', str(reporte.resolve())]
    if ragas:
        comando.append('--ragas')
    proceso = subprocess.run(comando, capture_output=True, text=True, encoding='utf-8',
                             cwd=raiz / 'data/oficial')
    (carpeta / f'evaluacion_{etiqueta}.log').write_text(proceso.stdout + proceso.stderr, encoding='utf-8')
    if proceso.returncode:
        raise RuntimeError('El evaluador falló. Revisar su archivo .log')
    resultado = json.loads(reporte.read_text(encoding='utf-8'))
    if resultado['validacion']['errores']:
        raise ValueError('El evaluador reportó errores de forma. Revisar su reporte JSON')
    incompatibles = {r['id'] for r in respuestas if r['formato'] == 'multiple_choice'
                     and r['abstencion'] and r['respuesta_correcta'] is None}
    otros = [e for e in errores if not (e['id'] in incompatibles and e['campo'] == 'respuesta_correcta')]
    if otros:
        raise ValueError('Hay errores de esquema adicionales a la contradicción de abstención cerrada')
    claves = {q['id']: q for q in preguntas}
    citas = cargar_citaciones(raiz)
    detalles = []
    campos = {'multiple_choice': ['justificacion'], 'semi_open': ['respuesta', 'referencia_legal'],
              'open_ended': ['marco_normativo', 'analisis', 'jurisprudencia', 'conclusion']}
    for respuesta in respuestas:
        clave = claves[respuesta['id']]
        texto = ' '.join(respuesta[k] for k in campos[respuesta['formato']])
        respaldo = set().union(*(citas.extract(p['texto']) for p in respuesta['pasajes_recuperados'][:10]))
        detalles.append({'id': respuesta['id'], 'area': clave['area'], 'formato': respuesta['formato'],
                         'abstencion': respuesta['abstencion'],
                         'acierto_cerrada': (not respuesta['abstencion'] and respuesta['respuesta_correcta'] == clave['respuesta_correcta'])
                         if respuesta['formato'] == 'multiple_choice' else None,
                         'citas': citas.score(texto, clave.get('legal_basis') or '', respaldo),
                         'respuesta_evaluada': texto})
    guardar_json(carpeta / 'diagnostico_por_pregunta.json', detalles)
    return {**resultado, 'errores_esquema': len(errores),
            'abstencion_cerrada_pendiente_aclaracion': sorted(incompatibles)}


def huellas(raiz):
    raiz = Path(raiz)
    rutas = ['data/oficial/data/sample_50.jsonl', 'data/oficial/schema/submission.schema.json',
             'data/oficial/scripts/evaluate.py', 'data/oficial/scripts/citations.py',
             'data/oficial/scripts/common.py', 'data/processed/corpus/documentos.jsonl',
             'data/processed/corpus/unidades.jsonl', 'configs/modelos.json']
    rutas += [p.relative_to(raiz).as_posix() for p in sorted((raiz / 'scripts/auxiliares').glob('*.py'))]
    return {r: sha256(raiz / r) for r in rutas}


def abrir_experimento(base, nombre, configuracion, reanudar=''):
    from datetime import datetime, timezone
    base = Path(base)
    if reanudar and Path(reanudar).name != reanudar:
        raise ValueError('Reanudar debe ser el nombre de una carpeta, sin rutas')
    carpeta = base / nombre / (reanudar or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if reanudar:
        anterior = json.loads((carpeta / 'configuracion.json').read_text(encoding='utf-8'))
        if hash_json(anterior) != hash_json(configuracion):
            raise ValueError('La configuración cambió. Iniciar otra ejecución')
    else:
        carpeta.mkdir(parents=True, exist_ok=False)
        guardar_json(carpeta / 'configuracion.json', configuracion)
    return carpeta
