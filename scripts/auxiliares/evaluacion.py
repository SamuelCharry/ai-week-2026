import json
import os
import re
import subprocess
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path


CAMPOS_ENTRADA = ['id', 'formato', 'pregunta', 'opciones', 'area', 'sub_tarea', 'tema', 'complejidad']
CAMPOS_RESERVADOS = {
    'legal_basis', 'respuesta_correcta', 'respuesta_esperada', 'respuestas_esperadas',
    'justificacion', 'referencia_legal', 'ground_truth', 'answer', 'correct_answer',
    'expected_answer', 'texto_respuesta_correcta', 'is_correct', 'solution', 'solucion', 'respuesta', 'marco_normativo',
    'analisis', 'jurisprudencia', 'conclusion', 'palabras_clave', 'descarte_opciones',
}
CAMPOS_TIEMPO = {
    'latencia_ms', 'latencia_s', 'latency_ms', 'latency_s', 'duration_ms',
    'duracion_ms', 'duracion_s', 'elapsed_seconds', 'tiempo_ms', 'tiempo_s',
}


def cargar_jsonl(ruta):
    registros = []
    ids = set()
    for numero, linea in enumerate(Path(ruta).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not linea.strip():
            continue
        try:
            registro = json.loads(linea)
        except json.JSONDecodeError as error:
            raise ValueError(f'JSON inválido en la línea {numero}: {error.msg}') from error
        if not isinstance(registro, dict) or 'id' not in registro:
            raise ValueError(f'Falta un objeto con id en la línea {numero}')
        clave = json.dumps(registro['id'], sort_keys=True)
        if clave in ids:
            raise ValueError(f'Id repetido en la línea {numero}: {registro["id"]}')
        ids.add(clave)
        registros.append(registro)
    if not registros:
        raise ValueError('El archivo JSONL está vacío')
    return registros


def comprobar_oficiales(carpeta):
    carpeta = Path(carpeta)
    rutas = {
        'preguntas': carpeta / 'data/sample_50.jsonl',
        'fuentes': carpeta / 'data/seed_targets.json',
        'esquema': carpeta / 'schema/submission.schema.json',
        'evaluador': carpeta / 'scripts/evaluate.py',
        'citas': carpeta / 'scripts/citations.py',
        'comun': carpeta / 'scripts/common.py',
        'dependencias': carpeta / 'scripts/requirements-evaluador.txt',
    }
    faltantes = [str(ruta) for ruta in rutas.values() if not ruta.is_file()]
    if faltantes:
        raise FileNotFoundError('Faltan archivos oficiales:\n' + '\n'.join(faltantes))
    return rutas


def preparar_entrada(registro, campos=None):
    campos = CAMPOS_ENTRADA if campos is None else campos
    reservados = {campo.casefold() for campo in campos} & CAMPOS_RESERVADOS
    if reservados:
        raise ValueError('Campos de evaluación prohibidos en la entrada: ' + ', '.join(sorted(reservados)))
    if 'id' not in campos or 'formato' not in campos:
        raise ValueError('La entrada debe conservar id y formato')
    entrada = {campo: registro[campo] for campo in campos if campo in registro}

    def revisar(valor):
        if isinstance(valor, dict):
            if {str(k).casefold() for k in valor} & CAMPOS_RESERVADOS:
                raise ValueError('La entrada contiene un campo de respuesta o legal_basis')
            for hijo in valor.values():
                revisar(hijo)
        elif isinstance(valor, list):
            for hijo in valor:
                revisar(hijo)

    revisar(entrada)
    if 'id' not in entrada or 'formato' not in entrada:
        raise ValueError('La pregunta oficial no contiene id y formato. Revisar el contrato del lunes.')
    campos_metadata = {'id', 'formato', 'area', 'subtarea', 'complejidad', 'opciones'}
    if not any(isinstance(v, str) and v.strip() for k, v in entrada.items() if k not in campos_metadata):
        raise ValueError('No se encontró el texto de la pregunta. Revisar --input-fields con el archivo oficial.')
    return entrada


def generar(entrada, command=None, endpoint=None, timeout=180):
    if bool(command) == bool(endpoint):
        raise ValueError('Usar un comando o un endpoint local')
    cuerpo = json.dumps(entrada, ensure_ascii=False).encode('utf-8')
    inicio = time.perf_counter()
    if command:
        if not isinstance(command, list) or not all(isinstance(s, str) for s in command):
            raise ValueError('command debe ser una lista JSON de argumentos')
        entorno = dict(os.environ)
        entorno.pop('OPENROUTER_API_KEY', None)
        proceso = subprocess.run(command, input=cuerpo, capture_output=True, timeout=timeout, check=False, env=entorno)
        if proceso.returncode:
            detalle = proceso.stderr.decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'El generador terminó con código {proceso.returncode}: {detalle[:1000]}')
        contenido = proceso.stdout.decode('utf-8')
    else:
        destino = urllib.parse.urlparse(endpoint)
        if destino.scheme != 'http' or destino.hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('El endpoint debe ser HTTP local')
        if destino.username or destino.password:
            raise ValueError('El endpoint no admite credenciales en la URL')
        peticion = urllib.request.Request(endpoint, data=cuerpo, headers={'Content-Type': 'application/json'})

        class SinRedireccion(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                raise ValueError('El endpoint local no debe redirigir la petición')

        cliente = urllib.request.build_opener(urllib.request.ProxyHandler({}), SinRedireccion())
        with cliente.open(peticion, timeout=timeout) as respuesta:
            contenido = respuesta.read().decode('utf-8')
    latencia = time.perf_counter() - inicio
    try:
        salida = json.loads(contenido)
    except json.JSONDecodeError as error:
        raise ValueError('El generador debe devolver solo un objeto JSON, sin Markdown ni registros por stdout') from error
    if not isinstance(salida, dict):
        raise ValueError('El generador no devolvió un objeto JSON')
    if type(salida.get('id')) is not type(entrada['id']) or salida.get('id') != entrada['id'] or salida.get('formato') != entrada['formato']:
        raise ValueError('El generador cambió el id o el formato')
    json.dumps(salida, allow_nan=False)
    return salida, latencia


def validar_esquema(registros, ruta):
    try:
        import jsonschema
    except ImportError as error:
        raise RuntimeError('Falta jsonschema. Instalar con python -m pip install jsonschema') from error
    esquema = json.loads(Path(ruta).read_text(encoding='utf-8-sig'))
    clase = jsonschema.validators.validator_for(esquema)
    clase.check_schema(esquema)
    validador = clase(esquema)
    errores = []
    for registro in registros:
        for error in validador.iter_errors(registro):
            errores.append({'id': registro.get('id'), 'campo': '.'.join(map(str, error.path)), 'error': error.message})
    return errores


def sin_tiempos(valor):
    if isinstance(valor, dict):
        return {k: sin_tiempos(v) for k, v in valor.items() if k not in CAMPOS_TIEMPO}
    if isinstance(valor, list):
        return [sin_tiempos(v) for v in valor]
    return valor


def comparar_corridas(primera, segunda):
    a = {json.dumps(r['id'], sort_keys=True): sin_tiempos(r) for r in primera}
    b = {json.dumps(r['id'], sort_keys=True): sin_tiempos(r) for r in segunda}
    diferencias = [json.loads(clave) for clave in sorted(set(a) | set(b)) if a.get(clave) != b.get(clave)]
    return {'iguales': not diferencias, 'ids_distintos': diferencias, 'campos_excluidos': sorted(CAMPOS_TIEMPO)}


def normalizar(texto):
    texto = unicodedata.normalize('NFD', str(texto).casefold())
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', texto.replace('_', ' ')).strip()


PATRON_NORMA = re.compile(r'\b(decreto(?:[\s\-‐‑–—]+ley)?|ley)\s+(\d+)\s+de\s+(\d{4})\b')


def clave_norma(nombre):
    nombre = normalizar(nombre)
    coincidencia = PATRON_NORMA.fullmatch(nombre)
    if coincidencia:
        tipo, numero, anio = coincidencia.groups()
        tipo = 'decreto' if tipo.startswith('decreto') else tipo
        return f'{tipo} {int(numero)} de {anio}'
    return nombre


def catalogo_normas(manifiesto):
    alias = {}
    docs = {}

    def agregar(nombre, doc_id):
        clave = clave_norma(nombre)
        if clave in alias and alias[clave] != doc_id:
            alias[clave] = None
        else:
            alias[clave] = doc_id

    for d in manifiesto:
        doc_id = d['doc_id']
        docs[doc_id] = doc_id
        docs[doc_id + '.txt'] = doc_id
        nombre = normalizar(f"{d.get('tipo', '')} {d.get('numero', '')} de {d.get('anio', '')}")
        titulo = normalizar(d.get('titulo', ''))
        if d.get('tipo') in {'ley', 'decreto', 'decreto ley', 'decreto-ley', 'constitucion', 'decision'}:
            agregar(nombre, doc_id)
            if titulo:
                agregar(titulo, doc_id)
    return alias, docs


def auditar_citas(respuesta, manifiesto):
    alias, docs = catalogo_normas(manifiesto)

    def textos(valor):
        if isinstance(valor, str):
            return [valor]
        if isinstance(valor, dict):
            return [t for v in valor.values() for t in textos(v)]
        if isinstance(valor, list):
            return [t for v in valor for t in textos(v)]
        return []

    contenido = '\n'.join(textos({k: v for k, v in respuesta.items() if k not in {'id', 'formato', 'abstencion', 'pasajes_recuperados'}}))
    texto = normalizar(contenido)
    menciones = []
    for m in PATRON_NORMA.finditer(texto):
        menciones.append((m.start(), m.end(), m.group(), alias.get(clave_norma(m.group()))))
    for nombre, doc_id in sorted(alias.items(), key=lambda x: -len(x[0])):
        for m in re.finditer(r'(?<!\w)' + re.escape(nombre) + r'(?!\w)', texto):
            if any(m.start() < fin and m.end() > inicio for inicio, fin, _, _ in menciones):
                continue
            menciones.append((m.start(), m.end(), nombre, doc_id))
    citas = []
    for inicio, fin, nombre, doc_id in sorted(menciones):
        antes = texto[max(0, inicio - 100):inicio]
        numero = re.search(r'articulo\s+(\d+(?:\.\d+)*[a-z]?)[º°]?\s+(?:de\s+la|de|del)\s*$', antes)
        articulo = numero.group(1) if numero else None
        if articulo is None:
            numero = re.match(r'\s*[,.:]?\s*articulo\s+(\d+(?:\.\d+)*[a-z]?)', texto[fin:fin + 70])
            articulo = numero.group(1) if numero else None
        citas.append({'norma': nombre, 'doc_id': doc_id, 'articulo': articulo})
    pasajes = respuesta.get('pasajes_recuperados', [])
    if not isinstance(pasajes, list):
        pasajes = []
    for cita in citas:
        cita['indeterminada'] = cita['doc_id'] is None
        if cita['indeterminada']:
            cita['pasajes'] = []
            cita['respaldada'] = None
            continue
        indices = []
        for indice, pasaje in enumerate(pasajes[:10]):
            if not isinstance(pasaje, dict):
                continue
            doc = docs.get(pasaje.get('doc_id'))
            contenido_pasaje = normalizar(pasaje.get('texto', ''))
            norma_presente = bool(cita['doc_id'] and doc == cita['doc_id']) or bool(re.search(r'(?<!\w)' + re.escape(cita['norma']) + r'(?!\w)', contenido_pasaje))
            articulo = cita['articulo']
            articulo_presente = articulo is None or normalizar(pasaje.get('articulo', '')) == articulo or bool(re.search(r'\barticulo\s+' + re.escape(articulo) + r'(?!\w|\.\d)', contenido_pasaje))
            if norma_presente and articulo_presente and contenido_pasaje:
                indices.append(indice + 1)
        cita['pasajes'] = indices
        cita['respaldada'] = bool(indices)
    referencias_sin_norma = bool(re.search(r'\barticulos?\b', texto)) and not citas
    plural = bool(re.search(r'\barticulos\s+\d', texto))
    indeterminadas = sum(c['indeterminada'] for c in citas)
    return {
        'id': respuesta.get('id'),
        'citas': citas,
        'sin_respaldo': sum(c['respaldada'] is False for c in citas),
        'indeterminadas': indeterminadas,
        'revision_manual': referencias_sin_norma or plural or not citas or bool(indeterminadas),
        'alcance': 'Coincidencia de norma y artículo en los primeros 10 pasajes. No evalúa corrección jurídica.',
    }


def comando_evaluador(python, evaluador, submission, ragas=False):
    comando = [str(python), str(Path(evaluador).resolve()), '--submission', str(Path(submission).resolve()), '--split', 'sample']
    if ragas:
        comando.append('--ragas')
    return comando
