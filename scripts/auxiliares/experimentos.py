import csv
import gc
import importlib.metadata
import json
import platform
import shutil
import socket
import statistics
import time
import urllib.error
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.auxiliares.oficial import (
    Evidencia, cargar_muestra, consulta, evaluar_entrega, generar_oficial,
    guardar_json, huellas, metricas_recuperacion,
)
from scripts.auxiliares.recuperacion import hash_json, leer_jsonl, sha256


def _leer(ruta):
    return json.loads(Path(ruta).read_text(encoding='utf-8'))


def _ahora():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def _csv(ruta, filas):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    campos = list(dict.fromkeys(k for fila in filas for k in fila))
    temporal = ruta.with_name(ruta.name + '.tmp')
    with temporal.open('w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(filas)
    temporal.replace(ruta)


def _versiones():
    versiones = {'python': platform.python_version(), 'sistema': platform.system()}
    for paquete in ('torch', 'transformers', 'numpy', 'faiss-cpu', 'jsonschema'):
        try:
            versiones[paquete] = importlib.metadata.version(paquete)
        except importlib.metadata.PackageNotFoundError:
            versiones[paquete] = None
    return versiones


def _configuracion(raiz, variantes=None):
    config = _leer(Path(raiz) / 'configs/experimentos.json')
    if config['encoder'] != 'bge-m3' or config['decoder'] != 'salamandra-7b-instruct':
        raise ValueError('Este experimento usa BGE-M3 y Salamandra')
    if config['contexto'] != 8192 or config['max_tokens'] != 1024:
        raise ValueError('El contexto debe ser 8192 y la salida 1024')
    if config.get('temperatura', 0) != 0 or config.get('semilla', 0) != 0:
        raise ValueError('La temperatura y la semilla deben ser cero')
    elegidas = config['variantes']
    if variantes is not None:
        nombres = list(variantes)
        if len(set(nombres)) != len(nombres):
            raise ValueError('Hay variantes repetidas')
        catalogo = {v['nombre']: v for v in elegidas}
        if set(nombres) - catalogo.keys():
            raise ValueError('Variante desconocida')
        elegidas = [catalogo[n] for n in nombres]
    if not elegidas:
        raise ValueError('Seleccionar al menos una variante')
    for variante in elegidas:
        nombre = variante['nombre']
        if not nombre or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for c in nombre):
            raise ValueError('Nombre de variante inválido')
        if variante.get('max_unidades', config.get('max_unidades', 5)) != 5:
            raise ValueError('Se conservan cinco artículos por pregunta')
        if variante['candidatos'] < 5:
            raise ValueError('Faltan candidatos para los cinco artículos')
    return config, elegidas


def _huellas(raiz):
    return {**huellas(raiz), 'configs/experimentos.json': sha256(Path(raiz) / 'configs/experimentos.json')}


def _compatible(manifiesto, ficha, corpus_hash, config):
    if not isinstance(manifiesto, dict):
        return False
    opciones = manifiesto.get('configuracion', {})
    if not isinstance(opciones, dict):
        return False
    encoder = opciones.get('encoder', {})
    if not isinstance(encoder, dict):
        return False
    return (
        manifiesto.get('sha256_corpus') == corpus_hash
        and encoder.get('repo_id') == ficha['repo_id']
        and encoder.get('revision') == ficha['revision']
        and opciones.get('tamano_tokens') == config['tamano_tokens']
        and opciones.get('solapamiento_tokens') == config['solapamiento_tokens']
        and opciones.get('precision', 'float32') == config['precision']
    )


def _verificar_indice(carpeta, manifiesto):
    nombres = {'indice.faiss', 'vectores.npy', 'ventanas.jsonl', 'unidades.jsonl'}
    hashes = manifiesto.get('sha256_archivos', {})
    if set(hashes) != nombres or hash_json(hashes) != manifiesto.get('sha256_indice'):
        raise ValueError('Manifiesto del índice incompleto o alterado')
    config = {k: v for k, v in manifiesto['configuracion'].items()
              if k not in ('salida', 'corpus', 'seleccion')}
    if hash_json(config) != manifiesto['sha256_configuracion']:
        raise ValueError('Cambió la configuración del índice')
    for nombre, esperado in hashes.items():
        if sha256(carpeta / nombre) != esperado:
            raise ValueError('Cambió el archivo del índice: ' + nombre)
    if hash_json(leer_jsonl(carpeta / 'unidades.jsonl')) != manifiesto['sha256_corpus']:
        raise ValueError('El contenido del índice no coincide con su corpus')


def _buscar_indice(raiz, fuentes, ficha, corpus_hash, config):
    fuentes = [Path(f) for f in fuentes]
    carpetas, paquetes = set(), set()
    for fuente in fuentes:
        if fuente.is_dir():
            carpetas.update(p.parent for p in fuente.rglob('manifest.json'))
            paquetes.update(fuente.rglob('*.zip'))
        elif fuente.suffix.lower() == '.zip' and fuente.is_file():
            paquetes.add(fuente)
        elif fuente.name == 'manifest.json' and fuente.is_file():
            carpetas.add(fuente.parent)
    for carpeta in sorted(carpetas):
        manifiesto = _leer(carpeta / 'manifest.json')
        if _compatible(manifiesto, ficha, corpus_hash, config):
            _verificar_indice(carpeta, manifiesto)
            return carpeta, manifiesto
    for paquete in sorted(paquetes):
        with zipfile.ZipFile(paquete) as archivo:
            for nombre in archivo.namelist():
                if Path(nombre).name != 'manifest.json':
                    continue
                manifiesto = json.loads(archivo.read(nombre))
                if not _compatible(manifiesto, ficha, corpus_hash, config):
                    continue
                identificador = manifiesto.get('sha256_indice', '')
                if len(identificador) != 64 or any(c not in '0123456789abcdef' for c in identificador):
                    raise ValueError('Hash del índice inválido')
                destino = Path(raiz) / 'data/index/experimentos' / identificador
                destino.mkdir(parents=True, exist_ok=True)
                if not (destino / 'manifest.json').is_file():
                    prefijo = nombre[:-len('manifest.json')]
                    permitidos = {'indice.faiss', 'vectores.npy', 'ventanas.jsonl', 'unidades.jsonl'}
                    if set(manifiesto.get('sha256_archivos', {})) != permitidos:
                        raise ValueError('El ZIP no contiene un índice completo')
                    for base in sorted(permitidos):
                        temporal = destino / (base + '.tmp')
                        with archivo.open(prefijo + base) as entrada, temporal.open('wb') as salida:
                            shutil.copyfileobj(entrada, salida, length=8 * 1024 * 1024)
                        temporal.replace(destino / base)
                    _verificar_indice(destino, manifiesto)
                    guardar_json(destino / 'manifest.json', manifiesto)
                _verificar_indice(destino, manifiesto)
                return destino, manifiesto
    return None, None


def _liberar(recuperador=None, reordenador=None):
    if reordenador is not None:
        reordenador.cerrar()
    if recuperador is not None:
        recuperador.encoder = None
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _guardar_indice(carpeta, manifiesto, resultados):
    destino = resultados / 'indices' / (manifiesto['sha256_indice'] + '.zip')
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.is_file():
        return destino
    temporal = destino.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporal, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as paquete:
        for nombre in ['manifest.json', *sorted(manifiesto['sha256_archivos'])]:
            paquete.write(carpeta / nombre, arcname=nombre)
    with zipfile.ZipFile(temporal) as paquete:
        if paquete.testzip() is not None:
            raise ValueError('El ZIP del índice quedó incompleto')
    temporal.replace(destino)
    return destino


def _verificar_recuperaciones(carpeta):
    config = _leer(carpeta / 'configuracion.json')
    if _leer(carpeta / 'estado.json')['estado'] != 'completo':
        raise ValueError('La recuperación quedó incompleta')
    for nombre, esperado in config['archivos'].items():
        ruta = (carpeta / nombre).resolve()
        if not ruta.is_relative_to(carpeta.resolve()) or sha256(ruta) != esperado:
            raise ValueError('Cambió un archivo de recuperación')
    return config


def preparar_recuperaciones(raiz, resultados, fuentes_indices=(), variantes=None,
                           device='cuda', batch_size=4, permitir_construir=False, reanudar=''):
    from scripts.auxiliares.recuperacion import BM25, cargar_corpus, cargar_indice, construir_indice
    from scripts.auxiliares.entorno import hardware

    raiz, resultados = Path(raiz).resolve(), Path(resultados).resolve()
    config, elegidas = _configuracion(raiz, variantes)
    catalogo = _leer(raiz / 'configs/modelos.json')
    ficha = next(f for f in catalogo['encoders'] if f['nombre'] == 'bge-m3')
    corpus = raiz / 'data/processed/corpus'
    unidades, _ = cargar_corpus(corpus)
    equipo = hardware() if str(device).startswith('cuda') else 'CPU: ' + platform.processor()
    identidad = {'huellas': _huellas(raiz), 'sha256_corpus': hash_json(unidades),
                 'encoder': ficha, 'configuracion': config, 'variantes': elegidas,
                 'device': device, 'hardware': equipo, 'versiones': _versiones()}
    firma = hash_json(identidad)
    if reanudar and Path(reanudar).name != reanudar:
        raise ValueError('Reanudar debe ser el nombre de la recuperación')
    anteriores = [resultados / 'recuperaciones' / reanudar] if reanudar else sorted(
        (resultados / 'recuperaciones').glob('*'), reverse=True)
    for anterior in anteriores:
        if not (anterior / 'configuracion.json').is_file():
            continue
        if _leer(anterior / 'configuracion.json').get('firma') != firma:
            if reanudar:
                raise ValueError('La configuración de recuperación cambió')
            continue
        if (anterior / 'estado.json').is_file() and _leer(anterior / 'estado.json')['estado'] == 'completo':
            _verificar_recuperaciones(anterior)
            print('Recuperación reutilizada:', anterior.name, flush=True)
            return anterior
    if reanudar:
        raise ValueError('La recuperación no está completa. Preparar una nueva')
    fuentes = [*fuentes_indices, resultados / 'indices', raiz / 'data/index']
    ruta_indice, manifiesto = _buscar_indice(raiz, fuentes, ficha, identidad['sha256_corpus'], config)
    if ruta_indice is None and not permitir_construir:
        raise FileNotFoundError('No aparece un índice BGE-M3 compatible. Añadir su carpeta o ZIP a fuentes_indices. '
                                'Para construirlo, usar permitir_construir=True')
    import torch
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('No se detectó CUDA. Seleccionar una GPU antes de recuperar')
    carpeta = resultados / 'recuperaciones' / _ahora()
    carpeta.mkdir(parents=True, exist_ok=False)
    registro = {'firma': firma, 'identidad': identidad, 'archivos': {}}
    guardar_json(carpeta / 'configuracion.json', registro)
    guardar_json(carpeta / 'estado.json', {'estado': 'en_curso'})
    recuperador, reordenador = None, None
    try:
        if ruta_indice is None:
            ruta_indice = raiz / 'data/index/experimentos' / carpeta.name
            opciones = dict(corpus=str(corpus), salida=str(ruta_indice), encoder=ficha, device=device,
                            precision=config['precision'], tamano_tokens=config['tamano_tokens'],
                            solapamiento_tokens=config['solapamiento_tokens'], batch_size=batch_size, hibrido=False)
            manifiesto, recuperador = construir_indice(opciones)
            _guardar_indice(ruta_indice, manifiesto, resultados)
        else:
            recuperador = cargar_indice(ruta_indice, device=device)
        registro.update(indice=str(ruta_indice), sha256_indice=manifiesto['sha256_indice'])
        preguntas, entradas = cargar_muestra(raiz)
        guardar_json(carpeta / 'entradas.json', entradas)
        evidencia = Evidencia(raiz, corpus)
        referencias = {q['id']: q for q in preguntas}
        filas_metricas = []
        for variante in elegidas:
            nombre = variante['nombre']
            recuperador.bm25 = BM25([v['texto_busqueda'] for v in recuperador.ventanas]) if variante['hibrido'] else None
            if variante.get('reranker') and reordenador is None:
                from scripts.auxiliares.reordenamiento import Reordenador
                reordenador = Reordenador(config['reranker'], device=device)
            filas = []
            for numero, entrada in enumerate(entradas, 1):
                inicio = time.perf_counter()
                candidatos = recuperador.buscar(consulta(entrada), k=variante['candidatos'])
                recuperados = reordenador.ordenar(consulta(entrada), candidatos, k=5) if variante.get('reranker') else candidatos[:5]
                pasajes, avisos = evidencia.preparar(recuperados, max_unidades=5)
                evidencia.verificar(pasajes)
                segundos = time.perf_counter() - inicio
                metrica = metricas_recuperacion(referencias[entrada['id']], recuperados, evidencia.documentos, evidencia.citas)
                metrica.pop('recall_normas_10', None)
                metrica.update(variante=nombre, segundos=segundos, candidatos=len(candidatos),
                               articulos=len(recuperados), avisos=len(avisos),
                               reranker_truncados=sum(bool(p.get('reranker_truncado')) for p in candidatos)
                               if not variante.get('reranker') else reordenador.ultimo_resumen.get('truncados', 0))
                filas.append({'id': entrada['id'], 'pasajes': pasajes, 'avisos': avisos, 'metricas': metrica,
                              'candidatos': [{'unidad_id': p['unidad_id'], 'score': p['score']} for p in candidatos],
                              'seleccionados': [{k: p[k] for k in ('unidad_id', 'score', 'rerank_score', 'reranker_truncado')
                                                if k in p} for p in recuperados]})
                guardar_json(carpeta / 'recuperaciones' / f'{nombre}.json', filas)
                filas_metricas.append(metrica)
                print(f'{nombre}: {numero}/{len(entradas)} recuperadas', flush=True)
        _csv(carpeta / 'recuperacion_por_pregunta.csv', filas_metricas)
        archivos = ['entradas.json', 'recuperacion_por_pregunta.csv'] + [f"recuperaciones/{v['nombre']}.json" for v in elegidas]
        registro['archivos'] = {p: sha256(carpeta / p) for p in archivos}
        guardar_json(carpeta / 'configuracion.json', registro)
        guardar_json(carpeta / 'estado.json', {'estado': 'completo', 'preguntas': len(entradas)})
        guardar_json(resultados / 'ultima_recuperacion.json', {'carpeta': str(carpeta.relative_to(resultados))})
        return carpeta
    except BaseException as error:
        guardar_json(carpeta / 'estado.json', {'estado': 'incompleto', 'error': type(error).__name__, 'detalle': str(error)})
        raise
    finally:
        _liberar(recuperador, reordenador)


def _checkpoint(ruta, entrada, firma, evidencia):
    if not ruta.is_file():
        return None
    fila = _leer(ruta)
    contenido = {k: v for k, v in fila.items() if k != 'sha256_contenido'}
    if fila.get('huella_ejecucion') != firma or fila.get('sha256_contenido') != hash_json(contenido):
        raise ValueError('Checkpoint incompatible o alterado: ' + str(ruta))
    respuesta = fila['respuesta']
    if respuesta.get('id') != entrada['id'] or respuesta.get('formato') != entrada['formato']:
        raise ValueError('Checkpoint de otra pregunta: ' + str(ruta))
    import jsonschema
    from scripts.auxiliares.generacion import esquema_local
    contenido_respuesta = {k: v for k, v in respuesta.items() if k not in ('id', 'formato', 'pasajes_recuperados')}
    jsonschema.validate(contenido_respuesta, esquema_local(entrada['formato']))
    evidencia.verificar(respuesta['pasajes_recuperados'])
    return fila


def _guardado_relativo(carpeta, resultados):
    return str(carpeta.relative_to(resultados)) if carpeta.is_relative_to(resultados) else str(carpeta)


def _carpeta_recuperacion(config, resultados):
    ruta = Path(config['recuperacion'])
    return ruta if ruta.is_absolute() else resultados / ruta


def _verificar_config_ejecucion(config):
    contenido = {k: v for k, v in config.items() if k not in ('firma', 'recuperacion')}
    if hash_json(contenido) != config['firma'] or config['recuperacion'] != config['base']['recuperacion']:
        raise ValueError('La configuración de ejecución fue alterada')


def ejecutar_experimentos(raiz, carpeta_recuperacion, resultados, variantes=None, reanudar='',
                         permitir_cpu=False, cache_modelos=None, timeout=600):
    from scripts.auxiliares.entorno import hardware, preparar_decoder_persistente, runtime
    from scripts.auxiliares.generacion import ServidorLocal

    raiz, resultados = Path(raiz).resolve(), Path(resultados).resolve()
    carpeta_recuperacion = Path(carpeta_recuperacion).resolve()
    config, elegidas = _configuracion(raiz, variantes)
    recup = _verificar_recuperaciones(carpeta_recuperacion)
    if recup['identidad']['huellas'] != _huellas(raiz):
        raise ValueError('Cambió el código o el corpus desde la recuperación. Preparar otra recuperación')
    entradas = _leer(carpeta_recuperacion / 'entradas.json')
    if entradas != cargar_muestra(raiz)[1]:
        raise ValueError('Las preguntas de recuperación no coinciden con la muestra')
    nombres = [v['nombre'] for v in elegidas]
    if any(f'recuperaciones/{n}.json' not in recup['archivos'] for n in nombres):
        raise ValueError('Falta la recuperación de alguna variante')
    base = {'huellas': _huellas(raiz), 'sha256_recuperacion': sha256(carpeta_recuperacion / 'configuracion.json'),
            'recuperacion': _guardado_relativo(carpeta_recuperacion, resultados), 'variantes': nombres,
            'contexto': config['contexto'], 'max_tokens': config['max_tokens'], 'temperatura': 0,
            'semilla': 0, 'timeout': timeout, 'permitir_cpu': permitir_cpu}
    if reanudar == 'auto':
        puntero = resultados / 'ultima_ejecucion.json'
        reanudar = Path(_leer(puntero)['carpeta']).name if puntero.is_file() else ''
    if reanudar and Path(reanudar).name != reanudar:
        raise ValueError('Reanudar debe ser el nombre de la ejecución')
    carpeta = resultados / 'ejecuciones' / (reanudar or _ahora())
    evidencia = Evidencia(raiz, raiz / 'data/processed/corpus')
    anterior = _leer(carpeta / 'configuracion.json') if reanudar else None
    if anterior is not None:
        _verificar_config_ejecucion(anterior)
    if anterior is not None and anterior['base'] != base:
        raise ValueError('La configuración cambió. Iniciar otra ejecución')
    if anterior is not None:
        completas = all(_checkpoint(carpeta / n / 'respuestas' / f"{p['id']}.json", p,
                                    anterior['firma'], evidencia) is not None for n in nombres for p in entradas)
        if completas:
            evaluar_experimentos(raiz, carpeta)
            guardar_json(carpeta / 'estado.json', {'estado': 'completo', 'variantes': nombres, 'preguntas': len(entradas)})
            return carpeta
    catalogo = _leer(raiz / 'configs/modelos.json')
    ficha = next(f for f in catalogo['decoders'] if f['nombre'] == 'salamandra-7b-instruct')
    if ficha['parametros'] > 8000000000 or ficha['licencia'] != 'apache-2.0':
        raise ValueError('Salamandra debe cumplir el límite y la licencia abierta')
    try:
        equipo = hardware()
    except RuntimeError:
        if not permitir_cpu:
            raise
        equipo = 'CPU: ' + platform.processor()
    servidor, fuente = runtime(raiz, catalogo)
    modelo = preparar_decoder_persistente(raiz, ficha, catalogo, servidor, fuente, cache=cache_modelos)
    ejecucion = {'base': base, 'hardware': equipo, 'versiones': _versiones(), 'decoder': ficha,
                 'sha256_modelo': sha256(modelo), 'sha256_runtime': sha256(servidor),
                 'runtime_commit': catalogo['runtime']['commit']}
    firma = hash_json(ejecucion)
    ejecucion['firma'] = firma
    ejecucion['recuperacion'] = base['recuperacion']
    if anterior is not None and anterior != ejecucion:
        raise ValueError('El modelo, runtime o hardware cambió. Iniciar otra ejecución')
    carpeta.mkdir(parents=True, exist_ok=bool(reanudar))
    guardar_json(carpeta / 'configuracion.json', ejecucion)
    guardar_json(resultados / 'ultima_ejecucion.json', {'carpeta': str(carpeta.relative_to(resultados)),
                                                       'recuperacion': base['recuperacion']})
    guardar_json(carpeta / 'estado.json', {'estado': 'en_curso', 'variantes': nombres})
    try:
        with ServidorLocal(servidor, modelo, carpeta / 'servidor', contexto=config['contexto'],
                           capas='0' if equipo.startswith('CPU:') else 'auto') as motor:
            motor.cliente.timeout = timeout
            if not permitir_cpu and (not motor.capas_gpu or motor.capas_gpu['cargadas'] == 0):
                raise RuntimeError('Salamandra no cargó capas en GPU. Revisar servidor/servidor.log')
            guardar_json(carpeta / 'servidor' / 'configuracion.json',
                         {'comando': motor.comando, 'capas_gpu': motor.capas_gpu, 'propiedades': motor.propiedades})
            for nombre in nombres:
                datos = {r['id']: r for r in _leer(carpeta_recuperacion / 'recuperaciones' / f'{nombre}.json')}
                for numero, entrada in enumerate(entradas, 1):
                    destino = carpeta / nombre / 'respuestas' / f"{entrada['id']}.json"
                    if _checkpoint(destino, entrada, firma, evidencia) is not None:
                        print(f'{nombre}: {numero}/{len(entradas)} guardada', flush=True)
                        continue
                    pasajes = datos[entrada['id']]['pasajes']
                    evidencia.verificar(pasajes)
                    inicio_generacion = time.perf_counter()
                    segundos_reintentos = 0.0
                    for intento in range(2):
                        raw = carpeta / nombre / 'raw' / str(entrada['id']) / str(intento + 1)
                        inicio_intento = time.perf_counter()
                        try:
                            respuesta, registro = generar_oficial(motor.cliente, entrada, pasajes, evidencia, raw,
                                                                  contexto=config['contexto'], max_tokens=config['max_tokens'])
                            registro['segundos_con_validacion'] = time.perf_counter() - inicio_generacion
                            registro['reintentos'] = intento
                            registro['intentos'] = intento + 1
                            registro['segundos_reintentos'] = segundos_reintentos
                            break
                        except (TimeoutError, socket.timeout, ConnectionError, urllib.error.URLError) as error:
                            if isinstance(error, urllib.error.HTTPError) or intento:
                                raise
                            segundos_reintentos += time.perf_counter() - inicio_intento
                            guardar_json(raw / 'reintento.json', {'error': type(error).__name__, 'detalle': str(error)})
                    fila = {'huella_ejecucion': firma, 'respuesta': respuesta, 'registro': registro}
                    fila['sha256_contenido'] = hash_json(fila)
                    guardar_json(destino, fila)
                    print(f'{nombre}: {numero}/{len(entradas)} respondidas', flush=True)
                evaluar_experimentos(raiz, carpeta, variantes=[nombre])
        evaluar_experimentos(raiz, carpeta)
        guardar_json(carpeta / 'estado.json', {'estado': 'completo', 'variantes': nombres, 'preguntas': len(entradas)})
        return carpeta
    except BaseException as error:
        guardar_json(carpeta / 'estado.json', {'estado': 'incompleto', 'error': type(error).__name__, 'detalle': str(error)})
        raise


def _promedio(filas, campo):
    valores = [f[campo] for f in filas if isinstance(f.get(campo), (int, float))]
    return sum(valores) / len(valores) if valores else None


def evaluar_experimentos(raiz, carpeta, variantes=None):
    from scripts.auxiliares.evaluacion import auditar_citas

    raiz, carpeta = Path(raiz).resolve(), Path(carpeta).resolve()
    config = _leer(carpeta / 'configuracion.json')
    _verificar_config_ejecucion(config)
    if config['base']['huellas'] != _huellas(raiz):
        raise ValueError('El código o las fuentes cambiaron desde esta ejecución')
    resultados = carpeta.parent.parent
    recuperacion = _carpeta_recuperacion(config, resultados)
    _verificar_recuperaciones(recuperacion)
    if sha256(recuperacion / 'configuracion.json') != config['base']['sha256_recuperacion']:
        raise ValueError('Cambió la recuperación de esta ejecución')
    preguntas, entradas = cargar_muestra(raiz)
    referencias = {q['id']: q for q in preguntas}
    evidencia = Evidencia(raiz, raiz / 'data/processed/corpus')
    todas, resumen = [], []
    seleccion = set(config['base']['variantes'] if variantes is None else variantes)
    if seleccion - set(config['base']['variantes']):
        raise ValueError('Variante desconocida en la ejecución')
    for nombre in config['base']['variantes']:
        destino = carpeta / nombre
        guardadas = [_checkpoint(destino / 'respuestas' / f"{p['id']}.json", p, config['firma'], evidencia) for p in entradas]
        guardadas = [g for g in guardadas if g is not None]
        reporte = None
        if len(guardadas) == len(entradas) and nombre in seleccion:
            reporte = evaluar_entrega(raiz, destino, [g['respuesta'] for g in guardadas])
            guardar_json(destino / 'metricas.json', reporte)
        elif (destino / 'metricas.json').is_file() and len(guardadas) == len(entradas):
            reporte = _leer(destino / 'metricas.json')
        diagnostico = {d['id']: d for d in _leer(destino / 'diagnostico_por_pregunta.json')} if reporte else {}
        recuperaciones = {p['id']: p for p in _leer(recuperacion / 'recuperaciones' / f'{nombre}.json')}
        filas, auditorias = [], []
        for guardada in guardadas:
            respuesta, registro = guardada['respuesta'], guardada['registro']
            identificador = respuesta['id']
            pregunta = referencias[identificador]
            detalle = diagnostico.get(identificador, {})
            metrica = recuperaciones[identificador]['metricas']
            auditoria = auditar_citas(respuesta, list(evidencia.documentos.values()))
            auditorias.append(auditoria)
            articulos_sin_respaldo = sum(c['articulo'] is not None and c['respaldada'] is False
                                        for c in auditoria['citas'])
            filas.append({'variante': nombre, 'id': identificador, 'area': pregunta['area'], 'formato': pregunta['formato'],
                          'abstencion': respuesta['abstencion'], 'acierto_cerrada': detalle.get('acierto_cerrada'),
                          'recall_normas_5': metrica['recall_normas_5'],
                          'segundos_recuperacion': metrica['segundos'],
                          'segundos_generacion': registro.get('segundos_con_validacion', registro.get('segundos')),
                          'reintentos': registro.get('reintentos', 0),
                          'segundos_reintentos': registro.get('segundos_reintentos', 0),
                          'abstencion_programatica': registro.get('abstencion_programatica', False),
                          'motivo': registro.get('motivo', ''), 'finish_reason': registro.get('finish_reason'),
                          'contextos_omitidos': len(registro.get('omitidos', [])),
                          'articulos_sin_respaldo': articulos_sin_respaldo,
                          'revision_articulos': auditoria['revision_manual'] or articulos_sin_respaldo > 0,
                          'json_valido': registro.get('json_valido'), 'claves_validas': registro.get('claves_validas'),
                          'avisos_fuente': metrica.get('avisos', 0),
                          'reranker_truncados': metrica.get('reranker_truncados', 0),
                          'citas_sin_respaldo_modelo': len(registro.get('citas_sin_respaldo_oficial', [])),
                          'tokens_entrada': registro.get('tokens_prompt_estimados'),
                          'tokens_salida': registro.get('uso', {}).get('completion_tokens'),
                          **{f'citas_{k}': v for k, v in detalle.get('citas', {}).items() if k != 'detalle'}})
        guardar_json(destino / 'auditoria_articulos.json', auditorias)
        todas.extend(filas)
        ragas = _leer(destino / 'metricas_ragas.json').get('correccion_ragas', {}) if (destino / 'metricas_ragas.json').is_file() else {}
        tiempos_generacion = [f['segundos_generacion'] for f in filas if f['segundos_generacion'] is not None]
        tiempos_totales = [f['segundos_generacion'] + f['segundos_recuperacion'] for f in filas
                          if f['segundos_generacion'] is not None]
        total_50 = reporte['total_automatico']['obtenidos'] if reporte else None
        ragas_completo = ragas.get('puntos') is not None and not ragas.get('n_fallidos', 0)
        resumen.append({'variante': nombre, 'estado': 'completo' if reporte else 'incompleto', 'respuestas': len(filas),
                        'preguntas': len(entradas), 'puntos_offline_50': reporte['total_automatico']['obtenidos'] if reporte else None,
                        'total_50': total_50,
                        'total_80': total_50 + ragas['puntos'] if total_50 is not None and ragas_completo else None,
                        'cerradas_20': reporte['cerradas']['puntos'] if reporte else None,
                        'citas_20': reporte['citas']['puntos'] if reporte else None,
                        'abstencion_10': reporte['abstencion']['puntos'] if reporte else None,
                        'ragas_30': ragas.get('puntos'),
                        'ragas_estado': 'evaluado' if ragas_completo else ('parcial' if ragas.get('puntos') is not None else 'pendiente'),
                        'ragas_fallidos': ragas.get('n_fallidos'),
                        'recall_normas_5': _promedio(filas, 'recall_normas_5'),
                        'segundos_generacion_media': _promedio(filas, 'segundos_generacion'),
                        'mediana_generacion_s': statistics.median(tiempos_generacion) if tiempos_generacion else None,
                        'mediana_total_estimado_s': statistics.median(tiempos_totales) if tiempos_totales else None,
                        'recuperacion_s': _promedio(filas, 'segundos_recuperacion'),
                        'reintentos': sum(f['reintentos'] for f in filas),
                        'abstenciones': sum(f['abstencion'] for f in filas),
                        'abstenciones_programaticas': sum(f['abstencion_programatica'] for f in filas),
                        'contextos_omitidos': sum(f['contextos_omitidos'] for f in filas),
                        'respuestas_con_omisiones': sum(f['contextos_omitidos'] > 0 for f in filas),
                        'articulos_sin_respaldo': sum(f['articulos_sin_respaldo'] for f in filas),
                        'respuestas_revision_articulos': sum(f['revision_articulos'] for f in filas),
                        'errores_esquema': reporte.get('errores_esquema') if reporte else None,
                        'abstenciones_cerradas_null': len(reporte['abstencion_cerrada_pendiente_aclaracion']) if reporte else None})
    _csv(carpeta / 'resumen.csv', resumen)
    _csv(carpeta / 'por_pregunta.csv', todas)
    for agrupacion in ('area', 'formato'):
        grupos = []
        for variante, grupo in sorted({(f['variante'], f[agrupacion]) for f in todas}):
            filas = [f for f in todas if f['variante'] == variante and f[agrupacion] == grupo]
            grupos.append({'variante': variante, agrupacion: grupo, 'preguntas': len(filas),
                           'acierto_cerradas': _promedio(filas, 'acierto_cerrada'),
                           'recall_normas_5': _promedio(filas, 'recall_normas_5'),
                           'recall_citas': _promedio(filas, 'citas_recall_citas'),
                           'precision_citas': _promedio(filas, 'citas_precision_citas'),
                           'abstenciones': sum(f['abstencion'] for f in filas)})
        _csv(carpeta / f'por_{agrupacion}.csv', grupos)
    return resumen


def evaluar_ragas(raiz, carpeta, variante, repetir=False):
    raiz, carpeta = Path(raiz).resolve(), Path(carpeta).resolve()
    config = _leer(carpeta / 'configuracion.json')
    if variante not in config['base']['variantes']:
        raise ValueError('Variante desconocida')
    evaluar_experimentos(raiz, carpeta, variantes=[variante])
    destino = carpeta / variante
    if not (destino / 'metricas.json').is_file():
        raise ValueError('Completar las 50 respuestas antes del juez oficial')
    identidad = {'firma': config['firma'], 'sha256_submission': sha256(destino / 'submissions.jsonl'),
                 'sha256_evaluador': sha256(raiz / 'data/oficial/scripts/evaluate.py')}
    huella = destino / 'evaluacion_ragas_configuracion.json'
    archivo_oficial = destino / 'evaluacion_con_ragas.json'
    if archivo_oficial.is_file() and not repetir:
        if not huella.is_file() or _leer(huella) != identidad:
            raise ValueError('El reporte del juez no tiene una huella compatible. Revisarlo antes de repetir el pago')
        oficial = _leer(archivo_oficial)
        determinista = _leer(destino / 'metricas.json')
        if oficial.get('validacion', {}).get('errores') != 0 or oficial.get('split') != 'sample':
            raise ValueError('El reporte guardado del juez tiene errores de validación')
        if any(oficial.get(k) != determinista.get(k) for k in ('cerradas', 'citas', 'abstencion')):
            raise ValueError('El reporte del juez corresponde a otra entrega')
        if oficial.get('correccion_ragas', {}).get('puntos') is None:
            raise ValueError('El reporte guardado no contiene evaluación del juez')
        reporte = {**oficial, 'errores_esquema': determinista['errores_esquema'],
                   'abstencion_cerrada_pendiente_aclaracion': determinista['abstencion_cerrada_pendiente_aclaracion']}
    else:
        guardar_json(huella, identidad)
        reporte = evaluar_entrega(raiz, destino, leer_jsonl(destino / 'submissions.jsonl'), ragas=True)
    guardar_json(destino / 'metricas_ragas.json', reporte)
    evaluar_experimentos(raiz, carpeta, variantes=[])
    return reporte
