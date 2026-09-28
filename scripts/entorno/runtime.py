import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def sha256_archivo(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(8 * 1024 * 1024), b''):
            digest.update(bloque)
    return digest.hexdigest()


def crear_paquete(raiz, nombre='ai-week-colab.zip'):
    raiz = Path(raiz).resolve()
    if Path(nombre).name != nombre or not nombre.endswith('.zip'):
        raise ValueError('El paquete debe tener un nombre ZIP sin carpetas')
    carpetas = ['notebooks', 'scripts', 'configs', 'data/raw', 'data/processed/corpus', 'data/oficial']
    rutas = []
    for carpeta in carpetas:
        for p in (raiz / carpeta).rglob('*'):
            if not p.is_file() or p.is_symlink():
                continue
            if any(x.startswith('.') or x == '__pycache__' for x in p.relative_to(raiz).parts):
                continue
            if p.suffix.lower() not in {'.py', '.ipynb', '.json', '.jsonl', '.txt', '.csv', '.md', '.pdf', '.html', '.htm'}:
                continue
            rutas.append(p)
    for archivo_raiz in ['README.md', 'CORPUS.md', 'corpus_manifest.json', 'requirements.txt',
                         'requirements-experimentos.txt', 'requirements-colab.txt', 'reports/a_muestra.csv']:
        p = raiz / archivo_raiz
        if p.is_file():
            rutas.append(p)
    destino = raiz / 'data/colab' / nombre
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix('.zip.part')
    hashes = {}
    with zipfile.ZipFile(temporal, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as paquete:
        for p in sorted(set(rutas)):
            nombre = p.relative_to(raiz).as_posix()
            contenido = p.read_bytes()
            hashes[nombre] = hashlib.sha256(contenido).hexdigest()
            paquete.writestr(nombre, contenido)
        paquete.writestr('paquete.json', json.dumps({'sha256_archivos': hashes}, indent=2))
    temporal.replace(destino)
    return destino


def preparar_decoder_persistente(raiz, modelo, catalogo, servidor, fuente=None, cache=None):
    raiz = Path(raiz).resolve()
    destino = raiz / modelo['ruta']
    registro = destino.with_name('conversion.json')
    carpeta_cache = None
    if cache is not None and modelo['nombre'] == 'salamandra-7b-instruct':
        carpeta_cache = (Path(cache) / modelo['nombre'] / modelo['revision'] /
                         catalogo['runtime']['commit'] / 'Q4_K_M')
        pesos_cache = carpeta_cache / destino.name
        registro_cache = carpeta_cache / 'conversion.json'
        if not destino.exists() and pesos_cache.is_file() and registro_cache.is_file():
            info = json.loads(registro_cache.read_text(encoding='utf-8'))
            coincide = (info.get('revision_modelo') == modelo['revision'] and
                        info.get('revision_runtime') == catalogo['runtime']['commit'] and
                        info.get('cuantizacion') == 'Q4_K_M' and
                        info.get('sha256') == sha256_archivo(pesos_cache))
            if not coincide:
                raise ValueError('La copia de Salamandra en Drive no coincide con su registro')
            destino.parent.mkdir(parents=True, exist_ok=True)
            temporal = destino.with_name(destino.name + '.part')
            print('Restaurando Salamandra desde la copia guardada', flush=True)
            shutil.copy2(pesos_cache, temporal)
            if sha256_archivo(temporal) != info['sha256']:
                raise ValueError('La copia local de Salamandra quedó incompleta')
            temporal.replace(destino)
            shutil.copy2(registro_cache, registro)
    preparado = preparar_decoder(raiz, modelo, catalogo, servidor, fuente)
    if carpeta_cache is not None:
        carpeta_cache.mkdir(parents=True, exist_ok=True)
        info = json.loads(registro.read_text(encoding='utf-8'))
        pesos_cache = carpeta_cache / destino.name
        if not pesos_cache.is_file() or sha256_archivo(pesos_cache) != info['sha256']:
            print('Guardando Salamandra para la siguiente sesión', flush=True)
            temporal = pesos_cache.with_name(pesos_cache.name + '.part')
            shutil.copy2(preparado, temporal)
            if sha256_archivo(temporal) != info['sha256']:
                raise ValueError('La copia persistente de Salamandra quedó incompleta')
            temporal.replace(pesos_cache)
        temporal = carpeta_cache / 'conversion.json.tmp'
        shutil.copy2(registro, temporal)
        temporal.replace(carpeta_cache / 'conversion.json')
    return preparado


def hardware():
    try:
        proceso = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap',
                                  '--format=csv,noheader'], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError('No se detectó una GPU NVIDIA. En Colab seleccionar GPU A100 y reconectar') from error
    return proceso.stdout.strip()


def runtime(raiz, catalogo):
    raiz = Path(raiz)
    if os.name == 'nt':
        destino = raiz / catalogo['runtime']['ruta']
        if not (destino / 'llama-server.exe').is_file():
            from scripts.entorno.modelos import preparar_runtime
            preparar_runtime(catalogo)
        return destino / 'llama-server.exe', None
    commit = catalogo['runtime']['commit']
    fuente = raiz / 'data/runtime' / f'fuente-{commit}'
    if not (fuente / '.git').is_dir():
        fuente.mkdir(parents=True, exist_ok=True)
        for comando in [
            ['git', 'init', str(fuente)],
            ['git', '-C', str(fuente), 'remote', 'add', 'origin', 'https://github.com/ggml-org/llama.cpp.git'],
            ['git', '-C', str(fuente), 'fetch', '--depth', '1', 'origin', commit],
            ['git', '-C', str(fuente), 'checkout', '--detach', 'FETCH_HEAD'],
        ]:
            subprocess.run(comando, check=True)
    revision = subprocess.check_output(['git', '-C', str(fuente), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != commit:
        raise ValueError('La fuente de llama.cpp no coincide con el catálogo')
    servidor = fuente / 'build/bin/llama-server'
    if not servidor.is_file():
        arquitectura = hardware().splitlines()[0].split(',')[-1].strip().replace('.', '')
        subprocess.run(['cmake', '-S', str(fuente), '-B', str(fuente / 'build'), '-DGGML_CUDA=ON',
                        '-DBUILD_SHARED_LIBS=OFF', '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=OFF',
                        '-DCMAKE_BUILD_TYPE=Release', f'-DCMAKE_CUDA_ARCHITECTURES={arquitectura}'], check=True)
        subprocess.run(['cmake', '--build', str(fuente / 'build'), '--target', 'llama-server',
                        'llama-quantize', '-j', str(min(os.cpu_count() or 2, 4))], check=True)
    subprocess.run([str(servidor), '--version'], check=True)
    return servidor, fuente


def preparar_decoder(raiz, modelo, catalogo, servidor, fuente=None):
    from scripts.entorno.modelos import descargar

    raiz = Path(raiz)
    if modelo['parametros'] > 8000000000 or modelo['licencia'] != 'apache-2.0':
        raise ValueError('Decoder fuera del catálogo abierto admitido')
    destino = raiz / modelo['ruta']
    if modelo['nombre'] != 'salamandra-7b-instruct':
        for archivo in modelo['archivos']:
            descargar(archivo)
        return destino
    registro = destino.with_name('conversion.json')
    if destino.is_file() and registro.is_file():
        info = json.loads(registro.read_text(encoding='utf-8'))
        if (info['revision_modelo'] == modelo['revision'] and info['revision_runtime'] == catalogo['runtime']['commit']
                and sha256_archivo(destino) == info['sha256']):
            return destino
        raise ValueError('La conversión existente de Salamandra no coincide con su registro')
    if shutil.disk_usage(raiz).free < 45 * 2**30:
        raise RuntimeError('Se requieren 45 GiB libres para descargar y convertir Salamandra')
    for archivo in modelo['archivos']:
        descargar(archivo)
    if fuente is None:
        from scripts.entorno.modelos import preparar_fuente
        fuente = preparar_fuente(catalogo)
    entorno = dict(os.environ, PYTHONPATH=str(fuente / 'gguf-py'), PYTHONIOENCODING='utf-8')
    intermedio = destino.with_name('salamandra-F16.gguf')
    temporal = destino.with_suffix('.gguf.part')
    cuantizador = Path(servidor).with_name('llama-quantize.exe' if os.name == 'nt' else 'llama-quantize')
    subprocess.run([sys.executable, str(fuente / 'convert_hf_to_gguf.py'), str(raiz / modelo['snapshot']),
                    '--outfile', str(intermedio), '--outtype', 'f16', '--use-temp-file'], env=entorno, check=True)
    subprocess.run([str(cuantizador), str(intermedio), str(temporal), 'Q4_K_M', '4'], check=True)
    temporal.replace(destino)
    registro.write_text(json.dumps({'repo_id': modelo['repo_id'], 'revision_modelo': modelo['revision'],
                                    'revision_runtime': catalogo['runtime']['commit'], 'cuantizacion': 'Q4_K_M',
                                    'sha256': sha256_archivo(destino), 'bytes': destino.stat().st_size}, indent=2), encoding='utf-8')
    if intermedio.resolve().is_relative_to(raiz.resolve()):
        intermedio.unlink()
    return destino
