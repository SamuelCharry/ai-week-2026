import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ))
from scripts.evaluacion.comparar import comparar_corridas
from scripts.evaluacion.entrega import (
    CAMPOS_ENTRADA,
    auditar_citas,
    cargar_jsonl,
    comando_evaluador,
    comprobar_oficiales,
    generar,
    preparar_entrada,
    validar_esquema,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluar las 50 preguntas oficiales cuando estén disponibles.')
    parser.add_argument('--official-dir', type=Path, default=RAIZ / 'data/oficial')
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument('--command', help='Lista JSON del comando. Recibe una pregunta por stdin y devuelve un objeto JSON por stdout.')
    modo.add_argument('--endpoint', help='Endpoint HTTP local que recibe y devuelve un objeto JSON.')
    parser.add_argument('--input-fields', default=','.join(CAMPOS_ENTRADA))
    parser.add_argument('--manifest', type=Path, default=RAIZ / 'data/raw/manifest.json')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--ragas', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()

    oficiales = comprobar_oficiales(args.official_dir)
    preguntas = cargar_jsonl(oficiales['preguntas'])
    if len(preguntas) != 50:
        raise ValueError(f'Se esperaban las 50 preguntas oficiales. El archivo tiene {len(preguntas)}. Revisar el material recibido.')
    campos = [campo.strip() for campo in args.input_fields.split(',') if campo.strip()]
    entradas = [preparar_entrada(pregunta, campos) for pregunta in preguntas]
    json.loads(oficiales['esquema'].read_text(encoding='utf-8-sig'))
    validar_esquema([], oficiales['esquema'])
    json.loads(oficiales['fuentes'].read_text(encoding='utf-8-sig'))
    print(f'Material oficial encontrado. {len(preguntas)} entradas disponibles.')
    if args.check_only:
        return 0
    if not args.command and not args.endpoint:
        raise ValueError('Falta --command o --endpoint para ejecutar el sistema')
    if args.timeout <= 0:
        raise ValueError('timeout debe ser mayor que cero')
    if args.ragas and not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError('Falta OPENROUTER_API_KEY para el evaluador oficial con --ragas')
    command = json.loads(args.command) if args.command else None
    manifiesto = json.loads(args.manifest.read_text(encoding='utf-8-sig'))
    carpeta = args.output_dir or RAIZ / 'data/evaluacion' / datetime.now().strftime('lunes_%Y%m%d_%H%M%S')
    carpeta.mkdir(parents=True, exist_ok=False)
    resumen = {
        'estado': 'en_ejecucion',
        'campos_entrada': campos,
        'archivos_oficiales': {k: {'ruta': str(v.resolve()), 'sha256': hashlib.sha256(v.read_bytes()).hexdigest()} for k, v in oficiales.items()},
        'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        'corridas': [],
    }
    corridas = []
    try:
        for numero in [1, 2]:
            respuestas = []
            tiempos = []
            ruta_salida = carpeta / f'submissions_{numero}.jsonl'
            with ruta_salida.open('w', encoding='utf-8', newline='\n') as f:
                for entrada in entradas:
                    respuesta, segundos = generar(entrada, command=command, endpoint=args.endpoint, timeout=args.timeout)
                    f.write(json.dumps(respuesta, ensure_ascii=False, allow_nan=False) + '\n')
                    f.flush()
                    errores = validar_esquema([respuesta], oficiales['esquema'])
                    if errores:
                        raise ValueError('Salida fuera del esquema oficial: ' + json.dumps(errores, ensure_ascii=False))
                    respuestas.append(respuesta)
                    tiempos.append({'id': entrada['id'], 'segundos': segundos})
                    print(f'Corrida {numero}, id {entrada["id"]}: {segundos:.2f} s', flush=True)
            auditoria = [auditar_citas(r, manifiesto) for r in respuestas]
            (carpeta / f'citas_{numero}.json').write_text(json.dumps(auditoria, ensure_ascii=False, indent=2), encoding='utf-8')
            (carpeta / f'latencias_{numero}.json').write_text(json.dumps(tiempos, ensure_ascii=False, indent=2), encoding='utf-8')
            segundos = [t['segundos'] for t in tiempos]
            resumen['corridas'].append({
                'numero': numero,
                'respuestas': len(respuestas),
                'abstenciones': sum(r['abstencion'] is True for r in respuestas),
                'latencia_media_s': statistics.mean(segundos),
                'latencia_mediana_s': statistics.median(segundos),
                'latencia_p95_s': sorted(segundos)[math.ceil(len(segundos) * .95) - 1],
                'latencia_max_s': max(segundos),
                'mas_de_22_s': sum(s > 22 for s in segundos),
                'estimacion_992_s': statistics.mean(segundos) * 992,
                'citas_sin_respaldo_mecanico': sum(a['sin_respaldo'] for a in auditoria),
                'respuestas_citas_revision_manual': sum(a['revision_manual'] for a in auditoria),
            })
            corridas.append(respuestas)
        resumen['determinismo'] = comparar_corridas(*corridas)
        for con_ragas in ([False, True] if args.ragas else [False]):
            comando = comando_evaluador(sys.executable, oficiales['evaluador'], carpeta / 'submissions_1.jsonl', ragas=con_ragas)
            etiqueta = 'con_ragas' if con_ragas else 'sin_ragas'
            with (carpeta / f'evaluador_{etiqueta}.log').open('w', encoding='utf-8') as registro:
                resultado = subprocess.run(comando, cwd=args.official_dir.resolve(), stdout=registro, stderr=subprocess.STDOUT, check=False)
            resumen[f'evaluador_{etiqueta}'] = {'comando': comando, 'codigo_salida': resultado.returncode}
            if resultado.returncode:
                raise RuntimeError(f'El evaluador oficial falló. Revisar evaluador_{etiqueta}.log')
        resumen['estado'] = 'completo' if resumen['determinismo']['iguales'] else 'revisar_determinismo'
    except Exception as error:
        resumen['estado'] = 'error'
        resumen['error'] = str(error)
        raise
    finally:
        (carpeta / 'resumen.json').write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Resultados guardados en {carpeta}')
    return 0 if resumen['determinismo']['iguales'] else 3


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f'No se pudo completar la evaluación. {error}', file=sys.stderr)
        raise SystemExit(2)
