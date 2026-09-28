import argparse
from pathlib import Path
import sys

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))


def main():
    parser = argparse.ArgumentParser(description='Experimentos con BGE-M3 y Salamandra')
    parser.add_argument('--resultados', type=Path, default=RAIZ / 'data/experimentos_sistema')
    sub = parser.add_subparsers(dest='accion', required=True)
    recuperar = sub.add_parser('recuperar', help='Preparar evidencia para las 50 preguntas')
    recuperar.add_argument('--fuente-indice', type=Path, action='append', default=[])
    recuperar.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    recuperar.add_argument('--batch-size', type=int, default=4)
    recuperar.add_argument('--permitir-construir', action='store_true')
    generar = sub.add_parser('generar', help='Generar y evaluar las respuestas')
    generar.add_argument('--recuperacion', type=Path, required=True)
    generar.add_argument('--reanudar', default='')
    generar.add_argument('--cache-modelos', type=Path)
    generar.add_argument('--permitir-cpu', action='store_true')
    generar.add_argument('--timeout', type=int, default=600)
    evaluar = sub.add_parser('evaluar', help='Reconstruir las métricas sin generar')
    evaluar.add_argument('carpeta', type=Path)
    evaluar.add_argument('--ragas', action='store_true', help='Usar el juez oficial y su llave')
    evaluar.add_argument('--variante', help='Variante para el juez oficial')
    evaluar.add_argument('--repetir-ragas', action='store_true', help='Repetir el juez y consumir presupuesto')
    for comando in (recuperar, generar):
        comando.add_argument('--variantes', nargs='+', choices=['densa', 'hibrida', 'densa_reranker'])
    args = parser.parse_args()
    from scripts.experimentos.orquestador import (
        preparar_recuperaciones, ejecutar_experimentos, evaluar_experimentos, evaluar_ragas,
    )
    if args.accion == 'recuperar':
        carpeta = preparar_recuperaciones(RAIZ, args.resultados, args.fuente_indice, args.variantes,
                                         device=args.device, batch_size=args.batch_size,
                                         permitir_construir=args.permitir_construir)
    elif args.accion == 'generar':
        carpeta = ejecutar_experimentos(RAIZ, args.recuperacion, args.resultados, args.variantes,
                                        reanudar=args.reanudar, permitir_cpu=args.permitir_cpu,
                                        cache_modelos=args.cache_modelos, timeout=args.timeout)
    else:
        carpeta = args.carpeta
        if args.ragas:
            if not args.variante:
                parser.error('--ragas requiere --variante')
            evaluar_ragas(RAIZ, carpeta, args.variante, repetir=args.repetir_ragas)
        else:
            evaluar_experimentos(RAIZ, carpeta)
    print(carpeta)


if __name__ == '__main__':
    main()
