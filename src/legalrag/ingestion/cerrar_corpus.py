"""Cierra el corpus: descarga lo que el propio corpus cita y todavía no tiene.

La auditoría del 29 de septiembre dejó 2.865 referencias abiertas (1.351 sentencias T,
813 decretos, 435 leyes, 252 sentencias C, 8 actos legislativos y 6 SU). Tienen dos causas:

1. El ZIP 1 fue parcial: la ronda 5 ya tenía URL para parte de ellas (Gestor Normativo,
   jurisprudencia y Sala de Consulta del Gestor, circulares de Supersociedades y
   Superfinanciera), pero esos orígenes no entraron al inventario.
2. Cada descarga trae documentos que citan otros nuevos, y la última ronda no volvió a
   pasar por `objetivos`. Además, las sentencias se medían solo por citas desde normas
   (`--sentencias-por-primarias`), y la auditoría las mide contra todo el corpus.

Este script repite el ciclo hasta que no queda nada abierto o deja de avanzar:

    grafo -> objetivos (mismo criterio que la auditoría) -> reconstruir -> exclusiones

Al final corre la auditoría completa. La doctrina masiva (conceptos de Función Pública y
Supersociedades) sigue fuera, como se decidió, salvo con --con-doctrina.

Uso:
    python -m legalrag.ingestion.cerrar_corpus                 # hasta 3 iteraciones
    python -m legalrag.ingestion.cerrar_corpus --iteraciones 5 --hilos 4
    python -m legalrag.ingestion.cerrar_corpus --solo-grafo    # solo las citadas, sin las fuentes nuevas de la ronda 5
    python -m legalrag.ingestion.cerrar_corpus --solo-contar   # cuántas faltan, sin descargar
    python -m legalrag.ingestion.cerrar_corpus --omitir relatoria_cc   # si la relatoría bloquea
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

# Orígenes de la ronda 5 que ya entraron en el ZIP 1; con --solo-grafo no se suma ninguno más.
ORIGENES_ZIP1 = ["cc_datos_abiertos", "senado_arbol", "sic_circular_unica"]
RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
SRC = Path(__file__).resolve().parents[2]


def correr(modulo, *argumentos):
    """Ejecuta un paso del corpus como proceso aparte; los pasos usan sys.exit y argparse propios."""
    comando = [sys.executable, "-m", f"legalrag.ingestion.{modulo}", *map(str, argumentos)]
    print(f"\n$ {' '.join(comando[1:])}", flush=True)
    entorno = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    codigo = subprocess.run(comando, cwd=RAIZ, env=entorno).returncode
    if codigo:
        raise SystemExit(f"{modulo} terminó con código {codigo}; corregir y volver a correr (se reanuda).")


def tipo(clave):
    m = re.match(r"sentencia_(cc|csj|ce)_([a-z]+)\d", clave)
    return f"sentencia_{m.group(1)}_{m.group(2)}" if m else (re.match(r"([a-z_]+?)_\d", clave) or [None, clave])[1]


def abiertas(umbral_normas, umbral_sentencias):
    """Referencias citadas por encima del umbral que no están ni tienen exclusión (criterio de la auditoría)."""
    ruta = RAIZ / "data/raw/grafo_citas.csv"
    if not ruta.is_file():
        return None
    presentes = {re.sub(r"^co_", "", d["doc_id"])
                 for d in json.loads((RAIZ / "data/raw/manifest.json").read_text(encoding="utf-8"))}
    ruta_excl = RAIZ / "configs/corpus_exclusiones.json"
    excluidas = {e["clave"] for e in json.loads(ruta_excl.read_text(encoding="utf-8"))["exclusiones"]} \
        if ruta_excl.is_file() else set()
    faltan = []
    with ruta.open(encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            umbral = umbral_sentencias if fila["clave"].startswith("sentencia") else umbral_normas
            if (int(fila["documentos_que_citan"]) >= umbral and re.sub(r"^co_", "", fila["clave"]) not in presentes
                    and fila["clave"] not in excluidas):
                faltan.append(fila["clave"])
    return faltan


def informar(faltan, titulo):
    print(f"\n{titulo}: {len(faltan)} abiertas", flush=True)
    for t, n in Counter(tipo(c) for c in faltan).most_common():
        print(f"  {t:22} {n}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iteraciones", type=int, default=3)
    ap.add_argument("--umbral-normas", type=int, default=4, help="documentos que citan una ley o decreto")
    ap.add_argument("--umbral-sentencias", type=int, default=8, help="documentos que citan una sentencia")
    ap.add_argument("--hilos", type=int, default=4, help="documentos en paralelo al descargar")
    ap.add_argument("--omitir", nargs="*", default=[], help="servidores que no se tocan (p. ej. relatoria_cc)")
    ap.add_argument("--cc-senado", action="store_true", help="sentencias C primero al Senado")
    ap.add_argument("--con-doctrina", action="store_true", help="incluye la doctrina masiva de la ronda 5")
    ap.add_argument("--solo-grafo", action="store_true",
                    help="no suma las fuentes de la ronda 5 que faltaban (Gestor Normativo, etc.): solo lo citado")
    ap.add_argument("--solo-contar", action="store_true", help="recalcula el grafo y cuenta, sin descargar")
    args = ap.parse_args()

    if not (RAIZ / "data/raw/manifest.json").is_file():
        raise SystemExit("No está data/raw/manifest.json: primero python -m legalrag.ingestion.reconstruir")
    anteriores = None
    for iteracion in range(1, args.iteraciones + 1):
        print(f"\n===== Iteración {iteracion}/{args.iteraciones}", flush=True)
        correr("grafo", "--minimo", min(args.umbral_normas, args.umbral_sentencias))
        faltan = abiertas(args.umbral_normas, args.umbral_sentencias)
        informar(faltan, "Grafo")
        if args.solo_contar or not faltan:
            break
        if anteriores is not None and len(faltan) >= anteriores:
            print("No hubo avance respecto a la iteración anterior: lo que queda no se pudo descargar.")
            break
        anteriores = len(faltan)
        correr("objetivos", "--umbral-grafo", args.umbral_normas, "--umbral-sentencias", args.umbral_sentencias,
               *(["--con-doctrina"] if args.con_doctrina else []),
               *(["--ronda5-origenes", *ORIGENES_ZIP1] if args.solo_grafo else []))
        correr("reconstruir", "--hilos", args.hilos, *(["--cc-senado"] if args.cc_senado else []),
               *(["--omitir", *args.omitir] if args.omitir else []))
        correr("exclusiones")
    if not args.solo_contar:
        correr("auditoria", "--umbral-grafo", args.umbral_normas)
        informar(abiertas(args.umbral_normas, args.umbral_sentencias) or [], "Al cerrar")
        print("\nSiguiente paso: python -m legalrag.ingestion.empaquetar_raw   (ZIP para el Drive)")


if __name__ == "__main__":
    main()
