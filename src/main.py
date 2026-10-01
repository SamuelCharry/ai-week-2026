"""Cerberus de punta a punta, en un solo comando: responde la muestra (o el examen) y pasa el evaluador oficial.

    python3 src/main.py                 # Mark 43 sobre las 50 de muestra + evaluador oficial
    python3 src/main.py --mark 42       # otro mark (ver MARKS)
    python3 src/main.py --split test    # las 992 del examen -> submissions.jsonl (sin evaluador)

Antes de empezar comprueba que estén el índice (`index/`), el corpus procesado (`data/processed/`), el manifiesto
(`corpus_manifest.json`) y el material oficial (`data/oficial/`). Cada corrida reemplaza la anterior del mismo mark
(se aparta como `<salida>.anterior.jsonl`).
"""
import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

# Opciones de cada mark (las mismas de `legalrag.cli run`); ver MARKS.md.
MARK_42 = {"hybrid": True, "use_hyde": False, "mc_thinking": True, "multi_query": True}
MARKS = {
    42: MARK_42,
    43: {**MARK_42, "herramientas_v2": True, "normalizador_citas": True},
}


def comprobar(config):
    faltan = [str(p.relative_to(RAIZ)) for p in (config.index_dir / "build.json", config.prepared,
                                                 RAIZ / "data/oficial/data/sample_50.jsonl")
              if not p.exists()]
    if config.normalizador_citas and not (RAIZ / "corpus_manifest.json").is_file():
        faltan.append("corpus_manifest.json (lo usa el normalizador de citas)")
    if faltan:
        sys.exit("Faltan datos en esta carpeta: " + ", ".join(faltan) + "\nEnlacen o copien index/, data/ y "
                 "corpus_manifest.json desde la carpeta donde se construyó el índice (la de Mark 42), por ejemplo:\n"
                 "  ln -s /ruta/a/mark42/index index && ln -s /ruta/a/mark42/data data && "
                 "ln -s /ruta/a/mark42/corpus_manifest.json corpus_manifest.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mark", type=int, default=max(MARKS), choices=sorted(MARKS))
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    ap.add_argument("--ragas", action="store_true", help="evalúa también texto libre (OPENROUTER_API_KEY)")
    args = ap.parse_args()

    from legalrag.silencio import silenciar
    silenciar()
    from legalrag.config import CONFIG
    from legalrag.pipeline import run

    config = replace(CONFIG, root=RAIZ, **MARKS[args.mark])
    comprobar(config)
    if args.split == "test":
        preguntas, salida = RAIZ / "data/oficial/data/test_992.jsonl", RAIZ / "submissions.jsonl"
    else:
        preguntas, salida = RAIZ / "data/oficial/data/sample_50.jsonl", RAIZ / f"salidas/mark{args.mark}.jsonl"
    if not preguntas.is_file():
        sys.exit(f"No está {preguntas.relative_to(RAIZ)}")
    if salida.exists():
        anterior = salida.with_name(salida.stem + ".anterior.jsonl")
        shutil.move(salida, anterior)
        for extra in (".run.json", ".trace.jsonl"):
            viejo = salida.with_suffix(extra)
            if viejo.exists():
                viejo.unlink()
        print(f"La corrida anterior quedó en {anterior.relative_to(RAIZ)}")
    print(f"Cerberus Mark {args.mark} · {preguntas.name} · opciones {MARKS[args.mark]}", flush=True)
    run(config, preguntas, salida, expected_count=992 if args.split == "test" else None)
    if args.split == "test":
        print(f"Entrega: {salida.relative_to(RAIZ)}")
        return
    reporte = salida.with_name(salida.stem + "_oficial.json")
    subprocess.run([sys.executable, str(RAIZ / "data/oficial/scripts/evaluate.py"), "--submission", str(salida),
                    "--split", "sample", "--out", str(reporte)] + (["--ragas"] if args.ragas else []),
                   check=True, capture_output=True)
    d = json.loads(reporte.read_text(encoding="utf-8"))
    print(f"\nMARK {args.mark} (evaluador oficial, muestra de 50)\n"
          f"  total {d['total_automatico']['obtenidos']} de {d['total_automatico']['posibles']} · cerradas "
          f"{d['cerradas']['aciertos']}/{d['cerradas']['n']} ({d['cerradas']['puntos']}) · citas {d['citas']['puntos']} "
          f"(recall {d['citas']['recall_citas_ponderado']}) · abstención {d['abstencion']['puntos']}"
          + (f" · RAGAS {d['correccion_ragas']['puntos']}" if args.ragas else "")
          + f"\n  reporte: {reporte.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
