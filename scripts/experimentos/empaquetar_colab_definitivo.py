"""Crea ZIP privado mínimo para ejecutar E06 en Colab; no incluye data/raw."""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/colab/ai-week-2026-corpus-definitivo.zip"


def main():
    manifest = json.loads((ROOT / "data/releases/corpus_eval_v1/corpus_manifest.json").read_text(encoding="utf-8"))
    paths = [ROOT / row["texto_archivo"] for row in manifest]
    paths += list((ROOT / "data/releases/corpus_eval_v1").glob("*.json"))
    paths += list((ROOT / "data/oficial").rglob("*.py"))
    paths += list((ROOT / "data/oficial").rglob("*.json"))
    paths += list((ROOT / "data/oficial").rglob("*.jsonl"))
    paths += list((ROOT / "data/oficial").rglob("*.txt"))
    paths += [ROOT / "reports/reporte_evaluacion/modelos_verificados.json",
              ROOT / "notebooks/experimentos/e06_corpus_definitivo.ipynb",
              ROOT / "requirements-notebook-definitivo.txt", ROOT / "requirements.txt",
              ROOT / "GUIA_NOTEBOOK_DEFINITIVO.md"]
    paths += [ROOT / "scripts/corpus/ingesta.py", ROOT / "scripts/generacion/cliente.py",
              ROOT / "scripts/evaluacion/entrega.py", ROOT / "scripts/experimentos/corpus_definitivo.py"]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix(".tmp")
    with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as z:
        for i, path in enumerate(dict.fromkeys(paths), 1):
            z.write(path, "ai-week-2026/" + path.relative_to(ROOT).as_posix())
            if i % 1000 == 0:
                print(f"Empaquetados {i}/{len(paths)}", end="\r")
    temp.replace(OUT)
    print(f"{OUT} ({OUT.stat().st_size/2**20:.1f} MiB, {len(paths)} archivos)")


if __name__ == "__main__":
    sys.exit(main())
