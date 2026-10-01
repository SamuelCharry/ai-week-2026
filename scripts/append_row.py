"""Agrega una fila a reports/ablaciones.csv con el resultado de una variante."""
from __future__ import annotations

import csv
import json
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "reports" / "ablaciones.csv"
COLS = ["variante", "cerradas_acc", "cerradas_pts", "citas_pts", "recall",
        "sin_respaldo", "abstencion_pts", "total_pts", "respondidas",
        "abstenciones", "s_por_pregunta", "errores_validacion"]


def main() -> int:
    name = sys.argv[1]
    report_path = Path(sys.argv[2])
    submission_path = Path(sys.argv[3])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    subs = [json.loads(l) for l in submission_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    latencias = [s.get("latencia_ms", 0) / 1000 for s in subs]
    row = {
        "variante": name,
        "cerradas_acc": report["cerradas"]["accuracy"],
        "cerradas_pts": report["cerradas"]["puntos"],
        "citas_pts": report["citas"]["puntos"],
        "recall": report["citas"]["recall_citas_ponderado"],
        "sin_respaldo": report["citas"]["tasa_sin_respaldo"],
        "abstencion_pts": report["abstencion"]["puntos"],
        "total_pts": report["cerradas"]["puntos"] + report["citas"]["puntos"] + report["abstencion"]["puntos"],
        "respondidas": sum(1 for s in subs if not s.get("abstencion")),
        "abstenciones": sum(1 for s in subs if s.get("abstencion")),
        "s_por_pregunta": round(statistics.mean(latencias), 2) if latencias else 0,
        "errores_validacion": report["validacion"]["errores"],
    }
    CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = CSV.exists()
    with CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[ablate] {name} -> total {row['total_pts']}/50 "
          f"(cer {row['cerradas_pts']}, cit {row['citas_pts']}, abs {row['abstencion_pts']}) "
          f"recall={row['recall']} sin_respaldo={row['sin_respaldo']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
