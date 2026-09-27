import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.auxiliares.recuperacion import BM25, cargar_indice, escribir_jsonl, leer_jsonl, sha256
from scripts.auxiliares.sondas import crear_sondas, evaluar_sondas, resumir

parser = argparse.ArgumentParser()
parser.add_argument("--indice", required=True)
parser.add_argument("--sondas")
parser.add_argument("--salida", required=True)
args = parser.parse_args()
salida = Path(args.salida)
salida.mkdir(parents=True, exist_ok=False)
recuperador = cargar_indice(args.indice)
sondas = leer_jsonl(args.sondas) if args.sondas else crear_sondas(list(recuperador.unidades.values()))
escribir_jsonl(salida / "sondas.jsonl", sondas)
resumenes = []
filas = []
for hibrido in (False, True):
    metodo = "hibrido" if hibrido else "denso"
    recuperador.bm25 = BM25([v["texto_busqueda"] for v in recuperador.ventanas]) if hibrido else None
    primeras = evaluar_sondas(recuperador, sondas)
    segundas = evaluar_sondas(recuperador, sondas)
    iguales = all(
        {k: v for k, v in a.items() if k != "segundos"} ==
        {k: v for k, v in b.items() if k != "segundos"}
        for a, b in zip(primeras, segundas)
    )
    escribir_jsonl(salida / f"{metodo}_1.jsonl", primeras)
    escribir_jsonl(salida / f"{metodo}_2.jsonl", segundas)
    resumenes.append({"metodo": metodo, **resumir(primeras), "repeticion_identica": iguales})
    filas.extend({"metodo": metodo, **r} for r in primeras)
    if not iguales:
        raise ValueError("Cambió la recuperación al repetir las sondas")
tabla = pd.DataFrame(filas)
metricas = ["recall_1", "recall_5", "recall_10", "mrr_10"]
for columna, nombre in (("doc_id", "norma"), ("areas", "area")):
    datos = tabla.explode("areas") if columna == "areas" else tabla
    agrupados = datos.groupby(["metodo", columna])[metricas].mean()
    agrupados["sondas"] = datos.groupby(["metodo", columna]).size()
    agrupados.reset_index().to_csv(salida / f"por_{nombre}.csv", index=False)
tabla[tabla.articulo.astype(str).isin(["24", "42"])].drop(columns=["top_10"]).to_csv(
    salida / "articulos_24_42.csv", index=False
)
reporte = {
    "indice": args.indice,
    "sha256_manifest": sha256(Path(args.indice) / "manifest.json"),
    "sha256_codigo": {p: sha256(p) for p in ["scripts/auxiliares/recuperacion.py", "scripts/auxiliares/sondas.py", "scripts/verificar_recuperacion.py"]},
    "unidades": len(recuperador.unidades),
    "ventanas": len(recuperador.ventanas),
    "resultados": resumenes,
}
(salida / "resumen.json").write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(reporte, ensure_ascii=False, indent=2))
