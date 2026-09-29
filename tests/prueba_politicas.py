"""Prueba sin GPU de saneo="cita" y citar_evidencia="usadas".

Uso: python -m tests.prueba_politicas
"""
import json
import sys
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))

from legalrag.generation import politica as gen  # noqa: E402
from legalrag.experimentos.v04 import evidencia, muestra  # noqa: E402


def main():
    ev = evidencia(RAIZ)
    citas = ev.citas
    _, entradas = muestra(RAIZ)
    unidades = []
    with (RAIZ / "data/processed/corpus/unidades.jsonl").open(encoding="utf-8") as f:
        for linea in f:
            if '"ley_84_1873"' in linea or '"decreto_2663_1950"' in linea:
                u = json.loads(linea)
                if u.get("articulo") and sum(x["doc_id"] == u["doc_id"] for x in unidades) < 2:
                    unidades.append({k: u[k] for k in ("doc_id", "unidad_id", "inicio", "fin", "texto", "articulo")} | {"score": 1})
            if len(unidades) == 4:
                break
    pasajes, _ = ev.reconstruir(unidades)
    respaldo = ev.respaldo(pasajes)
    fallos = []

    texto = ("La Sentencia SU-455 de 2020 unificó el criterio sobre despidos. Según el Código Sustantivo del Trabajo, "
             "el empleador debe motivar. La Ley 9999 de 2021 lo reitera y el Código Penal no aplica.")
    limpio, quitadas = gen.sanear_cita(texto, respaldo, citas)
    print("saneo=cita  ->", limpio)
    print("saneo=oracion ->", gen.sanear(texto, respaldo, citas)[0])
    if citas.bodies(citas.extract(limpio)) - respaldo:
        fallos.append("quedaron citas sin respaldo")
    if "despidos" not in limpio or "Código Sustantivo del Trabajo" not in limpio:
        fallos.append("se perdió razonamiento o una cita con respaldo")

    entrada = next(e for e in entradas if e["formato"] == "semi_open")
    crudo = {"respuesta": texto, "palabras_clave": ["despido"], "referencia_legal": "Código Sustantivo del Trabajo"}
    for politica in ({"citar_evidencia": "todas", "saneo": "oracion"}, {"citar_evidencia": "usadas", "saneo": "cita"}):
        r, _ = gen.postprocesar(entrada, crudo, pasajes, ev, politica)
        print(politica, "\n   respuesta:", r["respuesta"], "\n   referencia_legal:", r["referencia_legal"])
        if citas.bodies(citas.extract(r["respuesta"] + " " + r["referencia_legal"])) - respaldo:
            fallos.append(f"sin respaldo con {politica}")
    assert not fallos, fallos
    print("OK: políticas 05 sin citas sin respaldo")


if __name__ == "__main__":
    main()
