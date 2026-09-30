"""¿Dónde se pierden los puntos? Recuperación frente a generación, pregunta por pregunta.

Lee una corrida de `main.py` / `pipeline` y, solo para puntuar, el fundamento y la respuesta de la
muestra. Para cada pregunta dice si la norma de referencia llegó a los pasajes entregados (si no,
es un problema de recuperación o de corpus) y si la respuesta la citó o eligió la letra correcta
(si llegó y aun así falla, es de generación).

    python3 src/diagnostico.py                                   # data/reproduccion/submissions_sample.jsonl
    python3 src/diagnostico.py --entrega data/comparacion/qwen3-4b-2507/submissions.jsonl
"""
import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrega", type=Path, default=RAIZ / "data/reproduccion/submissions_sample.jsonl")
    args = ap.parse_args()

    scripts = RAIZ / "data/oficial/scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location("evaluate_oficial", scripts / "evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    citas = ev.citations
    muestra = {r["id"]: r for r in ev.read_jsonl(RAIZ / "data/oficial/data/sample_50.jsonl")}
    entrega = {r["id"]: r for r in ev.read_jsonl(args.entrega)}
    carpeta = args.entrega.with_name(args.entrega.stem + "_respuestas")

    filas, causas = [], Counter()
    for qid, ref in muestra.items():
        respuesta = entrega.get(qid)
        if respuesta is None:
            continue
        referencia = citas.bodies(citas.extract(ref.get("legal_basis") or ""))
        evidencia = citas.bodies(set().union(*(citas.extract(p.get("texto") or "")
                                               for p in respuesta.get("pasajes_recuperados", [])[:10])) or set())
        citadas = citas.bodies(citas.extract(ev.answer_text(respuesta)))
        detalle = json.loads((carpeta / f"{qid}.json").read_text(encoding="utf-8")) if (carpeta / f"{qid}.json").is_file() else {}
        registro = detalle.get("registro") or {}
        fila = {"id": qid, "formato": ref["formato"], "area": ref.get("area", "")[:24],
                "ref_en_evidencia": bool(referencia & evidencia) if referencia else None,
                "ref_citada": bool(referencia & citadas) if referencia else None,
                "problema": detalle.get("problema"), "segundos": round(detalle.get("segundos", 0), 1)}
        if ref["formato"] == "multiple_choice":
            fila["letra"], fila["correcta"] = respuesta.get("respuesta_correcta"), ref.get("respuesta_correcta")
            probs = registro.get("probabilidades_letras") or {}
            fila["prob_correcta"] = round(probs.get(ref.get("respuesta_correcta"), 0), 2) if probs else None
            fila["acierto"] = fila["letra"] == fila["correcta"]
        if not referencia:
            causa = "sin cita extraíble en el fundamento (doctrina o prosa)"
        elif not fila["ref_en_evidencia"]:
            causa = "recuperación/corpus: la norma de referencia no llegó a los pasajes"
        elif not fila["ref_citada"]:
            causa = "generación: la norma estaba en los pasajes y no se citó"
        else:
            causa = "cita correcta"
        if ref["formato"] == "multiple_choice" and not fila.get("acierto"):
            causa += " · letra incorrecta"
        fila["causa"] = causa
        causas[causa] += 1
        filas.append(fila)

    for f in filas:
        extra = (f" letra={f['letra']} correcta={f['correcta']} p(correcta)={f['prob_correcta']}"
                 if f["formato"] == "multiple_choice" else "")
        print(f"{f['id']:5} {f['formato']:15} {f['area']:24} ref_en_pasajes={str(f['ref_en_evidencia']):5} "
              f"ref_citada={str(f['ref_citada']):5}{extra}  [{f['problema']}]")
    print("\nCausas:")
    for causa, n in causas.most_common():
        print(f"  {n:3}  {causa}")
    salida = args.entrega.with_name(args.entrega.stem + "_diagnostico.json")
    salida.write_text(json.dumps({"causas": dict(causas), "preguntas": filas}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("\nDetalle:", salida)


if __name__ == "__main__":
    main()
