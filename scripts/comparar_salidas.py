"""Compara dos salidas sobre la muestra (iteración de Cerberus): puntaje oficial, cambios pregunta por pregunta y
RAGAS≈, sin volver a generar.

    python scripts/comparar_salidas.py salidas/exp_phase6.jsonl salidas/mark43.jsonl
    python scripts/comparar_salidas.py salidas/exp_phase6.jsonl salidas/mark43.jsonl --sin-ragas-local

RAGAS≈ aproxima los 30 puntos de texto libre del evaluador (answer correctness contra la respuesta esperada:
0,75 · TP / (TP + 0,5·(FP + FN)) + 0,25 · similitud): la similitud con el mismo encoder del evaluador
(intfloat/multilingual-e5-large) y la parte factual con un NLI multilingüe (mDeBERTa-xnli). Ordena variantes;
no reemplaza al juez oficial (--ragas en el evaluador).
"""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
OFICIAL = RAIZ / "data/oficial"
NLI = ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "b5113eb38ab63efdd7f280f8c144ea8b13f978ce")
E5 = ("intfloat/multilingual-e5-large", "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3")
_ORACION = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÁÉÍÓÚÑ¿(\"“])")


def evaluador():
    sys.path.insert(0, str(OFICIAL / "scripts"))
    spec = importlib.util.spec_from_file_location("evaluador_oficial", OFICIAL / "scripts/evaluate.py")
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


def por_pregunta(ev, salida, muestra):
    subs = {s["id"]: s for s in ev.read_jsonl(salida)}
    citas, resultado = ev.citations, {}
    for p in muestra:
        s = subs.get(p["id"])
        fila = {"formato": p["formato"], "abstencion": bool(s and s.get("abstencion"))}
        if p["formato"] == "multiple_choice" and p["id"] not in ev.FLAWED_IDS:
            fila["letra"] = (s or {}).get("respuesta_correcta")
            fila["cerrada"] = bool(s) and not fila["abstencion"] and fila["letra"] == p["respuesta_correcta"]
        ref = citas.extract(p.get("legal_basis") or "")
        if ref:
            fila.update(citas=0.0, ref=len(citas.bodies(ref)))
            if s and not fila["abstencion"]:
                r = citas.score(ev.answer_text(s), p.get("legal_basis") or "", ev.citas_respaldadas(s))
                fila.update(citas=r["aciertos_respaldados"] + 0.5 * r["aciertos_sin_respaldo"], ref=r["n_ref"])
        resultado[p["id"]] = fila
    return resultado


def cambios(antes, despues):
    lineas = []
    for qid, a in antes.items():
        d = despues.get(qid, {})
        if "cerrada" in a and (a["cerrada"], a["letra"]) != (d.get("cerrada"), d.get("letra")):
            signo = "~" if a["cerrada"] == d.get("cerrada") else "+" if d.get("cerrada") else "-"
            lineas.append((signo, qid, a["formato"], f"letra {a['letra']} → {d.get('letra')}"))
        if "ref" in a and a["citas"] != d.get("citas"):
            signo = "+" if d.get("citas", 0) > a["citas"] else "-"
            lineas.append((signo, qid, a["formato"], f"citas {a['citas']:g}/{a['ref']} → {d.get('citas', 0):g}/{a['ref']}"))
    return sorted(lineas, key=lambda l: ("+-~".index(l[0]), l[1]))


class RagasLocal:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tn = AutoTokenizer.from_pretrained(NLI[0], revision=NLI[1])
        self.nli = AutoModelForSequenceClassification.from_pretrained(NLI[0], revision=NLI[1]).to("cuda").eval()
        etiquetas = {str(v).lower(): int(k) for k, v in self.nli.config.id2label.items()}
        self.implica = etiquetas["entailment"]
        self.te = AutoTokenizer.from_pretrained(E5[0], revision=E5[1])
        self.enc = AutoModel.from_pretrained(E5[0], revision=E5[1]).to("cuda").eval()

    def _vector(self, texto):
        t = self.te([texto], truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            h = self.enc(**t).last_hidden_state
            m = t["attention_mask"].unsqueeze(-1)
            return self.torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)[0]

    def _implicadas(self, hipotesis, premisa):
        if not hipotesis:
            return 0
        t = self.tn([premisa] * len(hipotesis), hipotesis, truncation="only_first", max_length=512, padding=True,
                    return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            p = self.torch.softmax(self.nli(**t).logits.float(), dim=-1)[:, self.implica]
        return int((p >= 0.5).sum())

    def puntuar(self, respuesta, esperada):
        sim = float(self._vector(respuesta) @ self._vector(esperada))
        propias = [o for o in _ORACION.split(respuesta) if len(o.strip()) > 15]
        esperadas = [o for o in _ORACION.split(esperada) if len(o.strip()) > 15]
        if not propias or not esperadas:
            return 0.25 * sim
        tp = self._implicadas(propias, esperada)
        fn = len(esperadas) - self._implicadas(esperadas, respuesta)
        fp = len(propias) - tp
        return 0.75 * (tp / (tp + 0.5 * (fp + fn)) if tp else 0.0) + 0.25 * sim

    def puntos(self, ev, salida, muestra):
        subs = {s["id"]: s for s in ev.read_jsonl(salida)}
        valores = []
        for p in muestra:
            if p["formato"] == "multiple_choice":
                continue
            s = subs.get(p["id"])
            valores.append(0.0 if not s or s.get("abstencion") else
                           self.puntuar(ev.ragas_text(s), p.get("respuesta_esperada") or ""))
        return round(30 * sum(valores) / max(len(valores), 1), 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anterior", type=Path)
    ap.add_argument("nueva", type=Path)
    ap.add_argument("--sin-ragas-local", action="store_true")
    args = ap.parse_args()
    ev = evaluador()
    muestra = ev.read_jsonl(OFICIAL / "data/sample_50.jsonl")
    for salida in (args.anterior, args.nueva):
        reporte = salida.with_name(salida.stem + "_oficial.json")
        subprocess.run([sys.executable, str(OFICIAL / "scripts/evaluate.py"), "--submission", str(salida),
                        "--split", "sample", "--out", str(reporte)], check=True, capture_output=True)
        d = json.loads(reporte.read_text(encoding="utf-8"))
        print(f"{salida.name}: total {d['total_automatico']['obtenidos']} · cerradas {d['cerradas']['aciertos']}/"
              f"{d['cerradas']['n']} · citas {d['citas']['puntos']} (recall {d['citas']['recall_citas_ponderado']}) · "
              f"abstención {d['abstencion']['puntos']}")
    lista = cambios(por_pregunta(ev, args.anterior, muestra), por_pregunta(ev, args.nueva, muestra))
    print(f"\n{args.nueva.name} frente a {args.anterior.name}: {sum(l[0] == '+' for l in lista)} mejoras, "
          f"{sum(l[0] == '-' for l in lista)} empeoras")
    for signo, qid, formato, detalle in lista:
        print(f"  {signo} {qid:>4} {formato:<15} {detalle}")
    if not args.sin_ragas_local:
        juez = RagasLocal()
        print(f"\nRAGAS≈ (de 30): {args.anterior.name} {juez.puntos(ev, args.anterior, muestra)} · "
              f"{args.nueva.name} {juez.puntos(ev, args.nueva, muestra)}")


if __name__ == "__main__":
    main()
