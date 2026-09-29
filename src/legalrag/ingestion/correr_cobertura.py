"""Uso: python -m legalrag.ingestion.correr_cobertura  (desde la raíz del proyecto)."""
import json
import sys
from collections import Counter
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))

from legalrag.ingestion.cobertura import auditar, auditar_banco, indice_corpus  # noqa: E402


def main():
    preguntas = [json.loads(l) for l in (RAIZ / "data/oficial/data/sample_50.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    indice = indice_corpus(RAIZ)
    filas = auditar(RAIZ, preguntas, indice=indice)
    banco = auditar_banco(RAIZ, indice=indice)
    salida = RAIZ / "data/experimentos_v04"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "cobertura_muestra.json").write_text(json.dumps(filas, ensure_ascii=False, indent=1), encoding="utf-8")
    (salida / "cobertura_banco.json").write_text(json.dumps(banco, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Muestra 50, citas de referencia por estado:", dict(Counter(f["estado_corpus"] for f in filas)))
    por_pregunta = {}
    for f in filas:
        por_pregunta.setdefault(f["id"], []).append(f["estado_corpus"])
    alguna = sum(any(e == "presente" for e in v) for v in por_pregunta.values())
    print(f"Preguntas con al menos una norma de referencia presente: {alguna}/{len(por_pregunta)}")
    total = sum(b["items_del_banco"] for b in banco)
    cubierto = sum(b["items_del_banco"] for b in banco if b["en_corpus"])
    print(f"Banco completo: {cubierto}/{total} menciones cubiertas ({cubierto/total:.0%})")
    print("\nFaltantes con más ítems del banco:")
    for b in sorted((b for b in banco if not b["en_corpus"]), key=lambda b: -b["items_del_banco"])[:15]:
        print(f"  {b['items_del_banco']:3d}  {b['norma']}")
    print("\nDetalle muestra (no presentes):")
    for f in filas:
        if f["estado_corpus"] != "presente":
            print(f"  {f['id']:5d} {f['formato']:16s} {f['estado_corpus']:18s} {f.get('cita')}")


if __name__ == "__main__":
    main()
