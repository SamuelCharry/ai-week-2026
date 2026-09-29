"""Vigencia y nivel de cada documento: separar el núcleo del material que puede meter ruido.

En RAG, más documentos no siempre es mejor: una norma derogada o una sentencia marginal
pueden responder "semánticamente" y ser jurídicamente incorrectas. Nada se borra (el banco
pregunta por vigencia temporal y conflicto normativo, que necesitan la norma anterior);
cada documento queda marcado para que el índice decida qué usa:

    vigencia   derogada          el Senado anota la derogación de la norma completa en el
                                 encabezado ("<Ley derogada ... por el artículo 211 de la
                                 Ley 2056 de 2020>") o casi todos sus artículos la llevan
               inexequible       declarada inexequible en su totalidad
               con_notas         vigente con artículos derogados, modificados o condicionados
                                 (las notas quedan en el texto de cada artículo)
               sin_marca         la fuente no anota vigencia (jurisprudencia, doctrina, o
                                 compilaciones sin notas); no se certifica
    nivel      nucleo            índice principal: Constitución, códigos, normas no derogadas,
                                 jurisprudencia curada (seed, hitos, grafo, unificación,
                                 Sala Penal), circulares y doctrina curada
               complementario    índice secundario o de desempate: normas derogadas o
                                 inexequibles, leyes de tratados, honoríficas o ambientales,
                                 sentencias C/SU del barrido completo que el corpus no cita,
                                 decretos del Gestor fuera del árbol del Senado

La decisión de qué nivel indexar se toma midiendo sobre la muestra oficial
(scripts/evaluate.py), no a ojo.

Correr con la descarga detenida, después de scripts.corpus.areas; reescribe
data/raw/manifest.json.

Uso: python -m scripts.corpus.niveles [--dry-run]
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.corpus.areas import plano  # noqa: E402
from scripts.corpus.grafo import texto_en_cache  # noqa: E402

NORMAS = ("ley", "decreto", "acto_legislativo", "constitucion")
# Anotación de la norma completa, antes del epígrafe: "<Ley derogada ...>", "<Decreto derogado ...>".
DEROGADA_TOTAL = re.compile(r"<\s*(ley|decreto|decreto ley|acto legislativo|norma|codigo)\b[^>]{0,60}\bderogad[oa]")
INEXEQUIBLE_TOTAL = re.compile(r"<\s*(ley|decreto|decreto ley|acto legislativo|norma)\b[^>]{0,60}\binexequible\b")
ARTICULO = re.compile(r"\barticulo\s+\d+")
ARTICULO_DEROGADO = re.compile(r"<\s*articulo\s+(derogado|subrogado)|<\s*articulo[^>]{0,40}\binexequible\b")
NOTA = re.compile(r"<\s*(articulo|inciso|numeral|literal|paragrafo)[^>]{0,40}\b(modificado|derogado|adicionado|"
                  r"inexequible|exequible|subrogado)")
# Orígenes de barrido completo: entran al núcleo solo si el corpus los cita (grafo).
BARRIDO = {"cc_datos_abiertos", "gestor_normas", "gestor_jurisprudencia"}
# Mismo umbral del grafo normativo (scripts.corpus.objetivos --umbral-grafo 4): con el corpus
# completo casi toda sentencia C es citada por alguna otra, así que una sola cita no distingue.
UMBRAL_CITAS = 4
ALCANCE_COMPLEMENTARIO = {"tratado_internacional", "honorifica", "ambiental"}


def vigencia(texto):
    cabeza = plano(texto[:6000])
    epigrafe = re.search(r"\bpor (medio de )?(la|el) cual\b", cabeza)
    encabezado = cabeza[:epigrafe.start()] if epigrafe else cabeza[:1500]
    if DEROGADA_TOTAL.search(encabezado):
        return "derogada"
    if INEXEQUIBLE_TOTAL.search(encabezado):
        return "inexequible"
    cuerpo = plano(texto)
    articulos = len(set(ARTICULO.findall(cuerpo)))
    if articulos >= 3 and len(ARTICULO_DEROGADO.findall(cuerpo)) >= 0.9 * articulos:
        return "derogada"
    return "con_notas" if NOTA.search(cuerpo) else "sin_marca"


def citados_por_el_corpus():
    ruta = RAIZ / "data/raw/grafo_citas.csv"
    if not ruta.is_file():
        return set()
    return {f["clave"] for f in csv.DictReader(ruta.open(encoding="utf-8-sig"))
            if int(f.get("documentos_que_citan") or 0) >= UMBRAL_CITAS}


def nivel(entrada, vig, citados):
    if entrada.get("alcance") in ALCANCE_COMPLEMENTARIO or vig in ("derogada", "inexequible"):
        return "complementario"
    if entrada.get("origen_ampliacion") in BARRIDO and entrada["doc_id"] not in citados:
        return "complementario"
    return "nucleo"


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="solo muestra el resumen, no escribe")
    args = ap.parse_args()
    ruta = RAIZ / "data/raw/manifest.json"
    manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
    citados = citados_por_el_corpus()
    conteo = collections.Counter()
    for entrada in manifiesto:
        vig = vigencia(texto_en_cache(RAIZ, entrada)) if entrada.get("tipo") in NORMAS else "sin_marca"
        entrada["vigencia_fuente"] = vig
        entrada["nivel"] = nivel(entrada, vig, citados)
        conteo[(entrada["nivel"], vig)] += 1
    for (niv, vig), n in sorted(conteo.items()):
        print(f"  {niv:15} {vig:12} {n}")
    print("  total:", dict(collections.Counter(e["nivel"] for e in manifiesto)))
    if not args.dry_run:
        ruta.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"{ruta.relative_to(RAIZ)} actualizado")


if __name__ == "__main__":
    main()
