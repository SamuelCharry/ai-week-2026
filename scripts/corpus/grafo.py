"""Grafo normativo del corpus: qué normas y sentencias citan nuestros documentos y no tenemos.

Lee los originales de data/raw (con el extractor de la ingesta), extrae las
referencias a leyes, decretos, actos legislativos y sentencias, y cuenta cuántos
documentos distintos del corpus citan cada una. Una norma citada por muchos
documentos del propio corpus y ausente de él es una brecha probable: si el CGP,
el Estatuto Tributario y cuarenta sentencias remiten a ella, las preguntas del
banco también pueden hacerlo.

No usa el banco de preguntas ni sus respuestas: solo el texto oficial descargado.

Escribe data/raw/grafo_citas.csv (todas las referencias) y
data/raw/grafo_faltantes.csv (las ausentes del inventario, ordenadas por documentos
que las citan). El texto extraído se guarda en caché en data/raw/.texto_grafo/.

Uso: python -m scripts.corpus.grafo [--minimo 3]
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.corpus.reconstruir import inventario, texto_de  # noqa: E402

MESES = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
# "Ley 1564 de 2012", "Ley 100 del 23 de diciembre de 1993", "Decreto-ley 2663 de 1950",
# "Decreto Legislativo 806 de 2020", "Acto Legislativo 01 de 2005", "Ley Estatutaria 1581 de 2012".
NORMA = re.compile(
    rf"(?i)\b(ley(?:\s+estatutaria|\s+org[aá]nica)?|decreto(?:[\s-]+ley|\s+legislativo|\s+reglamentario|\s+[uú]nico(?:\s+reglamentario)?)?"
    rf"|acto\s+legislativo)\s+(?:n[uú]mero\s+|no\.?\s*|n[°º]\.?\s*)?(\d{{1,2}}(?:\.\d{{3}})+|\d{{1,5}})"
    rf"\s*(?:de|del)\s+(?:\d{{1,2}}\s+de\s+{MESES}\s+(?:de|del)\s+)?(\d{{4}})\b")
# "C-355 de 2006", "C-355/06", "Sentencia T-760 de 2008", "SU-214/16", "SU016-20".
# "SU.917/10" (con punto) también aparece en la relatoría.
SENT_CC = re.compile(r"\b(SU|C|T)\s*[-.]?\s*(\d{1,4})\s*(?:/|de|del)\s*((?:19|20)\d{2}|\d{2})\b"
                     r"|\b((?i:su|c|t))\s*[-.]\s*(\d{1,4})\s*-\s*((?:19|20)\d{2}|\d{2})\b")
# Corte Suprema: "SL3385-2022", "SC5191-2020", "STC1234-2019", "AP1234-2020".
SENT_CSJ = re.compile(r"\b(SL|SC|SP|STL|STC|STP|AL|AC|AP)\s?-?\s?(\d{2,5})\s*-\s*((?:19|20)\d{2})\b")


PROYECTO = re.compile(r"[\s,\-–—(]*(?:de\s+(?:la|el)\s+)?(?:Senado|C[aá]mara)\b", re.I)


def anio_completo(anio):
    anio = int(anio)
    return anio if anio > 100 else (1900 + anio if anio >= 91 else 2000 + anio)


def referencias(texto):
    for m in NORMA.finditer(texto):
        tipo = "acto_legislativo" if m.group(1).lower().startswith("acto") else \
            "decreto" if m.group(1).lower().startswith("decreto") else "ley"
        numero = int(m.group(2).replace(".", ""))
        anio = int(m.group(3))
        # "Ley 111 de 2006 Senado", "Proyecto de Acto Legislativo 002 de 2016 Cámara": es la
        # numeración del trámite en el Congreso, no una norma.
        if PROYECTO.match(texto, m.end()) or re.search(r"(?i)proyecto\s+de\s*$", texto[max(0, m.start() - 20):m.start()]):
            continue
        if 1821 <= anio <= 2026 and numero > 0:
            yield f"{tipo}_{numero}_{anio}", f"{tipo.replace('_', ' ').capitalize()} {numero} de {anio}"
    for m in SENT_CC.finditer(texto):
        grupos = m.groups()[:3] if m.group(1) else m.groups()[3:]
        sala, numero, anio = grupos[0].upper(), int(grupos[1]), anio_completo(grupos[2])
        if 1992 <= anio <= 2026 and numero > 0:
            yield f"sentencia_cc_{sala.lower()}{numero:03d}_{anio}", f"Sentencia {sala}-{numero:03d} de {anio}"
    for m in SENT_CSJ.finditer(texto):
        sala, numero, anio = m.group(1).upper(), int(m.group(2)), int(m.group(3))
        if 1990 <= anio <= 2026:
            yield f"sentencia_csj_{sala.lower()}{numero}_{anio}", f"Sentencia {sala}{numero}-{anio}"


def clave_inventario(doc_id):
    """Misma norma aunque el doc_id del corpus lleve prefijo ("co_ley_1437_2011")."""
    return re.sub(r"^co_", "", doc_id)


def texto_en_cache(raiz, entrada):
    cache = raiz / "data/raw/.texto_grafo"
    cache.mkdir(exist_ok=True)
    huella = hashlib.sha256("".join(a["sha256"] + (a.get("texto_derivado") or {}).get("sha256", "")
                                    for a in entrada["archivos_raw"]).encode()).hexdigest()[:16]
    ruta = cache / f"{entrada['doc_id']}.{huella}.txt"
    if ruta.is_file():
        return ruta.read_text(encoding="utf-8")
    partes = [{"contenido": (raiz / "data/raw" / a["archivo"]).read_bytes(),
               "derivado": {"texto": (raiz / "data/raw" / a["texto_derivado"]["archivo"]).read_text(encoding="utf-8")}
               if a.get("texto_derivado") else None}
              for a in entrada["archivos_raw"]]
    texto = texto_de(partes)
    ruta.write_text(texto, encoding="utf-8")
    return texto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minimo", type=int, default=3, help="documentos que citan, mínimo, para listar una brecha")
    args = ap.parse_args()

    manifiesto = json.loads((RAIZ / "data/raw/manifest.json").read_text(encoding="utf-8"))
    tenemos = {clave_inventario(d["doc_id"]) for d in inventario(RAIZ)}
    # Los códigos se guardan con su norma de origen; el alias por nombre no cuenta como brecha.
    citantes, menciones, nombres, areas = defaultdict(set), Counter(), {}, defaultdict(Counter)
    # Fuentes primarias: normas y núcleo curado (todo lo que no entró por el grafo). Cierran
    # la jurisprudencia: una sentencia que solo citan otras sentencias del grafo no la exige.
    primarias = {d["doc_id"] for d in manifiesto
                 if d.get("tipo") not in ("sentencia", "auto") or d.get("origen_ampliacion") != "grafo_normativo"}
    for n, entrada in enumerate(manifiesto, 1):
        try:
            texto = texto_en_cache(RAIZ, entrada)
        except Exception as error:
            print(f"  {entrada['doc_id']}: {type(error).__name__}: {error}")
            continue
        propia = clave_inventario(entrada["doc_id"])
        for clave, nombre in referencias(texto):
            if clave == propia:
                continue
            citantes[clave].add(entrada["doc_id"])
            menciones[clave] += 1
            nombres[clave] = nombre
            for area in entrada.get("areas") or []:
                areas[clave][area] += 1
        if n % 50 == 0:
            print(f"Leídos {n}/{len(manifiesto)}", flush=True)

    filas = []
    for clave in citantes:
        filas.append({"clave": clave, "norma": nombres[clave], "documentos_que_citan": len(citantes[clave]),
                      "citan_fuentes_primarias": len(citantes[clave] & primarias),
                      "menciones": menciones[clave], "en_corpus": clave in tenemos,
                      "areas": ", ".join(a for a, _ in areas[clave].most_common(4)),
                      "ejemplos_citantes": ", ".join(sorted(citantes[clave])[:5])})
    filas.sort(key=lambda f: (-f["documentos_que_citan"], -f["menciones"]))
    campos = list(filas[0]) if filas else ["clave"]
    for nombre, seleccion in (("grafo_citas.csv", filas),
                              ("grafo_faltantes.csv", [f for f in filas if not f["en_corpus"]
                                                       and f["documentos_que_citan"] >= args.minimo])):
        with (RAIZ / "data/raw" / nombre).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(seleccion)
    faltan = [f for f in filas if not f["en_corpus"] and f["documentos_que_citan"] >= args.minimo]
    print(f"\nReferencias distintas: {len(filas)} | en corpus: {sum(f['en_corpus'] for f in filas)} | "
          f"ausentes citadas por >= {args.minimo} documentos: {len(faltan)}")
    for f in faltan[:60]:
        print(f"  {f['documentos_que_citan']:4d} docs {f['menciones']:5d} menc.  {f['norma']:32s} {f['areas']}")


if __name__ == "__main__":
    main()
