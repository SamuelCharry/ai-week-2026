"""Genera configs/corpus_exclusiones.json: lo que el grafo pide y no entra, con evidencia.

Cada referencia que el corpus cita y no se incorpora queda en una de cinco categorías:

    cita_errada           el documento que la cita se equivoca de año o número; la norma
                          real ya está en el corpus (norma_real).
    proyecto_de_ley       es la numeración de un proyecto en el Congreso ("Ley 111 de 2006
                          Senado"), no una norma.
    compilada             el contenido vigente está compilado en un decreto único que ya
                          está en el corpus (norma_real).
    sin_texto_oficial     norma real, casi siempre derogada, cuyo texto no publica ninguna
                          fuente oficial en línea. No se toma de fuentes privadas.
    enlace_roto_en_fuente un índice oficial enlaza el documento y el servidor responde 404 o
                          un archivo vacío.

La evidencia se arma sola: el fragmento donde se cita, qué documentos la citan y qué
fuentes oficiales se consultaron con su respuesta (data/raw/reconstruccion.csv). La
auditoría (sección G) exige que toda exclusión tenga motivo y evidencia.

Uso: python -m legalrag.ingestion.exclusiones
"""
import csv
import glob
import json
import re
import sys
from datetime import date
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())

# Clasificación revisada a mano sobre el contexto de cada cita (ver evidencia generada).
CITA_ERRADA = {
    "ley_100_1991": ("ley_100_1993", "la fuente misma anota '<sic, la Ley 100 referida corresponde al año 1993>'"),
    "ley_906_2005": ("ley_906_2004", "Código de Procedimiento Penal; la Ley 906 es de 2004"),
    "ley_788_2003": ("ley_788_2002", "reforma tributaria; la Ley 788 es de 2002"),
    "ley_789_2003": ("ley_789_2002", "reforma laboral y de aportes; la Ley 789 es de 2002"),
    "ley_617_2001": ("ley_617_2000", "saneamiento fiscal territorial; la Ley 617 es de 2000"),
    "ley_142_1993": ("ley_142_1994", "servicios públicos domiciliarios; la Ley 142 es de 1994"),
    "decreto_2067_1992": ("decreto_2067_1991", "régimen procedimental ante la Corte Constitucional; es de 1991"),
    "acto_legislativo_3_2003": ("acto_legislativo_3_2002", "sistema penal acusatorio; el Acto Legislativo 03 es de 2002"),
    "sentencia_cc_su159_2000": ("sentencia_cc_su159_2002", "la relatoría no tiene SU-159/00; la SU-159 es de 2002"),
    "sentencia_cc_c517_1998": ("sentencia_cc_c517_1999", "la relatoría no tiene C-517/98; la C-517 es de 1999"),
    "ley_100_1980": ("decreto_100_1980", "el Código Penal de 1980 es el Decreto 100, no una ley"),
}
PROYECTO_DE_LEY = {
    "ley_111_2006": "'ley número 111 de 2006 Senado, 144 de 2005 Cámara, por la cual se expide el Código Penal Militar' (hoy Ley 1407 de 2010)",
    "ley_169_2003": "'Ley 169 de 2003 Cámara'",
    "ley_184_2010": "'ley Estatutaria número 184 de 2010 Senado, 046 de 2010 Cámara' (hoy Ley 1581 de 2012)",
    "ley_1_2003": "'ley 001 de 2003 Cámara, por la cual se expide el Código de Procedimiento Penal' (hoy Ley 906 de 2004)",
    "ley_57_2002": "'Ley 57 de 2002 Senado'",
    "ley_72_2000": "'ley No. 072 de 2000 Cámara'",
    "ley_80_2002": "'ley No. 80 de 2002-Cámara'",
    "acto_legislativo_2_2016": "'Acto Legislativo número 002 de 2016 Cámara, acumulado con el 003 de 2016 Cámara, 002 de 2017 Senado'",
}
COMPILADA = {
    "decreto_2920_1982": ("decreto_1068_2015", "la definición de captación masiva y habitual está compilada en el "
                          "art. 2.18.2.1 del Decreto 1068 de 2015 (Superfinanciera, 'Normas de captación ilegal')"),
}


# Orígenes leídos de un índice oficial (rondas 4 y 5): si el índice enlaza un documento que el
# servidor no entrega, el enlace está roto en la fuente, no es una cita que falte.
DESDE_INDICE = {"ce_unificacion", "csj_compendio", "csj_spa_providencia", "doctrina_dian", "doctrina_fp",
                "cc_datos_abiertos", "senado_arbol", "sic_circular_unica", "supersociedades_concepto",
                "supersociedades_jurisprudencia", "supersociedades_circular", "superfinanciera_circular",
                "gestor_normas", "gestor_jurisprudencia", "gestor_sala_consulta", "gestor_conceptos_fp"}


def fragmento(clave, textos, ejemplos):
    m = re.fullmatch(r"(ley|decreto|acto_legislativo)_(\d+)_(\d{4})", clave)
    if m:
        tipo = m.group(1).replace("_", " ")
        patron = re.compile(rf"(?i){tipo}[^\d]{{0,25}}0*{m.group(2)}\s+del?\s+(\d+\s+de\s+\w+\s+de\s+)?{m.group(3)}.{{0,140}}")
    else:
        m = re.fullmatch(r"sentencia_cc_([a-z]+)(\d+)_(\d{4})", clave)
        if not m:
            return ""
        patron = re.compile(rf"(?i)\b{m.group(1)}\s*[-.]?\s*0*{int(m.group(2))}\s*(?:/|de)\s*(?:{m.group(3)}|{m.group(3)[2:]}).{{0,140}}")
    for doc in ejemplos:
        ruta = textos.get(doc)
        if ruta:
            encontrado = patron.search(Path(ruta).read_text(encoding="utf-8"))
            if encontrado:
                return f"{doc}: «{re.sub(r'\s+', ' ', encontrado.group())[:220]}»"
    return ""


def main():
    raw = {d["doc_id"] for d in json.loads((RAIZ / "data/raw/manifest.json").read_text(encoding="utf-8"))}
    sin_prefijo = {re.sub(r"^co_", "", d) for d in raw}
    grafo = {r["clave"]: r for r in csv.DictReader((RAIZ / "data/raw/grafo_citas.csv").open(encoding="utf-8-sig"))}
    intentos = {}
    for fila in csv.DictReader((RAIZ / "data/raw/reconstruccion.csv").open(encoding="utf-8-sig")):
        intentos.setdefault(fila["doc_id"], []).append(f"{fila['url']} -> {fila['estado']}")
    textos = {Path(f).name.split(".")[0]: f for f in glob.glob(str(RAIZ / "data/raw/.texto_grafo/*.txt"))}
    inventario = [d["doc_id"] for d in json.loads((RAIZ / "corpus_manifest.json").read_text(encoding="utf-8"))["documentos"]]
    inventario += [d["doc_id"] for d in json.loads((RAIZ / "configs/corpus_objetivos.json").read_text(encoding="utf-8"))["documentos"]]
    pendientes = sorted({d for d in inventario if re.sub(r"^co_", "", d) not in sin_prefijo})
    origen = {d["doc_id"]: d.get("origen") for d in
              json.loads((RAIZ / "configs/corpus_objetivos.json").read_text(encoding="utf-8"))["documentos"]}

    exclusiones = []
    for clave in pendientes:
        g = grafo.get(clave, {})
        ejemplos = [e for e in g.get("ejemplos_citantes", "").split(", ") if e]
        base = {"clave": clave, "norma": g.get("norma", clave), "documentos_que_citan": int(g.get("documentos_que_citan") or 0),
                "fecha_verificacion": date.today().isoformat()}
        cita = fragmento(clave, textos, ejemplos)
        if clave in CITA_ERRADA:
            real, nota = CITA_ERRADA[clave]
            if real not in sin_prefijo:
                print(f"AVISO: {clave} apunta a {real}, que no está en el corpus")
            exclusiones.append({**base, "motivo": "cita_errada", "norma_real": real,
                                "evidencia": f"{nota}. Cita: {cita}".strip()})
        elif clave in PROYECTO_DE_LEY:
            exclusiones.append({**base, "motivo": "proyecto_de_ley",
                                "evidencia": f"Numeración del trámite en el Congreso: {PROYECTO_DE_LEY[clave]}. Cita: {cita}"})
        elif clave in COMPILADA:
            real, nota = COMPILADA[clave]
            exclusiones.append({**base, "motivo": "compilada", "norma_real": real, "evidencia": f"{nota}. Cita: {cita}"})
        elif origen.get(clave) in DESDE_INDICE and intentos.get(clave) and \
                all(re.search(r"-> HTTP (404|200)$", i) for i in intentos[clave]):
            # "HTTP 200" sin más es un cuerpo vacío: el Consejo de Estado responde así a archivos
            # que su listado enlaza y ya no existen.
            nota = (" Las providencias completas de ese tema, enlazadas en la misma página, sí están en el corpus "
                    "(origen csj_spa_providencia)." if clave.startswith("csj_compendio") else "")
            exclusiones.append({**base, "motivo": "enlace_roto_en_fuente",
                                "evidencia": (f"El índice oficial ({origen[clave]}) lo enlaza pero el servidor responde 404 "
                                              f"o un archivo vacío: {'; '.join(intentos[clave])}.{nota}")})
        else:
            consultadas = intentos.get(clave, [])
            exclusiones.append({**base, "motivo": "sin_texto_oficial",
                                "evidencia": (f"Ninguna de las fuentes oficiales consultadas publica el texto: "
                                              f"{'; '.join(consultadas) or 'relatoría de la Corte Constitucional'}. "
                                              f"Cita: {cita}")})
    ruta = RAIZ / "configs/corpus_exclusiones.json"
    ruta.write_text(json.dumps({"descripcion": __doc__.strip().splitlines()[0], "exclusiones": exclusiones},
                               ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    from collections import Counter
    print(f"{ruta.relative_to(RAIZ)}: {len(exclusiones)} exclusiones {dict(Counter(e['motivo'] for e in exclusiones))}")


if __name__ == "__main__":
    sys.exit(main())
