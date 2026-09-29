"""Genera configs/corpus_objetivos.json: lo que se agrega al inventario del corpus.

Tres grupos, todos de fuentes oficiales colombianas (Senado, relatorías de la Corte
Constitucional y la Corte Suprema, Gestor Normativo, compilación del ICBF). SUIN-Juriscol
no se usa: ahora es una aplicación JavaScript y su URL no entrega el texto de la norma.

1. seed_targets: normas que seed_targets.json declara y el corpus no tenía. No se
   incluyen los identificadores que no existen ("Ley 11500 de 2007", "Ley 116 de 2006",
   "Ley 1150 de 2005", "Ley 964 de 2006", "Decreto 1563 de 2012", "Decreto 875 de 2008"):
   el evaluador compara el identificador literal y no se sustituyen.
2. normas: leyes y decretos troncales de las diez áreas del banco (sección 4.2 del
   enunciado) que el seed no lista. El enunciado dice que el seed es deliberadamente
   no exhaustivo y que cerrar esa brecha es el trabajo de la semana.
4. grafo_normativo: normas y sentencias que citan >= N documentos del propio corpus
   y no están en él (scripts.corpus.grafo). Incluye leyes aprobatorias de tratados
   citadas por el corpus: decisión del equipo, por el bloque de constitucionalidad.
3. hitos: sentencias de la Corte Constitucional que fijan reglas citadas como
   precedente en las sub-tareas de complejidad media y alta (precedente, sentido del
   fallo, ponderación). Solo el texto oficial de la relatoría.

No se lee el banco de preguntas ni sus respuestas: la selección sale del seed, de la
composición por áreas del enunciado y de normas de uso general en cada área. Se
excluyen derecho ambiental e internacional, que el banco no cubre.

Uso: python -m scripts.corpus.objetivos   (luego: python -m scripts.corpus.reconstruir)
"""
import json
import re
import sys
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())

PRIMARIAS = False
AREA_SLUG_OFICIAL = {
    "Derecho constitucional": "constitucional", "Derecho administrativo": "administrativo",
    "Derecho penal": "penal", "Derecho procesal": "procesal", "Derecho comercial y sociedades": "comercial",
    "Derecho civil": "civil", "Derecho de familia": "familia", "Derecho tributario": "tributario",
    "Derecho laboral": "laboral",
    "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]": "mercados",
}
A = {
    "const": "Derecho constitucional", "adm": "Derecho administrativo", "penal": "Derecho penal",
    "proc": "Derecho procesal", "com": "Derecho comercial y sociedades", "civil": "Derecho civil",
    "fam": "Derecho de familia", "trib": "Derecho tributario", "lab": "Derecho laboral",
    "merc": "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]",
}
GESTOR = "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i="
SENADO = "http://www.secretariasenado.gov.co/senado/basedoc"

# (doc_id, título, áreas, url declarada o None para usar Senado, urls alternas)
SEED = [
    ("decreto_46_2024", "Decreto 46 de 2024 - conflictos de interés de administradores", ["com"], GESTOR + "228530", []),
    ("decreto_175_2025", "Decreto 175 de 2025 - medidas tributarias conmoción interior Catatumbo", ["trib"], GESTOR + "259629", []),
    ("decreto_24_2016", "Decreto 24 de 2016 - adiciona el Decreto 1074 de 2015", ["com"], GESTOR + "67536", []),
    ("decreto_405_2025", "Decreto 405 de 2025 - multa por despido de víctima de acoso sexual", ["lab"], GESTOR + "259517", []),
    ("decreto_4436_2005", "Decreto 4436 de 2005 - divorcio ante notario", ["fam"], GESTOR + "18346", []),
    ("decreto_780_2016", "Decreto Único Reglamentario del Sector Salud y Protección Social", ["penal", "lab"], GESTOR + "77813", []),
    ("decreto_2737_1989", "Código del Menor", ["fam"], "http://www.secretariasenado.gov.co/senado/basedoc/codigo_menor.html",
     ["https://www.icbf.gov.co/cargues/avance/compilacion/docs/codigo_menor.htm"]),
    ("ley_1909_2018", "Estatuto de la Oposición política", ["const"], None, []),
    ("acuerdo_cc_02_2015", "Acuerdo 02 de 2015 - Reglamento de la Corte Constitucional", ["const", "proc"],
     "https://sidn.ramajudicial.gov.co/SIDN/NORMATIVA/TEXTOS_COMPLETOS/3_ACUERDOS/ACUERDOS%202015/"
     "CC%20Acuerdo%202%20de%202015%20(Unifica%20y%20actualiza%20el%20Reglamento%20de%20la%20Corte%20Constitucional).pdf",
     ["https://www.corteconstitucional.gov.co/inicio/Reforma%20Reglamento.pdf"]),
    ("sentencia_csj_sp1945_2019", "Sentencia SP1945-2019", ["penal"],
     "https://cortesuprema.gov.co/corte/wp-content/uploads/relatorias/pe/b1ago2019/SP1945-2019(50523).PDF", []),
    ("sentencia_csj_sc1121_2018", "Sentencia SC1121-2018", ["civil"],
     "https://cortesuprema.gov.co/corte/wp-content/uploads/2018/10/SC1121-2018-2007-00128-01.pdf", []),
    ("sentencia_csj_sc8453_2016", "Sentencia SC8453-2016", ["com"],
     "https://cortesuprema.gov.co/corte/wp-content/uploads/2022/03/SC8453-2016-2014-02243-00-C.pdf", []),
    ("sentencia_csj_sc18392_2017", "Sentencia SC18392-2017", ["com"],
     "https://www.cortesuprema.gov.co/corte/wp-content/uploads/2019/02/SC18392-2017-2011-00081-01-1-47.pdf", []),
    ("sentencia_csj_sc3085_2024", "Sentencia SC3085-2024", ["fam"],
     "https://archivodigitalapi.cortesuprema.gov.co/share/2024/12/Sentencias/SC3085-2024.pdf", []),
    ("sentencia_csj_sl1050_2023", "Sentencia SL1050-2023", ["lab"],
     "https://cortesuprema.gov.co/corte/wp-content/uploads/2023/07/SL1050-2023.pdf", []),
]
# Sentencias de la Corte Constitucional del seed: la relatoría tiene patrón de URL.
SEED_CC = [("T", 248, 2025, ["penal"])]

# Leyes y decretos troncales que el seed no lista. Leyes: Senado por patrón.
NORMAS = [
    # administrativo y constitucional
    ("ley_734_2002", "Código Disciplinario Único (derogado por la Ley 1952 de 2019)", ["adm"], None),
    ("ley_1882_2018", "Ley 1882 de 2018 - contratación pública", ["adm"], None),
    ("ley_2022_2020", "Ley 2022 de 2020 - pliegos tipo", ["adm"], None),
    ("ley_42_1993", "Ley 42 de 1993 - control fiscal", ["adm"], None),
    ("ley_1864_2017", "Ley 1864 de 2017 - delitos electorales", ["const", "penal"], None),
    ("ley_163_1994", "Ley 163 de 1994 - disposiciones electorales", ["const"], None),
    ("ley_1621_2013", "Ley Estatutaria 1621 de 2013 - inteligencia y contrainteligencia", ["const"], None),
    ("ley_1123_2007", "Código Disciplinario del Abogado", ["proc", "adm"], None),
    # penal
    ("ley_1121_2006", "Ley 1121 de 2006 - financiación del terrorismo", ["penal"], None),
    ("ley_1959_2019", "Ley 1959 de 2019 - violencia intrafamiliar", ["penal", "fam"], None),
    ("ley_2081_2021", "Ley 2081 de 2021 - imprescriptibilidad de delitos sexuales contra menores", ["penal"], None),
    ("ley_1236_2008", "Ley 1236 de 2008 - delitos sexuales", ["penal"], None),
    ("ley_2098_2021", "Ley 2098 de 2021 - prisión perpetua revisable", ["penal"], None),
    ("ley_1918_2018", "Ley 1918 de 2018 - inhabilidades por delitos sexuales", ["penal"], None),
    ("ley_1407_2010", "Código Penal Militar", ["penal"], None),
    # procesal, civil y familia
    ("decreto_806_2020", "Decreto Legislativo 806 de 2020 - virtualidad en actuaciones judiciales", ["proc"], GESTOR + "127580"),
    ("ley_258_1996", "Ley 258 de 1996 - afectación a vivienda familiar", ["fam", "civil"], None),
    ("ley_861_2003", "Ley 861 de 2003 - patrimonio de familia de la mujer cabeza de familia", ["fam"], None),
    ("ley_70_1931", "Ley 70 de 1931 - patrimonio de familia inembargable", ["fam", "civil"], None),
    ("ley_311_1996", "Ley 311 de 1996 - registro nacional de protección familiar", ["fam"], None),
    ("ley_1404_2010", "Ley 1404 de 2010 - escuela para padres", ["fam"], None),
    ("decreto_2820_1974", "Decreto 2820 de 1974 - igualdad jurídica de los sexos", ["fam", "civil"], GESTOR + "80962"),
    ("ley_1183_2008", "Ley 1183 de 2008 - titulación de la posesión", ["civil"], None),
    ("ley_1306_2009", "Ley 1306 de 2009 - personas con discapacidad mental (derogada parcialmente)", ["fam", "civil"], None),
    # comercial
    ("decreto_2555_2010", "Decreto Único del sector financiero, asegurador y del mercado de valores", ["com"], GESTOR + "40032"),
    ("ley_1231_2008", "Ley 1231 de 2008 - factura como título valor", ["com"], None),
    ("ley_43_1990", "Ley 43 de 1990 - profesión de contador público", ["com", "trib"], None),
    ("ley_1735_2014", "Ley 1735 de 2014 - sociedades especializadas en depósitos y pagos electrónicos", ["com"], None),
    ("ley_590_2000", "Ley 590 de 2000 - mipymes", ["com"], None),
    ("ley_905_2004", "Ley 905 de 2004 - reforma de la ley de mipymes", ["com"], None),
    # tributario
    ("ley_1430_2010", "Ley 1430 de 2010 - control tributario", ["trib"], None),
    ("ley_223_1995", "Ley 223 de 1995 - racionalización tributaria", ["trib"], None),
    ("ley_44_1990", "Ley 44 de 1990 - impuesto predial unificado", ["trib"], None),
    ("ley_1111_2006", "Ley 1111 de 2006 - reforma tributaria", ["trib"], None),
    ("ley_383_1997", "Ley 383 de 1997 - evasión y contrabando", ["trib"], None),
    ("ley_2068_2020", "Ley 2068 de 2020 - turismo", ["trib", "com"], None),
    ("ley_2294_2023", "Plan Nacional de Desarrollo 2022-2026", ["trib", "adm"], None),
    # laboral
    ("ley_1846_2017", "Ley 1846 de 2017 - jornada nocturna", ["lab"], None),
    ("ley_1857_2017", "Ley 1857 de 2017 - protección integral de la familia", ["lab", "fam"], None),
    ("ley_1496_2011", "Ley 1496 de 2011 - igualdad salarial entre mujeres y hombres", ["lab"], None),
    ("ley_1280_2009", "Ley 1280 de 2009 - licencia por luto", ["lab"], None),
    ("ley_1468_2011", "Ley 1468 de 2011 - licencia de maternidad", ["lab"], None),
    ("ley_1780_2016", "Ley 1780 de 2016 - empleo juvenil", ["lab"], None),
    ("ley_1636_2013", "Ley 1636 de 2013 - mecanismo de protección al cesante", ["lab"], None),
    ("ley_1233_2008", "Ley 1233 de 2008 - cooperativas y precooperativas de trabajo asociado", ["lab"], None),
    ("ley_1955_2019", "Plan Nacional de Desarrollo 2018-2022", ["lab", "adm"], None),
    ("ley_2040_2020", "Ley 2040 de 2020 - empleo de adultos mayores", ["lab"], None),
    # mercados
    ("ley_1341_2009", "Ley 1341 de 2009 - tecnologías de la información y las comunicaciones", ["merc"], None),
    ("ley_1978_2019", "Ley 1978 de 2019 - modernización del sector TIC", ["merc"], None),
    ("ley_1450_2011", "Plan Nacional de Desarrollo 2010-2014", ["merc", "adm"], None),
]

# Hitos de la Corte Constitucional: (sala, número, año, áreas).
HITOS = [
    ("T", 406, 1992, ["const"]), ("T", 6, 1992, ["const", "proc"]), ("T", 2, 1992, ["const"]),
    ("C", 37, 1996, ["const", "proc"]), ("SU", 47, 1999, ["const", "proc"]), ("C", 836, 2001, ["const", "proc", "civil"]),
    ("C", 590, 2005, ["const", "proc"]), ("T", 25, 2004, ["const", "adm"]), ("T", 881, 2002, ["const"]),
    ("C", 1052, 2001, ["const", "proc"]), ("C", 551, 2003, ["const"]), ("C", 141, 2010, ["const"]),
    ("C", 288, 2012, ["const", "trib"]), ("C", 634, 2011, ["adm", "const"]), ("C", 539, 2011, ["adm", "const"]),
    ("SU", 917, 2010, ["adm"]), ("C", 818, 2011, ["adm"]), ("C", 951, 2014, ["adm", "const"]),
    ("C", 713, 2008, ["const", "proc"]),
    ("C", 239, 1997, ["penal", "const"]), ("C", 233, 2021, ["penal", "const"]), ("C", 221, 1994, ["penal", "const"]),
    ("C", 591, 2005, ["penal", "proc"]), ("C", 1154, 2005, ["penal", "proc"]), ("C", 209, 2007, ["penal", "proc"]),
    ("C", 228, 2002, ["penal", "proc"]), ("C", 792, 2014, ["penal", "proc"]), ("SU", 146, 2020, ["penal", "proc"]),
    ("C", 75, 2007, ["fam"]), ("C", 577, 2011, ["fam", "const"]), ("C", 683, 2015, ["fam"]),
    ("C", 71, 2015, ["fam"]), ("T", 968, 2009, ["fam"]),
    ("C", 776, 2003, ["trib", "const"]), ("C", 1011, 2008, ["merc", "const"]),
]


# La cita "C-517 de 1998" del corpus corresponde a la C-517 de 1999 (ver exclusiones).
CORRECCIONES_CC = [("C", 517, 1999, ["lab", "const"])]


def radical_cc(sala, n, anio):
    return f"{sala.lower()}{'' if sala == 'SU' else '-'}{n:03d}-{str(anio)[2:]}"


# ~37.000 conceptos que parafrasean normas: en recuperación compiten con la norma misma y
# pueden desplazarla (ruido). Fuera del inventario salvo --con-doctrina, que se decide
# midiendo sobre la muestra oficial.
DOCTRINA_MASIVA = {"supersociedades_concepto", "gestor_conceptos_fp"}


def ficha_cc(sala, n, anio, areas, origen):
    return {"doc_id": f"sentencia_cc_{sala.lower()}{n:03d}_{anio}", "titulo": f"Sentencia {sala}-{n:03d} de {anio}",
            "fuente": "Corte Constitucional - Relatoría",
            "url": f"https://www.corteconstitucional.gov.co/relatoria/{anio}/{radical_cc(sala, n, anio)}.htm",
            "areas": [A[a] for a in areas], "origen": origen}


def ficha(doc_id, titulo, areas, url, alternas, origen):
    fila = {"doc_id": doc_id, "titulo": titulo, "areas": [A[a] for a in areas], "origen": origen}
    if url:
        fila["url"] = url
        fila["fuente"] = ("Corte Suprema de Justicia - Relatoría" if "cortesuprema" in url else
                          "SUIN-Juriscol - Ministerio de Justicia y del Derecho" if "suin-juriscol" in url else
                          "ICBF - Compilación jurídica" if "icbf.gov.co" in url else
                          "Corte Constitucional" if "corteconstitucional" in url else
                          "Rama Judicial - SIDN" if "ramajudicial" in url else
                          "Secretaría General del Senado - Base documental" if "secretariasenado" in url else
                          "Departamento Administrativo de la Función Pública - Gestor Normativo")
    else:
        fila["fuente"] = "Secretaría General del Senado - Base documental"
    if alternas:
        fila["urls_alternas"] = alternas
    if doc_id == "acuerdo_cc_02_2015":
        fila.update(tipo="acuerdo", numero="02", anio=2015, organo_emisor="Corte Constitucional")
    return fila


def desde_grafo(umbral, umbral_sentencias=None):
    """Brechas del grafo normativo (scripts.corpus.grafo) citadas por >= umbral documentos.

    Leyes, decretos y actos legislativos van a Senado con respaldo del Gestor Normativo;
    las sentencias de la Corte Constitucional, a su relatoría. Las de la Corte Suprema
    no tienen patrón de URL y quedan fuera (se listan como pendientes).
    """
    import csv
    ruta = RAIZ / "data/raw/grafo_faltantes.csv"
    if not ruta.is_file():
        return [], []
    slug_a_area = {s: n for n, s in AREA_SLUG_OFICIAL.items()}
    documentos, pendientes = [], []
    with ruta.open(encoding="utf-8-sig") as f:
        for fila in csv.DictReader(f):
            es_sentencia = fila["clave"].startswith("sentencia")
            # Sentencias: cuentan solo las citas desde fuentes primarias (normas y núcleo curado);
            # si no, cada sentencia agregada arrastra las que ella cita y el grafo no cierra.
            citan = int(fila.get("citan_fuentes_primarias") or fila["documentos_que_citan"]) \
                if es_sentencia and PRIMARIAS else int(fila["documentos_que_citan"])
            if citan < ((umbral_sentencias or umbral) if es_sentencia else umbral):
                continue
            areas = [slug_a_area[a.strip()] for a in fila["areas"].split(",") if a.strip() in slug_a_area][:3]
            base = {"doc_id": fila["clave"], "titulo": fila["norma"], "areas": areas, "origen": "grafo_normativo",
                    "justificacion": f"citada por {fila['documentos_que_citan']} documentos del corpus "
                                     f"({fila['menciones']} menciones), p. ej. {fila['ejemplos_citantes'][:120]}"}
            m = re.fullmatch(r"sentencia_cc_(c|t|su)(\d+)_(\d{4})", fila["clave"])
            if m:
                documentos.append({**ficha_cc(m.group(1).upper(), int(m.group(2)), int(m.group(3)), [], "grafo_normativo"),
                                   "areas": areas, "justificacion": base["justificacion"]})
            elif re.fullmatch(r"(ley|decreto|acto_legislativo)_\d+_\d{4}", fila["clave"]):
                documentos.append({**base, "fuente": "Secretaría General del Senado - Base documental"})
            else:
                pendientes.append({"norma": fila["norma"], "motivo": "sin patrón de URL (Corte Suprema u otra)"})
    return documentos, pendientes


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--umbral-grafo", type=int, default=4,
                    help="documentos del corpus que deben citar una norma ausente para incorporarla (0 = no usar el grafo)")
    ap.add_argument("--umbral-sentencias", type=int, default=None,
                    help="umbral propio para sentencias: cada sentencia cita decenas de sentencias y con el "
                         "mismo umbral de las normas el grafo no converge")
    ap.add_argument("--ronda5-origenes", nargs="*",
                    help="solo estos orígenes de la ronda 5 (p. ej. para cerrar una entrega parcial)")
    ap.add_argument("--con-doctrina", action="store_true",
                    help="incluye la doctrina masiva de la ronda 5 (conceptos de Supersociedades y Función Pública)")
    ap.add_argument("--sentencias-por-primarias", action="store_true",
                    help="el umbral de sentencias se mide sobre citas desde normas y núcleo curado")
    args = ap.parse_args()
    global PRIMARIAS
    PRIMARIAS = args.sentencias_por_primarias
    # Las rondas anteriores del grafo se conservan: el grafo se recalcula sobre el corpus
    # ampliado y ya no lista lo que se incorporó antes.
    ruta_previa = RAIZ / "configs/corpus_objetivos.json"
    previas = [d for d in json.loads(ruta_previa.read_text(encoding="utf-8"))["documentos"]
               if d.get("origen") == "grafo_normativo"] if ruta_previa.is_file() else []
    iteracion = max((d.get("iteracion", 1) for d in previas), default=0) + 1
    existentes = {d["doc_id"] for d in json.loads((RAIZ / "corpus_manifest.json").read_text(encoding="utf-8"))["documentos"]}
    documentos = [ficha(*f, "seed_targets") for f in SEED]
    documentos += [ficha_cc(*f, "seed_targets") for f in SEED_CC]
    documentos += [ficha(d, t, a, u, [], "norma_troncal") for d, t, a, u in NORMAS]
    documentos += [ficha_cc(*f, "hito_jurisprudencial") for f in HITOS]
    # Ronda 4: jurisprudencia de cierre y doctrina oficial (scripts.corpus.fuentes_ronda4).
    ruta_r4 = RAIZ / "configs/corpus_ronda4.json"
    if ruta_r4.is_file():
        documentos += json.loads(ruta_r4.read_text(encoding="utf-8"))["documentos"]
    # Correcciones de citas erradas cuya norma real no estaba (configs/corpus_exclusiones.json).
    documentos += [ficha_cc(*f, "cita_corregida") for f in CORRECCIONES_CC]
    del_grafo, pendientes_grafo = desde_grafo(args.umbral_grafo, args.umbral_sentencias) if args.umbral_grafo else ([], [])
    ruta_excl = RAIZ / "configs/corpus_exclusiones.json"
    excluidas = {e["clave"] for e in json.loads(ruta_excl.read_text(encoding="utf-8"))["exclusiones"]} \
        if ruta_excl.is_file() else set()
    del_grafo = [d for d in del_grafo if d["doc_id"] not in excluidas]
    curados = {d["doc_id"] for d in documentos}
    previos = {d["doc_id"] for d in previas}
    nuevos = [{**d, "iteracion": iteracion, "umbral": args.umbral_sentencias if d["doc_id"].startswith("sentencia")
               and args.umbral_sentencias else args.umbral_grafo,
               "criterio": "fuentes_primarias" if PRIMARIAS and d["doc_id"].startswith("sentencia") else "todo_el_corpus"}
              for d in del_grafo if d["doc_id"] not in curados | previos]
    del_grafo = [{"iteracion": 1, "umbral": 4, **d} for d in previas
                 if d["doc_id"] not in curados and d["doc_id"] not in excluidas] + nuevos
    documentos += del_grafo
    # Ronda 5: universo completo de las fuentes primarias (scripts.corpus.fuentes_ronda5). Lo
    # que ya entró por otra vía (curado o grafo) conserva su ficha y su procedencia.
    ruta_r5 = RAIZ / "configs/corpus_ronda5.json"
    if ruta_r5.is_file():
        ya = {d["doc_id"] for d in documentos}
        documentos += [d for d in json.loads(ruta_r5.read_text(encoding="utf-8"))["documentos"]
                       if d["doc_id"] not in ya and d["doc_id"] not in excluidas
                       and (args.con_doctrina or d.get("origen") not in DOCTRINA_MASIVA)
                       and (not args.ronda5_origenes or d.get("origen") in args.ronda5_origenes)]
    # Los códigos llevan prefijo co_ ("co_ley_1564_2012", "co_decreto_1082_2015"): son la misma norma.
    ruta_raw = RAIZ / "data/raw/manifest.json"
    if ruta_raw.is_file():
        existentes |= {d["doc_id"] for d in json.loads(ruta_raw.read_text(encoding="utf-8")) if d["doc_id"].startswith("co_")}
    repetidos = [d["doc_id"] for d in documentos if d["doc_id"] in existentes or "co_" + d["doc_id"] in existentes]
    documentos = [d for d in documentos if d["doc_id"] not in repetidos]
    salida = {"descripcion": __doc__.strip().splitlines()[0], "documentos": documentos,
              "pendientes_manuales": [
                  {"norma": "Sentencia SC425-2024", "motivo": "la relatoría no la publica como archivo propio"},
                  {"norma": "Sentencia SC3674-2021", "motivo": "la relatoría no la publica como archivo propio"},
                  {"norma": "Sentencia SL1972-2025", "motivo": "la relatoría no la publica como archivo propio"},
              ] + pendientes_grafo,
              "descartados_identificador_inexistente": [
                  "Ley 11500 de 2007", "Ley 116 de 2006", "Ley 1150 de 2005", "Ley 964 de 2006",
                  "Ley 2737 de 1989", "Ley 23 de 1961", "Decreto 1563 de 2012", "Decreto 875 de 2008",
                  "Sentencia SU-6 de 1991",
                  "Ley 1692 de 2017 (la Ley 1692 es de 2013 y aprueba un convenio tributario internacional)",
                  "Sentencia SU-488 de 2011 (la relatoría no la publica; existe la T-488 de 2011)"]}
    ruta = RAIZ / "configs/corpus_objetivos.json"
    ruta.write_text(json.dumps(salida, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{ruta.relative_to(RAIZ)}: {len(documentos)} documentos ({len(del_grafo)} del grafo, "
          f"{len(nuevos)} nuevos en la iteración {iteracion}); "
          f"omitidos por existir: {len(repetidos)}")


if __name__ == "__main__":
    sys.exit(main())
