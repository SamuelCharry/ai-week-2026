"""Auditoría del corpus de punta a punta: qué falta, qué está mal y qué viola las reglas.

Cada comprobación produce hallazgos con severidad:

    CRITICO  invalida la entrega o descalifica (fuga del conjunto de evaluación,
             original alterado, documento del inventario sin descargar, campos
             obligatorios del manifiesto ausentes).
    ALTO     pierde puntos seguro (documento incompleto, página de error guardada
             como norma, duplicados, texto dañado, fuente no oficial).
    AVISO    revisar a mano (sentencia sin parte resolutiva visible, norma que el
             grafo muestra citada y ausente, PDF con páginas sin texto).

Secciones:
    A. Cumplimiento del reto: campos del manifiesto (enunciado, paso 5), fuentes
       oficiales colombianas, derecho extranjero o internacional, derecho ambiental,
       fuga de preguntas y respuestas de la muestra dentro del corpus (sección 8).
    B. Integridad de los originales: todo el inventario descargado, hashes, que el
       texto nombre la norma, duplicados por contenido y por identidad.
    C. Completitud: numeración de artículos sin huecos, candidata elegida frente a
       las demás fuentes, cadena de tramos de Senado cerrada, parte resolutiva.
    D. Calidad del texto: mojibake, caracteres de reemplazo, PDF sin texto.
    E. Cobertura: seed_targets, fundamento de la muestra y brechas del grafo.
    G. Cierre del grafo normativo: toda referencia citada por encima del umbral está en
       el corpus o excluida con motivo y evidencia (configs/corpus_exclusiones.json).
    F. Ingesta (si existe data/processed/corpus): documentos con error, documentos
       del manifiesto que no llegaron al corpus procesado y comprobaciones fallidas.

Escribe reports/auditoria_corpus.csv (un hallazgo por fila), reports/auditoria_corpus.json
(resumen y hallazgos) y reports/auditoria_corpus.md (lectura rápida). Sale con código 1
si hay hallazgos críticos, para poder usarla antes de congelar el índice.

Uso: python -m legalrag.ingestion.auditoria [--sin-texto] [--umbral-grafo 4]
"""
import argparse
import csv
import hashlib
import json
import re
import sys
import unicodedata
import urllib.parse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from legalrag.preprocessing.ocr import es_legible, legibilidad  # noqa: E402

SEVERIDADES = ("CRITICO", "ALTO", "AVISO")
CAMPOS_MANIFIESTO = ("doc_id", "titulo", "fuente", "url", "fecha_consulta", "areas")
# Dominios de publicadores oficiales colombianos (y la CAN, cuyas decisiones rigen en Colombia).
DOMINIOS_OFICIALES = (".gov.co", "comunidadandina.org")
TRATADO = re.compile(r"(?i)\bpor\s+medio\s+de\s+la\s+cual\s+se\s+aprueba|\baprueba\s+(el|la|los|las)\s+"
                     r"[\"“]?(convenio|convenci[oó]n|tratado|acuerdo|protocolo|enmienda|estatuto)")
AMBIENTAL = {"ley_99_1993", "ley_1333_2009", "decreto_1076_2015", "decreto_2811_1974", "ley_2111_2021"}
MOJIBAKE = re.compile(r"�|Ã[\u0080-¿]|Â[\u0080-¿]|â[€\u0080-¿]")
IDENTIFICADORES_INEXISTENTES_SEED = {
    ("ley", "11500", "2007"), ("ley", "116", "2006"), ("ley", "1150", "2005"), ("ley", "964", "2006"),
    ("ley", "2737", "1989"), ("ley", "23", "1961"), ("decreto", "1563", "2012"), ("decreto", "875", "2008"),
    ("jurisprudencia", "SU-6", "1991"), ("ley", "1692", "2017"), ("jurisprudencia", "SU-488", "2011"),
}
CODIGOS = {"constitucion": "constitucion_1_1991", "codigo_general_proceso": "ley_1564_2012",
           "codigo_sustantivo_trabajo": "decreto_2663_1950", "estatuto_tributario": "decreto_624_1989",
           "decision_andina_486": "decision_486_2000", "estatuto_consumidor": "ley_1480_2011",
           "codigo_infancia": "ley_1098_2006", "codigo_disciplinario": "ley_1952_2019",
           "codigo_nacional_policia": "ley_1801_2016", "codigo_civil": "ley_84_1873",
           "codigo_comercio": "decreto_410_1971", "codigo_penal": "ley_599_2000",
           "codigo_procedimiento_penal": "ley_906_2004", "cpaca": "ley_1437_2011"}


class Auditoria:
    def __init__(self):
        self.hallazgos = []
        self.metricas = {}

    def agregar(self, seccion, severidad, doc_id, comprobacion, detalle):
        assert severidad in SEVERIDADES
        self.hallazgos.append({"seccion": seccion, "severidad": severidad, "doc_id": doc_id or "",
                               "comprobacion": comprobacion, "detalle": str(detalle)[:500]})


def _leer_json(ruta, defecto=None):
    ruta = Path(ruta)
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.is_file() else defecto


def _leer_csv(ruta):
    ruta = Path(ruta)
    if not ruta.is_file():
        return []
    with ruta.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def clave(doc_id):
    """Identidad de la norma sin el prefijo de origen: co_ley_1437_2011 == ley_1437_2011."""
    return re.sub(r"^co_", "", doc_id)


def normalizar(texto):
    texto = unicodedata.normalize("NFD", str(texto).casefold())
    return re.sub(r"\s+", " ", "".join(c for c in texto if unicodedata.category(c) != "Mn")).strip()


# ----------------------------------------------------------------- A. cumplimiento

def auditar_cumplimiento(aud, inventario, raw):
    for doc in inventario:
        faltan = [c for c in CAMPOS_MANIFIESTO if not doc.get(c)]
        entrada = raw.get(doc["doc_id"], {})
        # fecha_consulta, url y areas los completa la descarga (y legalrag.ingestion.areas) en
        # data/raw/manifest.json, que es el manifiesto que se entrega: basta con que estén allí.
        faltan = [c for c in faltan if not entrada.get(c)]
        if faltan:
            aud.agregar("A", "CRITICO", doc["doc_id"], "campos_obligatorios_manifiesto", f"faltan: {faltan}")
        for url in [entrada.get("url") or doc.get("url")] + [a.get("url_final") for a in entrada.get("archivos_raw", [])]:
            if not url:
                continue
            host = urllib.parse.urlsplit(url).netloc.lower()
            if not any(host.endswith(d) for d in DOMINIOS_OFICIALES):
                aud.agregar("A", "ALTO", doc["doc_id"], "fuente_no_oficial", url)
                break
        titulo = " ".join(str(doc.get(c) or "") for c in ("titulo", "epigrafe"))
        if TRATADO.search(titulo) or entrada.get("alcance") == "tratado_internacional":
            aud.agregar("A", "AVISO", doc["doc_id"], "ley_aprobatoria_de_tratado",
                        "ley colombiana que incorpora un tratado; incluida por decisión del equipo "
                        "(bloque de constitucionalidad, art. 93 C.P.)")
        if clave(doc["doc_id"]) in AMBIENTAL or entrada.get("alcance") == "ambiental":
            aud.agregar("A", "AVISO", doc["doc_id"], "derecho_ambiental",
                        "el enunciado (4.2) excluye derecho ambiental del banco")


def auditar_fuga(aud, textos, raw=None):
    """Sección 8 del enunciado: ningún ítem de evaluación ni material con respuestas en el índice.

    Busca en el corpus fragmentos literales (24 palabras) de preguntas, opciones y
    respuestas esperadas de la muestra. Un documento oficial puede citar la misma
    norma, pero no reproducir 24 palabras seguidas de una pregunta redactada por juristas.
    """
    muestra = RAIZ / "data/oficial/data/sample_50.jsonl"
    if not muestra.is_file():
        muestra = RAIZ / "data/sample_50.jsonl"
    if not muestra.is_file():
        aud.agregar("A", "AVISO", "", "fuga_evaluacion", "no se encontró sample_50.jsonl; comprobación omitida")
        return
    sondas = []
    for linea in muestra.read_text(encoding="utf-8-sig").splitlines():
        if not linea.strip():
            continue
        item = json.loads(linea)
        for campo in ("pregunta", "respuesta_esperada", "texto_respuesta_correcta", "justificacion"):
            palabras = normalizar(item.get(campo) or "").split()
            for i in range(0, max(0, len(palabras) - 24) + 1, 12):
                if len(palabras) >= 24:
                    sondas.append((item["id"], campo, " ".join(palabras[i:i + 24])))
    aud.metricas["fuga_sondas"] = len(sondas)
    oficiales = {d for d, e in (raw or {}).items()
                 if all(any(urllib.parse.urlsplit(a.get("url") or "").netloc.lower().endswith(o) for o in DOMINIOS_OFICIALES)
                        for a in e.get("archivos_raw") or [{}])}
    for doc_id, texto in textos.items():
        plano = normalizar(texto)
        for item_id, campo, sonda in sondas:
            if sonda not in plano:
                continue
            if campo == "pregunta" or doc_id not in oficiales:
                # Una pregunta redactada por juristas no aparece en una fuente oficial, y una
                # fuente no oficial con la respuesta es exactamente lo que prohíbe la sección 8.
                aud.agregar("A", "CRITICO", doc_id, "fuga_evaluacion",
                            f"contiene texto literal del ítem {item_id} ({campo}): '{sonda[:80]}…'")
            else:
                aud.agregar("A", "AVISO", doc_id, "respuesta_cita_fuente_oficial",
                            f"la respuesta esperada del ítem {item_id} de la muestra copia este documento oficial "
                            f"(no es fuga: el texto es de la fuente, no del banco): '{sonda[:60]}…'")
            break
    ruta_proc = RAIZ / "data/processed"
    for archivo in ruta_proc.rglob("*") if ruta_proc.is_dir() else []:
        if archivo.is_file() and re.search(r"(?i)sample_50|seed_targets|questions|submission", archivo.name):
            aud.agregar("A", "CRITICO", str(archivo.relative_to(RAIZ)), "fuga_evaluacion",
                        "archivo del material de evaluación dentro de data/processed")


# ------------------------------------------------------------- B. integridad

def auditar_integridad(aud, inventario, raw, textos):
    from legalrag.ingestion.reconstruir import ficha_base, identidad_valida

    ids_inventario = {d["doc_id"] for d in inventario}
    for doc in inventario:
        entrada = raw.get(doc["doc_id"])
        if not entrada:
            if doc["doc_id"] in exclusiones():
                continue
            aud.agregar("B", "CRITICO", doc["doc_id"], "sin_descargar", "está en el inventario y no en data/raw/manifest.json")
            continue
        for archivo in [a for x in entrada.get("archivos_raw") or [] for a in (x, x.get("texto_derivado")) if a]:
            ruta = RAIZ / "data/raw" / archivo["archivo"]
            if not ruta.is_file():
                aud.agregar("B", "CRITICO", doc["doc_id"], "original_ausente", archivo["archivo"])
                continue
            contenido = ruta.read_bytes()
            if len(contenido) != archivo["bytes"] or hashlib.sha256(contenido).hexdigest() != archivo["sha256"]:
                aud.agregar("B", "CRITICO", doc["doc_id"], "original_alterado", archivo["archivo"])
        texto = textos.get(doc["doc_id"])
        if texto is not None and not identidad_valida(texto, {**ficha_base(doc), **{k: v for k, v in entrada.items() if k in ("tipo", "numero", "anio", "organo_emisor")}}):
            aud.agregar("B", "ALTO", doc["doc_id"], "no_nombra_la_norma",
                        "el texto no contiene el identificador esperado (¿portada o página de error?)")
    for doc_id in set(raw) - ids_inventario:
        aud.agregar("B", "AVISO", doc_id, "fuera_del_inventario", "en data/raw/manifest.json pero no en el inventario")
    # Duplicados: el mismo original bajo dos documentos, o la misma norma con dos doc_id.
    por_hash = defaultdict(list)
    for doc_id, entrada in raw.items():
        huella = "".join(a["sha256"] for a in entrada.get("archivos_raw") or [])
        por_hash[huella].append(doc_id)
    for ids in por_hash.values():
        if len(ids) > 1:
            aud.agregar("B", "ALTO", ids[0], "duplicado_por_contenido", f"mismos originales que {ids[1:]}")
    por_clave = defaultdict(list)
    for doc_id in raw:
        por_clave[clave(doc_id)].append(doc_id)
    for ids in por_clave.values():
        if len(ids) > 1:
            aud.agregar("B", "ALTO", ids[0], "duplicado_por_identidad", f"la misma norma también como {ids[1:]}")


# ------------------------------------------------------------- C. completitud

def auditar_completitud(aud, raw, textos):
    from legalrag.ingestion.reconstruir import ficha_base, metricas

    candidatas = defaultdict(list)
    for fila in _leer_csv(RAIZ / "data/raw/reconstruccion.csv"):
        candidatas[fila["doc_id"]].append(fila)
    resumen = Counter()
    for doc_id, entrada in raw.items():
        texto = textos.get(doc_id)
        if texto is None:
            continue
        judicial = entrada.get("tipo") in ("sentencia", "auto")
        m = metricas(texto, judicial)
        if judicial:
            if not m["cierre"]:
                resumen["sentencias_sin_cierre"] += 1
                aud.agregar("C", "AVISO", doc_id, "sentencia_sin_parte_resolutiva",
                            "no aparece 'RESUELVE' ni 'notifíquese', 'cópiese' o firmas al final: posible texto parcial")
            if m["caracteres"] < 8000:
                aud.agregar("C", "ALTO", doc_id, "sentencia_corta", f"{m['caracteres']} caracteres: posible texto parcial")
        elif entrada.get("tipo") in ("ley", "decreto", "acto_legislativo", "constitucion", "decision"):
            if m["articulos"] == 0:
                aud.agregar("C", "ALTO", doc_id, "norma_sin_articulos", "no se detectó ningún artículo")
            elif m["huecos"] and m["huecos"] / max(m["max_articulo"], 1) > 0.10 and m["articulos"] > 10:
                resumen["normas_con_huecos"] += 1
                aud.agregar("C", "AVISO", doc_id, "huecos_en_numeracion",
                            f"{m['articulos']} artículos de 1..{m['max_articulo']} ({m['huecos']} huecos). "
                            "Normal en leyes modificatorias o con artículos derogados omitidos; revisar si no lo es")
        # ¿Otra fuente tenía más?
        filas = candidatas.get(doc_id, [])
        elegida = next((f for f in filas if f.get("elegida") == "True"), None)
        for f in filas:
            if elegida and f is not elegida and f.get("valida") == "True" and not judicial:
                if int(f.get("articulos") or 0) > int(elegida.get("articulos") or 0):
                    aud.agregar("C", "ALTO", doc_id, "fuente_mas_completa_descartada", f"{f['url']} tiene más artículos")
        # Cadena de Senado: el último tramo no debe enlazar un tramo siguiente no descargado.
        archivos = entrada.get("archivos_raw") or []
        if archivos and "secretariasenado" in (archivos[0].get("url") or ""):
            ultimo = (RAIZ / "data/raw" / archivos[-1]["archivo"])
            if ultimo.is_file():
                base = re.sub(r"(_pr\d+)?\.html?$", "", archivos[0]["url"].rsplit("/", 1)[1], flags=re.I)
                enlazados = set(re.findall(re.escape(base) + r"_pr\d+\.html?", ultimo.read_bytes().decode("latin-1"), re.I))
                descargados = {a["url"].rsplit("/", 1)[1].lower() for a in archivos}
                pendientes = {e for e in enlazados if e.lower() not in descargados}
                if pendientes:
                    aud.agregar("C", "ALTO", doc_id, "cadena_senado_incompleta", f"tramos sin descargar: {sorted(pendientes)[:5]}")
    aud.metricas.update(resumen)


# ------------------------------------------------------------- D. calidad del texto

def auditar_calidad(aud, textos):
    for doc_id, texto in textos.items():
        dañados = len(MOJIBAKE.findall(texto))
        if dañados > 20 or (texto and dañados / len(texto) > 1e-4):
            aud.agregar("D", "ALTO", doc_id, "texto_con_mojibake", f"{dañados} secuencias dañadas")
        # PDF escaneado con capa de texto basura ("Repdbli鍛deColombia"): el mojibake clásico
        # no lo detecta. Misma medida que decide el OCR en la reconstrucción.
        no_latinas, funcionales = legibilidad(texto)
        if not es_legible(texto):
            aud.agregar("D", "ALTO", doc_id, "texto_ilegible_requiere_ocr",
                        f"{no_latinas:.1%} letras no latinas; {funcionales:.1%} palabras funcionales (lo normal es > 30 %)")
        if len(texto.strip()) < 1500:
            aud.agregar("D", "ALTO", doc_id, "texto_muy_corto", f"{len(texto)} caracteres")
        pies = len(re.findall(r"normas de uso de la informaci[oó]n aqu[ií] contenida", texto, re.I))
        if pies > 1:
            aud.agregar("D", "AVISO", doc_id, "pie_editorial_en_original",
                        f"{pies} pies editoriales de Avance Jurídico en el original; la ingesta 0.5.0 los retira")


# ------------------------------------------------------------- E. cobertura

def _doc_de_canonico(canonico, ids):
    tipo, numero, anio = canonico
    if numero is None:
        objetivo = CODIGOS.get(tipo)
        return objetivo if objetivo in ids else None
    if tipo == "jurisprudencia":
        sala, num = numero.split("-")
        num = int(num)
        candidatos = ([f"sentencia_cc_{sala.lower()}{num:03d}_{anio}"] if sala in ("C", "T", "SU")
                      else [f"sentencia_csj_{sala.lower()}{num}_{anio}"])
    elif tipo == "acuerdo":
        candidatos = [f"acuerdo_{int(numero)}_{anio}", f"acuerdo_cc_{numero}_{anio}", f"acuerdo_cc_{int(numero):02d}_{anio}"]
    else:
        candidatos = [f"{tipo}_{int(numero)}_{anio}"]
    return next((c for c in candidatos if c in ids), None)


def auditar_cobertura(aud, raw, umbral_grafo):
    ids = {clave(d) for d in raw}
    seed = _leer_json(RAIZ / "data/oficial/data/seed_targets.json") or _leer_json(RAIZ / "data/seed_targets.json")
    if seed:
        total = cubiertas = 0
        for objetivo in seed["documentos"]:
            canonico = tuple(objetivo["canonico"])
            if canonico in IDENTIFICADORES_INEXISTENTES_SEED:
                continue
            total += objetivo["items_del_banco"]
            if _doc_de_canonico(canonico, ids):
                cubiertas += objetivo["items_del_banco"]
            else:
                aud.agregar("E", "ALTO" if objetivo["items_del_banco"] > 1 else "AVISO", "",
                            "seed_target_ausente", f"{objetivo['norma']} ({objetivo['items_del_banco']} ítems del banco)")
        aud.metricas["seed_menciones_cubiertas"] = f"{cubiertas}/{total}"
    for fila in _leer_csv(RAIZ / "data/raw/grafo_faltantes.csv"):
        if int(fila["documentos_que_citan"]) >= umbral_grafo and clave(fila["clave"]) not in ids:
            aud.agregar("E", "AVISO", fila["clave"], "citada_por_el_corpus_y_ausente",
                        f"{fila['norma']}: la citan {fila['documentos_que_citan']} documentos ({fila['areas']})")


# ------------------------------------------------------------- F. ingesta

def auditar_identidad_codigos(aud, raw):
    """Los códigos que el extractor oficial reconoce solo por nombre deben poder citarse.

    Si la identidad del documento no incluye el cuerpo del código, toda cita a
    "Código Sustantivo del Trabajo, art. 64" cuenta como no respaldada (penalización doble).
    """
    try:
        from legalrag.evaluation.oficial import cargar_citaciones, identidad
        citas = cargar_citaciones(RAIZ)
    except Exception as error:
        aud.agregar("E", "AVISO", "", "identidad_codigos", f"sin extractor oficial: {error}")
        return
    presentes = set()
    for d in raw.values():
        cuerpos = set(identidad(d, citas))
        titulo = citas.norm(d.get("titulo") or "")
        if titulo.startswith(("codigo", "estatuto", "constitucion")):
            cuerpos |= citas.bodies(citas.extract(titulo))
        presentes |= {c[0] for c in cuerpos if c[1] is None}
    for codigo in citas.CODES:
        if codigo not in presentes:
            aud.agregar("E", "CRITICO" if codigo in CODIGOS else "ALTO", "", "codigo_sin_identidad",
                        f"ningún documento se reconoce como '{codigo}': sus citas contarían como no respaldadas")


def exclusiones(raiz=RAIZ):
    """Referencias del grafo que no se incorporan, cada una con su motivo y evidencia."""
    ruta = Path(raiz) / "configs/corpus_exclusiones.json"
    return {e["clave"]: e for e in _leer_json(ruta, {"exclusiones": []})["exclusiones"]}


def auditar_cierre_grafo(aud, raw, umbral_normas=4, umbral_sentencias=8):
    """G. El grafo normativo está cerrado: nada citado por encima del umbral queda suelto.

    Cada referencia que el corpus cita en >= umbral documentos debe estar en el corpus o
    en configs/corpus_exclusiones.json con motivo (cita errada, inexistente, sin fuente
    pública) y evidencia. Requiere un grafo recalculado sobre el corpus actual.
    """
    citas = _leer_csv(RAIZ / "data/raw/grafo_citas.csv")
    if not citas:
        aud.agregar("G", "ALTO", "", "grafo_sin_calcular", "correr python -m legalrag.ingestion.grafo")
        return
    ids = {clave(d) for d in raw}
    excluidas = exclusiones()
    abiertas = excl = 0
    for fila in citas:
        umbral = umbral_sentencias if fila["clave"].startswith("sentencia") else umbral_normas
        if int(fila["documentos_que_citan"]) < umbral or clave(fila["clave"]) in ids:
            continue
        if fila["clave"] in excluidas:
            excl += 1
            continue
        abiertas += 1
        aud.agregar("G", "CRITICO", fila["clave"], "grafo_abierto",
                    f"{fila['norma']}: la citan {fila['documentos_que_citan']} documentos del corpus y no está "
                    "ni tiene exclusión justificada")
    for k, e in excluidas.items():
        if not e.get("motivo") or not e.get("evidencia"):
            aud.agregar("G", "ALTO", k, "exclusion_sin_evidencia", "toda exclusión necesita motivo y evidencia")
    aud.metricas.update(grafo_abiertas=abiertas, grafo_excluidas_con_evidencia=excl,
                        grafo_umbrales=f"normas >= {umbral_normas}, sentencias >= {umbral_sentencias}")


def auditar_ingesta(aud, raw):
    corpus = RAIZ / "data/processed/corpus"
    if not (corpus / "documentos.jsonl").is_file():
        aud.agregar("F", "AVISO", "", "ingesta_pendiente", "no existe data/processed/corpus/documentos.jsonl")
        return
    documentos = [json.loads(l) for l in (corpus / "documentos.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    procesados = {d["doc_id"]: d for d in documentos}
    for doc_id in set(raw) - set(procesados):
        aud.agregar("F", "CRITICO", doc_id, "no_llego_al_corpus", "descargado pero ausente del corpus procesado")
    for d in documentos:
        if d.get("estado_extraccion") == "error":
            aud.agregar("F", "CRITICO", d["doc_id"], "error_de_extraccion", d.get("error"))
        if d.get("apta_para_busqueda") is False:
            aud.agregar("F", "ALTO", d["doc_id"], "excluido_de_busqueda", d.get("rechazos_fuente"))
        if raw.get(d["doc_id"]) and d.get("archivos_raw") != raw[d["doc_id"]].get("archivos_raw"):
            aud.agregar("F", "CRITICO", d["doc_id"], "corpus_desactualizado",
                        "los originales cambiaron después de la ingesta: volver a ingerir")
    for c in _leer_csv(corpus / "comprobaciones.csv"):
        if c.get("resultado") != "ok":
            aud.agregar("F", "CRITICO", "", "comprobacion_ingesta_fallida", c.get("comprobacion"))


# ------------------------------------------------------------- salida

def escribir(aud, destino):
    destino.mkdir(exist_ok=True)
    conteo = Counter(h["severidad"] for h in aud.hallazgos)
    por_comprobacion = Counter((h["severidad"], h["comprobacion"]) for h in aud.hallazgos)
    resumen = {"fecha": datetime.now().isoformat(timespec="seconds"),
               "hallazgos": {s: conteo.get(s, 0) for s in SEVERIDADES}, "metricas": aud.metricas,
               "por_comprobacion": [{"severidad": s, "comprobacion": c, "cantidad": n}
                                    for (s, c), n in sorted(por_comprobacion.items(), key=lambda x: (SEVERIDADES.index(x[0][0]), -x[1]))]}
    orden = sorted(aud.hallazgos, key=lambda h: (SEVERIDADES.index(h["severidad"]), h["seccion"], h["comprobacion"], h["doc_id"]))
    with (destino / "auditoria_corpus.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["severidad", "seccion", "comprobacion", "doc_id", "detalle"])
        w.writeheader()
        w.writerows(orden)
    (destino / "auditoria_corpus.json").write_text(json.dumps({**resumen, "detalle": orden}, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")
    lineas = [f"# Auditoría del corpus ({resumen['fecha']})", "",
              " | ".join(f"**{s}**: {conteo.get(s, 0)}" for s in SEVERIDADES), "",
              "| Severidad | Comprobación | Cantidad |", "|---|---|---:|"]
    lineas += [f"| {r['severidad']} | {r['comprobacion']} | {r['cantidad']} |" for r in resumen["por_comprobacion"]]
    lineas += ["", "## Métricas", ""] + [f"- {k}: {v}" for k, v in aud.metricas.items()]
    lineas += ["", "## Críticos y altos", ""] + [f"- `{h['severidad']}` {h['comprobacion']} `{h['doc_id']}`: {h['detalle']}"
                                                 for h in orden if h["severidad"] != "AVISO"][:300]
    (destino / "auditoria_corpus.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-texto", action="store_true", help="omite las comprobaciones que extraen texto (rápida)")
    ap.add_argument("--umbral-grafo", type=int, default=4)
    args = ap.parse_args()

    from legalrag.ingestion.reconstruir import inventario

    aud = Auditoria()
    inv = inventario(RAIZ)
    raw = {d["doc_id"]: d for d in _leer_json(RAIZ / "data/raw/manifest.json", [])}
    textos = {}
    if not args.sin_texto:
        from legalrag.ingestion.grafo import texto_en_cache
        for n, (doc_id, entrada) in enumerate(sorted(raw.items()), 1):
            try:
                textos[doc_id] = texto_en_cache(RAIZ, entrada)
            except Exception as error:
                aud.agregar("B", "CRITICO", doc_id, "original_ilegible", f"{type(error).__name__}: {error}")
            if n % 100 == 0:
                print(f"Texto extraído: {n}/{len(raw)}", flush=True)
    aud.metricas.update(inventario=len(inv), descargados=len(raw), con_texto=len(textos))

    auditar_cumplimiento(aud, inv, raw)
    auditar_integridad(aud, inv, raw, textos)
    if textos:
        auditar_fuga(aud, textos, raw)
        auditar_completitud(aud, raw, textos)
        auditar_calidad(aud, textos)
    auditar_cobertura(aud, raw, args.umbral_grafo)
    auditar_identidad_codigos(aud, raw)
    auditar_cierre_grafo(aud, raw)
    auditar_ingesta(aud, raw)

    resumen = escribir(aud, RAIZ / "reports")
    print(json.dumps({k: resumen[k] for k in ("hallazgos", "metricas")}, ensure_ascii=False, indent=1))
    for r in resumen["por_comprobacion"]:
        print(f"  {r['severidad']:8s} {r['comprobacion']:36s} {r['cantidad']}")
    print("Detalle: reports/auditoria_corpus.md, reports/auditoria_corpus.csv")
    return 1 if resumen["hallazgos"]["CRITICO"] else 0


if __name__ == "__main__":
    sys.exit(main())
