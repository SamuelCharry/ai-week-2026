import hashlib
import time
from collections import defaultdict

import numpy as np


def seleccionar_muestra(unidades, por_area=40):
    por_tema = defaultdict(list)
    for unidad in unidades:
        if unidad.get("articulo") is None or len(unidad["texto"]) < 160:
            continue
        for area in unidad["areas"]:
            por_tema[area].append(unidad)
    elegidas = {}
    for area, candidatas in sorted(por_tema.items()):
        candidatas.sort(key=lambda u: hashlib.sha256(u["unidad_id"].encode()).hexdigest())
        for unidad in candidatas[:por_area]:
            elegidas[unidad["unidad_id"]] = unidad
    for unidad in unidades:
        if str(unidad.get("articulo")) in ("24", "42"):
            elegidas[unidad["unidad_id"]] = unidad
    return sorted(elegidas)


def crear_sondas(unidades, por_area=3):
    por_tema = defaultdict(list)
    for unidad in unidades:
        if unidad.get("articulo") is not None and len(unidad["texto"]) >= 160:
            for area in unidad["areas"]:
                por_tema[area].append(unidad)
    seleccionadas = {}
    for area, candidatas in sorted(por_tema.items()):
        candidatas.sort(key=lambda u: hashlib.sha256(u["unidad_id"].encode()).hexdigest())
        documentos = set()
        for unidad in candidatas:
            if unidad["doc_id"] in documentos:
                continue
            seleccionadas[unidad["unidad_id"]] = unidad
            documentos.add(unidad["doc_id"])
            if len(documentos) == por_area:
                break
    por_doc = defaultdict(dict)
    for unidad in unidades:
        if str(unidad.get("articulo")) in ("24", "42"):
            por_doc[unidad["doc_id"]][str(unidad["articulo"])] = unidad
    for doc, articulos in sorted(por_doc.items()):
        if set(articulos) == {"24", "42"}:
            for unidad in articulos.values():
                seleccionadas[unidad["unidad_id"]] = unidad
            break
    sondas = []
    for unidad in sorted(seleccionadas.values(), key=lambda u: u["unidad_id"]):
        inicio = min(len(unidad["texto"]) // 3, 200)
        literal = unidad["texto"][inicio:inicio + 120]
        for tipo, consulta in [("referencia", f"artículo {unidad['articulo']} de {unidad['titulo']}"),
                               ("literal", literal)]:
            sondas.append({"id": unidad["unidad_id"] + "_" + tipo, "tipo_sonda": tipo,
                           "consulta": consulta, "unidad_id": unidad["unidad_id"],
                           "doc_id": unidad["doc_id"], "articulo": unidad["articulo"],
                           "inicio": unidad["inicio"], "fin": unidad["fin"], "areas": unidad["areas"],
                           "literal_inicio": unidad["inicio"] + inicio if tipo == "literal" else None,
                           "literal_fin": unidad["inicio"] + inicio + len(literal) if tipo == "literal" else None})
    return sondas


def evaluar_sondas(recuperador, sondas):
    resultados = []
    for sonda in sondas:
        inicio = time.perf_counter()
        pasajes = recuperador.buscar(sonda["consulta"], k=10)
        segundos = time.perf_counter() - inicio
        puestos = [p["puesto"] for p in pasajes if p["unidad_id"] == sonda["unidad_id"] and
                   p["doc_id"] == sonda["doc_id"] and str(p["articulo"]) == str(sonda["articulo"]) and
                   p["inicio"] == sonda["inicio"] and p["fin"] == sonda["fin"]]
        esperado = recuperador.unidades[sonda["unidad_id"]]
        for pasaje in pasajes:
            fuente = recuperador.unidades[pasaje["unidad_id"]]
            campos = ("doc_id", "articulo", "inicio", "fin", "texto")
            if any(pasaje[c] != fuente[c] for c in campos) or len(pasaje["texto"]) != pasaje["fin"] - pasaje["inicio"]:
                raise ValueError("La recuperación alteró el texto fuente")
            if not pasaje["inicio"] <= pasaje["ventana_inicio"] < pasaje["ventana_fin"] <= pasaje["fin"]:
                raise ValueError("La ventana está fuera del artículo")
        if sonda["tipo_sonda"] == "literal":
            a, b = sonda["literal_inicio"] - esperado["inicio"], sonda["literal_fin"] - esperado["inicio"]
            if esperado["texto"][a:b] != sonda["consulta"]:
                raise ValueError("La sonda literal no coincide con el artículo")
        puesto = min(puestos) if puestos else None
        resultados.append({**sonda, "puesto": puesto, "recall_1": int(puesto == 1),
                           "recall_5": int(puesto is not None and puesto <= 5),
                           "recall_10": int(puesto is not None), "mrr_10": 1 / puesto if puesto else 0,
                           "segundos": segundos, "offsets_correctos": True,
                           "top_10": [p["unidad_id"] for p in pasajes]})
    return resultados


def resumir(resultados):
    return {"sondas": len(resultados), **{k: float(np.mean([r[k] for r in resultados]))
            for k in ("recall_1", "recall_5", "recall_10", "mrr_10")},
            "consulta_mediana_s": float(np.median([r["segundos"] for r in resultados])),
            "consulta_p95_s": float(np.percentile([r["segundos"] for r in resultados], 95))}
