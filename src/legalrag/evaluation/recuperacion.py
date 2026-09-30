"""Medición de la recuperación sin decoder, comparable con E06 (`evaluate_retrieval`).

El `legal_basis` de la muestra solo se lee después de recuperar, para puntuar; nunca entra a la
consulta ni al índice. La muestra no publica doc_id relevantes, así que no hay recall documental:
se mide si alguna norma citada en el fundamento aparece literalmente en la evidencia.

    respaldo_literal_top10  alguna norma del fundamento en los 10 primeros fragmentos (métrica de E06)
    MRR_cita_top10          recíproco del puesto del primer fragmento que la contiene
    respaldo_en_pasajes     lo mismo sobre los pasajes finales tal como se entregan (con encabezado de norma)
"""
import time


def evaluar(recuperador, preguntas, citaciones):
    from legalrag.citations.normas import EvidenciaCorpus
    from legalrag.evaluation.entrega import preparar_entrada

    evaluables = aciertos = aciertos_pasajes = 0
    reciprocos, segundos, detalle = [], [], []
    for pregunta in preguntas:
        entrada = preparar_entrada(pregunta)
        inicio = time.perf_counter()
        ranking = recuperador.ranking(entrada)
        pasajes = recuperador.pasajes(ranking)
        segundos.append(time.perf_counter() - inicio)
        referencias = citaciones.bodies(citaciones.extract(pregunta.get("legal_basis") or ""))
        if not referencias:
            continue
        evaluables += 1
        filas = recuperador.fragmentos.filas([f for f, _ in ranking[:10]])
        encontrados = [bool(referencias & citaciones.bodies(citaciones.extract(f["texto"]))) for f in filas]
        en_pasajes = any(referencias & citaciones.bodies(citaciones.extract(EvidenciaCorpus.texto_entregado(p)))
                         for p in pasajes)
        if any(encontrados):
            aciertos += 1
            reciprocos.append(1 / (encontrados.index(True) + 1))
        aciertos_pasajes += en_pasajes
        detalle.append({"id": pregunta["id"], "respaldo_top10": any(encontrados),
                        "puesto": encontrados.index(True) + 1 if any(encontrados) else None,
                        "respaldo_en_pasajes": en_pasajes, "segundos": round(segundos[-1], 3)})
    return {"n": len(preguntas), "items_con_cita_extraible": evaluables,
            "respaldo_literal_top10": aciertos / evaluables if evaluables else None,
            "MRR_cita_top10": sum(reciprocos) / evaluables if evaluables else None,
            "respaldo_en_pasajes": aciertos_pasajes / evaluables if evaluables else None,
            "segundos_promedio": sum(segundos) / len(segundos) if segundos else None,
            "referencia_E06_R03": {"respaldo_literal_top10": 0.8536585365853658, "MRR_cita_top10": 0.5463414634146341},
            "detalle": detalle}
