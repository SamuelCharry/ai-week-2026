"""Auditoría de cobertura del corpus frente al fundamento legal de las preguntas.

Separa cuatro causas de fallo por cada cita de referencia:
    norma_ausente      la norma no existe en el corpus
    articulo_ausente   la norma existe, pero no hay una unidad con ese artículo
    no_recuperada      la norma está, pero no aparece entre los pasajes recuperados
    recuperada         la norma aparece en los pasajes que recibió el decoder

Solo usa CPU. No modifica el corpus ni el índice.
"""
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from scripts.auxiliares.oficial import cargar_citaciones, identidad


def _norm_articulo(valor):
    if valor is None:
        return None
    texto = unicodedata.normalize("NFD", str(valor).casefold())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[º°ªo]\b", "", texto)
    coincidencia = re.match(r"\s*(\d+(?:\.\d+)*[a-z]?)", texto)
    return coincidencia.group(1) if coincidencia else texto.strip(" .-") or None


def _leer_jsonl(ruta, campos=None):
    with Path(ruta).open(encoding="utf-8") as archivo:
        for linea in archivo:
            if linea.strip():
                fila = json.loads(linea)
                yield {k: fila.get(k) for k in campos} if campos else fila


def indice_corpus(raiz):
    """Mapa cuerpo normativo -> doc_ids y doc_id -> artículos presentes."""
    raiz = Path(raiz)
    citas = cargar_citaciones(raiz)
    corpus = raiz / "data/processed/corpus"
    documentos = {d["doc_id"]: d for d in _leer_jsonl(corpus / "documentos.jsonl")}
    from scripts.v04.evidencia import EvidenciaV04
    ev = EvidenciaV04(raiz, corpus)
    por_cuerpo = defaultdict(set)
    for doc_id, doc in documentos.items():
        # Identidad ampliada: incluye "Código Civil", "Código Sustantivo del Trabajo", etc.
        for cuerpo in ev.identidad(doc_id):
            por_cuerpo[cuerpo].add(doc_id)
    articulos = defaultdict(set)
    for unidad in _leer_jsonl(corpus / "unidades.jsonl", ["doc_id", "articulo", "apta_para_busqueda"]):
        if unidad["articulo"] is not None:
            articulos[unidad["doc_id"]].add(_norm_articulo(unidad["articulo"]))
    return citas, documentos, por_cuerpo, articulos


def cuerpos_recuperados(pasajes, documentos, citas, limite=10):
    """Cuerpos normativos que el evaluador oficial vería como respaldo."""
    encontrados = set()
    for pasaje in (pasajes or [])[:limite]:
        encontrados |= citas.bodies(citas.extract(str(pasaje.get("texto") or "")))
    return encontrados


def auditar(raiz, preguntas, recuperaciones=None, indice=None):
    """Una fila por (pregunta, cita de referencia).

    recuperaciones: {variante: {id: pasajes}} opcional. Sin ella solo se mide
    presencia en el corpus.
    """
    citas, documentos, por_cuerpo, articulos = indice or indice_corpus(raiz)
    recuperaciones = recuperaciones or {}
    filas = []
    for pregunta in preguntas:
        referencia = citas.extract(pregunta.get("legal_basis") or "")
        if not referencia:
            filas.append({"id": pregunta["id"], "area": pregunta["area"], "formato": pregunta["formato"],
                          "cita": None, "estado_corpus": "sin_cita_extraible"})
            continue
        for cita in sorted(referencia, key=lambda c: tuple(str(x) for x in c)):
            cuerpo, articulo = cita[:3], _norm_articulo(cita[3])
            docs = por_cuerpo.get(cuerpo, set())
            if not docs:
                estado = "norma_ausente"
            elif articulo is not None and not any(articulo in articulos[d] for d in docs):
                estado = "articulo_ausente"
            else:
                estado = "presente"
            fila = {"id": pregunta["id"], "area": pregunta["area"], "formato": pregunta["formato"],
                    "cita": " ".join(str(x) for x in cita if x is not None), "cuerpo": "|".join(str(x) for x in cuerpo),
                    "articulo": articulo, "doc_ids": sorted(docs), "estado_corpus": estado}
            for variante, por_id in recuperaciones.items():
                obtenidos = cuerpos_recuperados(por_id.get(pregunta["id"]), documentos, citas)
                if estado == "norma_ausente":
                    fila[variante] = "norma_ausente"
                else:
                    fila[variante] = "recuperada" if cuerpo in obtenidos else "no_recuperada"
            filas.append(fila)
    return filas


def auditar_banco(raiz, indice=None):
    """Cobertura sobre seed_targets.json, ponderada por ítems del banco de 992."""
    citas, documentos, por_cuerpo, _ = indice or indice_corpus(raiz)
    objetivos = json.loads((Path(raiz) / "data/oficial/data/seed_targets.json").read_text(encoding="utf-8"))["documentos"]
    filas = []
    for objetivo in objetivos:
        cuerpo = tuple(objetivo["canonico"])
        filas.append({"norma": objetivo["norma"], "items_del_banco": objetivo["items_del_banco"],
                      "en_corpus": bool(por_cuerpo.get(cuerpo)), "doc_ids": sorted(por_cuerpo.get(cuerpo, [])),
                      "areas": objetivo["areas"], "donde_buscar": objetivo["donde_buscar"]})
    return filas
