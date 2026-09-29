"""Corpus admitido, construcción del índice FAISS y carga verificada por hashes."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from legalrag.chunking.ventanas import VERSION_TEXTO_BUSQUEDA, crear_ventanas
from legalrag.comun import escribir_jsonl, hash_json, leer_jsonl, sha256
from legalrag.encoding.encoder import Encoder
from legalrag.retrieval.recuperador import Recuperador


def cargar_corpus(ruta, seleccion=None):
    ruta = Path(ruta)
    documentos = {d["doc_id"]: d for d in leer_jsonl(ruta / "documentos.jsonl")}
    unidades = leer_jsonl(ruta / "unidades.jsonl")
    textos = {}
    validas = []
    rechazadas = Counter()
    requeridos = ["doc_id", "tipo", "numero", "anio", "organo_emisor", "vigencia", "url"]
    for unidad in unidades:
        if any(c in unidad for c in ("respuesta_esperada", "legal_basis", "pregunta")):
            raise ValueError("La entrada contiene campos de evaluación")
        doc = documentos[unidad["doc_id"]]
        for campo in requeridos + ["areas"]:
            if unidad.get(campo) != doc.get(campo):
                raise ValueError("Metadato distinto del documento: " + campo + " en " + unidad["unidad_id"])
        if doc["doc_id"] not in textos:
            path = (ruta / doc["texto_archivo"]).resolve()
            if not path.is_relative_to(ruta.resolve()):
                raise ValueError("Texto fuera del corpus")
            textos[doc["doc_id"]] = path.read_text(encoding="utf-8")
            if hashlib.sha256(textos[doc["doc_id"]].encode()).hexdigest() != doc["sha256_texto"]:
                raise ValueError("Cambió el texto fuente: " + doc["doc_id"])
        texto = textos[doc["doc_id"]]
        if not 0 <= unidad["inicio"] < unidad["fin"] <= len(texto):
            raise ValueError("Offsets inválidos: " + unidad["unidad_id"])
        if texto[unidad["inicio"]:unidad["fin"]] != unidad["texto"]:
            raise ValueError("El pasaje no coincide con la fuente")
        if any(unidad.get(c) in (None, "") for c in requeridos):
            rechazadas["metadatos_incompletos"] += 1
            continue
        if not unidad.get("apta_para_busqueda", True):
            rechazadas["fuente_o_estructura_no_apta"] += 1
            continue
        if not any(c.isalpha() for c in unidad["texto"]):
            rechazadas["solo_numero_o_signos"] += 1
            continue
        if unidad["estado_segmentacion"] != "segmentado":
            rechazadas["segmentacion_por_revisar"] += 1
            continue
        if unidad["tipo"] not in ("sentencia", "auto") and unidad.get("articulo") is None:
            rechazadas["norma_sin_articulo"] += 1
            continue
        fila = dict(unidad)
        fila["titulo"] = doc["titulo"]
        fila["sha256_documento"] = doc["sha256_texto"]
        validas.append(fila)
    if seleccion:
        ids = set(json.loads(Path(seleccion).read_text(encoding="utf-8")))
        validas = [u for u in validas if u["unidad_id"] in ids]
        if {u["unidad_id"] for u in validas} != ids:
            raise ValueError("La selección no coincide con unidades admitidas")
    validas.sort(key=lambda u: (u["doc_id"], u["inicio"]))
    if not validas:
        raise ValueError("No hay unidades admitidas")
    return validas, dict(rechazadas)


def construir_indice(config, encoder=None):
    import faiss
    import psutil
    import torch

    carpeta = Path(config["salida"])
    if (carpeta / "CONGELADO.json").exists():
        raise ValueError("Índice congelado. Usar otra carpeta")
    carpeta.mkdir(parents=True, exist_ok=True)
    unidades, excluidas = cargar_corpus(config["corpus"], config.get("seleccion"))
    inicio = time.perf_counter()
    encoder = encoder or Encoder(config["encoder"], config.get("device", "cpu"), config.get("precision", "float32"))
    if encoder.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    ventanas = crear_ventanas(unidades, encoder, config["tamano_tokens"], config["solapamiento_tokens"])
    vectores = encoder.encode([v["texto_busqueda"] for v in ventanas], batch_size=config.get("batch_size", 4))
    if not np.isfinite(vectores).all() or not np.allclose(np.linalg.norm(vectores, axis=1), 1, atol=1e-5):
        raise ValueError("Vectores inválidos")
    faiss.omp_set_num_threads(1)
    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores)
    faiss.write_index(indice, str(carpeta / "indice.faiss"))
    np.save(carpeta / "vectores.npy", vectores)
    escribir_jsonl(carpeta / "ventanas.jsonl", ventanas)
    escribir_jsonl(carpeta / "unidades.jsonl", unidades)
    configuracion = {k: v for k, v in config.items() if k not in ("salida", "corpus", "seleccion")}
    hashes = {nombre: sha256(carpeta / nombre) for nombre in ["indice.faiss", "vectores.npy", "ventanas.jsonl", "unidades.jsonl"]}
    manifiesto = {"configuracion": config, "sha256_configuracion": hash_json(configuracion),
                  "version_texto_busqueda": VERSION_TEXTO_BUSQUEDA,
                  "sha256_corpus": hash_json(unidades), "sha256_archivos": hashes,
                  "sha256_indice": hash_json(hashes), "unidades": len(unidades), "ventanas": len(ventanas),
                  "dimensiones": vectores.shape[1], "parametros_encoder": encoder.parametros, "excluidas": excluidas,
                  "precision_real": encoder.precision_real,
                  "segundos": time.perf_counter() - inicio,
                  "ram_rss_mib": psutil.Process().memory_info().rss / 2**20,
                  "vram_pico_mib": torch.cuda.max_memory_allocated() / 2**20 if encoder.device == "cuda" else 0}
    (carpeta / "manifest.json").write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifiesto, Recuperador(indice, ventanas, unidades, encoder, config.get("hibrido", False))


def cargar_indice(ruta, device=None):
    import faiss

    ruta = Path(ruta)
    manifiesto = json.loads((ruta / "manifest.json").read_text(encoding="utf-8"))
    for nombre, esperado in manifiesto["sha256_archivos"].items():
        if sha256(ruta / nombre) != esperado:
            raise ValueError("Cambió un archivo del índice: " + nombre)
    config = manifiesto["configuracion"]
    semantica = {k: v for k, v in config.items() if k not in ("salida", "corpus", "seleccion")}
    if hash_json(semantica) != manifiesto["sha256_configuracion"]:
        raise ValueError("Cambió la configuración del índice")
    if manifiesto.get("version_texto_busqueda") != VERSION_TEXTO_BUSQUEDA:
        raise ValueError("Cambió el texto de búsqueda. Reconstruir el índice con la versión actual")
    unidades = leer_jsonl(ruta / "unidades.jsonl")
    if hash_json(unidades) != manifiesto["sha256_corpus"]:
        raise ValueError("Cambió el corpus del índice")
    encoder = Encoder(config["encoder"], device or config.get("device", "cpu"), config.get("precision", "float32"))
    return Recuperador(faiss.read_index(str(ruta / "indice.faiss")), leer_jsonl(ruta / "ventanas.jsonl"),
                       unidades, encoder, config.get("hibrido", False))
