import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def hash_json(datos):
    return hashlib.sha256(json.dumps(datos, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def leer_jsonl(path):
    with Path(path).open(encoding="utf-8") as archivo:
        return [json.loads(linea) for linea in archivo if linea.strip()]


def escribir_jsonl(path, filas):
    with Path(path).open("w", encoding="utf-8", newline="\n") as archivo:
        for fila in filas:
            archivo.write(json.dumps(fila, ensure_ascii=False, sort_keys=True) + "\n")


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


def metadatos_busqueda(unidad):
    articulo = "" if unidad.get("articulo") is None else "Artículo " + str(unidad["articulo"])
    return f"{unidad['titulo']}\n{articulo}\n"


class Encoder:
    def __init__(self, ficha, device="cpu", precision="float32"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub import snapshot_download

        torch.manual_seed(0)
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.ficha = ficha
        self.device = device
        self.jina = "jina-embeddings-v3" in ficha["repo_id"]
        self.e5 = "multilingual-e5" in ficha["repo_id"]
        self.prefijos = {"query": "query: " if self.e5 else "", "passage": "passage: " if self.e5 else ""}
        ruta_modelo = ficha["repo_id"]
        if self.jina:
            ruta_modelo = snapshot_download(ficha["repo_id"], revision=ficha["revision"],
                                            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
        self.tokenizer = AutoTokenizer.from_pretrained(ruta_modelo, revision=ficha["revision"])
        opciones = {"revision": ficha["revision"], "trust_remote_code": self.jina,
                    "torch_dtype": getattr(torch, precision)}
        if self.jina:
            opciones.update(code_revision=ficha["code_revision"], use_flash_attn=False)
        else:
            opciones["attn_implementation"] = "eager"
        self.modelo = AutoModel.from_pretrained(ruta_modelo, **opciones).to(device).eval()
        self.parametros = sum(p.numel() for _, p in torch.nn.Module.named_parameters(self.modelo))
        self.precision_real = sorted({str(p.dtype) for _, p in torch.nn.Module.named_parameters(self.modelo)})
        if self.jina:
            self.prefijos = {t: self.modelo._task_instructions["retrieval." + t] for t in ("query", "passage")}
        self.limite = 512 if self.e5 else 8192

    def encode(self, textos, tarea="passage", batch_size=4):
        import torch

        vectores = []
        for inicio in range(0, len(textos), batch_size):
            lote = [self.prefijos[tarea] + t for t in textos[inicio:inicio + batch_size]]
            tokens = self.tokenizer(lote, padding=True, truncation=False, return_tensors="pt")
            if tokens["input_ids"].shape[1] > self.limite:
                raise ValueError("La entrada supera el contexto del encoder")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            with torch.inference_mode():
                if self.jina:
                    tipo = self.modelo._adaptation_map["retrieval." + tarea]
                    salida = self.modelo(**tokens, adapter_mask=torch.full(
                        (len(lote),), tipo, dtype=torch.int32, device=self.device))
                else:
                    salida = self.modelo(**tokens)
                hidden = salida.last_hidden_state
                if self.e5 or self.jina:
                    mascara = tokens["attention_mask"].unsqueeze(-1)
                    pooling = (hidden.float() * mascara).sum(dim=1) / mascara.sum(dim=1)
                else:
                    pooling = hidden[:, 0]
                pooling = torch.nn.functional.normalize(pooling.float(), p=2, dim=1)
                vectores.append(pooling.cpu().numpy())
        return np.vstack(vectores).astype("float32")


def crear_ventanas(unidades, encoder, tamano=320, solapamiento=48):
    if not 0 <= solapamiento < tamano:
        raise ValueError("Solapamiento fuera de rango")
    ventanas = []
    for unidad in unidades:
        texto = unidad["texto"]
        cabecera = metadatos_busqueda(unidad)
        margen = len(encoder.tokenizer(encoder.prefijos["passage"] + cabecera)["input_ids"]) + 4
        limite = min(tamano, encoder.limite - margen)
        if limite <= solapamiento:
            raise ValueError("Metadatos demasiado largos para el encoder")
        offsets = encoder.tokenizer(texto, add_special_tokens=False, truncation=False,
                                    return_offsets_mapping=True)["offset_mapping"]
        offsets = [(a, b) for a, b in offsets if b > a]
        if not offsets:
            continue
        posicion = 0
        while posicion < len(offsets):
            fin_token = min(posicion + limite, len(offsets))
            inicio = 0 if posicion == 0 else offsets[posicion][0]
            fin = len(texto) if fin_token == len(offsets) else offsets[fin_token][0]
            if fin <= inicio:
                raise ValueError("Ventana vacía")
            pasaje = texto[inicio:fin]
            while len(encoder.tokenizer(encoder.prefijos["passage"] + cabecera + pasaje)["input_ids"]) > encoder.limite:
                fin_token -= 1
                fin = offsets[fin_token][0]
                pasaje = texto[inicio:fin]
            fila = dict(unidad)
            fila.update(inicio=unidad["inicio"] + inicio, fin=unidad["inicio"] + fin,
                        unidad_inicio=unidad["inicio"], unidad_fin=unidad["fin"], texto=pasaje,
                        texto_busqueda=cabecera + pasaje, recuperar_unidad_completa=True)
            fila["fragmento_id"] = hash_json([unidad["unidad_id"], fila["inicio"], fila["fin"]])[:24]
            ventanas.append(fila)
            if fin_token == len(offsets):
                break
            posicion = max(posicion + 1, fin_token - solapamiento)
    return ventanas


class BM25:
    def __init__(self, textos):
        self.terminos = [Counter(re.findall(r"\w+", t.casefold())) for t in textos]
        self.longitudes = np.array([sum(t.values()) for t in self.terminos])
        self.promedio = max(1, self.longitudes.mean())
        df = Counter(t for doc in self.terminos for t in doc)
        self.idf = {t: math.log(1 + (len(textos) - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.postings = defaultdict(list)
        for i, doc in enumerate(self.terminos):
            for termino, cantidad in doc.items():
                self.postings[termino].append((i, cantidad))

    def buscar(self, consulta):
        scores = np.zeros(len(self.terminos))
        for termino in sorted(set(re.findall(r"\w+", consulta.casefold()))):
            for i, frecuencia in self.postings.get(termino, []):
                norma = 1.2 * (0.25 + 0.75 * self.longitudes[i] / self.promedio)
                scores[i] += self.idf[termino] * frecuencia * 2.2 / (frecuencia + norma)
        return scores


class Recuperador:
    def __init__(self, indice, ventanas, unidades, encoder, hibrido=False):
        self.indice = indice
        self.ventanas = ventanas
        self.unidades = {u["unidad_id"]: u for u in unidades}
        self.encoder = encoder
        self.bm25 = BM25([v["texto_busqueda"] for v in ventanas]) if hibrido else None

    def buscar(self, consulta, k=10):
        vector = self.encoder.encode([consulta], "query")
        scores, posiciones = self.indice.search(vector, len(self.ventanas))
        orden = sorted(zip(posiciones[0].tolist(), scores[0].tolist()), key=lambda x: (-x[1], x[0]))
        if self.bm25:
            lexical = self.bm25.buscar(consulta)
            lexical_orden = sorted(range(len(lexical)), key=lambda i: (-lexical[i], i))
            fusion = defaultdict(float)
            representante = {}
            for ranking in ([i for i, _ in orden], [i for i in lexical_orden if lexical[i] > 0]):
                padres = set()
                for i in ranking:
                    padre = self.ventanas[i]["unidad_id"]
                    if padre in padres:
                        continue
                    padres.add(padre)
                    representante.setdefault(padre, i)
                    fusion[padre] += 1 / (60 + len(padres))
                    if len(padres) == max(100, k):
                        break
            orden = sorted(((representante[padre], valor) for padre, valor in fusion.items()),
                           key=lambda x: (-x[1], x[0]))
        resultado = []
        vistas = set()
        for posicion, score in orden:
            ventana = self.ventanas[posicion]
            if ventana["unidad_id"] in vistas:
                continue
            vistas.add(ventana["unidad_id"])
            unidad = dict(self.unidades[ventana["unidad_id"]])
            unidad.update(puesto=len(resultado) + 1, score=float(score), fragmento_id=ventana["fragmento_id"],
                          ventana_inicio=ventana["inicio"], ventana_fin=ventana["fin"])
            resultado.append(unidad)
            if len(resultado) == k:
                break
        return resultado


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
    unidades = leer_jsonl(ruta / "unidades.jsonl")
    if hash_json(unidades) != manifiesto["sha256_corpus"]:
        raise ValueError("Cambió el corpus del índice")
    encoder = Encoder(config["encoder"], device or config.get("device", "cpu"), config.get("precision", "float32"))
    return Recuperador(faiss.read_index(str(ruta / "indice.faiss")), leer_jsonl(ruta / "ventanas.jsonl"),
                       unidades, encoder, config.get("hibrido", False))
