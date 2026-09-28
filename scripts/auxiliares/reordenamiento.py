import gc
import os
import re
from copy import deepcopy

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class Reordenador:
    def __init__(self, ficha, device="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.ficha = deepcopy(ficha)
        self.device = str(torch.device(device))
        self.max_length = int(ficha.get("max_length", 1024))
        self.batch_size = int(ficha.get("batch_size", 2))
        self.normalizar = bool(ficha.get("normalizar", False))
        self.ultimo_resumen = {}
        self.modelo = None
        self.tokenizer = None
        if ficha.get("repo_id") != "BAAI/bge-reranker-v2-m3":
            raise ValueError("El reranker admitido es BAAI/bge-reranker-v2-m3")
        if not re.fullmatch(r"[0-9a-f]{40}", ficha.get("revision", "")):
            raise ValueError("El reranker necesita una revisión exacta")
        if ficha.get("licencia") != "apache-2.0":
            raise ValueError("La licencia del reranker debe ser Apache 2.0")
        if not 8 <= self.max_length <= 8192 or self.batch_size < 1:
            raise ValueError("Longitud o tamaño de lote fuera de rango")
        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("Se solicitó CUDA, pero no hay GPU disponible para el reranker")
            indice = torch.device(self.device).index
            if indice is not None and indice >= torch.cuda.device_count():
                raise RuntimeError("La GPU solicitada no existe")
        elif self.device != "cpu":
            raise ValueError("Usar cpu o cuda para el reranker")

        precision = ficha.get("precision", "float16")
        if precision not in ("float16", "float32", "bfloat16"):
            raise ValueError("Precisión no admitida")
        if self.device == "cpu" and precision != "float32":
            raise ValueError("En CPU, configurar el reranker con precision='float32'")
        torch.manual_seed(int(ficha.get("semilla", 0)))
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        opciones = {"revision": ficha["revision"], "trust_remote_code": False}
        self.tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], **opciones)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            ficha["repo_id"], **opciones, use_safetensors=True,
            torch_dtype=getattr(torch, precision), attn_implementation="eager",
        ).to(self.device).eval()
        if self.modelo.config.num_labels != 1:
            self.cerrar()
            raise ValueError("Se esperaba un único puntaje por pareja")

    def ordenar(self, consulta, unidades, k=5):
        import torch

        if self.modelo is None:
            raise RuntimeError("El reranker está cerrado")
        if not isinstance(consulta, str) or not consulta.strip():
            raise ValueError("La consulta está vacía")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k debe ser un entero positivo")
        candidatos = list(unidades)
        self.ultimo_resumen = {"candidatos": len(candidatos), "truncados": 0,
                               "max_length": self.max_length}
        if not candidatos:
            return []
        tokens_consulta = len(self.tokenizer(
            consulta, add_special_tokens=False, truncation=False, verbose=False,
        )["input_ids"])
        especiales = self.tokenizer.num_special_tokens_to_add(pair=True)
        if tokens_consulta + especiales >= self.max_length:
            raise ValueError("La consulta no deja espacio para el artículo en el reranker")

        ordenadas = []
        for inicio in range(0, len(candidatos), self.batch_size):
            lote = candidatos[inicio:inicio + self.batch_size]
            pasajes = []
            for unidad in lote:
                if not isinstance(unidad.get("texto"), str) or not unidad["texto"].strip():
                    raise ValueError("Un candidato no tiene texto")
                cabecera = unidad.get("titulo", "")
                if unidad.get("articulo") is not None:
                    cabecera += "\nArtículo " + str(unidad["articulo"])
                pasajes.append(cabecera + "\n" + unidad["texto"])
            longitudes = [len(ids) + tokens_consulta + especiales for ids in self.tokenizer(
                pasajes, add_special_tokens=False, truncation=False, verbose=False,
            )["input_ids"]]
            entradas = self.tokenizer(
                [consulta] * len(lote), text_pair=pasajes, padding=True,
                truncation="only_second", max_length=self.max_length, return_tensors="pt",
            )
            usados = entradas["attention_mask"].sum(dim=1).tolist()
            entradas = {clave: valor.to(self.device) for clave, valor in entradas.items()}
            with torch.inference_mode():
                logits = self.modelo(**entradas, return_dict=True).logits.view(-1).float()
                if not torch.isfinite(logits).all():
                    raise RuntimeError("El reranker produjo puntajes no finitos")
                scores = logits.cpu().tolist()
                normalizados = torch.sigmoid(logits).cpu().tolist() if self.normalizar else None
            for posicion, (unidad, score, largo, usado) in enumerate(zip(lote, scores, longitudes, usados)):
                fila = deepcopy(unidad)
                fila["puesto_recuperacion"] = unidad.get("puesto", inicio + posicion + 1)
                fila["rerank_score"] = float(score)
                if normalizados is not None:
                    fila["rerank_score_normalizado"] = float(normalizados[posicion])
                fila["reranker_tokens_originales"] = largo
                fila["reranker_tokens_usados"] = usado
                fila["reranker_truncado"] = largo > usado
                self.ultimo_resumen["truncados"] += int(fila["reranker_truncado"])
                ordenadas.append(fila)

        # Los empates conservan el orden de recuperación. El texto fuente no se recorta.
        ordenadas.sort(key=lambda unidad: -unidad["rerank_score"])
        for puesto, unidad in enumerate(ordenadas[:k], start=1):
            unidad["puesto"] = puesto
        return ordenadas[:k]

    def cerrar(self):
        import torch

        self.modelo = None
        self.tokenizer = None
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()
