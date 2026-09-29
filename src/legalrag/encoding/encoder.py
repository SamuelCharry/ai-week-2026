"""Encoder abierto: vectores normalizados y deterministas para consultas y pasajes."""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np  # noqa: E402


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
