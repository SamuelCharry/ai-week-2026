import os
from legalrag.preprocessing.clean import clean_text


def reproducible(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class Encoder:
    def __init__(self, config):
        reproducible(config.seed)
        import torch
        from sentence_transformers import SentenceTransformer
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA no disponible. Instala PyTorch CUDA y revisa el controlador NVIDIA.")
        self.config = config
        self.model = SentenceTransformer(config.embedding_model, revision=config.embedding_revision,
                                         device=config.device, trust_remote_code=False)
        self.model.max_seq_length = config.embedding_max_tokens
        self.tokenizer = self.model.tokenizer
        self.revision = getattr(self.model[0].auto_model.config, "_commit_hash", None)

    def encode(self, texts):
        import numpy as np
        result = self.model.encode(texts, batch_size=self.config.batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True,
                                  show_progress_bar=False)
        return np.asarray(result, dtype="float32")

    def chunks(self, chunks):
        texts = [c["encabezado"] + "\n" + clean_text(c["texto"]) for c in chunks]
        lengths = [len(self.tokenizer.encode(t, add_special_tokens=True)) for t in texts]
        if max(lengths, default=0) > self.config.embedding_max_tokens:
            raise ValueError("Una ventana excede el encoder. Reduce window_tokens antes de indexar.")
        return self.encode(texts)

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
