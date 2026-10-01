class Reranker:
    def __init__(self, config):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.reranker_model,
                                                      revision=config.reranker_revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.reranker_model, revision=config.reranker_revision,
            dtype=torch.float16 if config.device == "cuda" else torch.float32,
            trust_remote_code=False).to(config.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)

    def rank(self, question, passages):
        import torch
        if not passages:
            return []
        scores = []
        for start in range(0, len(passages), self.config.reranker_batch_size):
            batch = passages[start:start + self.config.reranker_batch_size]
            inputs = self.tokenizer([[question, p["encabezado"] + "\n" + p["texto"]] for p in batch],
                                    padding=True, truncation="only_second", max_length=1024,
                                    return_tensors="pt").to(self.config.device)
            with torch.inference_mode():
                scores.extend(torch.sigmoid(self.model(**inputs).logits.float().reshape(-1)).cpu().tolist())
        # Devolvemos TODO el ranking para que la capa superior pueda diversificar.
        ranked = sorted(({**p, "reranker_score": float(s)} for p, s in zip(passages, scores)),
                        key=lambda p: (-p["reranker_score"], p["chunk_id"]))
        return ranked

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
