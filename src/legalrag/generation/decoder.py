"""Decoder de la entrega (opción A): Qwen2.5-7B-Instruct con transformers.

Generación greedy (`do_sample=False`), equivalente a temperatura 0: la misma pregunta
con la misma evidencia produce el mismo texto, como exige la verificación en vivo.
El contexto y la longitud de salida vienen fijos de la configuración, no de la VRAM,
para que el resultado no cambie entre equipos.
"""
from legalrag.generation.cliente import mensajes


class DecoderTransformers:
    def __init__(self, config, limite_parametros=8_000_000_000):
        if config["decoder"]["parametros"] > limite_parametros:
            raise ValueError("El decoder supera el límite de 8.000 millones de parámetros")
        self.config = config
        self.tokenizer = self.modelo = None

    def abrir(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(0)
        ficha = self.config["decoder"]
        self.tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], revision=ficha["revision"])
        self.modelo = AutoModelForCausalLM.from_pretrained(
            ficha["repo_id"], revision=ficha["revision"], torch_dtype=getattr(torch, self.config["dtype"]),
            device_map="cuda").eval()
        reales = sum(p.numel() for p in self.modelo.parameters())
        if reales != ficha["parametros"]:
            raise ValueError(f"El decoder cargado tiene {reales} parámetros, no {ficha['parametros']}")

    def cerrar(self):
        self.modelo = None

    def _tokens(self, entrada, pasajes):
        return self.tokenizer.apply_chat_template(mensajes(entrada, pasajes), tokenize=True,
                                                  add_generation_prompt=True)

    def seleccionar(self, entrada, pasajes):
        """Pasajes en orden de ranking mientras quepan junto a la salida en el contexto."""
        presupuesto = self.config["contexto"] - self.config["max_nuevos_tokens"] - 64
        elegidos = []
        for pasaje in pasajes:
            if len(self._tokens(entrada, elegidos + [pasaje])) <= presupuesto:
                elegidos.append(pasaje)
        return elegidos

    def generar(self, entrada, pasajes):
        import torch

        tokens = torch.tensor([self._tokens(entrada, pasajes)], device="cuda")
        with torch.inference_mode():
            salida = self.modelo.generate(tokens, max_new_tokens=self.config["max_nuevos_tokens"], do_sample=False,
                                          pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(salida[0, tokens.shape[-1]:], skip_special_tokens=True)
