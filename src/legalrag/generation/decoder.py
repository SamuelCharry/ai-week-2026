"""Decoder de la entrega: un modelo instruct abierto (≤ 8.000 M) con transformers.

Generación greedy (`do_sample=False`), equivalente a temperatura 0: la misma pregunta con la
misma evidencia produce el mismo texto, como exige la verificación en vivo. El prompt es el de
la versión 04/05 (`generation.politica.mensajes`): pasajes numerados con el nombre de su norma,
letra obligatoria en cerradas y longitudes por campo. La respuesta se abre con "{" para que el
modelo escriba solo el objeto JSON, y una penalización de repetición corta los bucles que
dejaban el JSON truncado. Contexto y salida vienen fijos de la configuración, no de la VRAM.

En cerradas, `probabilidades_letras` elige la opción sin generar texto: con la respuesta abierta en
`{"respuesta_correcta": "`, compara la probabilidad del siguiente token para cada letra. Después
`generar` escribe la justificación con esa letra ya fijada en el prefijo. En el modo "razonada" el
orden se invierte: primero se genera la justificación y luego se comparan las letras con ella escrita.
"""
from legalrag.generation import politica


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
            ficha["repo_id"], revision=ficha["revision"], dtype=getattr(torch, self.config["dtype"]),
            device_map=self.config.get("dispositivo", "cuda")).eval()
        self.parametros_cargados = sum(p.numel() for p in self.modelo.parameters())
        if self.parametros_cargados > 8_000_000_000:
            raise ValueError(f"El decoder cargado tiene {self.parametros_cargados} parámetros (> 8.000 M)")

    def cerrar(self):
        self.modelo = None

    def mensajes(self, entrada, pasajes, evidencia):
        return politica.mensajes(entrada, pasajes, evidencia, self.config.get("max_caracteres_prompt", 1800))

    def _tokens(self, mensajes, prefijo="{"):
        opciones = {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
        if "Qwen3" in self.config["decoder"]["repo_id"]:
            opciones["enable_thinking"] = False
        # return_dict=False: en transformers 5 el valor por defecto devuelve un diccionario.
        try:
            plantilla = self.tokenizer.apply_chat_template(mensajes, **opciones)
        except Exception:
            # Plantillas sin rol de sistema (p. ej. Mistral): las instrucciones van al comienzo del usuario.
            unidos = [{"role": "user", "content": mensajes[0]["content"] + "\n\n" + mensajes[1]["content"]}]
            plantilla = self.tokenizer.apply_chat_template(unidos, **opciones)
        return list(plantilla) + self.tokenizer.encode(prefijo, add_special_tokens=False)

    def _id_letra(self, prefijo, letra):
        """Token de la letra tal como sigue al prefijo (en SentencePiece "A" suelta lleva un espacio: "▁A")."""
        base = self.tokenizer.encode(prefijo, add_special_tokens=False)
        junto = self.tokenizer.encode(prefijo + letra, add_special_tokens=False)
        if junto[:len(base)] == base and len(junto) == len(base) + 1:
            return junto[-1]
        return self.tokenizer.encode(letra, add_special_tokens=False)[-1]

    def seleccionar(self, entrada, pasajes, evidencia):
        """Pasajes en orden de ranking mientras quepan junto a la salida en el contexto."""
        presupuesto = self.config["contexto"] - self.config["max_nuevos_tokens"] - 64
        elegidos = []
        for pasaje in pasajes:
            if len(self._tokens(self.mensajes(entrada, elegidos + [pasaje], evidencia))) <= presupuesto:
                elegidos.append(pasaje)
        return elegidos

    def probabilidades_letras(self, entrada, pasajes, evidencia, prefijo='{"respuesta_correcta": "'):
        """{letra: probabilidad} del siguiente token tras `prefijo`, normalizada entre las opciones.

        El prefijo por defecto pide la letra de entrada; con la justificación ya escrita en el prefijo
        (modo "razonada") la letra se elige después de razonar."""
        import torch

        letras = list((entrada.get("opciones") or {}).keys())
        tokens = torch.tensor([self._tokens(self.mensajes(entrada, pasajes, evidencia), prefijo)],
                              device=self.config.get("dispositivo", "cuda"))
        ids = [self._id_letra(prefijo, letra) for letra in letras]
        with torch.inference_mode():
            logits = self.modelo(tokens).logits[0, -1].float()
        probabilidades = torch.softmax(logits[ids], dim=0).tolist()
        return dict(zip(letras, probabilidades))

    def generar(self, entrada, pasajes, evidencia, prefijo="{"):
        import torch

        tokens = torch.tensor([self._tokens(self.mensajes(entrada, pasajes, evidencia), prefijo)],
                              device=self.config.get("dispositivo", "cuda"))
        with torch.inference_mode():
            salida = self.modelo.generate(tokens, max_new_tokens=self.config["max_nuevos_tokens"], do_sample=False,
                                          repetition_penalty=self.config.get("repetition_penalty", 1.0),
                                          pad_token_id=self.tokenizer.eos_token_id)
        return prefijo + self.tokenizer.decode(salida[0, tokens.shape[-1]:], skip_special_tokens=True)
