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
import hashlib
import json
import sqlite3

from legalrag.generation import politica


def limite_de(ficha, limite=8_000_000_000):
    """Límite de parámetros para esta ficha. Solo un modelo que el enunciado sugiere por nombre (§3.1:
    Qwen/Qwen3-8B, meta-llama/Llama-3.1-8B-Instruct) puede pasar de 8.000 M, y debe decirlo en su ficha."""
    return max(limite, ficha["parametros"]) if ficha.get("admitido_por_enunciado") else limite


class DecoderTransformers:
    def __init__(self, config, limite_parametros=8_000_000_000):
        if config["decoder"]["parametros"] > limite_de(config["decoder"], limite_parametros):
            raise ValueError("El decoder supera el límite de 8.000 millones de parámetros")
        self.config = config
        self.tokenizer = self.modelo = self.cache = None

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
        if self.parametros_cargados > limite_de(ficha):
            raise ValueError(f"El decoder cargado tiene {self.parametros_cargados} parámetros (> 8.000 M)")
        if self.config.get("cache_generaciones"):
            self.cache = sqlite3.connect(self.config["cache_generaciones"])
            self.cache.execute("CREATE TABLE IF NOT EXISTS salidas (clave TEXT PRIMARY KEY, valor TEXT NOT NULL)")

    def cerrar(self):
        """Libera la GPU (en el sistema por etapas otro modelo se carga después)."""
        import gc

        self.modelo = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass
        if self.cache is not None:
            self.cache.close()
            self.cache = None

    def _en_cache(self, datos, calcular):
        """Solo para comparar variantes (`cache_generaciones`, lo pone src/comparar.py): con greedy, el mismo
        modelo y los mismos tokens de entrada dan la misma salida, así que una variante no vuelve a generar
        lo que otra ya generó con el mismo prompt. La entrega no lo usa: ahí siempre se genera."""
        if self.cache is None:
            return calcular()
        ficha = self.config["decoder"]
        clave = hashlib.sha256(json.dumps([ficha["repo_id"], ficha.get("revision"), self.config["dtype"], *datos])
                               .encode()).hexdigest()
        fila = self.cache.execute("SELECT valor FROM salidas WHERE clave = ?", (clave,)).fetchone()
        if fila:
            return json.loads(fila[0])
        valor = calcular()
        self.cache.execute("INSERT OR REPLACE INTO salidas VALUES (?, ?)", (clave, json.dumps(valor, ensure_ascii=False)))
        self.cache.commit()
        return valor

    def mensajes(self, entrada, pasajes, evidencia, extra=None):
        """Prompt v04/05; `extra` (p. ej. las verificaciones de CoVe) va justo antes de las instrucciones."""
        mensajes = politica.mensajes(entrada, pasajes, evidencia, self.config.get("max_caracteres_prompt", 1800),
                                     self.config.get("estilo"))
        if extra:
            usuario = mensajes[-1]["content"].replace("\nINSTRUCCIONES\n", f"\n{extra}\n\nINSTRUCCIONES\n", 1)
            mensajes = mensajes[:-1] + [{**mensajes[-1], "content": usuario}]
        return mensajes

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

    def probabilidades_letras(self, entrada, pasajes, evidencia, prefijo='{"respuesta_correcta": "', extra=None):
        """{letra: probabilidad} del siguiente token tras `prefijo`, normalizada entre las opciones.

        El prefijo por defecto pide la letra de entrada; con la justificación ya escrita en el prefijo
        (modo "razonada") la letra se elige después de razonar."""
        import torch

        letras = list((entrada.get("opciones") or {}).keys())
        entrada_tokens = self._tokens(self.mensajes(entrada, pasajes, evidencia, extra), prefijo)
        ids = [self._id_letra(prefijo, letra) for letra in letras]

        def calcular():
            return torch.softmax(self._logits_precisos(entrada_tokens, ids), dim=0).tolist()

        probabilidades = self._en_cache(["letras_fp32", list(entrada_tokens), ids], calcular)
        return dict(zip(letras, probabilidades))

    def _logits_precisos(self, entrada_tokens, ids):
        """Logits de `ids` en la última posición, con la última capa (lm_head) en float32.

        En bf16 los logits tienen ~3 cifras significativas: en la pregunta 748, A y C quedaron con la misma
        probabilidad y ganó la A por orden alfabético. Se captura la entrada de lm_head (el estado oculto ya
        normalizado) y el producto se hace en float32 solo para estos ids."""
        import torch

        capa = self.modelo.get_output_embeddings()
        capturado = {}
        gancho = capa.register_forward_hook(lambda modulo, entrada, salida: capturado.__setitem__("oculto", entrada[0]))
        tokens = torch.tensor([entrada_tokens], device=self.config.get("dispositivo", "cuda"))
        try:
            with torch.inference_mode():
                try:  # solo la última posición: con 10k tokens de contexto, todas pesarían ~3 GB en la GPU
                    self.modelo(tokens, logits_to_keep=1)
                except TypeError:
                    self.modelo(tokens)
                oculto = capturado["oculto"][0, -1].float()
                logits = capa.weight[ids].float() @ oculto
                if getattr(capa, "bias", None) is not None:
                    logits = logits + capa.bias[ids].float()
        finally:
            gancho.remove()
        return logits

    def generar(self, entrada, pasajes, evidencia, prefijo="{", repetition_penalty=None, extra=None):
        """Objeto JSON de la respuesta. `repetition_penalty` reemplaza el de la configuración (reintento
        cuando el JSON salió inválido); `extra` es un bloque adicional del prompt."""
        return prefijo + self._continuar(self.mensajes(entrada, pasajes, evidencia, extra), prefijo,
                                         self.config["max_nuevos_tokens"], repetition_penalty)

    def redactar(self, mensajes, max_nuevos):
        """Texto libre, sin JSON (hipótesis para la búsqueda, preguntas y respuestas de verificación)."""
        return self._continuar(mensajes, "", max_nuevos).strip()

    def _continuar(self, mensajes, prefijo, max_nuevos, repetition_penalty=None):
        import torch

        entrada_tokens = self._tokens(mensajes, prefijo)
        penalizacion = repetition_penalty or self.config.get("repetition_penalty", 1.0)

        def calcular():
            tokens = torch.tensor([entrada_tokens], device=self.config.get("dispositivo", "cuda"))
            with torch.inference_mode():
                salida = self.modelo.generate(tokens, max_new_tokens=max_nuevos, do_sample=False,
                                              repetition_penalty=penalizacion, pad_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.decode(salida[0, tokens.shape[-1]:], skip_special_tokens=True)

        return self._en_cache(["generar", list(entrada_tokens), max_nuevos, penalizacion], calcular)
