"""Sistema entregado (opción A): BM25 + BGE-M3 con RRF, reranker BGE y Qwen2.5-7B-Instruct.

El pipeline por lotes, el servicio de la interfaz y la reproducción solo hablan con
esta clase. Otra implementación se declara en `configs/sistema.json`
(`implementacion: "modulo:Clase"`) con los mismos métodos.

Recorrido de una pregunta:

    recuperar  legalrag.retrieval.hibrido     BM25 top 100 + BGE-M3 top 100 -> RRF -> reranker top 50
                                              -> hasta 10 unidades literales, con cabecera de la norma
    responder  legalrag.generation.decoder    pasajes que caben en el contexto -> Qwen2.5-7B greedy
               legalrag.citations.verificacion JSON por formato, citas respaldadas, esquema oficial;
                                              si algo falla, abstención con la evidencia conservada

Requisitos del enunciado que dependen de esta clase:

    - Temperatura 0: la salida debe ser reproducible en la verificación en vivo.
    - Toda norma citada debe figurar en `pasajes_recuperados` (solo cuentan los 10 primeros).
    - `abstencion: true` cuando el corpus no da fundamento suficiente.
    - Claves exactas por formato (ver data/oficial/schema/submission.schema.json).
    - Unos 22 segundos por pregunta en promedio (992 preguntas en seis horas).
"""
import importlib
import json
from pathlib import Path


class Sistema:
    def __init__(self, raiz, config):
        from legalrag.generation.decoder import DecoderTransformers
        from legalrag.retrieval.hibrido import RecuperadorHibrido

        self.raiz = Path(raiz)
        self.config = config
        self.recuperador = RecuperadorHibrido(self.raiz, config["recuperacion"])
        self.decoder = DecoderTransformers(config["generacion"])
        self.manifiesto = self.validador = None
        self.ultimo_problema = None

    def abrir(self):
        """Carga índice, encoder, reranker y decoder. Se llama una vez antes de responder."""
        import jsonschema

        ruta = self.raiz / self.config["recuperacion"]["manifiesto"]
        self.manifiesto = json.loads(ruta.read_text(encoding="utf-8"))
        esquema = json.loads((self.raiz / self.config["oficial"] / "schema/submission.schema.json")
                             .read_text(encoding="utf-8"))
        self.validador = jsonschema.validators.validator_for(esquema)(esquema)
        self.recuperador.abrir()
        self.decoder.abrir()

    def cerrar(self):
        self.decoder.cerrar()
        self.recuperador.cerrar()

    def recuperar(self, entrada):
        """Pasajes de la pregunta, ordenados por el reranker."""
        return self.recuperador.buscar(entrada)

    def responder(self, entrada, pasajes):
        """Objeto de entrega con el mismo id y formato. `ultimo_problema` dice por qué se abstuvo."""
        from legalrag.citations.verificacion import abstencion, respuesta_final

        usados = self.decoder.seleccionar(entrada, pasajes)
        if not usados:
            self.ultimo_problema = "sin_evidencia_en_contexto"
            return abstencion(entrada, pasajes)
        crudo = self.decoder.generar(entrada, usados)
        respuesta, self.ultimo_problema = respuesta_final(entrada, crudo, usados, self.manifiesto, self.validador)
        return respuesta

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, tipo, valor, traza):
        self.cerrar()


def cargar(raiz, config):
    """Instancia la clase declarada en `implementacion`."""
    modulo, _, clase = config["implementacion"].partition(":")
    return getattr(importlib.import_module(modulo), clase)(raiz, config)
