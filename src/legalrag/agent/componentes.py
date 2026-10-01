"""Sistema entregado: BM25 + BGE-M3 con RRF, reranker BGE y un decoder instruct abierto.

El pipeline por lotes, el servicio de la interfaz y la reproducción solo hablan con
esta clase. Otra implementación se declara en `configs/sistema.json`
(`implementacion: "modulo:Clase"`) con los mismos métodos.

Recorrido de una pregunta:

    recuperar  legalrag.retrieval.hibrido     BM25 + denso (+ búsqueda dentro de la norma que nombra la
                                              pregunta) -> RRF -> reranker -> hasta 10 unidades literales,
                                              cada una encabezada con el nombre de su norma
    responder  legalrag.generation.decoder    prompt v04/05 con los pasajes que caben -> decoder greedy
               legalrag.citations.verificacion JSON reparado, letra siempre en cerradas, citas sin
                                              respaldo quitadas (no se anula la respuesta), fundamento
                                              desde la evidencia; abstención solo sin evidencia

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
        self.validador = None
        self.ultimo_problema = self.ultimo_registro = None

    def abrir(self):
        """Carga índice, encoder, reranker y decoder. Se llama una vez antes de responder."""
        import jsonschema

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
        """Objeto de entrega con el mismo id y formato. `ultimo_problema` resume los arreglos."""
        from legalrag.citations.verificacion import abstencion, justificacion_de, respuesta_final

        evidencia = self.recuperador.evidencia
        usados = self.decoder.seleccionar(entrada, pasajes, evidencia)
        if not usados:
            self.ultimo_problema = "sin_evidencia_en_contexto"
            return abstencion(entrada, pasajes)
        probabilidades = eleccion = None
        modo = self.config["generacion"].get("letra_por_probabilidad")
        if entrada["formato"] == "multiple_choice" and modo == "razonada":
            # Primero razona (justificación) y después se comparan las letras con ese razonamiento escrito.
            crudo = self.decoder.generar(entrada, usados, evidencia, prefijo='{"justificacion": "')
            razon = justificacion_de(crudo)
            probabilidades = self.decoder.probabilidades_letras(
                entrada, usados, evidencia,
                prefijo='{"justificacion": ' + json.dumps(razon, ensure_ascii=False) + ', "respuesta_correcta": "')
            letra = max(sorted(probabilidades), key=probabilidades.get)
        elif entrada["formato"] == "multiple_choice" and modo:
            # La letra sale de comparar las opciones sin generar texto; el texto se genera ya con esa letra.
            gen = self.config["generacion"]
            if gen.get("permutar_opciones") or gen.get("descarte_mantener"):
                # Permutaciones (quita el sesgo por posición) y descarte en dos pasos (generation.eleccion).
                from legalrag.generation.eleccion import elegir
                letra, eleccion = elegir(
                    lambda opciones: self.decoder.probabilidades_letras({**entrada, "opciones": opciones}, usados,
                                                                        evidencia),
                    entrada["opciones"], gen.get("permutar_opciones", False), gen.get("descarte_mantener", 0))
                probabilidades = eleccion.get("final", eleccion["promedio"])
            else:
                probabilidades = self.decoder.probabilidades_letras(entrada, usados, evidencia)
                letra = max(sorted(probabilidades), key=probabilidades.get)
            crudo = self.decoder.generar(entrada, usados, evidencia,
                                         prefijo=f'{{"respuesta_correcta": "{letra}", "justificacion": "')
        else:
            crudo = self.decoder.generar(entrada, usados, evidencia)
        # Con entregar_todos, la evidencia entregada son los pasajes recuperados completos (máx. 10), aunque el
        # prompt solo haya usado los que caben: el respaldo de citas y el fundamento se miden sobre ellos.
        entregados = pasajes[:10] if self.config["generacion"].get("entregar_todos") else usados
        respuesta, registro = respuesta_final(entrada, crudo, entregados, evidencia,
                                              self.config["generacion"]["politica"], self.validador)
        if probabilidades and not respuesta.get("abstencion"):
            respuesta["respuesta_correcta"] = letra
            respuesta["descarte_opciones"] = {k: v for k, v in respuesta["descarte_opciones"].items() if k != letra}
        self.ultimo_problema = registro["problema"]
        self.ultimo_registro = {**registro, "crudo": crudo, "probabilidades_letras": probabilidades, "eleccion": eleccion,
                                "pasajes_en_prompt": len(usados), "pasajes_entregados": len(entregados)}
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
