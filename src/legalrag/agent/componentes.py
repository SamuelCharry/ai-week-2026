"""Sistema entregado: Cerberus, tres agentes alrededor de Qwen3-8B sobre BM25 + BGE-M3 y un reranker abierto.

El pipeline por lotes, el servicio de la interfaz y la reproducción solo hablan con esta clase. Otra
implementación se declara en `configs/sistema.json` (`implementacion: "modulo:Clase"`) con los mismos métodos.

Sigue el ciclo agéntico que propone el enunciado (anexo B): recuperar, reformular con el nombre de la norma
probable y volver a recuperar, redactar solo con la evidencia y verificar cada cita.

    recuperar  retrieval.hibrido          BM25 + denso (+ búsqueda dentro de la norma que nombra la pregunta)
                                          -> RRF -> reranker -> hasta 10 unidades literales con el nombre de su norma
               agente reformulador        (texto libre) escribe con qué figura y normas se resuelve el caso
                                          (generation.pasos) y se vuelve a recuperar con eso; las normas que nombra
                                          tienen lugares reservados en la evidencia
    responder  agente calculadora         montos de la pregunta en SMMLV y UVT con los decretos del corpus
               agente normalizador        avisa si la pregunta u opción cita una ley con el año equivocado
               generation.decoder         greedy; en cerradas, letra por probabilidad (float32) y, si hay nota de
                                          las herramientas, primero el razonamiento y después la letra
               citations.verificacion     JSON reparado, citas sin respaldo quitadas, normas de la evidencia
                                          agregadas, esquema oficial; abstención solo sin evidencia
               reintento                  si el JSON salió inválido, otra generación con distinta penalización

Requisitos del enunciado que dependen de esta clase:

    - Temperatura 0: la salida debe ser reproducible en la verificación en vivo (una pregunta a la vez).
    - Toda norma citada debe figurar en `pasajes_recuperados` (solo cuentan los 10 primeros).
    - `abstencion: true` cuando el corpus no da fundamento suficiente.
    - Claves exactas por formato (ver data/oficial/schema/submission.schema.json).
    - Unos 22 segundos por pregunta en promedio (992 preguntas en seis horas).
"""
import importlib
import json
import time
from pathlib import Path


def gravedad(problema):
    """Qué tan mala salió una respuesta, para decidir si el reintento la mejora."""
    if str(problema).startswith("esquema_oficial"):
        return 3  # se entrega como abstención
    return {"json_invalido": 2, "campos_rellenados": 1}.get(problema, 0)


class Sistema:
    def __init__(self, raiz, config):
        from legalrag.generation.decoder import DecoderTransformers
        from legalrag.retrieval.hibrido import RecuperadorHibrido

        self.raiz = Path(raiz)
        self.config = config
        self.recuperador = RecuperadorHibrido(self.raiz, config["recuperacion"])
        self.decoder = DecoderTransformers(config["generacion"])
        self.validador = self.calculadora = self.normalizador = None
        self.ultimo_problema = self.ultimo_registro = self.ultima_expansion = None

    def abrir(self):
        """Carga índice, encoder, reranker, herramientas y decoder. Se llama una vez antes de responder."""
        import jsonschema

        esquema = json.loads((self.raiz / self.config["oficial"] / "schema/submission.schema.json")
                             .read_text(encoding="utf-8"))
        self.validador = jsonschema.validators.validator_for(esquema)(esquema)
        self.recuperador.abrir()
        gen = self.config["generacion"]
        documentos = self.recuperador.evidencia.documentos.values()
        if gen.get("calculadora"):
            from legalrag.generation.calculadora import Calculadora
            textos = self.recuperador._ruta("textos")
            self.calculadora = Calculadora.desde_corpus(
                documentos, lambda doc_id: (textos / f"{doc_id}.txt").read_text(encoding="utf-8"))
        if gen.get("normalizador_citas"):
            from legalrag.citations.normalizador import NormalizadorCitas
            self.normalizador = NormalizadorCitas(documentos)
        self.decoder.abrir()

    def cerrar(self):
        self.decoder.cerrar()
        self.recuperador.cerrar()

    def recuperar(self, entrada):
        """Pasajes de la pregunta. En texto libre, el agente reformulador escribe con qué se resuelve el caso y se
        vuelve a recuperar con eso (`recuperacion.expansion`, `recuperacion.agente_expansion`)."""
        from legalrag.generation import pasos

        rec = self.config["recuperacion"]
        inicio = time.perf_counter()
        pasajes = self.recuperador.buscar(entrada)
        self.tiempos = {"recuperacion": time.perf_counter() - inicio}
        self.ultima_expansion = None
        if not rec.get("expansion") or entrada["formato"] == "multiple_choice":
            return pasajes
        agente = rec.get("agente_expansion", "hipotesis")
        if agente == "iterativo":
            from legalrag.generation import politica
            mensajes = pasos.mensajes_iterativo(entrada, politica.bloque_pasajes(
                pasajes[:5], self.recuperador.evidencia, rec.get("max_caracteres_iterativo", 600)))
        elif agente == "normas":
            mensajes = pasos.mensajes_reformulador(entrada)
        else:
            mensajes = pasos.mensajes_hipotesis(entrada)
        inicio = time.perf_counter()
        texto = self.decoder.redactar(mensajes, rec.get("max_tokens_hipotesis", 160))
        self.tiempos["reformulador"] = time.perf_counter() - inicio
        inicio = time.perf_counter()
        expandidos = self.recuperador.buscar(entrada, expansion=texto)
        self.tiempos["recuperacion_2"] = time.perf_counter() - inicio
        self.ultima_expansion = {"agente": agente, "hipotesis": texto, "docs_antes": [p["doc_id"] for p in pasajes],
                                 "docs_despues": [p["doc_id"] for p in expandidos]}
        return expandidos

    def responder(self, entrada, pasajes):
        """Objeto de entrega con el mismo id y formato. `ultimo_problema` resume los arreglos."""
        from legalrag.citations.verificacion import abstencion, justificacion_de, respuesta_final

        evidencia = self.recuperador.evidencia
        tiempos = getattr(self, "tiempos", None) or {}
        inicio = time.perf_counter()
        usados = self.decoder.seleccionar(entrada, pasajes, evidencia)
        if not usados:
            self.ultimo_problema = "sin_evidencia_en_contexto"
            return abstencion(entrada, pasajes)
        gen = self.config["generacion"]
        # Herramientas deterministas: calculadora (montos en SMMLV y UVT) y normalizador de citas (años equivocados).
        herramientas = [getattr(self, "calculadora", None), getattr(self, "normalizador", None)]
        nota = "\n\n".join(n for n in (h.nota(entrada) for h in herramientas if h) if n) or None
        con_nota = {"extra": nota} if nota else {}
        probabilidades = None
        modo = gen.get("letra_por_probabilidad")
        if modo is True and nota and gen.get("razonar_con_herramienta") and entrada["formato"] == "multiple_choice":
            # Con una nota de las herramientas (cálculo o aviso de norma), la letra se elige después de razonar: en una
            # pasada el modelo no hacía la cuenta (528: «30.000.000 = 17,1 SMMLV» en el prompt y aun así «menor cuantía»).
            modo = "razonada"
        prefijo = "{"
        if entrada["formato"] == "multiple_choice" and modo == "razonada":
            # Primero razona (justificación) y después se comparan las letras con ese razonamiento escrito.
            prefijo = '{"justificacion": "'
            crudo = self.decoder.generar(entrada, usados, evidencia, prefijo=prefijo, **con_nota)
            razon = justificacion_de(crudo)
            probabilidades = self.decoder.probabilidades_letras(
                entrada, usados, evidencia,
                prefijo='{"justificacion": ' + json.dumps(razon, ensure_ascii=False) + ', "respuesta_correcta": "',
                **con_nota)
            letra = max(sorted(probabilidades), key=probabilidades.get)
        elif entrada["formato"] == "multiple_choice" and modo:
            # La letra sale de comparar las opciones sin generar texto; el texto se genera ya con esa letra.
            probabilidades = self.decoder.probabilidades_letras(entrada, usados, evidencia, **con_nota)
            letra = max(sorted(probabilidades), key=probabilidades.get)
            prefijo = f'{{"respuesta_correcta": "{letra}", "justificacion": "'
            crudo = self.decoder.generar(entrada, usados, evidencia, prefijo=prefijo, **con_nota)
        else:
            crudo = self.decoder.generar(entrada, usados, evidencia, **con_nota)
        # Con entregar_todos, la evidencia entregada son los pasajes recuperados completos (máx. 10), aunque el
        # prompt solo haya usado los que caben: el respaldo de citas y el fundamento se miden sobre ellos.
        entregados = pasajes[:10] if gen.get("entregar_todos") else usados

        def final(texto):
            return respuesta_final(entrada, texto, entregados, evidencia, gen["politica"], self.validador)

        respuesta, registro = final(crudo)
        tiempos["generacion"] = time.perf_counter() - inicio
        inicio = time.perf_counter()
        reintento = None
        if gen.get("regenerar_json") and gravedad(registro["problema"]) >= 1:
            # Greedy repite la misma salida: el reintento cambia la penalización de repetición (los JSON
            # inválidos casi siempre son bucles que agotan los tokens).
            nuevo = self.decoder.generar(entrada, usados, evidencia, prefijo=prefijo, **con_nota,
                                         repetition_penalty=gen.get("repetition_penalty_reintento", 1.3))
            respuesta_r, registro_r = final(nuevo)
            reintento = {"problema_antes": registro["problema"], "problema_despues": registro_r["problema"]}
            if gravedad(registro_r["problema"]) < gravedad(registro["problema"]):
                crudo, respuesta, registro = nuevo, respuesta_r, registro_r
        if reintento:
            tiempos["reintento"] = time.perf_counter() - inicio
        if probabilidades and not respuesta.get("abstencion"):
            respuesta["respuesta_correcta"] = letra
            respuesta["descarte_opciones"] = {k: v for k, v in respuesta["descarte_opciones"].items() if k != letra}
        self.ultimo_problema = registro["problema"]
        self.ultimo_registro = {**registro, "crudo": crudo, "probabilidades_letras": probabilidades,
                                "pasajes_en_prompt": len(usados), "pasajes_entregados": len(entregados),
                                "expansion": self.ultima_expansion, "reintento_json": reintento, "calculadora": nota,
                                "modo_letra": modo if entrada["formato"] == "multiple_choice" else None,
                                "tiempos": {k: round(v, 2) for k, v in tiempos.items()}}
        self.tiempos = None
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
