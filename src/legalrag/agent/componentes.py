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

Pasos opcionales del agente (configs/sistema.json, legalrag.generation.pasos), cada uno medible en src/comparar.py:

    recuperacion.expansion = "debil"|"siempre"   hipótesis del decoder como consulta extra (HyDE/Query2doc),
                                                 solo con evidencia débil (CRAG); texto libre salvo expansion_cerradas
    generacion.regenerar_json                    un reintento con otra penalización de repetición si el JSON salió
                                                 inválido o incompleto; se queda con la mejor de las dos
    generacion.verificar                         CoVe: preguntas de verificación, respuestas solo con los pasajes y
                                                 respuesta otra vez con ellas a la vista
    generacion.verificador_nli                   cada oración de texto libre contra los pasajes con un modelo NLI
                                                 pequeño (citations.respaldo_nli); registra o quita las contradichas
    recuperacion.recuperar_por_opcion            en cerradas, una búsqueda más por opción (retrieval.hibrido)

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


def gravedad(problema):
    """Qué tan mala salió una respuesta, para decidir si un reintento o una verificación la mejora."""
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
        self.verificador = None
        if config["generacion"].get("verificador_nli"):
            from legalrag.citations.respaldo_nli import VerificadorNLI
            self.verificador = VerificadorNLI(config["generacion"]["verificador_nli"],
                                              config["generacion"].get("dispositivo", "cuda"))
        self.validador = None
        self.ultimo_problema = self.ultimo_registro = self.ultima_expansion = None

    def abrir(self):
        """Carga índice, encoder, reranker y decoder. Se llama una vez antes de responder."""
        import jsonschema

        esquema = json.loads((self.raiz / self.config["oficial"] / "schema/submission.schema.json")
                             .read_text(encoding="utf-8"))
        self.validador = jsonschema.validators.validator_for(esquema)(esquema)
        self.recuperador.abrir()
        self.decoder.abrir()
        if self.verificador:
            self.verificador.abrir()

    def cerrar(self):
        self.decoder.cerrar()
        self.recuperador.cerrar()
        if self.verificador:
            self.verificador.cerrar()

    def recuperar(self, entrada):
        """Pasajes de la pregunta, ordenados por el reranker.

        Con `recuperacion.expansion` ("debil" o "siempre") el decoder escribe una hipótesis de respuesta
        y se vuelve a buscar con ella (generation.pasos: HyDE/Query2doc, activado como en CRAG)."""
        from legalrag.generation import pasos

        rec = self.config["recuperacion"]
        pasajes = self.recuperador.buscar(entrada)
        self.ultima_expansion = None
        modo = rec.get("expansion")
        if not modo or (entrada["formato"] == "multiple_choice" and not rec.get("expansion_cerradas")):
            return pasajes
        debil, motivo = pasos.evidencia_debil(pasajes, rec.get("umbral_evidencia_debil"),
                                              criterio=rec.get("criterio_evidencia_debil", "ambos"))
        nombradas = (getattr(self.recuperador, "ultima_traza", None) or {}).get("normas_nombradas") or {}
        if debil and nombradas and rec.get("expansion_sin_norma_nombrada"):
            # Adaptive-RAG: si la pregunta ya nombra una norma del corpus, el enrutamiento la trae; no se gasta más.
            debil, motivo = False, "norma_nombrada"
        self.ultima_expansion = {"motivo": motivo, "puntaje_maximo": max((p["score"] for p in pasajes), default=None)}
        if modo == "debil" and not debil:
            return pasajes
        hipotesis = self.decoder.redactar(pasos.mensajes_hipotesis(entrada), rec.get("max_tokens_hipotesis", 160))
        expandidos = self.recuperador.buscar(entrada, expansion=hipotesis)
        self.ultima_expansion.update(hipotesis=hipotesis, docs_antes=[p["doc_id"] for p in pasajes],
                                     docs_despues=[p["doc_id"] for p in expandidos])
        return expandidos

    def responder(self, entrada, pasajes):
        """Objeto de entrega con el mismo id y formato. `ultimo_problema` resume los arreglos."""
        from legalrag.citations.verificacion import abstencion, justificacion_de, respuesta_final

        evidencia = self.recuperador.evidencia
        usados = self.decoder.seleccionar(entrada, pasajes, evidencia)
        if not usados:
            self.ultimo_problema = "sin_evidencia_en_contexto"
            return abstencion(entrada, pasajes)
        probabilidades = eleccion = None
        gen = self.config["generacion"]
        modo = gen.get("letra_por_probabilidad")
        prefijo = "{"
        if entrada["formato"] == "multiple_choice" and modo == "razonada":
            # Primero razona (justificación) y después se comparan las letras con ese razonamiento escrito.
            prefijo = '{"justificacion": "'
            crudo = self.decoder.generar(entrada, usados, evidencia, prefijo=prefijo)
            razon = justificacion_de(crudo)
            probabilidades = self.decoder.probabilidades_letras(
                entrada, usados, evidencia,
                prefijo='{"justificacion": ' + json.dumps(razon, ensure_ascii=False) + ', "respuesta_correcta": "')
            letra = max(sorted(probabilidades), key=probabilidades.get)
        elif entrada["formato"] == "multiple_choice" and modo:
            # La letra sale de comparar las opciones sin generar texto; el texto se genera ya con esa letra.
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
            prefijo = f'{{"respuesta_correcta": "{letra}", "justificacion": "'
            crudo = self.decoder.generar(entrada, usados, evidencia, prefijo=prefijo)
        else:
            crudo = self.decoder.generar(entrada, usados, evidencia)
        # Con entregar_todos, la evidencia entregada son los pasajes recuperados completos (máx. 10), aunque el
        # prompt solo haya usado los que caben: el respaldo de citas y el fundamento se miden sobre ellos.
        entregados = pasajes[:10] if gen.get("entregar_todos") else usados

        def final(texto):
            return respuesta_final(entrada, texto, entregados, evidencia, gen["politica"], self.validador)

        respuesta, registro = final(crudo)
        verificacion = None
        if gen.get("verificar") and entrada["formato"] in gen.get("verificar_formatos", ["semi_open", "open_ended"]) \
                and registro["problema"] != "json_invalido":
            # CoVe factorizado: preguntas desde el borrador, respuestas solo con los pasajes, y respuesta de nuevo.
            verificacion = self._verificar(entrada, usados, evidencia, respuesta)
            nuevo = self.decoder.generar(entrada, usados, evidencia, prefijo=prefijo, extra=verificacion["bloque"])
            respuesta_v, registro_v = final(nuevo)
            verificacion["aceptada"] = gravedad(registro_v["problema"]) <= gravedad(registro["problema"])
            if verificacion["aceptada"]:
                crudo, respuesta, registro = nuevo, respuesta_v, registro_v
        reintento = None
        if gen.get("regenerar_json") and gravedad(registro["problema"]) >= 1:
            # Greedy repite la misma salida: el reintento cambia la penalización de repetición (los JSON
            # inválidos casi siempre son bucles que agotan los tokens).
            nuevo = self.decoder.generar(entrada, usados, evidencia, prefijo=prefijo,
                                         repetition_penalty=gen.get("repetition_penalty_reintento", 1.3))
            respuesta_r, registro_r = final(nuevo)
            reintento = {"problema_antes": registro["problema"], "problema_despues": registro_r["problema"]}
            if gravedad(registro_r["problema"]) < gravedad(registro["problema"]):
                crudo, respuesta, registro = nuevo, respuesta_r, registro_r
        nli = None
        if getattr(self, "verificador", None) and not respuesta.get("abstencion"):
            # Verificador NLI (citations.respaldo_nli): revisa cada oración contra los pasajes sin generar texto.
            nli = self.verificador.verificar(respuesta, entregados, self.recuperador.citaciones)
        if probabilidades and not respuesta.get("abstencion"):
            respuesta["respuesta_correcta"] = letra
            respuesta["descarte_opciones"] = {k: v for k, v in respuesta["descarte_opciones"].items() if k != letra}
        self.ultimo_problema = registro["problema"]
        self.ultimo_registro = {**registro, "crudo": crudo, "probabilidades_letras": probabilidades, "eleccion": eleccion,
                                "pasajes_en_prompt": len(usados), "pasajes_entregados": len(entregados),
                                "expansion": self.ultima_expansion, "verificacion": verificacion,
                                "reintento_json": reintento, "verificacion_nli": nli}
        return respuesta

    def _verificar(self, entrada, usados, evidencia, borrador):
        from legalrag.citations.verificacion import texto_citable
        from legalrag.generation import pasos, politica

        gen = self.config["generacion"]
        maximo = gen.get("verificar_preguntas", 3)
        preguntas = pasos.preguntas_de(
            self.decoder.redactar(pasos.mensajes_preguntas(entrada, texto_citable(borrador), maximo), 120), maximo)
        bloque = politica.bloque_pasajes(usados, evidencia, gen.get("max_caracteres_prompt", 1800))
        respuestas = self.decoder.redactar(pasos.mensajes_respuestas(entrada, preguntas, bloque), 60 * len(preguntas) + 40) \
            if preguntas else ""
        return {"preguntas": preguntas, "respuestas": respuestas,
                "bloque": pasos.bloque_verificacion(preguntas, respuestas) if preguntas else None}

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, tipo, valor, traza):
        self.cerrar()


def cargar(raiz, config):
    """Instancia la clase declarada en `implementacion`."""
    modulo, _, clase = config["implementacion"].partition(":")
    return getattr(importlib.import_module(modulo), clase)(raiz, config)
