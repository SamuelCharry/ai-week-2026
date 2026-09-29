"""Recuperación y generación del sistema entregado.

El pipeline por lotes, el servicio de la interfaz y la reproducción solo hablan con
esta clase. Para conectar la versión final hay que implementar `abrir`, `recuperar`
y `responder`. Si se prefiere otra clase, se declara en `configs/sistema.json`
(`implementacion: "modulo:Clase"`) con los mismos métodos.

Piezas disponibles en el repositorio:

    legalrag.indexing.indice     cargar_indice, Recuperador.buscar, BM25
    legalrag.retrieval.reordenamiento   Reordenador (bge-reranker-v2-m3)
    legalrag.citations.evidencia    EvidenciaV04: artículos completos y cabeceras literales
    legalrag.generation.politica     generar y postprocesar: prompt, JSON, citas y abstención
    legalrag.generation.cliente      ServidorLocal: llama.cpp con temperatura 0 y semilla 0
    legalrag.generation.runtime         runtime, preparar_decoder_persistente
    legalrag.experimentos.v04        ejecutar(): el mismo flujo completo sobre la muestra

Requisitos del enunciado que dependen de esta clase:

    - Temperatura 0: la salida debe ser reproducible en la verificación en vivo.
    - Toda norma citada debe figurar en `pasajes_recuperados` (solo cuentan los 10 primeros).
    - `abstencion: true` cuando el corpus no da fundamento suficiente.
    - Claves exactas por formato (ver data/oficial/schema/submission.schema.json).
    - Unos 22 segundos por pregunta en promedio (992 preguntas en seis horas).
"""
import importlib

PENDIENTE = ("Sistema.{} está pendiente. Implementarlo en src/legalrag/agent/componentes.py "
             "o declarar otra clase en configs/sistema.json (implementacion).")


class Sistema:
    def __init__(self, raiz, config):
        self.raiz = raiz
        self.config = config

    def abrir(self):
        """Carga índice, encoder, reranker y decoder. Se llama una vez antes de responder."""
        raise NotImplementedError(PENDIENTE.format("abrir"))

    def cerrar(self):
        """Libera GPU y detiene el servidor del decoder."""

    def recuperar(self, entrada):
        """Devuelve los pasajes de la pregunta, ordenados por pertinencia.

        entrada: pregunta sin campos de evaluación (id, formato, pregunta, opciones,
        area, sub_tarea, tema, complejidad).
        """
        raise NotImplementedError(PENDIENTE.format("recuperar"))

    def responder(self, entrada, pasajes):
        """Devuelve el objeto de entrega de la pregunta, con el mismo id y formato.

        Debe incluir `abstencion` y `pasajes_recuperados`, además de las claves del
        formato: multiple_choice (respuesta_correcta, justificacion, descarte_opciones),
        semi_open (respuesta, palabras_clave, referencia_legal) u open_ended
        (marco_normativo, analisis, jurisprudencia, conclusion).
        """
        raise NotImplementedError(PENDIENTE.format("responder"))

    def __enter__(self):
        self.abrir()
        return self

    def __exit__(self, tipo, valor, traza):
        self.cerrar()


def cargar(raiz, config):
    """Instancia la clase declarada en `implementacion`."""
    modulo, _, clase = config["implementacion"].partition(":")
    return getattr(importlib.import_module(modulo), clase)(raiz, config)
