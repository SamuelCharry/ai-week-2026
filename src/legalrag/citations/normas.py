"""Nombres de norma que el evaluador oficial reconoce, para el corpus nuevo (corpus_eval_v1).

El evaluador cuenta una cita como respaldada solo si el *texto* de alguno de los 10 primeros
pasajes la contiene (`citations.extract`, a nivel de norma). Un artículo suelto ("ARTÍCULO 369.
Traslado...") no dice de qué ley es, así que citar "Ley 1564 de 2012" quedaba sin respaldo y
restaba el doble. Por eso cada pasaje entregado empieza con un encabezado como

    Código General del Proceso (Ley 1564 de 2012), artículo 369.

que el extractor oficial vuelve a leer como la misma norma (el ejemplo oficial de entrega hace lo
mismo). `EvidenciaCorpus` además sirve a `generation.politica` (prompt, saneo de citas y
fundamento) y ubica en el corpus las normas que la propia pregunta nombra.
"""
import re
from collections import defaultdict

from legalrag.citations.evidencia import NOMBRES, TIPOS, EvidenciaV04, _prioridad, nombre_cuerpo

_SENTENCIA = re.compile(r"sentencia_(cc|csj|ce)_([a-z]+)(\d+)_(\d{4})")
_TITULO_CODIGO = ("codigo", "código", "constitucion", "constitución", "estatuto", "decision", "decisión")
# Siglas de sala que el extractor oficial lee como un código: «Sentencia CP-147 de 2014» (Sala Penal de la
# Corte Suprema) salía como la Constitución Política. Con esas salas se usa el formato de la Corte, «CP147-2014».
_SALAS_AMBIGUAS = {"cp", "cc", "cpp", "cst", "cpt", "cpts", "cgp", "cco", "et", "cpaca"}


def nombre_numerado(meta):
    """"Ley 1564 de 2012", "Sentencia C-355 de 2006"...; None si el documento no tiene número."""
    tipo = str(meta.get("tipo") or "").lower().replace(" ", "_")
    numero, anio = meta.get("numero"), meta.get("anio")
    m = _SENTENCIA.fullmatch(meta.get("doc_id", ""))
    if m:
        if m.group(1) != "cc" and m.group(2) in _SALAS_AMBIGUAS:
            return f"Sentencia {m.group(2).upper()}{int(m.group(3))}-{m.group(4)}"
        return f"Sentencia {m.group(2).upper()}-{int(m.group(3))} de {m.group(4)}"
    if tipo == "constitucion":
        return NOMBRES["constitucion"]
    if tipo in TIPOS and numero and anio:
        return f"{TIPOS[tipo]} {numero} de {anio}"
    return None


class EvidenciaCorpus:
    """Misma interfaz que `citations.evidencia.EvidenciaV04` (etiqueta, respaldo, citas_evidencia)."""

    citas_evidencia = EvidenciaV04.citas_evidencia

    def __init__(self, citas, manifiesto):
        self.citas = citas
        self.documentos = {d["doc_id"]: d for d in manifiesto}
        self._etiquetas = {}
        self._por_cuerpo = defaultdict(list)
        for doc_id, meta in self.documentos.items():
            for cuerpo in self.cuerpos(doc_id):
                self._por_cuerpo[cuerpo].append(doc_id)

    def cuerpos(self, doc_id):
        """Cuerpos del extractor oficial que nombran este documento."""
        meta = self.documentos[doc_id]
        partes = [nombre_numerado(meta) or ""]
        titulo = str(meta.get("titulo") or "")
        if titulo.lower().startswith(_TITULO_CODIGO):
            partes.append(titulo)
        return self.citas.bodies(self.citas.extract(" ".join(partes)))

    def etiqueta(self, pasaje):
        """(nombre para el prompt, cuerpo) de la norma del pasaje; el nombre de código va primero."""
        doc_id = pasaje["doc_id"]
        if doc_id not in self._etiquetas:
            meta = self.documentos.get(doc_id, {"doc_id": doc_id})
            numerado = nombre_numerado(meta)
            elegido = None
            for cuerpo in sorted(self.cuerpos(doc_id) if doc_id in self.documentos else [],
                                 key=lambda c: (_prioridad(c), str(c))):
                nombre = nombre_cuerpo(cuerpo)
                if nombre and cuerpo in self.citas.bodies(self.citas.extract(nombre)):
                    elegido = (nombre, cuerpo)
                    break
            if elegido is None:
                elegido = (numerado or meta.get("titulo") or doc_id, None)
            nombre, cuerpo = elegido
            if numerado and numerado != nombre:
                nombre = f"{nombre} ({numerado})"
            self._etiquetas[doc_id] = (nombre, cuerpo)
        return self._etiquetas[doc_id]

    def encabezado(self, pasaje):
        nombre, _ = self.etiqueta(pasaje)
        articulo = str(pasaje.get("articulo") or "").strip()
        return f"{nombre}, artículo {articulo}." if articulo else f"{nombre}."

    @staticmethod
    def texto_entregado(pasaje):
        """Texto del pasaje tal como lo ve el evaluador: encabezado + tramo literal."""
        encabezado = pasaje.get("encabezado")
        return f"{encabezado}\n{pasaje['texto']}" if encabezado else pasaje["texto"]

    def respaldo(self, pasajes):
        """Cuerpos que el evaluador considera respaldados por los 10 primeros pasajes entregados."""
        return set().union(*(self.citas.bodies(self.citas.extract(self.texto_entregado(p)))
                             for p in (pasajes or [])[:10])) if pasajes else set()

    def normas_de(self, texto):
        """{doc_id: {artículos}} de las normas que un texto nombra y que están en el corpus."""
        encontradas = defaultdict(set)
        for tipo, numero, anio, articulo in self.citas.extract(texto or ""):
            for doc_id in self._por_cuerpo.get((tipo, numero, anio), []):
                if articulo:
                    encontradas[doc_id].add(articulo)
                else:
                    encontradas.setdefault(doc_id, set())
        return dict(encontradas)
