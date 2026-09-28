"""Evidencia y citas para la versión 04.

Corrige un desajuste del 03: `identidad()` reconoce el Código Civil como
("ley", "84", "1873") y el Código de Comercio como ("decreto", "410", "1971"),
pero el evaluador oficial normaliza "Código Civil" a ("codigo_civil", None, None).
Con la cabecera "LEY 84 DE 1873", una respuesta que cita "artículo 946 del Código
Civil" quedaba sin respaldo y no coincidía con la referencia.

Aquí la cabecera es un tramo literal del inicio del documento que contiene todos
los nombres con que el evaluador puede reconocer la norma (número y código).
"""
import re
from collections import OrderedDict

from scripts.evaluacion.oficial import Evidencia, identidad

NOMBRES = {
    "constitucion": "Constitución Política",
    "codigo_civil": "Código Civil",
    "codigo_penal": "Código Penal",
    "codigo_procedimiento_penal": "Código de Procedimiento Penal",
    "codigo_comercio": "Código de Comercio",
    "codigo_sustantivo_trabajo": "Código Sustantivo del Trabajo",
    "codigo_procesal_trabajo": "Código Procesal del Trabajo",
    "codigo_general_proceso": "Código General del Proceso",
    "cpaca": "Código de Procedimiento Administrativo y de lo Contencioso Administrativo",
    "estatuto_tributario": "Estatuto Tributario",
    "codigo_infancia": "Código de la Infancia y la Adolescencia",
    "codigo_nacional_policia": "Código Nacional de Seguridad y Convivencia Ciudadana",
    "codigo_disciplinario": "Código General Disciplinario",
    "estatuto_consumidor": "Estatuto del Consumidor",
    "decision_andina_486": "Decisión 486 de la Comisión de la Comunidad Andina",
}
TIPOS = {"ley": "Ley", "decreto": "Decreto", "acto_legislativo": "Acto Legislativo",
         "resolucion": "Resolución", "circular": "Circular", "acuerdo": "Acuerdo"}
MAX_CABECERA = 700
# Alcance de la búsqueda del tramo literal cuando las primeras líneas no bastan.
# Corto a propósito: más adentro ya no hay encabezados, solo coincidencias sueltas
# del cuerpo que el extractor lee como una cita.
MAX_BUSQUEDA_CABECERA = 4000
# Pasajes que nombran la norma y no son evidencia recuperada.
CABECERAS = ("cabecera_fuente", "cabecera_normalizada")


def nombre_cuerpo(cuerpo):
    """Texto que el extractor oficial vuelve a leer como el mismo cuerpo."""
    tipo, numero, anio = cuerpo
    if tipo in NOMBRES:
        return NOMBRES[tipo]
    if tipo == "jurisprudencia":
        return f"Sentencia {numero} de {anio}"
    if tipo in TIPOS and numero and anio:
        return f"{TIPOS[tipo]} {numero} de {anio}"
    return None


def _prioridad(cuerpo):
    # Los nombres de código son los que usan las preguntas y el fundamento oficial.
    return 0 if cuerpo[0] in NOMBRES else 1


class EvidenciaV04(Evidencia):
    def __init__(self, raiz, corpus):
        super().__init__(raiz, corpus)
        self._identidades = {}
        self._cabeceras_v04 = {}
        self._normalizadas = {}

    def identidad(self, doc_id):
        """Cuerpos con que el evaluador puede reconocer el documento."""
        if doc_id not in self._identidades:
            doc = self.documentos[doc_id]
            cuerpos = set(identidad(doc, self.citas))
            # Alias de código solo si el título ES el código ("Código Civil colombiano"),
            # no si lo menciona ("Reforma al Código de Comercio", "desarrolla el artículo 88 de la Constitución").
            titulo = self.citas.norm(doc.get("titulo") or "")
            if titulo.startswith(("codigo", "estatuto", "constitucion")):
                del_titulo = {c for c in self.citas.bodies(self.citas.extract(titulo)) if c[0] in NOMBRES}
                if len(del_titulo) == 1:
                    cuerpos |= del_titulo
            # También si una de las primeras líneas es el nombre del código por sí solo
            # ("CODIGO SUSTANTIVO DEL TRABAJO" en el Decreto 2663 de 1950).
            lineas = [l.strip() for l in self.texto(doc_id)[:3000].splitlines() if l.strip()][:8]
            for linea in lineas:
                normal = self.citas.norm(linea).strip(' ."')
                if len(normal) <= 60 and normal.startswith(("codigo", "estatuto")):
                    de_linea = {c for c in self.citas.bodies(self.citas.extract(normal)) if c[0] in NOMBRES}
                    if len(de_linea) == 1:
                        cuerpos |= de_linea
            self._identidades[doc_id] = cuerpos
        return self._identidades[doc_id]

    def cabecera(self, doc_id):
        """Tramo literal corto que nombra la norma con todos sus alias posibles."""
        if doc_id in self._cabeceras_v04:
            return self._cabeceras_v04[doc_id]
        texto = self.texto(doc_id)
        propia = self.identidad(doc_id)
        lineas = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\n]+", texto[:12000])]
        cubiertos = []
        for inicio, fin, linea in lineas:
            if len(linea) > 600:
                continue
            hallados = propia & self.citas.bodies(self.citas.extract(linea))
            if hallados:
                cubiertos.append((inicio, fin, hallados))
        elegida = None
        if cubiertos:
            objetivo = set().union(*(h for _, _, h in cubiertos))
            # Primer tramo contiguo que cubre todos los alias encontrados, sin exceder el máximo.
            for i, (inicio, _, _) in enumerate(cubiertos):
                vistos = set()
                for _, fin, hallados in cubiertos[i:]:
                    if fin - inicio > MAX_CABECERA:
                        break
                    vistos |= hallados
                    if vistos == objetivo:
                        elegida = (inicio, fin)
                        break
                if elegida:
                    break
            if elegida is None:
                # Si no cabe todo, preferir la línea con nombre de código.
                inicio, fin, _ = min(cubiertos, key=lambda c: (min(_prioridad(x) for x in c[2]), c[0]))
                elegida = (inicio, fin)
        if elegida is None and propia:
            elegida = self._tramo_citable(texto[:MAX_BUSQUEDA_CABECERA], propia)
        resultado = None
        if elegida:
            inicio, fin = elegida
            resultado = {"doc_id": doc_id, "inicio": inicio, "fin": fin, "texto": texto[inicio:fin],
                         "tipo_evidencia": "cabecera_fuente"}
        self._cabeceras_v04[doc_id] = resultado
        return resultado

    def _tramo_citable(self, texto, propia):
        """Tramo contiguo de líneas, el más corto, que nombra la norma completa.

        Evalúa el tramo completo y no cada línea por separado, porque la fuente
        parte la cita: "LEY 890" y "DE 2004" quedan en líneas seguidas y ninguna
        de las dos es reconocible por sí sola. Se prefiere el más corto para que
        salga el encabezado y no un párrafo que lo contenga de paso.
        """
        lineas = [(m.start(), m.end()) for m in re.finditer(r"[^\n]+", texto)]
        completo, parcial = None, None
        for i, (inicio, _) in enumerate(lineas):
            for _, fin in lineas[i:]:
                if fin - inicio > MAX_CABECERA:
                    break
                hallados = propia & self.citas.bodies(self.citas.extract(texto[inicio:fin]))
                if not hallados:
                    continue
                if hallados == propia:
                    if completo is None or fin - inicio < completo[1] - completo[0]:
                        completo = (inicio, fin)
                    break
                if parcial is None or fin - inicio < parcial[1] - parcial[0]:
                    parcial = (inicio, fin)
        return completo or parcial

    def cabecera_normalizada(self, doc_id):
        """Cita canónica del documento, tomada de los metadatos del manifiesto.

        Último recurso para las fuentes cuyo texto nunca se nombra a sí mismo en una
        forma que el extractor oficial reconozca: "LEY ESTATUTARIA 1581 DE 2012",
        "LEY 45 DE 5 DE MARZO DE 1936" o "LEY 1032" con la fecha en otra línea. No
        agrega prosa ni resume nada: es la referencia del propio documento, y va sin
        offsets porque no es un tramo literal de la fuente.
        """
        if doc_id in self._normalizadas:
            return self._normalizadas[doc_id]
        propia = self.identidad(doc_id)
        nombres = [nombre_cuerpo(c) for c in sorted(propia, key=lambda c: (_prioridad(c), str(c)))]
        canonico = "; ".join(n for n in nombres if n)
        # El título del manifiesto se lee mejor ("Sentencia C-030 de 2023" y no "C-30"),
        # así que se prefiere cuando el extractor lo reconoce como la misma norma.
        titulo = (self.documentos[doc_id].get("titulo") or "").strip()
        resultado = None
        for texto in (titulo, canonico):
            if texto and propia <= self.citas.bodies(self.citas.extract(texto)):
                resultado = {"doc_id": doc_id, "texto": texto, "tipo_evidencia": "cabecera_normalizada"}
                break
        self._normalizadas[doc_id] = resultado
        return resultado

    def reconstruir(self, pasajes_03, max_unidades=5):
        """Reutiliza las unidades recuperadas en el 03 y rehace sus cabeceras."""
        unidades = [p for p in pasajes_03 if p.get("tipo_evidencia") not in CABECERAS][:max_unidades]
        pasajes, avisos = [], []
        for unidad in unidades:
            if self.texto(unidad["doc_id"])[unidad["inicio"]:unidad["fin"]] != unidad["texto"]:
                raise ValueError("El artículo no conserva el texto literal")
            necesarios = self.identidad(unidad["doc_id"])
            en_texto = self.citas.bodies(self.citas.extract(unidad["texto"]))
            if not necesarios <= en_texto:
                cabecera = self.cabecera(unidad["doc_id"]) or self.cabecera_normalizada(unidad["doc_id"])
                if cabecera:
                    pasajes.append(dict(cabecera, grupo_evidencia=unidad.get("unidad_id"), score=unidad.get("score")))
                    if cabecera["tipo_evidencia"] == "cabecera_normalizada":
                        avisos.append({"doc_id": unidad["doc_id"], "motivo": "cabecera_normalizada_del_manifiesto"})
                else:
                    avisos.append({"doc_id": unidad["doc_id"], "motivo": "sin_cabecera_literal_reconocible"})
            pasajes.append(dict(unidad, tipo_evidencia="unidad"))
        if len(pasajes) > 10:
            raise ValueError("Se superaron los diez pasajes de evidencia")
        self.verificar(pasajes)
        return pasajes, avisos

    def respaldo(self, pasajes):
        """Citas que el evaluador considera respaldadas por estos pasajes."""
        return set().union(*(self.citas.bodies(self.citas.extract(str(p.get("texto") or "")))
                             for p in (pasajes or [])[:10])) if pasajes else set()

    def etiqueta(self, pasaje):
        """Nombre legible de la norma de un pasaje de tipo unidad, verificado contra el extractor."""
        respaldadas = self.identidad(pasaje["doc_id"])
        for cuerpo in sorted(respaldadas, key=lambda c: (_prioridad(c), str(c))):
            nombre = nombre_cuerpo(cuerpo)
            if nombre and cuerpo in self.citas.bodies(self.citas.extract(nombre)):
                return nombre, cuerpo
        doc = self.documentos[pasaje["doc_id"]]
        return doc.get("titulo") or pasaje["doc_id"], None

    def citas_evidencia(self, pasajes, limite=None, solo_cuerpos=None, incluir_primera=False):
        """Frase de fundamento construida solo con las unidades recuperadas.

        Devuelve el texto y los cuerpos que cita. Cada cuerpo se incluye solo si
        el extractor oficial lo encuentra también en los pasajes (sin respaldo = 0).
        """
        respaldo = self.respaldo(pasajes)
        grupos = OrderedDict()
        for pasaje in pasajes:
            if pasaje.get("tipo_evidencia") in CABECERAS:
                continue
            nombre, cuerpo = self.etiqueta(pasaje)
            if cuerpo is None or cuerpo not in respaldo:
                continue
            if solo_cuerpos is not None and cuerpo not in solo_cuerpos and not (incluir_primera and not grupos):
                continue
            articulos = grupos.setdefault((nombre, cuerpo), [])
            articulo = str(pasaje.get("articulo") or "").strip()
            numero = re.match(r"\d+[A-Za-z]?", articulo)
            if numero and numero.group() not in articulos:
                articulos.append(numero.group())
            if limite and len(grupos) >= limite:
                break
        partes = []
        for (nombre, cuerpo), articulos in grupos.items():
            if not articulos:
                partes.append(nombre)
            elif len(articulos) == 1:
                partes.append(f"{nombre}, artículo {articulos[0]}")
            else:
                partes.append(f"{nombre}, artículos {', '.join(articulos[:-1])} y {articulos[-1]}")
        texto = "; ".join(partes)
        cuerpos = self.citas.bodies(self.citas.extract(texto)) if texto else set()
        if cuerpos - respaldo:
            # Nunca debería ocurrir; si ocurre, no se agrega nada.
            return "", set()
        return texto, cuerpos
