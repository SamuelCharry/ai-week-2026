"""Agente normalizador de citas: avisa cuando la pregunta o una opción cita una norma con el año equivocado.

El banco trae errores de digitación en las normas: la respuesta correcta de la pregunta 58 es «Ley 1564 de
2002» (la Ley 1564 es de 2012, el Código General del Proceso), y la lista oficial de normas del banco
(seed_targets.json) incluye «Ley 1150 de 2005» (es de 2007) o «Ley 964 de 2006» (es de 2005). El modelo
descarta la opción porque el año no cuadra con la evidencia. Si una ley o decreto citado no existe en el
corpus con ese año pero sí con el mismo número y otro año, se agrega al prompt una nota como

    Opción A: «Ley 1564 de 2002» no está con ese año; la Ley 1564 es de 2012 (Código General del Proceso).
    Puede ser un error de digitación de esa misma norma.

Es una herramienta determinista sobre el inventario; no usa modelo y no cambia la evidencia.
"""
import re
from collections import defaultdict

_CITA = re.compile(r"\b(ley|decreto)\s+(?:no\.?\s*)?(\d{1,5})\s+de\s+(\d{4})\b", re.I)


class NormalizadorCitas:
    def __init__(self, documentos):
        """documentos: registros del inventario (tipo, numero, anio, titulo)."""
        self.por_numero = defaultdict(list)
        for doc in documentos:
            tipo = str(doc.get("tipo") or "").lower()
            if tipo in ("ley", "decreto") and doc.get("numero") and doc.get("anio"):
                numero = str(doc["numero"]).lstrip("0") or "0"
                self.por_numero[(tipo, numero)].append((int(doc["anio"]), doc.get("titulo") or ""))

    def avisos(self, texto):
        """[(cita escrita, año real, título)] para las citas cuyo año no existe pero el número sí."""
        resultado = []
        for m in _CITA.finditer(texto or ""):
            tipo, numero, anio = m.group(1).lower(), m.group(2).lstrip("0") or "0", int(m.group(3))
            existentes = self.por_numero.get((tipo, numero), [])
            if not existentes or any(a == anio for a, _ in existentes):
                continue
            anios = {a for a, _ in existentes}
            if len(anios) == 1:  # solo si no hay ambigüedad sobre a qué norma se refiere
                real, titulo = existentes[0]
                resultado.append((m.group(0), real, titulo))
        return resultado

    def nota(self, entrada):
        lineas = []
        partes = [("La pregunta", entrada.get("pregunta") or "")] + \
                 [(f"Opción {letra}", texto) for letra, texto in (entrada.get("opciones") or {}).items()]
        for donde, texto in partes:
            for escrita, real, titulo in self.avisos(texto):
                tipo, numero = escrita.split()[0].capitalize(), re.search(r"\d+", escrita).group()
                nombre = f" ({titulo})" if titulo and not titulo.lower().startswith(tipo.lower()) else ""
                lineas.append(f"- {donde}: «{escrita}» no existe con ese año; la {tipo} {numero} es de {real}{nombre}. "
                              "Puede ser un error de digitación de esa misma norma.")
        if not lineas:
            return None
        return "NOTA SOBRE LAS NORMAS CITADAS (comprobado contra el inventario del corpus)\n" + "\n".join(lineas)
