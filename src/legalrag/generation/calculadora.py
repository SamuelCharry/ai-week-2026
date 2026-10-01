"""Agente calculadora: convierte los montos de la pregunta a salarios mínimos y UVT, sin modelo.

Un modelo de 8B no convierte de forma confiable "30.000.000 COP" a salarios mínimos: en la pregunta 528
el artículo 25 del CGP (mínima cuantía hasta 40 SMMLV) llegó a los pasajes y aun así eligió "mayor
cuantía" con total seguridad. Como en PAL (Gao et al., ICML 2023) y Toolformer (Schick et al., NeurIPS
2023), la cuenta la hace una herramienta y el modelo razona con el resultado.

Los valores salen del corpus, no de una tabla escrita a mano: los decretos del salario mínimo y las
resoluciones de la UVT que agregó ingestion.agregar_puntuales, leídos en su artículo 1 (en los
considerandos aparecen los valores de años anteriores). Si la pregunta trae un monto en pesos, se agrega
al prompt una línea como

    30.000.000 pesos = 17,1 SMMLV de 2026 (Decreto 159 de 2026: $1.750.905) = 21,1 SMMLV de 2025 (...)
"""
import re

_TITULOS = {"smmlv": re.compile(r"Salario mínimo mensual legal .*?para el año (\d{4})", re.I),
            "uvt": re.compile(r"Unidad de Valor Tributario \(UVT\) para el año (\d{4})", re.I)}
_RANGOS = {"smmlv": (300_000, 10_000_000), "uvt": (10_000, 500_000)}
_ARTICULO_1 = re.compile(r"art[íi]culo\s*(?:1|primero)\b", re.I)
_PESOS = re.compile(r"\$\s*(\d{1,3}(?:\.\d{3})+)")
_MONTOS = [re.compile(r"\$\s*(\d{1,3}(?:[.,]\d{3})+)"),
           re.compile(r"(\d{1,3}(?:\.\d{3})+)\s*(?:COP|pesos|de pesos)", re.I),
           re.compile(r"(\d+(?:[.,]\d+)?)\s*millones", re.I)]


def _entero(texto):
    return int(re.sub(r"[.,]", "", texto))


def valor_del_articulo_1(texto, tipo):
    """Primer monto en pesos dentro del rango del tipo, después del artículo 1; None si no lo hay."""
    minimo, maximo = _RANGOS[tipo]
    inicio = _ARTICULO_1.search(texto)
    for m in _PESOS.finditer(texto, inicio.end() if inicio else 0):
        valor = _entero(m.group(1))
        if minimo <= valor <= maximo:
            return valor
    return None


def montos(texto, minimo=100_000):
    """Montos en pesos que menciona un texto ("$30.000.000", "30.000.000 COP", "30 millones")."""
    encontrados = []
    for patron in _MONTOS:
        for m in patron.finditer(texto or ""):
            numero = m.group(1)
            if "millones" in m.group(0).lower():
                valor = int(float(numero.replace(",", ".")) * 1_000_000)
            else:
                valor = _entero(numero)
            if valor >= minimo and valor not in encontrados:
                encontrados.append(valor)
    return encontrados[:3]


def _formato(numero, decimales=0):
    texto = f"{numero:,.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


class Calculadora:
    def __init__(self, valores):
        """valores: {"smmlv"|"uvt": {año: (valor, nombre de la norma)}}."""
        self.valores = valores

    @classmethod
    def desde_corpus(cls, documentos, leer_texto):
        """documentos: registros del inventario; leer_texto(doc_id) -> texto canónico."""
        from legalrag.citations.normas import nombre_numerado

        valores = {"smmlv": {}, "uvt": {}}
        for doc in documentos:
            for tipo, patron in _TITULOS.items():
                m = patron.search(doc.get("titulo") or "")
                if not m:
                    continue
                anio = int(m.group(1))
                # Si un año tiene dos normas (2026: el Decreto 1469 suspendido y el transitorio 0159), manda la vigente.
                if anio in valores[tipo] and "suspendido" in (doc.get("titulo") or "").lower():
                    continue
                valor = valor_del_articulo_1(leer_texto(doc["doc_id"]), tipo)
                if valor:
                    valores[tipo][anio] = (valor, nombre_numerado(doc) or doc["titulo"])
        return cls(valores)

    def nota(self, entrada, anios=2):
        """Bloque para el prompt con las conversiones, o None si la pregunta no trae montos."""
        texto = " ".join([entrada.get("pregunta") or "", *(entrada.get("opciones") or {}).values()])
        lineas = []
        for monto in montos(texto):
            partes = []
            for tipo, unidad in (("smmlv", "SMMLV"), ("uvt", "UVT")):
                for anio in sorted(self.valores[tipo], reverse=True)[:anios]:
                    valor, norma = self.valores[tipo][anio]
                    partes.append(f"{_formato(monto / valor, 1)} {unidad} de {anio} ({norma}: ${_formato(valor)})")
            if partes:
                lineas.append(f"- {_formato(monto)} pesos = " + " = ".join(partes))
        if not lineas:
            return None
        return ("VALORES DE REFERENCIA (conversiones calculadas con los decretos del salario mínimo y las resoluciones "
                "de la UVT del corpus; úsalas si la respuesta depende de cuantías o topes en salarios mínimos o UVT)\n"
                + "\n".join(lineas))
