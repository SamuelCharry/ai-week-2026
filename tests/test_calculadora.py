"""Agente calculadora: valores del artículo 1 de cada decreto y conversión de montos (sin modelos)."""
import unittest

from legalrag.generation.calculadora import Calculadora, montos, valor_del_articulo_1

DECRETO_2026 = """DECRETO 0159 DE 2026
CONSIDERANDO: Que el salario mínimo de 2025 fue fijado en un millón cuatrocientos veintitrés mil quinientos
pesos ($1.423.500) mediante el Decreto 1572 de 2024.
DECRETA:
ARTÍCULO 1. Fijar transitoriamente como salario mínimo mensual legal la suma de un millón setecientos
cincuenta mil novecientos cinco pesos ($1.750.905).
ARTÍCULO 2. Vigencia."""

RESOLUCION_UVT = """RESOLUCIÓN 000193
Artículo 1. Valor de la UVT. Fijar en cuarenta y nueve mil setecientos noventa y nueve pesos ($49.799) el valor
de la Unidad de Valor Tributario."""


class Calculo(unittest.TestCase):
    def test_valor_del_articulo_1_y_no_de_los_considerandos(self):
        self.assertEqual(valor_del_articulo_1(DECRETO_2026, "smmlv"), 1_750_905)
        self.assertEqual(valor_del_articulo_1(RESOLUCION_UVT, "uvt"), 49_799)
        self.assertIsNone(valor_del_articulo_1("ARTÍCULO 1. Sin valores.", "smmlv"))

    def test_montos(self):
        self.assertEqual(montos("pretensiones por 30.000.000 COP"), [30_000_000])
        self.assertEqual(montos("una multa de $1.500.000 o de 2,5 millones de pesos en 2024"), [1_500_000, 2_500_000])
        self.assertEqual(montos("artículo 2.2.3 del decreto, año 2024, 500 pesos"), [])

    def test_nota(self):
        documentos = [{"doc_id": "decreto_159_2026", "titulo": "Salario mínimo mensual legal fijado transitoriamente "
                       "para el año 2026", "tipo": "decreto", "numero": "159", "anio": 2026},
                      {"doc_id": "resolucion_dian_0193_2024", "titulo": "Valor de la Unidad de Valor Tributario (UVT) "
                       "para el año 2025", "tipo": "resolucion", "numero": "193", "anio": 2024}]
        textos = {"decreto_159_2026": DECRETO_2026, "resolucion_dian_0193_2024": RESOLUCION_UVT}
        calculadora = Calculadora.desde_corpus(documentos, textos.get)
        nota = calculadora.nota({"pregunta": "Pretensiones por 30.000.000 COP ¿qué cuantía?", "opciones": {}})
        self.assertIn("30.000.000 pesos = 17,1 SMMLV de 2026 (Decreto 159 de 2026: $1.750.905)", nota)
        self.assertIn("602,4 UVT de 2025 (Resolución 193 de 2024: $49.799)", nota)
        self.assertIsNone(calculadora.nota({"pregunta": "¿Qué es la tutela?", "opciones": {}}))


if __name__ == "__main__":
    unittest.main()
