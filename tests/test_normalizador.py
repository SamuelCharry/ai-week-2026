"""Normalizador de citas: años equivocados en la pregunta o las opciones (sin modelos)."""
import unittest

from legalrag.citations.normalizador import NormalizadorCitas

INVENTARIO = [{"tipo": "ley", "numero": "1564", "anio": 2012, "titulo": "Código General del Proceso"},
              {"tipo": "ley", "numero": "906", "anio": 2004, "titulo": "Código de Procedimiento Penal"},
              {"tipo": "decreto", "numero": "100", "anio": 1980, "titulo": "Decreto 100 de 1980"},
              {"tipo": "decreto", "numero": "100", "anio": 2001, "titulo": "Decreto 100 de 2001"}]


class Normalizador(unittest.TestCase):
    def setUp(self):
        self.n = NormalizadorCitas(INVENTARIO)

    def test_pregunta_58(self):
        nota = self.n.nota({"pregunta": "¿Qué normativa regula las actuaciones jurisdiccionales ante la SIC?",
                            "opciones": {"A": "Ley 1564 de 2002", "D": "Ley 906 de 2004"}})
        self.assertIn("Opción A: «Ley 1564 de 2002» no existe con ese año; la Ley 1564 es de 2012 "
                      "(Código General del Proceso)", nota)
        self.assertNotIn("Opción D", nota)  # la Ley 906 de 2004 existe tal cual

    def test_sin_aviso_si_esta_bien_no_esta_o_es_ambigua(self):
        self.assertIsNone(self.n.nota({"pregunta": "Según la Ley 1564 de 2012", "opciones": {}}))
        self.assertIsNone(self.n.nota({"pregunta": "Según la Ley 9999 de 2020", "opciones": {}}))
        # Dos decretos 100 (1980 y 2001): no se adivina a cuál se refiere.
        self.assertIsNone(self.n.nota({"pregunta": "Según el Decreto 100 de 1999", "opciones": {}}))


if __name__ == "__main__":
    unittest.main()
