"""Ventanas R03: presupuesto de tokens con cabecera, solapamiento y cobertura del padre."""
import re
import unittest

from legalrag.indexing.r03 import cabecera, ventanas


class TokenizadorPalabras:
    """Un token por palabra; [CLS] y [SEP] cuando add_special_tokens."""

    def __call__(self, texto, add_special_tokens=True, return_offsets_mapping=False):
        offsets = [m.span() for m in re.finditer(r"\S+", texto)]
        salida = {"input_ids": ([0] if add_special_tokens else []) + [1] * len(offsets)
                  + ([2] if add_special_tokens else [])}
        if return_offsets_mapping:
            salida["offset_mapping"] = offsets
        return salida


class VentanasTest(unittest.TestCase):
    tok = TokenizadorPalabras()

    def test_presupuesto_solapamiento_y_cobertura(self):
        texto = "xx " + " ".join(f"p{i}" for i in range(100)) + " yy"
        inicio, fin = 3, len(texto) - 3
        cortes = list(ventanas(texto, inicio, fin, self.tok, "Ley 1 | Artículo 2", tamano=24, solapamiento=5))
        for a, b in cortes:
            self.assertLessEqual(len(self.tok("Ley 1 | Artículo 2\n" + texto[a:b])["input_ids"]), 24)
            self.assertTrue(inicio <= a < b <= fin)
        self.assertEqual(cortes[0][0], inicio)
        self.assertEqual(cortes[-1][1], fin)
        for (a1, b1), (a2, b2) in zip(cortes, cortes[1:]):
            self.assertEqual(len(texto[a2:b1].split()), 5)  # 5 palabras compartidas

    def test_cabecera_larga_se_recorta_y_sin_espacio_falla(self):
        documento = {"titulo": " ".join(["palabra"] * 50)}
        self.assertEqual(len(cabecera(documento, {"articulo": "3"}, self.tok, 10).split()), 10)
        with self.assertRaises(ValueError):
            list(ventanas("a b c", 0, 5, self.tok, "t " * 20, tamano=10, solapamiento=2))


if __name__ == "__main__":
    unittest.main()
