"""Permutaciones y descarte: se corrige el sesgo por posición y se decide entre finalistas."""
import unittest

from legalrag.generation.eleccion import elegir, promedio_por_contenido, rotaciones

OPCIONES = {"A": "diez días", "B": "veinte días", "C": "tres días", "D": "un mes"}


def modelo_sesgado(opciones):
    """Prefiere 'veinte días' por contenido, pero le suma mucho a la letra A por posición."""
    base = {letra: (2.0 if texto == "veinte días" else 1.0 if texto == "un mes" else 0.5) for letra, texto in opciones.items()}
    base[list(opciones)[0]] += 2.2  # sesgo de posición
    total = sum(base.values())
    return {letra: valor / total for letra, valor in base.items()}


class EleccionTest(unittest.TestCase):
    def test_rotaciones(self):
        self.assertEqual(rotaciones(3), [[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    def test_sin_permutar_gana_el_sesgo(self):
        letra, _ = elegir(modelo_sesgado, OPCIONES, permutar=False, mantener=0)
        self.assertEqual(letra, "A")

    def test_permutar_corrige_el_sesgo(self):
        letra, registro = elegir(modelo_sesgado, OPCIONES, permutar=True, mantener=0)
        self.assertEqual(letra, "B")
        self.assertEqual(len(registro["rotaciones"]), 4)
        self.assertAlmostEqual(sum(registro["promedio"].values()), 1.0)

    def test_descarte_deja_dos_finalistas_y_elige_entre_ellas(self):
        vistas = []

        def puntuar(opciones):
            vistas.append(dict(opciones))
            return modelo_sesgado(opciones)

        letra, registro = elegir(puntuar, OPCIONES, permutar=True, mantener=2)
        self.assertEqual(letra, "B")
        self.assertEqual(registro["finalistas"], ["B", "D"])
        self.assertEqual(sorted(registro["descartadas"]), ["A", "C"])
        # La segunda ronda muestra solo las finalistas, como A y B, en sus 2 rotaciones.
        self.assertEqual(vistas[-2:], [{"A": "veinte días", "B": "un mes"}, {"A": "un mes", "B": "veinte días"}])

    def test_promedio_por_contenido_devuelve_letras_originales(self):
        promedio, _ = promedio_por_contenido(modelo_sesgado, OPCIONES)
        self.assertEqual(set(promedio), set(OPCIONES))


if __name__ == "__main__":
    unittest.main()
