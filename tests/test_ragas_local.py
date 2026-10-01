"""Aproximación local de RAGAS: misma fórmula de answer correctness que el evaluador (sin modelos)."""
import unittest

from legalrag.evaluation.ragas_local import RagasLocal, texto_ragas


class Falso(RagasLocal):
    def __init__(self, similitud, implicadas):
        self.similitud, self.implicadas = similitud, implicadas

        class NLI:
            @staticmethod
            def puntuar(oraciones, premisas, s=self):
                return [(1.0 if o in s.implicadas else 0.0, 0.0, 1) for o in oraciones]

        self.nli = NLI()

    def _vector(self, texto):
        return Vector(self.similitud)


class Vector:
    def __init__(self, valor):
        self.valor = valor

    def __matmul__(self, otro):
        return self.valor ** 0.5 * otro.valor ** 0.5


class Formula(unittest.TestCase):
    def test_answer_correctness(self):
        respuesta = "La tutela procede contra particulares. El plazo es de diez días hábiles. Lo decide el juez civil."
        esperada = "La tutela procede contra particulares. Debe resolverse en diez días hábiles."
        juez = Falso(0.8, {"La tutela procede contra particulares.", "El plazo es de diez días hábiles.",
                           "Debe resolverse en diez días hábiles."})
        puntaje, d = juez.puntuar(respuesta, esperada)
        # TP=2 (dos oraciones propias implicadas), FP=1 (la del juez civil), FN=0 (la esperada queda cubierta).
        self.assertEqual((d["tp"], d["fp"], d["fn"]), (2, 1, 0))
        self.assertAlmostEqual(puntaje, 0.75 * (2 / (2 + 0.5)) + 0.25 * 0.8)

    def test_abstencion_vale_cero_y_texto_por_formato(self):
        juez = Falso(0.9, set())
        muestra = [{"id": 1, "formato": "semi_open", "respuesta_esperada": "x"},
                   {"id": 2, "formato": "multiple_choice"}]
        self.assertEqual(juez.evaluar({1: {"abstencion": True}}, muestra)["correctness_aprox"], 0.0)
        self.assertEqual(texto_ragas({"formato": "semi_open", "respuesta": "a", "referencia_legal": "b"}), "a")
        self.assertEqual(texto_ragas({"formato": "open_ended", "marco_normativo": "a", "conclusion": "b"}), "a   b")


if __name__ == "__main__":
    unittest.main()
