"""Pasos extra del agente: expansión con evidencia débil, reintento de JSON y verificación (sin modelos)."""
import unittest
from unittest import mock

from legalrag.agent.componentes import Sistema, gravedad
from legalrag.generation import pasos


def pasaje(doc_id, tipo="ley", score=3.0):
    return {"doc_id": doc_id, "tipo": tipo, "score": score, "texto": "texto", "inicio": 0, "fin": 5}


class Recuperador:
    def __init__(self, primera, expandida):
        self.primera, self.expandida, self.expansiones = primera, expandida, []
        self.evidencia = None

    def buscar(self, entrada, expansion=None):
        self.expansiones.append(expansion)
        return self.expandida if expansion else self.primera


class Decoder:
    def __init__(self, salidas=()):
        self.salidas, self.llamadas = list(salidas), []

    def redactar(self, mensajes, max_nuevos):
        self.llamadas.append(("redactar", mensajes[-1]["content"][:40]))
        return "1. ¿Qué norma regula la acción popular?\nLey 472 de 1998, artículo 2."

    def seleccionar(self, entrada, pasajes, evidencia):
        return pasajes

    def generar(self, entrada, pasajes, evidencia, prefijo="{", repetition_penalty=None, extra=None):
        self.llamadas.append(("generar", repetition_penalty, extra is not None))
        return self.salidas.pop(0)


def sistema(config, recuperador, decoder):
    s = Sistema.__new__(Sistema)
    s.config, s.recuperador, s.decoder, s.validador = config, recuperador, decoder, None
    s.ultimo_problema = s.ultimo_registro = s.ultima_expansion = None
    return s


ABIERTA = {"id": 1, "formato": "open_ended", "pregunta": "¿Procede la acción popular?", "area": "Constitucional"}
CERRADA = {"id": 2, "formato": "multiple_choice", "pregunta": "¿Cuál?", "opciones": {"A": "x", "B": "y"}}


class EvidenciaDebil(unittest.TestCase):
    def test_puntaje_bajo_o_sin_normas(self):
        self.assertEqual(pasos.evidencia_debil([], 0.0), (True, "sin_pasajes"))
        self.assertTrue(pasos.evidencia_debil([pasaje("a", score=-1.2)], 0.0)[0])
        self.assertTrue(pasos.evidencia_debil([pasaje("a", tipo="sentencia")] * 3 + [pasaje("b")], 0.0)[0])
        self.assertEqual(pasos.evidencia_debil([pasaje("a", tipo="sentencia"), pasaje("b")], 0.0), (False, None))
        # Sin umbral solo cuenta que haya normas entre los primeros.
        self.assertFalse(pasos.evidencia_debil([pasaje("a", score=-5)], None)[0])

    def test_preguntas_de(self):
        texto = "Preguntas:\n1. ¿Qué artículo aplica?\n2) ¿Cuál es el plazo?\n- ¿Quién decide?\n4. ¿Otra más?"
        self.assertEqual(pasos.preguntas_de(texto, 3), ["¿Qué artículo aplica?", "¿Cuál es el plazo?", "¿Quién decide?"])


class Expansion(unittest.TestCase):
    def config(self, modo):
        return {"recuperacion": {"expansion": modo, "umbral_evidencia_debil": 0.0}, "generacion": {}}

    def test_solo_con_evidencia_debil(self):
        fuerte = [pasaje("ley_472_1998", score=4.0)]
        r, d = Recuperador(fuerte, [pasaje("otra")]), Decoder()
        self.assertEqual(sistema(self.config("debil"), r, d).recuperar(ABIERTA), fuerte)
        self.assertEqual(r.expansiones, [None])

        debil, nuevos = [pasaje("sentencia_x", tipo="sentencia", score=-2.0)], [pasaje("ley_472_1998")]
        r, d = Recuperador(debil, nuevos), Decoder()
        s = sistema(self.config("debil"), r, d)
        self.assertEqual(s.recuperar(ABIERTA), nuevos)
        self.assertTrue(r.expansiones[1])  # se buscó con la hipótesis del decoder
        self.assertEqual(s.ultima_expansion["docs_despues"], ["ley_472_1998"])

    def test_siempre_y_cerradas(self):
        fuerte = [pasaje("ley_472_1998", score=4.0)]
        r = Recuperador(fuerte, [pasaje("otra")])
        sistema(self.config("siempre"), r, Decoder()).recuperar(ABIERTA)
        self.assertEqual(len(r.expansiones), 2)
        r = Recuperador([], [pasaje("otra")])
        sistema(self.config("siempre"), r, Decoder()).recuperar(CERRADA)
        self.assertEqual(r.expansiones, [None])  # las cerradas conservan su evidencia


def final_falso(problemas):
    """respuesta_final que devuelve el problema según el texto crudo."""
    def final(entrada, crudo, pasajes, evidencia, politica, validador):
        return {"id": entrada["id"], "formato": entrada["formato"], "texto": crudo}, {"problema": problemas[crudo]}
    return final


class ReintentoYVerificacion(unittest.TestCase):
    CONFIG = {"recuperacion": {}, "generacion": {"politica": {}, "regenerar_json": True,
                                                 "repetition_penalty_reintento": 1.3}}

    def responder(self, salidas, problemas, config=None):
        s = sistema(config or self.CONFIG, Recuperador([], []), Decoder(salidas))
        s.recuperador.evidencia = None
        with mock.patch("legalrag.citations.verificacion.respuesta_final", final_falso(problemas)):
            respuesta = s.responder(ABIERTA, [pasaje("ley_472_1998")])
        return respuesta, s

    def test_reintento_reemplaza_json_invalido(self):
        respuesta, s = self.responder(["roto", "bueno"], {"roto": "json_invalido", "bueno": None})
        self.assertEqual(respuesta["texto"], "bueno")
        self.assertEqual(s.decoder.llamadas[-1], ("generar", 1.3, False))
        self.assertEqual(s.ultimo_registro["reintento_json"], {"problema_antes": "json_invalido", "problema_despues": None})

    def test_reintento_peor_se_descarta(self):
        respuesta, _ = self.responder(["incompleto", "roto"], {"incompleto": "campos_rellenados", "roto": "json_invalido"})
        self.assertEqual(respuesta["texto"], "incompleto")

    def test_sin_problema_no_reintenta(self):
        _, s = self.responder(["bueno"], {"bueno": "citas_saneadas"})
        self.assertEqual(len(s.decoder.llamadas), 1)

    def test_verificacion_genera_con_el_bloque(self):
        config = {"recuperacion": {}, "generacion": {"politica": {}, "verificar": True}}
        with mock.patch("legalrag.citations.verificacion.texto_citable", lambda r: r["texto"]), \
                mock.patch("legalrag.generation.politica.bloque_pasajes", lambda *a: "[1] Ley 472 de 1998"):
            respuesta, s = self.responder(["borrador", "verificada"], {"borrador": None, "verificada": None}, config)
        self.assertEqual(respuesta["texto"], "verificada")
        self.assertEqual([l[0] for l in s.decoder.llamadas], ["generar", "redactar", "redactar", "generar"])
        self.assertTrue(s.decoder.llamadas[-1][2])  # la segunda generación lleva las verificaciones
        self.assertTrue(s.ultimo_registro["verificacion"]["aceptada"])

    def test_gravedad(self):
        self.assertGreater(gravedad("esquema_oficial: falta campo"), gravedad("json_invalido"))
        self.assertGreater(gravedad("json_invalido"), gravedad("campos_rellenados"))
        self.assertEqual(gravedad(None), gravedad("citas_saneadas"))


if __name__ == "__main__":
    unittest.main()
