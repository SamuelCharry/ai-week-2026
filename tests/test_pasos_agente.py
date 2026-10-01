"""Cerberus: reformulador, herramientas, reintento de JSON y verificador NLI (sin modelos)."""
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
    s.calculadora = s.normalizador = None
    return s


ABIERTA = {"id": 1, "formato": "open_ended", "pregunta": "¿Procede la acción popular?", "area": "Constitucional"}
CERRADA = {"id": 2, "formato": "multiple_choice", "pregunta": "¿Cuál?", "opciones": {"A": "x", "B": "y"}}


class Expansion(unittest.TestCase):
    def config(self, modo):
        return {"recuperacion": {"expansion": modo, "umbral_evidencia_debil": 0.0}, "generacion": {}}

    def test_reformulador_iterativo_lee_los_primeros_pasajes(self):
        r, d = Recuperador([pasaje("sentencia_x", tipo="sentencia")], [pasaje("ley_472_1998")]), Decoder()
        vistos = []
        d.redactar = lambda mensajes, n: vistos.append(mensajes[-1]["content"]) or "Acción popular, Ley 472 de 1998."
        config = {"recuperacion": {"expansion": "siempre", "agente_expansion": "iterativo"}, "generacion": {}}
        with mock.patch("legalrag.generation.politica.bloque_pasajes", lambda p, e, n: "[1] Sentencia X\ntexto"):
            sistema(config, r, d).recuperar(ABIERTA)
        self.assertIn("PASAJES DE UNA PRIMERA BÚSQUEDA\n[1] Sentencia X", vistos[0])
        self.assertEqual(r.expansiones[1], "Acción popular, Ley 472 de 1998.")

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


class ReintentoYHerramientas(unittest.TestCase):
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

    def test_razona_antes_de_la_letra_si_hay_nota_de_herramienta(self):
        class Calculadora:
            @staticmethod
            def nota(entrada):
                return "VALORES DE REFERENCIA\n- 30.000.000 pesos = 17,1 SMMLV"

        class Decoder2(Decoder):
            def probabilidades_letras(self, entrada, pasajes, evidencia, prefijo=None, **otros):
                self.llamadas.append(("letras", prefijo))
                return {"A": 0.2, "B": 0.8}

        config = {"recuperacion": {}, "generacion": {"politica": {}, "letra_por_probabilidad": True,
                                                     "razonar_con_herramienta": True}}
        s = sistema(config, Recuperador([], []), Decoder2(['{"justificacion": "17 SMMLV es mínima cuantía"}']))
        s.calculadora = Calculadora()
        cerrada = {**CERRADA, "opciones": {"A": "x", "B": "y"}}

        def final(entrada, crudo, pasajes, evidencia, politica, validador):
            return ({"id": 2, "formato": "multiple_choice", "respuesta_correcta": "A",
                     "descarte_opciones": {"A": "-", "B": "-"}}, {"problema": None})

        with mock.patch("legalrag.citations.verificacion.respuesta_final", final):
            s.responder(cerrada, [pasaje("ley_1564_2012")])
        primera, segunda = s.decoder.llamadas[0], s.decoder.llamadas[1]
        self.assertEqual(primera[0], "generar")  # primero razona…
        self.assertEqual(segunda[0], "letras")   # …y después elige la letra con el razonamiento escrito
        self.assertIn("17 SMMLV", segunda[1])

    def test_gravedad(self):
        self.assertGreater(gravedad("esquema_oficial: falta campo"), gravedad("json_invalido"))
        self.assertGreater(gravedad("json_invalido"), gravedad("campos_rellenados"))
        self.assertEqual(gravedad(None), gravedad("citas_saneadas"))


class Fragmentos:
    def __init__(self):
        self.consultas = []

    def bm25(self, texto, k, doc_ids=None):
        self.consultas.append(texto)
        return [(1, 1.0)]

    def filas(self, ids):
        return []


class ReformuladorEnRecuperacion(unittest.TestCase):
    def ranking(self, entrada, **config):
        from legalrag.retrieval.hibrido import RecuperadorHibrido

        r = RecuperadorHibrido(".", {"bm25_top": 5, "denso_top": 5, "rrf_k": 60, "rerank_top": 5, "usar_denso": False,
                                     "usar_reranker": False, **config})
        r.fragmentos = Fragmentos()
        r.ranking(entrada)
        return r.fragmentos.consultas

    def test_articulos_del_reformulador_entran_como_candidatos(self):
        from legalrag.retrieval.hibrido import RecuperadorHibrido

        class Evidencia:
            def normas_de(self, texto):
                return {"ley_472_1998": {"2"}} if "472" in texto else {}

        class ConArticulos(Fragmentos):
            def por_articulo(self, doc_id, articulo, k=3):
                return [900]

        r = RecuperadorHibrido(".", {"bm25_top": 5, "denso_top": 5, "rrf_k": 60, "rerank_top": 5, "usar_denso": False,
                                     "usar_reranker": False, "expansion_articulos": True})
        r.fragmentos, r.evidencia = ConArticulos(), Evidencia()
        ranking = r.ranking(ABIERTA, expansion="Ley 472 de 1998, artículo 2")
        self.assertIn(900, [f for f, _ in ranking])
        r.config["expansion_articulos"] = False
        self.assertNotIn(900, [f for f, _ in r.ranking(ABIERTA, expansion="Ley 472 de 1998, artículo 2")])

    def test_norma_del_reformulador_reservada_en_la_evidencia(self):
        from legalrag.retrieval.hibrido import RecuperadorHibrido

        class Evidencia:
            def normas_de(self, texto):
                return {"ley_472_1998": set()} if "472" in texto else {}

        class ConDocs(Fragmentos):
            def bm25(self, texto, k, doc_ids=None):
                return [(900, 1.0)] if doc_ids else [(1, 3.0), (2, 2.0), (3, 1.0)]

            def filas(self, ids):
                return [{"id": i, "doc_id": "ley_472_1998" if i == 900 else f"d{i}"} for i in ids]

        r = RecuperadorHibrido(".", {"bm25_top": 5, "denso_top": 5, "rrf_k": 60, "rerank_top": 5, "usar_denso": False,
                                     "usar_reranker": False, "reservar_expansion": 2})
        r.fragmentos, r.evidencia = ConDocs(), Evidencia()
        r.ranking(ABIERTA, expansion="Ley 472 de 1998, artículo 2")
        self.assertEqual(r.ultima_traza["reservados"], [900])
        r.config["reservar_expansion"] = 0
        r.ranking(ABIERTA, expansion="Ley 472 de 1998, artículo 2")
        self.assertEqual(r.ultima_traza["reservados"], [])

class VerificadorNLIFalso(unittest.TestCase):
    def verificador(self, modo, puntajes):
        from legalrag.citations.respaldo_nli import VerificadorNLI

        v = VerificadorNLI({"modo": modo, "umbral_implica": 0.5, "umbral_contradice": 0.9})
        v.puntuar = lambda oraciones, premisas: puntajes[:len(oraciones)]
        return v

    class Citas:
        @staticmethod
        def extract(texto):
            return {("ley", "472", "1998")} if "Ley 472" in texto else set()

    def respuesta(self):
        return {"formato": "semi_open", "respuesta": "Procede la acción popular según la Ley 472 de 1998. "
                "El plazo es de cinco años. La acción la conoce el juez civil.",
                "palabras_clave": ["acción popular"], "referencia_legal": "Ley 472 de 1998"}

    def test_registrar_no_cambia_la_respuesta(self):
        r = self.respuesta()
        v = self.verificador("registrar", [(0.9, 0.0, 1), (0.1, 0.95, 2), (0.2, 0.1, 3)])
        registro = v.verificar(r, [pasaje("ley_472_1998")], self.Citas)
        self.assertEqual(r, self.respuesta())
        self.assertEqual((registro["oraciones"], registro["respaldadas"], registro["contradichas"]), (3, 1, 1))

    def test_quita_contradichas_sin_tocar_citas(self):
        r = self.respuesta()
        # La primera oración (con cita) también sale contradicha, pero las citas no se tocan.
        v = self.verificador("quitar_contradichas", [(0.1, 0.99, 1), (0.1, 0.95, 2), (0.2, 0.1, 3)])
        registro = v.verificar(r, [pasaje("ley_472_1998")], self.Citas)
        self.assertEqual(r["respuesta"], "Procede la acción popular según la Ley 472 de 1998. "
                                         "La acción la conoce el juez civil.")
        self.assertEqual(registro["quitadas"], ["El plazo es de cinco años."])


if __name__ == "__main__":
    unittest.main()
