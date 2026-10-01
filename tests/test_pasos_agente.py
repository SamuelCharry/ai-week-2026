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


class Calibracion(unittest.TestCase):
    def test_criterio_puntaje_ignora_normas(self):
        sin_normas = [pasaje("s", tipo="sentencia", score=2.0)] * 3
        self.assertTrue(pasos.evidencia_debil(sin_normas, 0.0, criterio="ambos")[0])
        self.assertFalse(pasos.evidencia_debil(sin_normas, 0.0, criterio="puntaje")[0])
        self.assertTrue(pasos.evidencia_debil([pasaje("s", score=-1.0)], 0.0, criterio="puntaje")[0])

    def test_no_expande_si_la_pregunta_nombra_una_norma(self):
        config = {"recuperacion": {"expansion": "debil", "umbral_evidencia_debil": 0.0,
                                   "criterio_evidencia_debil": "puntaje", "expansion_sin_norma_nombrada": True},
                  "generacion": {}}
        r = Recuperador([pasaje("ley_472_1998", score=-3.0)], [pasaje("otra")])
        r.ultima_traza = {"normas_nombradas": {"ley_472_1998": [2]}}
        s = sistema(config, r, Decoder())
        s.recuperar(ABIERTA)
        self.assertEqual(r.expansiones, [None])
        self.assertEqual(s.ultima_expansion["motivo"], "norma_nombrada")


class Fragmentos:
    def __init__(self):
        self.consultas = []

    def bm25(self, texto, k, doc_ids=None):
        self.consultas.append(texto)
        return [(1, 1.0)]

    def filas(self, ids):
        return []


class PorOpcion(unittest.TestCase):
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

    def test_una_busqueda_por_opcion_en_cerradas(self):
        cerrada = {**CERRADA, "pregunta": "¿Qué vicio configura?", "opciones": {"A": "Falsa motivación", "B": "Usurpación"}}
        consultas = self.ranking(cerrada, recuperar_por_opcion=True)
        self.assertEqual(len(consultas), 3)
        self.assertIn("¿Qué vicio configura?\nFalsa motivación", consultas)
        self.assertEqual(len(self.ranking(cerrada)), 1)
        self.assertEqual(len(self.ranking(ABIERTA, recuperar_por_opcion=True)), 1)


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


class Multiagente(unittest.TestCase):
    def test_orden_y_filtro_del_juez(self):
        p = [pasaje(f"d{i}") for i in range(4)]
        ordenados, aceptados = pasos.ordenar_por_juez(p, [0.2, 0.9, 0.6, 0.1])
        self.assertEqual([x["doc_id"] for x in ordenados], ["d1", "d2", "d0", "d3"])
        self.assertEqual([x["doc_id"] for x in aceptados], ["d1", "d2", "d0"])  # nunca menos de 3
        _, aceptados = pasos.ordenar_por_juez(p, [0.7, 0.9, 0.6, 0.1])
        self.assertEqual([x["doc_id"] for x in aceptados], ["d1", "d0", "d2"])

    def test_combinacion_por_confianza(self):
        combinada = pasos.combinar_probabilidades({"A": 0.6, "B": 0.4}, {"A": 0.2, "B": 0.8})
        self.assertAlmostEqual(combinada["B"], (0.6 * 0.4 + 0.8 * 0.8) / 1.4)
        self.assertGreater(combinada["B"], combinada["A"])  # el más seguro pesa más

    def test_etapas_segundo_agente_antes_del_principal(self):
        eventos = []

        class Principal:
            modelo = None

            def abrir(self):
                eventos.append("abre principal")
                self.modelo = object()

            def cerrar(self):
                eventos.append("cierra principal")
                self.modelo = None

            def seleccionar(self, entrada, pasajes, evidencia):
                self.vistos = [p["doc_id"] for p in pasajes]
                return pasajes

            def probabilidades_letras(self, entrada, pasajes, evidencia, prefijo=None):
                return {"A": 0.6, "B": 0.4}

            def generar(self, entrada, pasajes, evidencia, prefijo="{", **otros):
                return prefijo + "x"

        class Segundo:
            def __init__(self, config):
                eventos.append("crea segundo " + config["decoder"]["repo_id"])

            def abrir(self):
                eventos.append("abre segundo")

            def cerrar(self):
                eventos.append("cierra segundo")

            def probabilidad_si(self, mensajes):
                return 0.9 if "útil" in mensajes[-1]["content"] else 0.1

            def seleccionar(self, entrada, pasajes, evidencia):
                return pasajes

            def probabilidades_letras(self, entrada, pasajes, evidencia):
                return {"A": 0.1, "B": 0.9}

        pasajes = [{**pasaje(f"d{i}"), "texto": "útil" if i == 3 else "ruido"} for i in range(4)]
        config = {"recuperacion": {}, "generacion": {"politica": {}, "letra_por_probabilidad": True},
                  "agentes": {"segundo": {"repo_id": "otro"}, "juez_evidencia": True, "segunda_opinion": True}}
        s = sistema(config, Recuperador(pasajes, pasajes), Principal())
        s.preparado = {}
        cerrada = {**CERRADA, "opciones": {"A": "x", "B": "y"}}

        def final(entrada, crudo, pasajes, evidencia, politica, validador):
            return ({"id": entrada["id"], "formato": entrada["formato"], "respuesta_correcta": "A",
                     "descarte_opciones": {"A": "-", "B": "-"}}, {"problema": None})

        with mock.patch("legalrag.generation.decoder.DecoderTransformers", Segundo), \
                mock.patch("legalrag.citations.verificacion.respuesta_final", final):
            tiempos = s.preparar_lote([cerrada])
            respuesta = s.responder(cerrada, s.recuperar(cerrada))
        self.assertEqual(eventos, ["crea segundo otro", "abre segundo", "cierra segundo", "abre principal"])
        self.assertIn(cerrada["id"], tiempos)
        self.assertEqual(s.preparado[cerrada["id"]]["pasajes"][0]["doc_id"], "d3")  # el juez lo subió
        self.assertEqual(s.decoder.vistos[0], "d3")
        self.assertEqual(respuesta["respuesta_correcta"], "B")  # la segunda opinión, más segura, inclina la letra
        self.assertEqual(s.ultimo_registro["letras_principal"], {"A": 0.6, "B": 0.4})


if __name__ == "__main__":
    unittest.main()
