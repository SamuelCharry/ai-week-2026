"""Opción A sin GPU: SQLite FTS5 real, índice y modelos falsos, citas con el evaluador oficial."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np

from legalrag.agent.componentes import Sistema
from legalrag.citations.verificacion import abstencion, interpretar_json, respuesta_final
from legalrag.config import RAIZ, leer_config
from legalrag.generation.decoder import DecoderTransformers
from legalrag.retrieval.hibrido import Fragmentos, RecuperadorHibrido, consulta, rrf

MANIFIESTO = [{"doc_id": "co_ley_1564_2012", "tipo": "ley", "numero": "1564", "anio": 2012,
               "titulo": "Código General del Proceso"}]
CABECERA = "LEY 1564 DE 2012\n(julio 12)\nPor medio de la cual se expide el Código General del Proceso.\n\n"
ARTICULO = "ARTÍCULO 369. TRASLADO DE LA DEMANDA. Admitida la demanda se correrá traslado al demandado por veinte días."
ESQUEMA = json.loads((RAIZ / "data/oficial/schema/submission.schema.json").read_text(encoding="utf-8"))
VALIDADOR = jsonschema.validators.validator_for(ESQUEMA)(ESQUEMA)
SEMI = {"id": 7, "formato": "semi_open", "pregunta": "¿Cuál es el término de traslado de la demanda verbal?"}
CERRADA = {"id": 8, "formato": "multiple_choice", "pregunta": "¿Término de traslado?",
           "opciones": {"A": "diez días", "B": "veinte días", "C": "tres días", "D": "un mes"}}


def crear_fragmentos(ruta, textos):
    conexion = sqlite3.connect(ruta)
    conexion.executescript("""
    CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id TEXT, titulo TEXT, tipo TEXT, articulo TEXT,
      seccion TEXT, unidad_id TEXT, unidad_inicio INTEGER, unidad_fin INTEGER, inicio INTEGER, fin INTEGER,
      texto TEXT, texto_busqueda TEXT, avisos TEXT);
    CREATE VIRTUAL TABLE fts USING fts5(texto_busqueda, content='chunks', content_rowid='id',
      tokenize='unicode61 remove_diacritics 2');""")
    for doc_id, texto, inicio, fin, unidad in textos:
        fila = (doc_id, "Código General del Proceso", "ley", "369", None, unidad, inicio, fin, inicio, fin,
                texto[inicio:fin], "Código General del Proceso\n" + texto[inicio:fin], "[]")
        cursor = conexion.execute("INSERT INTO chunks(doc_id,titulo,tipo,articulo,seccion,unidad_id,unidad_inicio,"
                                  "unidad_fin,inicio,fin,texto,texto_busqueda,avisos) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", fila)
        conexion.execute("INSERT INTO fts(rowid,texto_busqueda) VALUES(?,?)", (cursor.lastrowid, fila[-2]))
    conexion.commit()
    conexion.close()


class IndiceFalso:
    def __init__(self, orden):
        self.orden = orden

    def search(self, vector, k):
        ids = self.orden[:k]
        return np.array([[1.0 - 0.1 * i for i in range(len(ids))]]), np.array([ids])


class EncoderFalso:
    def codificar(self, texto):
        return np.zeros((1, 4), dtype="float32")


class ReordenadorFalso:
    """Prefiere los candidatos que mencionan 'traslado'."""

    def puntuar(self, texto, candidatos):
        return [float("traslado" in c.lower()) for c in candidatos]


class RecuperacionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        raiz = Path(self.tmp.name)
        (raiz / "textos").mkdir()
        self.texto = CABECERA + ARTICULO + "\n\nARTÍCULO 370. OTRA COSA. Texto sin relación con el tema."
        (raiz / "textos/co_ley_1564_2012.txt").write_text(self.texto, encoding="utf-8")
        a = len(CABECERA)
        b = a + len(ARTICULO)
        crear_fragmentos(raiz / "chunks.sqlite", [
            ("co_ley_1564_2012", self.texto, a, b, "u369"),
            ("co_ley_1564_2012", self.texto, b + 2, len(self.texto), "u370")])
        config = {**leer_config()["recuperacion"], "fragmentos": "chunks.sqlite", "textos": "textos"}
        self.recuperador = RecuperadorHibrido(raiz, config)
        from legalrag.evaluation.oficial import cargar_citaciones
        self.recuperador.fragmentos = Fragmentos(raiz / "chunks.sqlite")
        self.recuperador.indice = IndiceFalso([1, 0])
        self.recuperador.encoder = EncoderFalso()
        self.recuperador.reordenador = ReordenadorFalso()
        self.recuperador.citaciones = cargar_citaciones(RAIZ)

    def tearDown(self):
        self.tmp.cleanup()

    def test_consulta_incluye_opciones(self):
        self.assertIn("B. veinte días", consulta(CERRADA))
        self.assertEqual(consulta(SEMI), SEMI["pregunta"] + "\n")

    def test_rrf_premia_coincidencias_y_desempata_por_id(self):
        self.assertEqual([i for i, _ in rrf([(3, 9), (1, 8)], [(1, 5), (2, 4)])], [1, 3, 2])
        self.assertEqual([i for i, _ in rrf([(5, 1)], [(4, 1)])], [4, 5])

    def test_bm25_encuentra_por_termino(self):
        self.assertEqual(self.recuperador.fragmentos.bm25("traslado demanda", 10)[0][0], 1)
        self.assertEqual(self.recuperador.fragmentos.bm25("", 10), [])

    def test_reranker_decide_y_pasajes_son_literales_con_cabecera(self):
        ranking = self.recuperador.ranking(SEMI)
        self.assertEqual(ranking[0][0], 1)
        pasajes = self.recuperador.pasajes(ranking)
        self.assertEqual(pasajes[0]["texto"], ARTICULO)
        self.assertEqual(self.texto[pasajes[0]["inicio"]:pasajes[0]["fin"]], ARTICULO)
        cabecera = pasajes[1]
        self.assertTrue(cabecera["unidad_id"].endswith("__cabecera_literal"))
        self.assertTrue(cabecera["texto"].startswith("LEY 1564 DE 2012"))
        self.assertEqual(len({p["unidad_id"] for p in pasajes}), len(pasajes))


class VerificacionTest(unittest.TestCase):
    pasajes = [{"doc_id": "co_ley_1564_2012", "inicio": 0, "fin": 10, "texto": CABECERA + ARTICULO, "score": 1.0,
                "titulo": "Código General del Proceso", "articulo": "369"}]

    def test_json_con_bloque_de_codigo(self):
        self.assertEqual(interpretar_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(interpretar_json('Respuesta: {"a": 1} fin'), {"a": 1})

    def test_abstencion_cerrada_cumple_esquema_oficial(self):
        respuesta = abstencion(CERRADA, self.pasajes)
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])
        self.assertTrue(respuesta["abstencion"])
        self.assertEqual(respuesta["pasajes_recuperados"][0]["doc_id"], "co_ley_1564_2012")
        self.assertNotIn("titulo", respuesta["pasajes_recuperados"][0])

    def test_cita_respaldada_se_entrega(self):
        crudo = json.dumps({"abstencion": False, "respuesta": "El traslado es de veinte días. Lo fija el artículo 369 "
                            "de la Ley 1564 de 2012. Aplica al proceso verbal.",
                            "palabras_clave": ["traslado"], "referencia_legal": "Artículo 369 de la Ley 1564 de 2012"})
        respuesta, problema = respuesta_final(SEMI, crudo, self.pasajes, MANIFIESTO, VALIDADOR)
        self.assertIsNone(problema)
        self.assertFalse(respuesta["abstencion"])

    def test_cita_sin_respaldo_se_abstiene_y_conserva_evidencia(self):
        crudo = json.dumps({"abstencion": False, "respuesta": "Lo fija el artículo 90 de la Ley 1564 de 2012.",
                            "palabras_clave": [], "referencia_legal": "Artículo 90 de la Ley 1564 de 2012"})
        respuesta, problema = respuesta_final(SEMI, crudo, self.pasajes, MANIFIESTO, VALIDADOR)
        self.assertEqual(problema, "cita_sin_respaldo_o_indeterminada")
        self.assertTrue(respuesta["abstencion"])
        self.assertEqual(len(respuesta["pasajes_recuperados"]), 1)

    def test_json_invalido_y_null_del_modelo_en_cerradas(self):
        respuesta, problema = respuesta_final(CERRADA, "no es json", self.pasajes, MANIFIESTO, VALIDADOR)
        self.assertTrue(problema.startswith("json_invalido"))
        crudo = json.dumps({"abstencion": True, "respuesta_correcta": None, "justificacion": "", "descarte_opciones": {}})
        respuesta, problema = respuesta_final(CERRADA, crudo, self.pasajes, MANIFIESTO, VALIDADOR)
        self.assertEqual(problema, "abstencion_del_modelo")
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])


class TokenizadorFalso:
    eos_token_id = 0

    def apply_chat_template(self, mensajes, tokenize, add_generation_prompt, return_dict):
        return list(range(sum(len(m["content"]) for m in mensajes) // 10))


class DecoderFalso(DecoderTransformers):
    def __init__(self, crudo, contexto):
        super().__init__({"decoder": {"parametros": 1}, "contexto": contexto, "max_nuevos_tokens": 10})
        self.tokenizer, self.crudo, self.vistos = TokenizadorFalso(), crudo, None

    def generar(self, entrada, pasajes):
        self.vistos = pasajes
        return self.crudo


class DecoderTest(unittest.TestCase):
    def test_limite_de_parametros(self):
        with self.assertRaises(ValueError):
            DecoderTransformers({"decoder": {"parametros": 8_190_735_360}})

    def test_seleccion_respeta_el_contexto(self):
        decoder = DecoderFalso("", contexto=300)
        pasajes = [{"doc_id": "a", "texto": "x" * 400}, {"doc_id": "b", "texto": "y" * 4000},
                   {"doc_id": "c", "texto": "z" * 400}]
        self.assertEqual([p["doc_id"] for p in decoder.seleccionar(SEMI, pasajes)], ["a", "c"])


class SistemaOpcionATest(unittest.TestCase):
    def sistema(self, crudo, contexto=100_000):
        sistema = Sistema(RAIZ, leer_config())
        sistema.decoder = DecoderFalso(crudo, contexto)
        sistema.manifiesto, sistema.validador = MANIFIESTO, VALIDADOR
        return sistema

    def test_responde_con_los_pasajes_que_caben(self):
        crudo = json.dumps({"abstencion": False, "respuesta_correcta": "B",
                            "justificacion": "Veinte días según el artículo 369 de la Ley 1564 de 2012.",
                            "descarte_opciones": {"A": "No.", "C": "No.", "D": "No."}})
        sistema = self.sistema(crudo)
        respuesta = sistema.responder(CERRADA, VerificacionTest.pasajes)
        self.assertEqual(respuesta["respuesta_correcta"], "B")
        self.assertIsNone(sistema.ultimo_problema)
        self.assertEqual(sistema.decoder.vistos, VerificacionTest.pasajes)

    def test_sin_contexto_se_abstiene(self):
        sistema = self.sistema("{}", contexto=10)
        respuesta = sistema.responder(SEMI, VerificacionTest.pasajes)
        self.assertTrue(respuesta["abstencion"])
        self.assertEqual(sistema.ultimo_problema, "sin_evidencia_en_contexto")


if __name__ == "__main__":
    unittest.main()
