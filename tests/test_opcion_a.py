"""Sistema sin GPU: SQLite FTS5 real, índice y modelos falsos, citas con el evaluador oficial."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import jsonschema
import numpy as np

from legalrag.agent.componentes import Sistema
from legalrag.citations.normas import EvidenciaCorpus
from legalrag.citations.verificacion import (abstencion, interpretar_json, letra_de, normalizar_campos,
                                             respuesta_final)
from legalrag.config import RAIZ, leer_config
from legalrag.evaluation.oficial import cargar_citaciones
from legalrag.generation.decoder import DecoderTransformers
from legalrag.retrieval.hibrido import Fragmentos, RecuperadorHibrido, consulta, rrf

CITAS = cargar_citaciones(RAIZ)
MANIFIESTO = [{"doc_id": "co_ley_1564_2012", "tipo": "ley", "numero": "1564", "anio": 2012,
               "titulo": "Código General del Proceso"},
              {"doc_id": "ley_84_1873", "tipo": "ley", "numero": "84", "anio": 1873, "titulo": "Código Civil"}]
EVIDENCIA = EvidenciaCorpus(CITAS, MANIFIESTO)
ARTICULO = "ARTÍCULO 369. TRASLADO DE LA DEMANDA. Admitida la demanda se correrá traslado al demandado por veinte días."
OTRO = "ARTÍCULO 90. ADMISIÓN, INADMISIÓN Y RECHAZO. El juez admitirá la demanda que reúna los requisitos."
ESQUEMA = json.loads((RAIZ / "data/oficial/schema/submission.schema.json").read_text(encoding="utf-8"))
VALIDADOR = jsonschema.validators.validator_for(ESQUEMA)(ESQUEMA)
POLITICA = {"citar_evidencia": "usadas", "abstener_libre": "sin_evidencia", "saneo": "cita"}
SEMI = {"id": 7, "formato": "semi_open", "pregunta": "¿Cuál es el término de traslado de la demanda verbal?",
        "tema": "Traslado de la demanda", "area": "Derecho procesal"}
CERRADA = {"id": 8, "formato": "multiple_choice", "pregunta": "¿Término de traslado?", "area": "Derecho procesal",
           "opciones": {"A": "diez días", "B": "veinte días", "C": "tres días", "D": "un mes"}}
PASAJE = {"doc_id": "co_ley_1564_2012", "inicio": 0, "fin": len(ARTICULO), "texto": ARTICULO, "score": 1.0,
          "titulo": "Código General del Proceso", "articulo": "369",
          "encabezado": EVIDENCIA.encabezado({"doc_id": "co_ley_1564_2012", "articulo": "369"})}


def crear_fragmentos(ruta, filas):
    conexion = sqlite3.connect(ruta)
    conexion.executescript("""
    CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id TEXT, titulo TEXT, tipo TEXT, articulo TEXT,
      seccion TEXT, unidad_id TEXT, unidad_inicio INTEGER, unidad_fin INTEGER, inicio INTEGER, fin INTEGER,
      texto TEXT, texto_busqueda TEXT, avisos TEXT);
    CREATE VIRTUAL TABLE fts USING fts5(texto_busqueda, content='chunks', content_rowid='id',
      tokenize='unicode61 remove_diacritics 2');""")
    for doc_id, texto, inicio, fin, articulo in filas:
        fila = (doc_id, "Código General del Proceso", "ley", articulo, None, f"u{articulo}", inicio, fin, inicio, fin,
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
    """Prefiere los candidatos que mencionan 'admisión' (para probar que el artículo fijado gana igual)."""

    def puntuar(self, texto, candidatos):
        return [float("admisi" in c.lower()) for c in candidatos]


class RecuperacionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        raiz = Path(self.tmp.name)
        (raiz / "textos").mkdir()
        self.texto = ARTICULO + "\n\n" + OTRO
        (raiz / "textos/co_ley_1564_2012.txt").write_text(self.texto, encoding="utf-8")
        crear_fragmentos(raiz / "chunks.sqlite", [
            ("co_ley_1564_2012", self.texto, 0, len(ARTICULO), "369"),
            ("co_ley_1564_2012", self.texto, len(ARTICULO) + 2, len(self.texto), "90")])
        config = {**leer_config()["recuperacion"], "fragmentos": "chunks.sqlite", "textos": "textos"}
        self.recuperador = RecuperadorHibrido(raiz, config)
        self.recuperador.fragmentos = Fragmentos(raiz / "chunks.sqlite")
        self.recuperador.indice = IndiceFalso([1, 0])
        self.recuperador.encoder = EncoderFalso()
        self.recuperador.reordenador = ReordenadorFalso()
        self.recuperador.citaciones = CITAS
        self.recuperador.evidencia = EVIDENCIA

    def tearDown(self):
        self.tmp.cleanup()

    def test_consulta_con_opciones_y_tema(self):
        self.assertIn("B. veinte días", consulta(CERRADA))
        self.assertTrue(consulta(SEMI, con_tema=True).startswith("Traslado de la demanda\n"))
        self.assertNotIn("Traslado de la demanda\n", consulta(SEMI))

    def test_rrf_premia_coincidencias_y_desempata_por_id(self):
        self.assertEqual([i for i, _ in rrf([(3, 9), (1, 8)], [(1, 5), (2, 4)])], [1, 3, 2])

    def test_bm25_general_y_dentro_de_una_norma(self):
        self.assertEqual(self.recuperador.fragmentos.bm25("traslado demanda", 10)[0][0], 1)
        self.assertEqual(self.recuperador.fragmentos.bm25("traslado", 10, doc_ids=["otra_norma"]), [])
        self.assertEqual(self.recuperador.fragmentos.por_articulo("co_ley_1564_2012", "369"), [1])

    def test_articulo_nombrado_se_fija_aunque_el_reranker_prefiera_otro(self):
        entrada = {**SEMI, "pregunta": "¿Qué ordena el artículo 369 del Código General del Proceso?"}
        ranking = self.recuperador.ranking(entrada)
        self.assertEqual(ranking[0][0], 1)
        self.assertEqual(self.recuperador.ultima_traza["normas_nombradas"], {"co_ley_1564_2012": ["369"]})
        self.recuperador.config = {**self.recuperador.config, "enrutar_normas": False}
        self.assertEqual(self.recuperador.ranking(entrada)[0][0], 2)

    def test_pasajes_literales_con_encabezado_que_respalda_la_cita(self):
        pasajes = self.recuperador.buscar(SEMI)
        self.assertEqual({p["texto"] for p in pasajes}, {ARTICULO, OTRO})
        for p in pasajes:
            self.assertEqual(self.texto[p["inicio"]:p["fin"]], p["texto"])
        self.assertTrue(pasajes[0]["encabezado"].startswith("Código General del Proceso (Ley 1564 de 2012), artículo"))
        self.assertIn(("codigo_general_proceso", None, None), EVIDENCIA.respaldo(pasajes))
        # Sin encabezado el artículo suelto no respalda la cita (el problema que se corrige).
        self.assertEqual(EVIDENCIA.respaldo([{**pasajes[0], "encabezado": None}]), set())


class DiversificacionTest(unittest.TestCase):
    """Norma nombrada sin artículo, tope por documento y mínimo de pasajes normativos."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        raiz = Path(self.tmp.name)
        (raiz / "textos").mkdir()
        conexion = sqlite3.connect(raiz / "chunks.sqlite")
        conexion.executescript("""
        CREATE TABLE chunks (id INTEGER PRIMARY KEY, doc_id TEXT, titulo TEXT, tipo TEXT, articulo TEXT,
          seccion TEXT, unidad_id TEXT, unidad_inicio INTEGER, unidad_fin INTEGER, inicio INTEGER, fin INTEGER,
          texto TEXT, texto_busqueda TEXT, avisos TEXT);
        CREATE VIRTUAL TABLE fts USING fts5(texto_busqueda, content='chunks', content_rowid='id');""")
        # 6 bloques de una sentencia que "se parece a todo" y 2 artículos de la ley que la pregunta nombra.
        filas = [("sentencia_cc_c145_2018", "Sentencia", None, f"b{i}", "reorganización fiducia patrimonio autónomo")
                 for i in range(6)] + [("ley_1116_2006", "Ley", str(i), f"a{i}", "reorganización empresarial") for i in (1, 2)]
        for doc_id in {f[0] for f in filas}:
            (raiz / "textos" / f"{doc_id}.txt").write_text("x" * 100, encoding="utf-8")
        for doc_id, tipo, articulo, unidad, texto in filas:
            cursor = conexion.execute("INSERT INTO chunks(doc_id,titulo,tipo,articulo,seccion,unidad_id,unidad_inicio,"
                                      "unidad_fin,inicio,fin,texto,texto_busqueda,avisos) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                      (doc_id, doc_id, tipo, articulo, None, unidad, 0, 10, 0, 10, texto, texto, "[]"))
            conexion.execute("INSERT INTO fts(rowid,texto_busqueda) VALUES(?,?)", (cursor.lastrowid, texto))
        conexion.commit()
        conexion.close()
        manifiesto = [{"doc_id": "ley_1116_2006", "tipo": "ley", "numero": "1116", "anio": 2006, "titulo": "Ley 1116 de 2006"},
                      {"doc_id": "sentencia_cc_c145_2018", "tipo": "sentencia", "numero": "C-145", "anio": 2018,
                       "titulo": "Sentencia C-145 de 2018"}]
        # Los ajustes parten apagados aquí, sin importar lo que tenga configs/sistema.json.
        self.recuperador = RecuperadorHibrido(raiz, {**leer_config()["recuperacion"], "fragmentos": "chunks.sqlite",
                                                     "textos": "textos", "max_pasajes": 4, "rerank_top": 20,
                                                     "reservar_nombradas": 0, "max_por_documento": None,
                                                     "min_normativos": 0, "ajustes_solo_texto_libre": False})
        self.recuperador.fragmentos = Fragmentos(raiz / "chunks.sqlite")
        self.recuperador.indice = IndiceFalso(list(range(8)))
        self.recuperador.encoder = EncoderFalso()
        self.recuperador.reordenador = type("SentenciasPrimero", (), {
            "puntuar": staticmethod(lambda texto, cands: [2.0 if "fiducia" in c else 1.0 for c in cands])})()
        self.recuperador.citaciones = CITAS
        self.recuperador.evidencia = EvidenciaCorpus(CITAS, manifiesto)
        self.entrada = {**SEMI, "pregunta": "¿Qué pasa con la fiducia en la reorganización de la ley 1116 de 2006?"}

    def tearDown(self):
        self.tmp.cleanup()

    def docs(self, **cambios):
        self.recuperador.config = {**self.recuperador.config, **cambios}
        return [p["doc_id"] for p in self.recuperador.buscar(self.entrada)]

    def test_sin_ajustes_la_sentencia_llena_la_evidencia(self):
        self.assertEqual(set(self.docs()), {"sentencia_cc_c145_2018"})

    def test_reserva_para_la_norma_nombrada(self):
        docs = self.docs(reservar_nombradas=2)
        self.assertEqual(docs[:2], ["ley_1116_2006", "ley_1116_2006"])
        self.assertEqual(len(self.recuperador.ultima_traza["reservados"]), 2)

    def test_tope_por_documento(self):
        docs = self.docs(max_por_documento=2)
        self.assertEqual(docs.count("sentencia_cc_c145_2018"), 2)
        self.assertEqual(docs.count("ley_1116_2006"), 2)

    def test_ajustes_solo_en_texto_libre(self):
        self.recuperador.config = {**self.recuperador.config, "reservar_nombradas": 2, "ajustes_solo_texto_libre": True}
        cerrada = {**self.entrada, "formato": "multiple_choice", "opciones": {"A": "sí", "B": "no"}}
        self.assertEqual({p["doc_id"] for p in self.recuperador.buscar(cerrada)}, {"sentencia_cc_c145_2018"})
        self.assertEqual(self.recuperador.config["reservar_nombradas"], 2)  # se restaura después
        self.assertIn("ley_1116_2006", [p["doc_id"] for p in self.recuperador.buscar(self.entrada)])

    def test_minimo_normativo_reemplaza_los_ultimos(self):
        docs = self.docs(min_normativos=1)
        self.assertEqual(docs[:3], ["sentencia_cc_c145_2018"] * 3)
        self.assertEqual(docs[3], "ley_1116_2006")


class VerificacionTest(unittest.TestCase):
    def test_json_con_bloque_de_codigo_texto_alrededor_y_truncado(self):
        self.assertEqual(interpretar_json('```json\n{"a": 1}\n```'), {"a": 1})
        self.assertEqual(interpretar_json('Respuesta: {"a": 1} fin'), {"a": 1})
        self.assertEqual(interpretar_json('{"a": "texto cortado'), {"a": "texto cortado"})
        self.assertEqual(letra_de('{"respuesta_correcta": "C", "justificacion": "trunc', ["A", "B", "C", "D"]), "C")

    def test_salidas_reales_de_modelo_pequeno(self):
        # Claves sin comillas y un segundo objeto al final (Qwen2.5-0.5B, pregunta 290).
        crudo = '{justificacion: "Art. 13 de la Ley 1150 de 2007.", respuesta_correcta: "C", descarte_opciones: ["A", "B"]}\n\n{"justificacion": "otra'
        self.assertEqual(interpretar_json(crudo)["respuesta_correcta"], "C")
        self.assertEqual(letra_de(crudo, ["A", "B", "C", "D"]), "C")
        # Llave repetida y comilla de clave cerrada tarde con una lista como valor (pregunta 79).
        crudo = '{{"respuesta":"La Ley 1010 de 2006","palabras_clave":["acoso laboral"],"referencia_legal:"[1, "Ley 1010 de 2006"]}}'
        objeto = interpretar_json(crudo)
        self.assertEqual(objeto["referencia_legal"], [1, "Ley 1010 de 2006"])
        self.assertEqual(normalizar_campos(objeto, "semi_open")["referencia_legal"], "1 Ley 1010 de 2006")
        self.assertEqual(normalizar_campos({"descarte_opciones": ["A", "B"], "respuesta_correcta": "c"},
                                           "multiple_choice"), {"descarte_opciones": {}, "respuesta_correcta": "C"})

    def test_abstencion_cerrada_cumple_esquema_oficial(self):
        respuesta = abstencion(CERRADA, [PASAJE])
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])
        self.assertTrue(respuesta["pasajes_recuperados"][0]["texto"].startswith("Código General del Proceso"))

    def test_cita_respaldada_se_entrega_con_fundamento(self):
        crudo = json.dumps({"respuesta": "El traslado es de veinte días según el artículo 369 del Código General del "
                            "Proceso. Aplica al proceso verbal. Corre desde la notificación.",
                            "palabras_clave": ["traslado"], "referencia_legal": "Artículo 369 del CGP"})
        respuesta, registro = respuesta_final(SEMI, crudo, [PASAJE], EVIDENCIA, POLITICA, VALIDADOR)
        self.assertFalse(respuesta["abstencion"])
        self.assertIn("Código General del Proceso", respuesta["referencia_legal"])
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])

    def test_fundamento_cita_todas_las_normas_de_la_evidencia(self):
        # Una cita respaldada nunca resta: con "todas", referencia_legal nombra cada norma recuperada
        # (cuenta para citas y abstención; el juez RAGAS de semiabiertas solo lee "respuesta").
        otro = {**PASAJE, "doc_id": "ley_84_1873", "articulo": "946", "texto": "ARTÍCULO 946. Reivindicación.",
                "encabezado": EVIDENCIA.encabezado({"doc_id": "ley_84_1873", "articulo": "946"})}
        crudo = json.dumps({"respuesta": "El traslado es de veinte días. Aplica al proceso verbal. Corre desde la "
                            "notificación.", "palabras_clave": ["traslado"], "referencia_legal": "CGP"})
        politica = {**POLITICA, "citar_evidencia": "todas"}
        respuesta, _ = respuesta_final(SEMI, crudo, [PASAJE, otro], EVIDENCIA, politica, VALIDADOR)
        citadas = CITAS.bodies(CITAS.extract(respuesta["referencia_legal"]))
        self.assertTrue({("codigo_general_proceso", None, None), ("codigo_civil", None, None)} <= citadas)
        self.assertEqual(citadas - EVIDENCIA.respaldo([PASAJE, otro]), set())
        self.assertNotIn("Código Civil", respuesta["respuesta"])

    def test_respaldo_cita_tambien_las_normas_que_el_pasaje_menciona(self):
        menciona = {**PASAJE, "texto": ARTICULO + " Se aplicará lo dispuesto en la Ley 80 de 1993 y la Sentencia C-355 de 2006."}
        crudo = json.dumps({"respuesta": "El traslado es de veinte días. Aplica al proceso verbal. Corre desde la "
                            "notificación.", "palabras_clave": ["traslado"], "referencia_legal": "CGP"})
        politica = {**POLITICA, "citar_evidencia": "respaldo"}
        respuesta, registro = respuesta_final(SEMI, crudo, [menciona], EVIDENCIA, politica, VALIDADOR)
        citadas = CITAS.bodies(CITAS.extract(respuesta["referencia_legal"]))
        self.assertTrue({("ley", "80", "1993"), ("jurisprudencia", "C-355", "2006"),
                         ("codigo_general_proceso", None, None)} <= citadas)
        self.assertEqual(citadas - EVIDENCIA.respaldo([menciona]), set())  # nada sin respaldo
        self.assertGreaterEqual(registro["normas_agregadas"], 2)
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])

    def test_modo_de_cita_tolera_mayusculas_y_rechaza_valores_desconocidos(self):
        crudo = json.dumps({"respuesta": "Veinte días. Aplica al proceso verbal. Corre desde la notificación.",
                            "palabras_clave": ["traslado"], "referencia_legal": "CGP"})
        respuesta, _ = respuesta_final(SEMI, crudo, [PASAJE], EVIDENCIA, {**POLITICA, "citar_evidencia": " Respaldo "}, VALIDADOR)
        self.assertIn("Código General del Proceso", respuesta["referencia_legal"])
        with self.assertRaises(ValueError) as error:
            respuesta_final(SEMI, crudo, [PASAJE], EVIDENCIA, {**POLITICA, "citar_evidencia": "respaldos"}, VALIDADOR)
        self.assertIn("no es válido", str(error.exception))

    def test_cita_inventada_se_quita_sin_anular_la_respuesta(self):
        crudo = json.dumps({"respuesta": "El traslado es de veinte días. Lo confirma la Ley 999 de 2019. "
                            "Corre desde la notificación.", "palabras_clave": ["traslado"],
                            "referencia_legal": "Ley 999 de 2019"})
        respuesta, registro = respuesta_final(SEMI, crudo, [PASAJE], EVIDENCIA, POLITICA, VALIDADOR)
        self.assertFalse(respuesta["abstencion"])
        self.assertEqual(registro["problema"], "citas_saneadas")
        self.assertNotIn("999", respuesta["respuesta"] + respuesta["referencia_legal"])
        self.assertIn("veinte días", respuesta["respuesta"])

    def test_cerrada_nunca_se_abstiene_y_recupera_la_letra(self):
        respuesta, registro = respuesta_final(CERRADA, '{"respuesta_correcta": "B", "justificacion": "El artí',
                                              [PASAJE], EVIDENCIA, POLITICA, VALIDADOR)
        self.assertFalse(respuesta["abstencion"])
        self.assertEqual(respuesta["respuesta_correcta"], "B")
        respuesta, registro = respuesta_final(CERRADA, "no es json", [PASAJE], EVIDENCIA, POLITICA, VALIDADOR)
        self.assertEqual(registro["problema"], "json_invalido")
        self.assertFalse(respuesta["abstencion"])
        self.assertIn(respuesta["respuesta_correcta"], "ABCD")
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])


class TokenizadorFalso:
    eos_token_id = 0

    def apply_chat_template(self, mensajes, **opciones):
        return list(range(sum(len(m["content"]) for m in mensajes) // 10))

    def encode(self, texto, add_special_tokens=False):
        return [1]


class DecoderFalso(DecoderTransformers):
    def __init__(self, crudo, contexto):
        super().__init__({"decoder": {"repo_id": "falso", "parametros": 1}, "contexto": contexto,
                          "max_nuevos_tokens": 10})
        self.tokenizer, self.crudo, self.vistos = TokenizadorFalso(), crudo, None

    def generar(self, entrada, pasajes, evidencia, prefijo="{", **otros):
        self.vistos, self.prefijo = pasajes, prefijo
        return self.crudo if prefijo == "{" else prefijo + self.crudo

    def probabilidades_letras(self, entrada, pasajes, evidencia, **otros):
        return {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1}


class DecoderTest(unittest.TestCase):
    def test_limite_de_parametros(self):
        with self.assertRaises(ValueError):
            DecoderTransformers({"decoder": {"parametros": 8_190_735_360}})

    def test_solo_pasa_del_limite_si_el_enunciado_lo_sugiere(self):
        DecoderTransformers({"decoder": {"parametros": 8_190_735_360, "admitido_por_enunciado": "§3.1"}})
        with self.assertRaises(ValueError):
            DecoderTransformers({"decoder": {"parametros": 9_150_000_000}})

    def test_seleccion_respeta_el_contexto(self):
        pasajes = [{**PASAJE, "doc_id": d, "texto": t} for d, t in
                   (("co_ley_1564_2012", "x" * 400), ("ley_84_1873", "y" * 1700), ("co_ley_1564_2012", "z " * 150))]
        decoder = DecoderFalso("", contexto=0)
        justo = len(decoder._tokens(decoder.mensajes(SEMI, [pasajes[0], pasajes[2]], EVIDENCIA)))
        decoder.config["contexto"] = justo + decoder.config["max_nuevos_tokens"] + 64
        elegidos = decoder.seleccionar(SEMI, pasajes, EVIDENCIA)
        self.assertEqual([p["texto"][0] for p in elegidos], ["x", "z"])


class RecuperadorFalso:
    evidencia = EVIDENCIA


class SistemaTest(unittest.TestCase):
    def sistema(self, crudo, contexto=100_000, letra_por_probabilidad=False):
        config = leer_config()
        config["generacion"]["letra_por_probabilidad"] = letra_por_probabilidad
        sistema = Sistema(RAIZ, config)
        sistema.decoder = DecoderFalso(crudo, contexto)
        sistema.recuperador = RecuperadorFalso()
        sistema.validador = VALIDADOR
        return sistema

    def test_responde_con_los_pasajes_que_caben(self):
        crudo = json.dumps({"respuesta_correcta": "B", "justificacion": "Veinte días según el artículo 369 del "
                            "Código General del Proceso.", "descarte_opciones": {"A": "No.", "C": "No.", "D": "No."}})
        sistema = self.sistema(crudo)
        respuesta = sistema.responder(CERRADA, [PASAJE])
        self.assertEqual(respuesta["respuesta_correcta"], "B")
        self.assertIsNone(sistema.ultimo_problema)
        self.assertEqual(sistema.decoder.vistos, [PASAJE])

    def test_letra_por_probabilidad_manda_sobre_el_texto(self):
        sistema = self.sistema('Según el artículo 369 del Código General del Proceso.", "descarte_opciones": '
                               '{"A": "No.", "B": "No.", "D": "No."}}', letra_por_probabilidad=True)
        respuesta = sistema.responder(CERRADA, [PASAJE])
        self.assertEqual(sistema.decoder.prefijo, '{"respuesta_correcta": "C", "justificacion": "')
        self.assertEqual(respuesta["respuesta_correcta"], "C")
        self.assertNotIn("C", respuesta["descarte_opciones"])
        self.assertEqual(sistema.ultimo_registro["probabilidades_letras"]["C"], 0.6)
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])

    def test_letra_razonada_elige_despues_de_la_justificacion(self):
        sistema = self.sistema('La demanda se traslada por veinte días según el artículo 369 del Código General del '
                               'Proceso.", "respuesta_correcta": "A", "descarte_opciones": {"A": "No.", "B": "No.", "D": "No."}}',
                               letra_por_probabilidad="razonada")
        vistos = []
        sistema.decoder.probabilidades_letras = lambda e, p, ev, prefijo: vistos.append(prefijo) or \
            {"A": 0.1, "B": 0.2, "C": 0.6, "D": 0.1}
        respuesta = sistema.responder(CERRADA, [PASAJE])
        self.assertEqual(sistema.decoder.prefijo, '{"justificacion": "')
        self.assertIn("veinte días", vistos[0])
        self.assertTrue(vistos[0].endswith('"respuesta_correcta": "'))
        self.assertEqual(respuesta["respuesta_correcta"], "C")
        self.assertEqual(list(VALIDADOR.iter_errors(respuesta)), [])

    def test_entregar_todos_los_pasajes_aunque_el_prompt_use_menos(self):
        otro = {**PASAJE, "doc_id": "ley_84_1873", "articulo": "946", "texto": "ARTÍCULO 946. Reivindicación. " + "x" * 3000,
                "encabezado": EVIDENCIA.encabezado({"doc_id": "ley_84_1873", "articulo": "946"})}
        crudo = json.dumps({"respuesta": "Veinte días. Aplica al proceso verbal. Corre desde la notificación.",
                            "palabras_clave": ["traslado"], "referencia_legal": "CGP"})
        for entregar_todos, esperados in ((False, 1), (True, 2)):
            sistema = self.sistema(crudo)
            sistema.config["generacion"]["entregar_todos"] = entregar_todos
            sistema.decoder.seleccionar = lambda e, pasajes, ev: pasajes[:1]  # el prompt solo alcanza para uno
            respuesta = sistema.responder(SEMI, [PASAJE, otro])
            self.assertEqual(len(respuesta["pasajes_recuperados"]), esperados)
            self.assertEqual(sistema.ultimo_registro["pasajes_en_prompt"], 1)
        self.assertIn("Código Civil", respuesta["referencia_legal"])  # la norma del segundo pasaje queda citada

    def test_sin_contexto_se_abstiene(self):
        sistema = self.sistema("{}", contexto=10)
        respuesta = sistema.responder(SEMI, [PASAJE])
        self.assertTrue(respuesta["abstencion"])
        self.assertEqual(sistema.ultimo_problema, "sin_evidencia_en_contexto")


if __name__ == "__main__":
    unittest.main()
