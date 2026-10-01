import json
import sys
import tempfile
import threading
import unittest
import urllib.request
import zipfile
from http.server import HTTPServer
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))
from legalrag.agent.componentes import Sistema, cargar  # noqa: E402
from legalrag.agent.pipeline import responder_lote
from legalrag.config import leer_config  # noqa: E402
from legalrag.agent.reproducir import _sha256, asegurar_corpus_indice  # noqa: E402
from legalrag.agent.servicio import manejador  # noqa: E402

PREGUNTAS = [
    {"id": 1, "formato": "semi_open", "pregunta": "¿Plazo para contestar?", "area": "Procesal",
     "respuesta_esperada": "no debe llegar al sistema", "legal_basis": ["Ley 1564 de 2012, art. 369"]},
    {"id": 2, "formato": "multiple_choice", "pregunta": "¿Cuál procede?", "opciones": {"A": "x", "B": "y"},
     "respuesta_correcta": "A"},
]


class Falso:
    """Misma interfaz que Sistema, sin modelos."""

    def __init__(self, raiz=None, config=None, fallar=()):
        self.vistas, self.fallar, self.abierto = [], set(fallar), False

    def abrir(self):
        self.abierto = True

    def cerrar(self):
        self.abierto = False

    def recuperar(self, entrada):
        self.vistas.append(entrada)
        if entrada["id"] in self.fallar:
            raise RuntimeError("falla simulada")
        return [{"doc_id": "ley_1564_2012", "texto": "ARTÍCULO 369...", "score": 0.9}]

    def responder(self, entrada, pasajes):
        base = {"id": entrada["id"], "formato": entrada["formato"], "abstencion": False, "pasajes_recuperados": pasajes}
        if entrada["formato"] == "multiple_choice":
            return {**base, "respuesta_correcta": "A", "justificacion": "…", "descarte_opciones": {"B": "…"}}
        return {**base, "respuesta": "Veinte días.", "palabras_clave": ["traslado"], "referencia_legal": "Art. 369 CGP"}


class SistemaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.carpeta = Path(self.tmp.name)
        self.entrada = self.carpeta / "preguntas.jsonl"
        self.entrada.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in PREGUNTAS), encoding="utf-8")
        self.salida = self.carpeta / "salida/submissions.jsonl"
        self.config = {**leer_config(), "oficial": str(RAIZ / "data/oficial")}

    def tearDown(self):
        self.tmp.cleanup()

    def test_config_versionada_carga_la_opcion_a(self):
        config = leer_config()
        sistema = cargar(RAIZ, config)
        self.assertIs(type(sistema), Sistema)
        self.assertEqual(config["generacion"]["decoder"]["repo_id"], "Qwen/Qwen3-8B")
        # Qwen3-8B tiene 8.190 M: solo pasa del límite porque la §3.1 del enunciado lo sugiere por nombre.
        from legalrag.generation.decoder import limite_de
        self.assertLessEqual(config["generacion"]["decoder"]["parametros"], limite_de(config["generacion"]["decoder"]))

    def test_lote_escribe_en_orden_sin_filtrar_respuestas(self):
        sistema = Falso()
        resumen = responder_lote(RAIZ, self.entrada, self.salida, self.config, sistema=sistema)
        filas = [json.loads(l) for l in self.salida.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([f["id"] for f in filas], [1, 2])
        self.assertEqual(resumen["respondidas"], 2)
        self.assertFalse(resumen["errores"])
        for vista in sistema.vistas:
            self.assertFalse({"respuesta_esperada", "legal_basis", "respuesta_correcta"} & set(vista))
        if (RAIZ / "data/oficial/schema/submission.schema.json").is_file():
            self.assertEqual(resumen["errores_esquema"], [])

    def test_reanuda_sin_repetir_y_repite_si_cambia_config(self):
        responder_lote(RAIZ, self.entrada, self.salida, self.config, sistema=Falso())
        segundo = Falso()
        responder_lote(RAIZ, self.entrada, self.salida, self.config, sistema=segundo)
        self.assertEqual(segundo.vistas, [])
        tercero = Falso()
        responder_lote(RAIZ, self.entrada, self.salida, {**self.config, "generacion": {"k": 3}}, sistema=tercero)
        self.assertEqual(len(tercero.vistas), 2)

    def test_una_falla_no_detiene_la_tanda(self):
        resumen = responder_lote(RAIZ, self.entrada, self.salida, self.config, sistema=Falso(fallar={1}))
        self.assertEqual([e["id"] for e in resumen["errores"]], [1])
        self.assertEqual(resumen["respondidas"], 1)

    def test_rechaza_cambio_de_id(self):
        class Malo(Falso):
            def responder(self, entrada, pasajes):
                return {**super().responder(entrada, pasajes), "id": 99}
        resumen = responder_lote(RAIZ, self.entrada, self.salida, self.config, sistema=Malo())
        self.assertEqual(len(resumen["errores"]), 2)

    def test_hueco_pendiente_detiene_la_tanda(self):
        class Incompleto(Falso):
            def responder(self, entrada, pasajes):
                raise NotImplementedError("pendiente")
        with self.assertRaises(NotImplementedError):
            responder_lote(RAIZ, self.entrada, self.salida, self.config, sistema=Incompleto())

    def test_corpus_indice_sin_enlace(self):
        config = {"corpus_indice": {"url": None, "sha256": None, "archivo": "x.zip", "extraer_en": "data",
                                    "requeridos": ["data/no_existe"]}}
        with self.assertRaises(RuntimeError):
            asegurar_corpus_indice(self.carpeta, config)

    def test_corpus_indice_descarga_verifica_y_extrae(self):
        zip_origen = self.carpeta / "origen.zip"
        with zipfile.ZipFile(zip_origen, "w") as z:
            z.writestr("processed/corpus/a.jsonl", "{}\n")
            z.writestr("index/corpus/manifiesto.json", "{}")
        config = {"corpus_indice": {"url": zip_origen.as_uri(), "sha256": _sha256(zip_origen),
                                    "archivo": "data/descargas/c.zip", "extraer_en": "data",
                                    "requeridos": ["data/processed/corpus", "data/index/corpus"]}}
        rutas = asegurar_corpus_indice(self.carpeta, config)
        self.assertTrue(all(r.is_dir() for r in rutas))
        config["corpus_indice"]["sha256"] = "0" * 64
        for r in rutas:
            for f in r.iterdir():
                f.unlink()
            r.rmdir()
        (self.carpeta / "data/descargas/c.zip").unlink()
        with self.assertRaises(ValueError):
            asegurar_corpus_indice(self.carpeta, config)

    def test_servicio_preguntar(self):
        servidor = HTTPServer(("127.0.0.1", 0), manejador(Falso()))
        hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
        hilo.start()
        try:
            url = f"http://127.0.0.1:{servidor.server_port}"
            cuerpo = json.dumps({"pregunta": "¿Plazo?", "formato": "semi_open"}).encode()
            peticion = urllib.request.Request(url + "/preguntar", data=cuerpo, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(peticion) as r:
                datos = json.loads(r.read())
                self.assertEqual(r.headers["Access-Control-Allow-Origin"], "*")
            self.assertEqual(datos["respuesta"]["formato"], "semi_open")
            self.assertEqual(datos["traza"]["k"], 1)
            malo = urllib.request.Request(url + "/preguntar", data=b'{"formato": "semi_open"}')
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(malo)
            self.assertEqual(error.exception.code, 400)
        finally:
            servidor.shutdown()
            servidor.server_close()


if __name__ == "__main__":
    unittest.main()
