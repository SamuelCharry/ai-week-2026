import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.auxiliares.generacion import ClienteLocal, ajustar_contexto, controlar_citas, generar, mensajes


RAIZ = Path(__file__).resolve().parents[2]


class ClientePrueba:
    def __init__(self, contenido=None):
        self.contenido = contenido
        self.peticiones = []

    def contar(self, conversacion):
        datos = json.loads(conversacion[-1]["content"])
        return 20 + sum(len(p["texto"]) for p in datos["pasajes_recuperados"])

    def aplicar_plantilla(self, conversacion):
        return json.dumps(conversacion, ensure_ascii=False)

    def pedir(self, ruta, datos):
        self.peticiones.append(datos)
        respuesta = {"content": self.contenido, "stop_type": "eos", "tokens": []}
        return respuesta, json.dumps(respuesta, ensure_ascii=False).encode()


class PruebasGeneracion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (RAIZ / "data/processed/corpus/unidades.jsonl").open(encoding="utf-8") as archivo:
            cls.unidad = next(u for linea in archivo if (u := json.loads(linea)).get("articulo")
                             and 160 <= len(u["texto"]) <= 1000 and u.get("estado_segmentacion") == "segmentado")
        cls.entrada = {"id": "prueba_formato", "formato": "semi_open",
                       "pregunta": f"artículo {cls.unidad['articulo']} de {cls.unidad['titulo']}"}

    def test_solo_endpoint_local(self):
        for url in ["https://127.0.0.1", "http://example.com", "http://usuario:clave@localhost"]:
            with self.assertRaises(ValueError):
                ClienteLocal(url)

    def test_separacion_de_campos_de_evaluacion(self):
        conversacion = mensajes({**self.entrada, "respuesta_esperada": "reservado", "legal_basis": []}, [])
        self.assertNotIn("respuesta_esperada", conversacion[-1]["content"])
        self.assertNotIn("legal_basis", conversacion[-1]["content"])

    def test_padre_completo(self):
        original = copy.deepcopy(self.unidad)
        usados, omitidos, _ = ajustar_contexto(ClientePrueba(), self.entrada, [original], contexto=4096)
        self.assertEqual(usados, [original])
        self.assertEqual(omitidos, [])
        self.assertEqual(original, self.unidad)

    def test_no_recorta_unidad_que_no_cabe(self):
        usados, omitidos, _ = ajustar_contexto(ClientePrueba(), self.entrada, [self.unidad], contexto=550)
        self.assertEqual(usados, [])
        self.assertEqual(omitidos[0]["unidad_id"], self.unidad["unidad_id"])

    def test_rechaza_ventana_con_offsets_coherentes(self):
        ventana = {**self.unidad, "texto": self.unidad["texto"][:100],
                   "fin": self.unidad["inicio"] + 100, "recuperar_unidad_completa": True,
                   "unidad_inicio": self.unidad["inicio"], "unidad_fin": self.unidad["fin"]}
        with self.assertRaises(ValueError):
            ajustar_contexto(ClientePrueba(), self.entrada, [ventana])

    def test_abstencion_programatica_separada(self):
        cliente = ClientePrueba()
        salida, registro = generar(cliente, self.entrada, [])
        self.assertTrue(salida["abstencion"])
        self.assertTrue(registro["abstencion_programatica"])
        self.assertEqual(cliente.peticiones, [])

    def test_raw_invalido_se_conserva(self):
        cliente = ClientePrueba("{")
        with tempfile.TemporaryDirectory() as tmp:
            salida, registro = generar(cliente, self.entrada, [], abstencion_automatica=False, carpeta_raw=tmp)
            self.assertIsNone(salida)
            self.assertFalse(registro["json_valido"])
            self.assertEqual((Path(tmp) / "respuesta.txt").read_bytes(), b"{")
        self.assertFalse(cliente.peticiones[0]["cache_prompt"])
        self.assertEqual(cliente.peticiones[0]["temperature"], 0)
        self.assertEqual(cliente.peticiones[0]["seed"], 0)

    def test_gramatica_no_cambia_el_prompt(self):
        contenido = json.dumps({"respuesta": "", "palabras_clave": [], "referencia_legal": "", "abstencion": True})
        a, b = ClientePrueba(contenido), ClientePrueba(contenido)
        salida_a, _ = generar(a, self.entrada, [], abstencion_automatica=False, gramatica=False)
        salida_b, _ = generar(b, self.entrada, [], abstencion_automatica=False, gramatica=True)
        self.assertEqual(a.peticiones[0]["prompt"], b.peticiones[0]["prompt"])
        self.assertEqual(salida_a, salida_b)
        self.assertNotIn("json_schema", a.peticiones[0])
        self.assertIn("json_schema", b.peticiones[0])

    def test_control_cita_sin_respaldo(self):
        salida = {"id": self.entrada["id"], "formato": self.entrada["formato"], "respuesta": "", "palabras_clave": [],
                  "referencia_legal": self.entrada["pregunta"], "abstencion": False,
                  "pasajes_recuperados": []}
        original = copy.deepcopy(salida)
        controlada, registro = controlar_citas(salida, [self.unidad])
        self.assertTrue(controlada["abstencion"])
        self.assertTrue(registro["abstencion_programatica"])
        self.assertEqual(registro["motivo"], "citas_sin_respaldo")
        self.assertEqual(salida, original)
        self.assertGreater(registro["auditoria_citas"]["sin_respaldo"], 0)

    def test_control_cita_respaldada(self):
        salida = {"id": self.entrada["id"], "formato": self.entrada["formato"], "respuesta": "", "palabras_clave": [],
                  "referencia_legal": self.entrada["pregunta"], "abstencion": False,
                  "pasajes_recuperados": [self.unidad]}
        controlada, registro = controlar_citas(salida, [self.unidad])
        self.assertEqual(controlada, salida)
        self.assertFalse(registro["abstencion_por_citas"])

    def test_control_indeterminado_se_registra(self):
        salida = {"id": self.entrada["id"], "formato": self.entrada["formato"], "respuesta": "", "palabras_clave": [],
                  "referencia_legal": "", "abstencion": True, "pasajes_recuperados": []}
        controlada, registro = controlar_citas(salida, [self.unidad])
        self.assertEqual(controlada, salida)
        self.assertTrue(registro["auditoria_citas"]["revision_manual"])
        self.assertFalse(registro["abstencion_por_citas"])

    def test_resumen_separa_indeterminadas(self):
        from scripts.comparar_decoders import resumen

        fila = {"modelo": "prueba", "gramatica": True, "corrida": 1, "con_contexto": True,
                "citas_detectadas": 2, "citas_respaldadas": 1, "citas_sin_respaldo": 0,
                "citas_indeterminadas": 1, "auditoria_disponible": True}
        resultado = resumen([fila])[0]
        self.assertEqual(resultado["citas_respaldadas_proporcion"], 0.5)
        self.assertEqual(resultado["citas_indeterminadas"], 1)

    def test_error_no_mejora_proporcion_de_citas(self):
        from scripts.comparar_decoders import resumen

        fila = {"modelo": "prueba", "gramatica": True, "corrida": 1, "con_contexto": True,
                "citas_detectadas": 1, "citas_respaldadas": 1, "auditoria_disponible": True}
        error = {"modelo": "prueba", "gramatica": True, "corrida": 2, "con_contexto": True,
                 "error": "fallo de lectura"}
        resultado = resumen([fila, error])[0]
        self.assertIsNone(resultado["citas_respaldadas_proporcion"])
        self.assertEqual(resultado["respuestas_no_auditables"], 1)


if __name__ == "__main__":
    unittest.main()
