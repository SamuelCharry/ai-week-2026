import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch


RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))
from scripts.auxiliares.evaluacion import (
    auditar_citas,
    cargar_jsonl,
    comando_evaluador,
    comparar_corridas,
    comprobar_oficiales,
    generar,
    preparar_entrada,
    validar_esquema,
)


MANIFIESTO = [{
    'doc_id': 'co_ley_1564_2012',
    'tipo': 'ley',
    'numero': '1564',
    'anio': 2012,
    'titulo': 'Código General del Proceso',
}]

DECRETO = {
    'doc_id': 'co_decreto_2591_1991',
    'tipo': 'decreto',
    'numero': '2591',
    'anio': 1991,
    'titulo': 'Decreto 2591 de 1991',
}


class EvaluacionTest(unittest.TestCase):
    def test_material_oficial_ausente(self):
        with tempfile.TemporaryDirectory() as carpeta:
            with self.assertRaises(FileNotFoundError) as error:
                comprobar_oficiales(carpeta)
            for nombre in ['sample_50.jsonl', 'seed_targets.json', 'submission.schema.json', 'evaluate.py', 'requirements-evaluador.txt']:
                self.assertIn(nombre, str(error.exception))

    def test_carga_jsonl(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'control.jsonl'
            ruta.write_text('{"id": 1}\n\n{"id": 2}\n', encoding='utf-8')
            self.assertEqual(len(cargar_jsonl(ruta)), 2)

    def test_rechaza_ids_repetidos(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'control.jsonl'
            ruta.write_text('{"id": 1}\n{"id": 1}\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Id repetido'):
                cargar_jsonl(ruta)

    def test_rechaza_json_invalido(self):
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'control.jsonl'
            ruta.write_text('{"id":', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'línea 1'):
                cargar_jsonl(ruta)

    def test_no_envia_respuestas_ni_fundamento(self):
        registro = {'id': 1, 'formato': 'control', 'pregunta': 'entrada de control', 'legal_basis': 'reservado', 'respuesta_correcta': 'reservado'}
        entrada = preparar_entrada(registro)
        self.assertEqual(set(entrada), {'id', 'formato', 'pregunta'})

    def test_rechaza_campos_reservados_configurados(self):
        with self.assertRaisesRegex(ValueError, 'prohibidos'):
            preparar_entrada({}, ['id', 'formato', 'LEGAL_BASIS'])

    def test_rechaza_campos_reservados_anidados(self):
        registro = {'id': 1, 'formato': 'control', 'pregunta': 'entrada de control', 'opciones': {'legal_basis': 'reservado'}}
        with self.assertRaisesRegex(ValueError, 'legal_basis'):
            preparar_entrada(registro)

    def test_no_acepta_entrada_sin_pregunta(self):
        with self.assertRaisesRegex(ValueError, 'texto de la pregunta'):
            preparar_entrada({'id': 1, 'formato': 'control'})

    def test_permite_nombre_oficial_configurable(self):
        registro = {'id': 1, 'formato': 'control', 'enunciado': 'entrada de control'}
        self.assertIn('enunciado', preparar_entrada(registro, ['id', 'formato', 'enunciado']))

    def test_comando_json_y_tiempo(self):
        comando = [sys.executable, '-c', 'import json,sys; r=json.load(sys.stdin); print(json.dumps({"id":r["id"],"formato":r["formato"],"abstencion":True}))']
        respuesta, segundos = generar({'id': 1, 'formato': 'control'}, command=comando)
        self.assertTrue(respuesta['abstencion'])
        self.assertGreaterEqual(segundos, 0)

    def test_llave_juez_no_llega_al_generador(self):
        comando = [sys.executable, '-c', 'import json,sys,os; r=json.load(sys.stdin); print(json.dumps({"id":r["id"],"formato":r["formato"],"llave": "OPENROUTER_API_KEY" in os.environ}))']
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'control_sin_validez'}):
            respuesta, _ = generar({'id': 1, 'formato': 'control'}, command=comando)
        self.assertFalse(respuesta['llave'])

    def test_rechaza_respuesta_fuera_de_json(self):
        with self.assertRaisesRegex(ValueError, 'solo un objeto JSON'):
            generar({'id': 1, 'formato': 'control'}, command=[sys.executable, '-c', 'print("texto de control")'])

    def test_rechaza_endpoint_externo(self):
        with self.assertRaisesRegex(ValueError, 'HTTP local'):
            generar({'id': 1, 'formato': 'control'}, endpoint='https://example.com')

    def test_endpoint_local_recibe_json(self):
        class Control(BaseHTTPRequestHandler):
            def do_POST(self):
                entrada = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                cuerpo = json.dumps({'id': entrada['id'], 'formato': entrada['formato'], 'abstencion': True}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(cuerpo)

            def log_message(self, *args):
                pass

        with HTTPServer(('127.0.0.1', 0), Control) as servidor:
            hilo = threading.Thread(target=servidor.handle_request, daemon=True)
            hilo.start()
            respuesta, _ = generar({'id': 1, 'formato': 'control'}, endpoint=f'http://127.0.0.1:{servidor.server_port}', timeout=5)
            hilo.join(timeout=5)
        self.assertTrue(respuesta['abstencion'])

    def test_endpoint_no_sigue_redirecciones(self):
        class Control(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(302)
                self.send_header('Location', 'https://example.com')
                self.end_headers()

            def log_message(self, *args):
                pass

        with HTTPServer(('127.0.0.1', 0), Control) as servidor:
            hilo = threading.Thread(target=servidor.handle_request, daemon=True)
            hilo.start()
            with self.assertRaisesRegex(ValueError, 'redirigir'):
                generar({'id': 1, 'formato': 'control'}, endpoint=f'http://127.0.0.1:{servidor.server_port}', timeout=5)
            hilo.join(timeout=5)

    def test_rechaza_id_booleano_en_lugar_de_entero(self):
        comando = [sys.executable, '-c', 'import json; print(json.dumps({"id":True,"formato":"control"}))']
        with self.assertRaisesRegex(ValueError, 'cambió el id'):
            generar({'id': 1, 'formato': 'control'}, command=comando)

    def test_valida_con_esquema_recibido(self):
        esquema = {'type': 'object', 'properties': {'id': {'type': 'integer'}, 'abstencion': {'type': 'boolean'}}, 'required': ['id', 'abstencion']}
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / 'control.schema.json'
            ruta.write_text(json.dumps(esquema), encoding='utf-8')
            self.assertEqual(validar_esquema([{'id': 1, 'abstencion': True}], ruta), [])
            self.assertEqual(len(validar_esquema([{'id': 1, 'abstencion': 'true'}], ruta)), 1)

    def test_determinismo_excluye_solo_tiempos(self):
        a = [{'id': 1, 'respuesta': 'control', 'latencia_s': 1.1}]
        b = [{'id': 1, 'latencia_s': 2.4, 'respuesta': 'control'}]
        self.assertTrue(comparar_corridas(a, b)['iguales'])
        b[0]['respuesta'] = 'distinto'
        self.assertFalse(comparar_corridas(a, b)['iguales'])

    def test_determinismo_conserva_orden_pasajes(self):
        a = [{'id': 1, 'pasajes_recuperados': [{'id': 1}, {'id': 2}]}]
        b = [{'id': 1, 'pasajes_recuperados': [{'id': 2}, {'id': 1}]}]
        self.assertFalse(comparar_corridas(a, b)['iguales'])

    def test_respaldo_norma_y_articulo(self):
        respuesta = {'id': 1, 'referencia_legal': 'Artículo 42 del Código General del Proceso', 'pasajes_recuperados': [{'doc_id': 'co_ley_1564_2012', 'texto': 'ARTÍCULO 42.'}]}
        resultado = auditar_citas(respuesta, MANIFIESTO)
        self.assertEqual(resultado['citas'][0]['articulo'], '42')
        self.assertEqual(resultado['sin_respaldo'], 0)
        self.assertFalse(resultado['revision_manual'])

    def test_no_confunde_42_con_420(self):
        respuesta = {'id': 1, 'referencia_legal': 'Artículo 42 del Código General del Proceso', 'pasajes_recuperados': [{'doc_id': 'co_ley_1564_2012', 'texto': 'ARTÍCULO 420.'}]}
        self.assertEqual(auditar_citas(respuesta, MANIFIESTO)['sin_respaldo'], 1)

    def test_solo_cuenta_primeros_diez(self):
        pasajes = [{'doc_id': 'control', 'texto': 'control'} for _ in range(10)]
        pasajes.append({'doc_id': 'co_ley_1564_2012', 'texto': 'ARTÍCULO 42.'})
        respuesta = {'id': 1, 'referencia_legal': 'Artículo 42 del Código General del Proceso', 'pasajes_recuperados': pasajes}
        self.assertEqual(auditar_citas(respuesta, MANIFIESTO)['sin_respaldo'], 1)

    def test_sin_norma_identificable_queda_pendiente(self):
        respuesta = {'id': 1, 'referencia_legal': 'Artículo 42', 'pasajes_recuperados': []}
        self.assertTrue(auditar_citas(respuesta, MANIFIESTO)['revision_manual'])

    def test_alias_decreto_ley_equivalentes(self):
        for nombre in ['Decreto 2591 de 1991', 'Decreto Ley 2591 de 1991', 'Decreto-Ley 2591 de 1991', 'Decreto–Ley 2591 de 1991', 'Decreto Ley 02591 de 1991']:
            with self.subTest(nombre=nombre):
                respuesta = {'id': 1, 'referencia_legal': f'Artículo 42 del {nombre}', 'pasajes_recuperados': [{'doc_id': DECRETO['doc_id'], 'texto': 'ARTÍCULO 42.'}]}
                resultado = auditar_citas(respuesta, [DECRETO])
                self.assertEqual(len(resultado['citas']), 1)
                self.assertEqual(resultado['citas'][0]['doc_id'], DECRETO['doc_id'])
                self.assertIs(resultado['citas'][0]['respaldada'], True)
                self.assertEqual(resultado['indeterminadas'], 0)
                self.assertEqual(resultado['sin_respaldo'], 0)
                self.assertFalse(resultado['revision_manual'])

    def test_alias_resuelto_exige_articulo_correcto(self):
        respuesta = {'id': 1, 'referencia_legal': 'Artículo 42 del Decreto Ley 2591 de 1991', 'pasajes_recuperados': [{'doc_id': DECRETO['doc_id'], 'texto': 'ARTÍCULO 41.'}]}
        resultado = auditar_citas(respuesta, [DECRETO])
        self.assertIs(resultado['citas'][0]['respaldada'], False)
        self.assertEqual(resultado['sin_respaldo'], 1)
        self.assertEqual(resultado['indeterminadas'], 0)

    def test_alias_no_resuelto_es_indeterminado(self):
        for nombre in ['Decreto Ley 2591 de 1992', 'Decreto Ley 2592 de 1991']:
            with self.subTest(nombre=nombre):
                respuesta = {'id': 1, 'referencia_legal': f'Artículo 42 del {nombre}', 'pasajes_recuperados': [{'doc_id': DECRETO['doc_id'], 'texto': 'ARTÍCULO 42.'}]}
                resultado = auditar_citas(respuesta, [DECRETO])
                self.assertIsNone(resultado['citas'][0]['doc_id'])
                self.assertIsNone(resultado['citas'][0]['respaldada'])
                self.assertEqual(resultado['sin_respaldo'], 0)
                self.assertEqual(resultado['indeterminadas'], 1)
                self.assertTrue(resultado['revision_manual'])

    def test_alias_ambiguo_es_indeterminado(self):
        documentos = [DECRETO, dict(DECRETO, doc_id='control_segunda_fuente')]
        respuesta = {'id': 1, 'referencia_legal': 'Artículo 42 del Decreto Ley 2591 de 1991', 'pasajes_recuperados': [{'doc_id': DECRETO['doc_id'], 'texto': 'ARTÍCULO 42.'}]}
        resultado = auditar_citas(respuesta, documentos)
        self.assertIsNone(resultado['citas'][0]['respaldada'])
        self.assertEqual(resultado['sin_respaldo'], 0)
        self.assertEqual(resultado['indeterminadas'], 1)
        self.assertTrue(resultado['revision_manual'])

    def test_control_produccion_conserva_alias_respaldado(self):
        from scripts.auxiliares.generacion import controlar_citas

        respuesta = {'id': 1, 'formato': 'semiabierta', 'abstencion': False, 'referencia_legal': 'Artículo 42 del Decreto Ley 2591 de 1991', 'pasajes_recuperados': [{'doc_id': DECRETO['doc_id'], 'texto': 'ARTÍCULO 42.'}]}
        salida, registro = controlar_citas(respuesta, [DECRETO])
        self.assertEqual(salida, respuesta)
        self.assertFalse(registro['abstencion_por_citas'])
        self.assertIs(registro['auditoria_citas']['citas'][0]['respaldada'], True)

    def test_comando_oficial_con_y_sin_ragas(self):
        comando = comando_evaluador('python', 'evaluate.py', 'entrega.jsonl')
        self.assertEqual(comando[-2:], ['--split', 'sample'])
        self.assertNotIn('--ragas', comando)
        self.assertEqual(comando_evaluador('python', 'evaluate.py', 'entrega.jsonl', ragas=True)[-1], '--ragas')


if __name__ == '__main__':
    unittest.main()
