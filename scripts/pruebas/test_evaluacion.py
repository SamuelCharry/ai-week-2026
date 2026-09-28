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


@unittest.skipUnless((RAIZ / 'data/oficial/scripts/citations.py').is_file(), 'Requiere el paquete oficial')
class OficialTest(unittest.TestCase):
    def test_muestra_no_filtra_respuestas_al_modelo(self):
        from scripts.auxiliares.oficial import cargar_muestra
        preguntas, entradas = cargar_muestra(RAIZ)
        self.assertEqual(len(entradas), 50)
        for entrada in entradas:
            self.assertNotIn('legal_basis', entrada)
            self.assertNotIn('respuesta_correcta', entrada)
            self.assertNotIn('respuesta_esperada', entrada)
            self.assertNotIn('texto_respuesta_correcta', entrada)
        with self.assertRaises(ValueError):
            preparar_entrada(preguntas[0], ['id', 'formato', 'pregunta', 'texto_respuesta_correcta'])

    def test_cabecera_literal_y_articulo_conservan_offsets(self):
        import hashlib
        from scripts.auxiliares.oficial import Evidencia, guardar_json
        texto = 'LEY 80 DE 1993\nARTÍCULO 1. Objeto de la contratación.'
        with tempfile.TemporaryDirectory() as carpeta:
            carpeta = Path(carpeta)
            (carpeta / 'norma.txt').write_text(texto, encoding='utf-8')
            doc = {'doc_id': 'ley80', 'tipo': 'ley', 'numero': '80', 'anio': 1993,
                   'texto_archivo': 'norma.txt', 'sha256_texto': hashlib.sha256(texto.encode()).hexdigest()}
            (carpeta / 'documentos.jsonl').write_text(json.dumps(doc) + '\n', encoding='utf-8')
            unidad = {'doc_id': 'ley80', 'unidad_id': 'art1', 'inicio': 15, 'fin': len(texto),
                      'texto': texto[15:], 'score': .5}
            evidencia = Evidencia(RAIZ, carpeta)
            pasajes, avisos = evidencia.preparar([unidad])
            self.assertEqual(len(pasajes), 2)
            self.assertEqual(avisos, [])
            evidencia.verificar(pasajes)
            self.assertEqual(pasajes[0]['texto'], 'LEY 80 DE 1993')
            pasajes[1]['texto'] += ' añadido'
            with self.assertRaises(ValueError):
                evidencia.verificar(pasajes)

    def test_cabecera_y_articulo_no_se_separan_por_contexto(self):
        from scripts.auxiliares.generacion import ajustar_contexto
        class Cliente:
            def contar(self, mensajes):
                datos = json.loads(mensajes[1]['content'])
                return 100 + sum(len(p['texto']) for p in datos['pasajes_recuperados'])
        pasajes = [{'doc_id': 'ley80', 'texto': 'LEY 80 DE 1993', 'inicio': 0, 'fin': 14, 'grupo_evidencia': 'art1'},
                   {'doc_id': 'ley80', 'texto': 'x' * 1000, 'inicio': 15, 'fin': 1015,
                    'unidad_id': 'art1', 'grupo_evidencia': 'art1'}]
        usados, omitidos, _ = ajustar_contexto(Cliente(), {'id': 1, 'formato': 'semi_open', 'pregunta': 'Consulta'},
                                              pasajes, contexto=700, salida=100)
        self.assertEqual(usados, [])
        self.assertEqual(len(omitidos), 1)

    def test_evaluador_oficial_reporta_conflicto_sin_inventar_opcion(self):
        from scripts.auxiliares.oficial import cargar_muestra, evaluar_entrega
        from scripts.auxiliares.generacion import abstenerse
        _, entradas = cargar_muestra(RAIZ)
        respuestas = [{'id': p['id'], 'formato': p['formato'], **abstenerse(p['formato']),
                       'pasajes_recuperados': []} for p in entradas]
        with tempfile.TemporaryDirectory() as carpeta:
            resultado = evaluar_entrega(RAIZ, carpeta, respuestas)
            self.assertEqual(resultado['validacion']['errores'], 0)
            self.assertEqual(resultado['errores_esquema'], 15)
            self.assertEqual(len(resultado['abstencion_cerrada_pendiente_aclaracion']), 15)
            self.assertTrue(all(r['respuesta_correcta'] is None for r in respuestas if r['formato'] == 'multiple_choice'))

    def test_reanudacion_rechaza_evidencia_distinta(self):
        from scripts.auxiliares.oficial import abrir_experimento
        with tempfile.TemporaryDirectory() as carpeta:
            ruta = abrir_experimento(carpeta, 'control', {'evidencia': 'a'})
            with self.assertRaises(ValueError):
                abrir_experimento(carpeta, 'control', {'evidencia': 'b'}, ruta.name)

    def test_paquete_excluye_llave_modelos_y_entorno(self):
        import zipfile
        from scripts.auxiliares.entorno import crear_paquete
        with tempfile.TemporaryDirectory() as carpeta:
            raiz = Path(carpeta)
            for nombre in ['scripts/auxiliares/base.py', 'data/oficial/scripts/evaluate.py',
                           'data/oficial/scripts/.env', '.venv/secreto.py', 'data/modelos/peso.gguf']:
                ruta = raiz / nombre
                ruta.parent.mkdir(parents=True, exist_ok=True)
                ruta.write_text('control')
            with zipfile.ZipFile(crear_paquete(raiz)) as z:
                self.assertIn('scripts/auxiliares/base.py', z.namelist())
                self.assertIn('data/oficial/scripts/evaluate.py', z.namelist())
                self.assertNotIn('data/oficial/scripts/.env', z.namelist())
                self.assertNotIn('.venv/secreto.py', z.namelist())
                self.assertNotIn('data/modelos/peso.gguf', z.namelist())

    def test_notebook_reanuda_sin_repetir_preguntas_guardadas(self):
        import contextlib
        import io
        import time
        from types import SimpleNamespace
        from scripts.auxiliares.oficial import guardar_json
        from scripts.auxiliares.generacion import abstenerse
        notebook = json.loads((RAIZ / 'notebooks/02_1_comparacion_decoders.ipynb').read_text(encoding='utf-8'))
        celda = next(''.join(c['source']) for c in notebook['cells']
                     if c['cell_type'] == 'code' and ''.join(c['source']).startswith('for ficha in fichas:'))
        class Servidor:
            def __init__(self, *args, **kwargs):
                self.comando, self.capas_gpu, self.propiedades = [], None, {}
                self.proceso, self.cliente = SimpleNamespace(pid=0), None
            def __enter__(self): return self
            def __exit__(self, *args): pass
        class Memoria(Servidor):
            def resultado(self): return {}
        llamadas = []
        def generar_control(cliente, entrada, *args, **kwargs):
            llamadas.append(entrada['id'])
            return {'id': entrada['id'], 'formato': 'semi_open', **abstenerse('semi_open'),
                    'pasajes_recuperados': []}, {}
        with tempfile.TemporaryDirectory() as carpeta:
            entorno = {'fichas': [{'nombre': 'control', 'ruta': 'control.gguf'}], 'carpeta': Path(carpeta),
                       'corridas': 2, 'recuperaciones': [{'entrada': {'id': i}, 'pasajes': [], 'recuperacion_s': .1}
                                                      for i in [1, 2]],
                       'ServidorLocal': Servidor, 'Medidor': Memoria, 'servidor': Path('control'),
                       'raiz': RAIZ, 'contexto': 8192, 'max_tokens': 1024, 'evidencia': None,
                       'guardar_json': guardar_json, 'generar_oficial': generar_control, 'time': time,
                       'subprocess': SimpleNamespace(run=lambda *a, **k: SimpleNamespace(stdout=''))}
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(celda, 'notebook:generacion', 'exec'), entorno)
                exec(compile(celda, 'notebook:generacion', 'exec'), entorno)
            self.assertEqual(llamadas, [1, 2, 1, 2])


if __name__ == '__main__':
    unittest.main()
