"""Pruebas de la reconstrucción del corpus, el grafo normativo y la auditoría.

No usan red ni el corpus descargado: cubren las reglas que deciden qué entra.
"""
import unittest

from legalrag.ingestion.auditoria import Auditoria, _doc_de_canonico, auditar_fuga, clave
from legalrag.ingestion.grafo import referencias
from legalrag.ingestion.reconstruir import elegir, ficha_base, identidad_valida, metricas

ARTICULOS = "\n".join(f"ARTÍCULO {n}. Texto del artículo {n}." for n in range(1, 21))


class Identidad(unittest.TestCase):
    def test_ley_con_fecha_intercalada(self):
        ficha = ficha_base({"doc_id": "decreto_2663_1950"})
        self.assertTrue(identidad_valida("DECRETO LEY 2663 DEL 5 DE AGOSTO DE 1950\n" + ARTICULOS, ficha))

    def test_anotacion_en_encabezado(self):
        ficha = ficha_base({"doc_id": "decreto_4334_2008"})
        self.assertTrue(identidad_valida("DECRETO <LEY> 4334 DE 2008\n" + ARTICULOS, ficha))

    def test_portada_con_200_no_pasa(self):
        ficha = ficha_base({"doc_id": "ley_1909_2018"})
        self.assertFalse(identidad_valida("Secretaría del Senado. Bienvenido. Ley 1910 de 2018." * 50, ficha))

    def test_sentencia_cc(self):
        ficha = ficha_base({"doc_id": "sentencia_cc_su016_2020"})
        self.assertTrue(identidad_valida("Sentencia SU016/20 ... " * 100, ficha))
        self.assertFalse(identidad_valida("Sentencia SU017/20 ... " * 100, ficha))
        self.assertTrue(identidad_valida("Sentencia\nSU.917/10\nCARGO DE CARRERA" * 100,
                                         ficha_base({"doc_id": "sentencia_cc_su917_2010"})))

    def test_csj_con_ruido_de_ocr(self):
        ficha = ficha_base({"doc_id": "sentencia_csj_sc8453_2016"})
        self.assertTrue(identidad_valida("CORTE SUPREMA DE JUSTICIA\nSsC8453-2016\nRadicación" * 40, ficha))
        self.assertFalse(identidad_valida("CORTE SUPREMA DE JUSTICIA\nSC8454-2016\nRadicación" * 40, ficha))

    def test_consejo_de_estado_por_radicado(self):
        ficha = ficha_base({"doc_id": "sentencia_ce_suj4005_2020"})
        self.assertTrue(identidad_valida("Sentencia de unificación 2020CE-SUJ-4-005 del 26 de noviembre de 2020" * 30, ficha))

    def test_acto_legislativo(self):
        ficha = ficha_base({"doc_id": "acto_legislativo_1_2005"})
        self.assertEqual(ficha["tipo"], "acto_legislativo")
        self.assertTrue(identidad_valida("ACTO LEGISLATIVO 01 DE 2005 (julio 22)" * 50, ficha))


    def test_sentencia_cc_con_letra(self):
        from legalrag.ingestion.reconstruir import candidatas
        ficha = ficha_base({"doc_id": "sentencia_cc_c155a_1993"})
        self.assertEqual(ficha["numero"], "C-155A")
        self.assertIn("https://www.corteconstitucional.gov.co/relatoria/1993/c-155a-93.htm", candidatas({"doc_id": "x"}, ficha))
        self.assertTrue(identidad_valida("Sentencia No. C-155A/93 ... " * 100, ficha))
        self.assertFalse(identidad_valida("Sentencia No. C-155/93 ... " * 100, ficha))

    def test_sala_penal_antigua_por_sala_y_fecha(self):
        from legalrag.ingestion.reconstruir import identidad_valida
        ficha = {"tipo": "sentencia", "numero": "30222(25-07-08)", "anio": 2008,
                 "organo_emisor": "Corte Suprema de Justicia"}
        texto = ("CORTE SUPREMA DE JUSTICIA SALA DE CASACIÓN PENAL Aprobado Acta No. 205 Bogotá, D.C., "
                 "veinticinco (25) de julio de dos mil ocho (2008) VISTOS: Dirime la Sala el conflicto")
        self.assertTrue(identidad_valida(texto, ficha))
        self.assertFalse(identidad_valida(texto.replace("(25)", "(26)"), ficha))
        self.assertFalse(identidad_valida(texto.replace("CASACIÓN PENAL", "CASACIÓN CIVIL"), ficha))
        en_letras = ("CORTE SUPREMA DE JUSTICIA SALA DE CASACIÓN PENAL Aprobado acta No. 382 Bogotá, D. C., "
                     "diecisiete de octubre de dos mil doce. Se pronuncia la Corte")
        ficha_letras = {**ficha, "numero": "33145(17-10-12)imp", "anio": 2012}
        self.assertTrue(identidad_valida(en_letras, ficha_letras))
        self.assertFalse(identidad_valida(en_letras.replace("dos mil doce", "dos mil trece"), ficha_letras))


class Completitud(unittest.TestCase):
    def test_huecos_y_citas_entre_comillas(self):
        texto = ARTICULOS.replace("ARTÍCULO 7.", "") + '\n"ARTÍCULO 900. Citado entre comillas."'
        m = metricas(texto, judicial=False)
        self.assertEqual(m["max_articulo"], 20)
        self.assertEqual(m["huecos"], 1)

    def test_sentencia_con_cierre(self):
        texto = "Considerandos. " * 400 + "Notifíquese, comuníquese y cúmplase. Magistrado ponente"
        self.assertTrue(metricas(texto, judicial=True)["cierre"])
        self.assertFalse(metricas("Considerandos. " * 400, judicial=True)["cierre"])
        # Parte resolutiva antes de un bloque largo de notas al pie.
        con_notas = "Considerandos. " * 100 + "R E S U E L V E: Declarar exequible. " + "[12] Nota al pie. " * 400
        self.assertTrue(metricas(con_notas, judicial=True)["cierre"])

    def test_cadena_cortada_pierde(self):
        cortada = {"valida": True, "estado": "cadena de más de 80 tramos", "orden": 0,
                   "metricas": {"articulos": 2574, "huecos": 0}}
        completa = {"valida": True, "estado": "ok", "orden": 1, "metricas": {"articulos": 2684, "huecos": 0}}
        self.assertIs(elegir([cortada, completa], judicial=False), completa)

    def test_senado_gana_empate(self):
        senado = {"valida": True, "estado": "ok", "orden": 0, "metricas": {"articulos": 487, "huecos": 5}}
        gestor = {"valida": True, "estado": "ok", "orden": 1, "metricas": {"articulos": 487, "huecos": 5}}
        self.assertIs(elegir([senado, gestor], judicial=False), senado)


class Respaldos(unittest.TestCase):
    def test_se_prueban_solo_si_no_hay_fuente_valida(self):
        from unittest import mock
        import legalrag.ingestion.reconstruir as r
        pedidas = []

        def descargar(url):
            pedidas.append(url)
            if "cancilleria" in url:
                return [{"contenido": b"x", "url": url}], "ok"
            return None, "HTTP 404"

        texto = "LEY 74 DE 1968 (diciembre 26)\n" + ARTICULOS * 5
        with mock.patch.object(r, "descargar_con_tramos", descargar), \
                mock.patch.object(r, "texto_de", lambda partes: texto), \
                mock.patch.object(r, "buscar_gestor", lambda *a: None):
            _, evaluadas, elegida = r.procesar({"doc_id": "ley_74_1968"})
        self.assertIn("cancilleria", elegida["url"])
        self.assertTrue(pedidas[0].startswith(r.SENADO))
        self.assertEqual(sum("cancilleria" in u for u in pedidas), 1)
        # Con una fuente principal válida, no se consultan respaldos.
        pedidas.clear()
        with mock.patch.object(r, "descargar_con_tramos", lambda url: (pedidas.append(url) or [{"contenido": b"x"}], "ok")), \
                mock.patch.object(r, "texto_de", lambda partes: texto), \
                mock.patch.object(r, "buscar_gestor", lambda *a: None):
            r.procesar({"doc_id": "ley_74_1968"})
        self.assertFalse(any("cancilleria" in u for u in pedidas))


class Intercalado(unittest.TestCase):
    def test_turnos_por_servidor(self):
        from legalrag.ingestion.reconstruir import intercalar
        docs = [{"doc_id": f"ley_{i}_2000"} for i in range(3)] + [{"doc_id": f"sentencia_cc_c00{i}_2000"} for i in range(3)]
        orden = [d["doc_id"][:5] for d in intercalar(docs)]
        self.assertEqual([o.startswith("sent") for o in orden], [False, True, False, True, False, True])
        self.assertEqual(len(intercalar(docs)), 6)


class LimpiezaEditorial(unittest.TestCase):
    def test_pie_y_navegacion_de_senado(self):
        from legalrag.preprocessing.ingesta import retirar_navegacion_editorial
        texto = ("Inicio\n|\nSiguiente\nARTICULO 1. Texto.\nARTICULO 2. Otro texto.\nAnterior\n"
                 "Disposiciones analizadas por Avance Jurídico Casa Editorial S.A.S.©\n"
                 "\"Leyes desde 1992 - Vigencia Expresa\"\nISSN [1657-6241 (En linea)]\n"
                 "Las notas de vigencia ... normas de uso de la información aquí contenida.\n"
                 "ARTICULO 3. Sigue la norma.\nNotas de Vigencia")
        limpio, retirado = retirar_navegacion_editorial(texto)
        self.assertEqual(retirado, {"navegacion": 4, "pie_editorial": 1})
        self.assertEqual(limpio, "ARTICULO 1. Texto.\nARTICULO 2. Otro texto.\nARTICULO 3. Sigue la norma.\nNotas de Vigencia")

    def test_pie_sin_cierre_no_se_toca(self):
        from legalrag.preprocessing.ingesta import retirar_navegacion_editorial
        texto = "Disposiciones analizadas por Avance Jurídico\nARTICULO 5. Texto que debe quedar."
        self.assertEqual(retirar_navegacion_editorial(texto)[0], texto)

    def test_titulo_canonico_de_codigos(self):
        self.assertEqual(ficha_base({"doc_id": "decreto_2663_1950", "titulo": "Decreto 2663 de 1950"})["titulo"],
                         "Código Sustantivo del Trabajo")


class Cortesia(unittest.TestCase):
    def test_simultaneas_y_pausa_por_servidor(self):
        import threading
        import time
        from unittest import mock
        import legalrag.ingestion.reconstruir as r
        en_curso, maximo, inicios, candado = [0], [0], [], threading.Lock()

        class Respuesta:
            status = 200
            headers = {}
            def __enter__(self):
                return self
            def __exit__(self, *a):
                with candado:
                    en_curso[0] -= 1
            def geturl(self):
                return "x"
            def read(self):
                time.sleep(0.2)
                return b"ok"

        def abrir(*a, **k):
            with candado:
                en_curso[0] += 1
                maximo[0] = max(maximo[0], en_curso[0])
                inicios.append(time.monotonic())
            return Respuesta()

        with mock.patch.object(r.urllib.request, "urlopen", abrir), mock.patch.object(r, "PAUSA_HOST", 0.05), \
                mock.patch.dict(r.SIMULTANEAS_POR_SERVIDOR, {"prueba.gov.co": 3}):
            r._cupos.pop("prueba.gov.co", None), r._locks.pop("prueba.gov.co", None), r._ultimo.pop("prueba.gov.co", None)
            hilos = [threading.Thread(target=r.pedir, args=("http://prueba.gov.co/x",)) for _ in range(8)]
            [h.start() for h in hilos]
            [h.join() for h in hilos]
        self.assertLessEqual(maximo[0], 3)
        self.assertGreater(maximo[0], 1)
        inicios.sort()
        self.assertTrue(all(b - a >= 0.045 for a, b in zip(inicios, inicios[1:])))


class Grafo(unittest.TestCase):
    def test_formas_de_cita(self):
        texto = ("según la Ley 100 del 23 de diciembre de 1993, el Decreto-Ley 2663 de 1950, "
                 "la Ley 1.564 de 2012, el Acto Legislativo 01 de 2005, la sentencia C-355/06, "
                 "la T-760 de 2008, la SU-214 de 2016, la SU.917/10, la c-207-19 y la SL3385-2022")
        claves = {c for c, _ in referencias(texto)}
        self.assertTrue({"ley_100_1993", "decreto_2663_1950", "ley_1564_2012", "acto_legislativo_1_2005",
                         "sentencia_cc_c355_2006", "sentencia_cc_t760_2008", "sentencia_cc_su214_2016",
                         "sentencia_csj_sl3385_2022", "sentencia_cc_su917_2010", "sentencia_cc_c207_2019"} <= claves)

    def test_proyectos_de_ley_no_son_normas(self):
        texto = ("el proyecto de ley número 111 de 2006 Senado, 144 de 2005 Cámara, que expide el Código; "
                 "la Ley 1_2003-Cámara; el Proyecto de Acto Legislativo 002 de 2016 y la ley 72 de 2000 Cámara; "
                 "pero sí la Ley 1407 de 2010 y la Ley 80 de 1993.")
        claves = {c for c, _ in referencias(texto)}
        self.assertEqual(claves, {"ley_1407_2010", "ley_80_1993"})

    def test_prefijo_co(self):
        self.assertEqual(clave("co_ley_1437_2011"), "ley_1437_2011")


class Cobertura(unittest.TestCase):
    def test_canonicos_del_seed(self):
        ids = {"ley_1564_2012", "sentencia_cc_c055_2022", "sentencia_csj_sl648_2018", "ley_80_1993"}
        self.assertEqual(_doc_de_canonico(("codigo_general_proceso", None, None), ids), "ley_1564_2012")
        self.assertEqual(_doc_de_canonico(("jurisprudencia", "C-55", "2022"), ids), "sentencia_cc_c055_2022")
        self.assertEqual(_doc_de_canonico(("jurisprudencia", "SL-648", "2018"), ids), "sentencia_csj_sl648_2018")
        self.assertEqual(_doc_de_canonico(("ley", "80", "1993"), ids), "ley_80_1993")
        self.assertIsNone(_doc_de_canonico(("ley", "81", "1993"), ids))
        self.assertEqual(_doc_de_canonico(("acuerdo", "02", "2015"), {"acuerdo_cc_02_2015"}), "acuerdo_cc_02_2015")


class Legibilidad(unittest.TestCase):
    def test_capa_de_texto_basura(self):
        from legalrag.preprocessing.ocr import es_legible
        self.assertFalse(es_legible("富 Repdbli鍛deColombia 的neS脚remaueJusti舶 LUISARMANDOTOL " * 50))
        self.assertTrue(es_legible("Decide la Corte sobre la solicitud de reconocimiento promovida por la "
                                   "sociedad demandante, respecto del laudo arbitral y de las pruebas. " * 20))


class Word(unittest.TestCase):
    def test_detecta_doc_y_guarda_texto_derivado(self):
        import tempfile
        from pathlib import Path
        from unittest import mock
        import legalrag.ingestion.reconstruir as r
        from legalrag.preprocessing.word import FIRMA_OLE, es_word
        doc = FIRMA_OLE + b"\x00" * 600
        self.assertTrue(es_word(doc))
        self.assertFalse(es_word(b"%PDF-1.4"))
        texto = "SENTENCIA DE UNIFICACIÓN\nRadicación 2020CE-SUJ-4-005 de 2020. Decide la Sala. " * 60
        parte = {"url": "https://servicios.consejodeestado.gov.co/x", "url_final": "x", "tipo_contenido": "application/msword",
                 "contenido": doc}
        with mock.patch.object(r, "descargar_con_tramos", lambda url: ([dict(parte)], "ok")), \
                mock.patch.object(r, "texto_word", lambda c: texto), mock.patch.object(r, "buscar_gestor", lambda *a: None):
            ficha, evaluadas, elegida = r.procesar({"doc_id": "sentencia_ce_suj4005_2020", "url": parte["url"]})
        self.assertTrue(elegida and elegida["valida"])
        with tempfile.TemporaryDirectory() as carpeta:
            (Path(carpeta) / "data/raw").mkdir(parents=True)
            entrada = r.guardar(Path(carpeta), ficha, elegida, "Consejo de Estado")
            archivo = entrada["archivos_raw"][0]
            self.assertTrue(archivo["archivo"].endswith(".doc"))
            self.assertTrue(archivo["texto_derivado"]["archivo"].endswith("000.txt"))
            self.assertEqual((Path(carpeta) / "data/raw" / archivo["texto_derivado"]["archivo"]).read_text(), texto)
            self.assertTrue(r.completo_en_disco(Path(carpeta), entrada))


class Areas(unittest.TestCase):
    def test_epigrafe_y_alcance(self):
        from legalrag.ingestion.areas import clasificar
        tratado = ("LEY 1346 DE 2009 (julio 31) Diario Oficial No. 47.427 <Ley declarada EXEQUIBLE, Sentencia C-293-10> "
                   "POR MEDIO DE LA CUAL SE APRUEBA LA \"CONVENCIÓN SOBRE LOS DERECHOS DE LAS PERSONAS CON DISCAPACIDAD\". "
                   "EL CONGRESO DE COLOMBIA DECRETA: ARTÍCULO 1o.")
        areas, alcance = clasificar(tratado)
        self.assertEqual(alcance, "tratado_internacional")
        self.assertIn("familia", areas)
        desarrollo = ("DECRETO 2177 DE 1989 por el cual se desarrolla la Ley 82 de 1988, aprobatoria del Convenio número 159 "
                      "sobre readaptación profesional y el empleo. EL PRESIDENTE DE LA REPÚBLICA")
        self.assertIsNone(clasificar(desarrollo)[1])
        self.assertIn("laboral", clasificar(desarrollo)[0])
        divorcio = "LEY 2442 DE 2024 Por medio de la cual se permite el divorcio por la sola voluntad de cualquiera de los cónyuges."
        self.assertEqual(clasificar(divorcio), (["familia"], None))
        # La anotación de Senado no aporta área: "sentencia" no convierte la ley en procesal.
        anotada = "LEY 48 DE 1993 <Sentencia C-561-95 proceso> Por la cual se reglamenta el servicio de reclutamiento"
        self.assertEqual(clasificar(anotada), (["administrativo"], None))


class Niveles(unittest.TestCase):
    def test_vigencia_por_anotacion_del_senado(self):
        from legalrag.ingestion.niveles import nivel, vigencia
        derogada = ("LEY 1530 DE 2012 (mayo 17) <Ley derogada a partir del 1 de enero de 2021 por el artículo 211 de "
                    "la Ley 2056 de 2020> Por la cual se regula el Sistema General de Regalías. ARTÍCULO 1o. OBJETO.")
        self.assertEqual(vigencia(derogada), "derogada")
        exequible = "LEY 100 DE 1993 <Ley declarada EXEQUIBLE por la Corte> Por la cual se crea el sistema. ARTÍCULO 1o."
        self.assertEqual(vigencia(exequible), "sin_marca")
        con_notas = ("LEY 1395 DE 2010 Por la cual se adoptan medidas. ARTÍCULO 1o. <Artículo derogado por el literal c) "
                     "del artículo 626 de la Ley 1564 de 2012> texto. ARTÍCULO 2o. texto. ARTÍCULO 3o. texto.")
        self.assertEqual(vigencia(con_notas), "con_notas")
        self.assertEqual(nivel({"doc_id": "ley_1530_2012"}, "derogada", set()), "complementario")
        barrido = {"doc_id": "sentencia_cc_c259_2009", "origen_ampliacion": "cc_datos_abiertos"}
        self.assertEqual(nivel(barrido, "sin_marca", set()), "complementario")
        self.assertEqual(nivel(barrido, "sin_marca", {"sentencia_cc_c259_2009"}), "nucleo")


class Fuga(unittest.TestCase):
    def test_fuente_oficial_citada_no_es_fuga_pero_fuente_no_oficial_si(self):
        from legalrag.evaluation import oficial  # noqa: F401  (solo para comprobar que el paquete importa)
        import json
        from pathlib import Path
        muestra = Path(__file__).resolve().parents[1] / "data/sample_50.jsonl"
        if not muestra.is_file() and not (Path(__file__).resolve().parents[1] / "data/oficial/data/sample_50.jsonl").is_file():
            self.skipTest("Se necesita la muestra de 50")
        ruta = muestra if muestra.is_file() else Path(__file__).resolve().parents[1] / "data/oficial/data/sample_50.jsonl"
        item = next(json.loads(l) for l in ruta.read_text(encoding="utf-8-sig").splitlines()
                    if len((json.loads(l).get("pregunta") or "").split()) >= 30)
        texto = "Texto oficial. " + item["pregunta"]
        raw = {"oficial": {"archivos_raw": [{"url": "https://www.corteconstitucional.gov.co/x.htm"}]},
               "blog": {"archivos_raw": [{"url": "https://blog.example.com/x"}]}}
        aud = Auditoria()
        auditar_fuga(aud, {"oficial": texto, "blog": texto}, raw)
        criticos = {h["doc_id"] for h in aud.hallazgos if h["severidad"] == "CRITICO"}
        # La pregunta es redacción de los juristas: es fuga en cualquier documento.
        self.assertEqual(criticos, {"oficial", "blog"})


if __name__ == "__main__":
    unittest.main()
