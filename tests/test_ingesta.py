"""Regresiones con originales reales y cadenas mecánicas que no entran al corpus."""
import hashlib
import json
import re
from pathlib import Path
import unittest

from legalrag.preprocessing.ingesta import (
    candidatos_articulo, segmentar_documento, unidades_de_fragmentos,
    extraer_pdf, extraer_html, evaluar_fuente, MAX_BLOQUE_JUDICIAL, _MARCADOR_PAGINA,
)


RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
SALIDA = RAIZ / "data/processed/corpus"


class SegmentacionReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (SALIDA / "documentos.jsonl").exists():
            raise unittest.SkipTest("Se necesitan los derivados locales de los originales del corpus")
        cls.documentos = {}
        for linea in (SALIDA / "documentos.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(linea)
            cls.documentos[d["doc_id"]] = d

    def segmentar(self, doc_id):
        documento = self.documentos[doc_id]
        texto = (SALIDA / documento["texto_archivo"]).read_text(encoding="utf-8")
        fragmentos = segmentar_documento(documento, texto)
        unidades = unidades_de_fragmentos(documento, texto, fragmentos)
        for anterior, siguiente in zip(unidades, unidades[1:]):
            self.assertLessEqual(anterior["fin"], siguiente["inicio"])
        cubierto = bytearray(len(texto))
        for f in fragmentos:
            self.assertEqual(f["texto"], texto[f["inicio"]:f["fin"]])
            self.assertLessEqual(len(f["texto"]), 1800)
            self.assertLessEqual(f["unidad_inicio"], f["inicio"])
            self.assertLessEqual(f["fin"], f["unidad_fin"])
            self.assertTrue(f["origenes"])
            cubierto[f["inicio"]:f["fin"]] = b"1" * len(f["texto"])
        for m in _MARCADOR_PAGINA.finditer(texto):
            cubierto[m.start():m.end()] = b"1" * len(m.group())
        self.assertFalse(any(not cubierto[i] and not ch.isspace() for i, ch in enumerate(texto)))
        return texto, fragmentos, unidades

    def test_cuatro_normas_con_cantidades_revisadas(self):
        for doc_id, cantidad in [("co_decreto_306_1992", 10), ("ley_1032_2006", 5),
                                 ("ley_1581_2012", 30), ("ley_2445_2025", 45)]:
            with self.subTest(doc_id=doc_id):
                texto, fragmentos, unidades = self.segmentar(doc_id)
                arts = [u["articulo"] for u in unidades if u["articulo"] is not None]
                self.assertEqual(arts, list(map(str, range(1, cantidad + 1))))
                if doc_id == "ley_1032_2006":
                    art3 = next(u for u in unidades if u["articulo"] == "3")
                    self.assertTrue({2, 3}.issubset(art3["paginas"]))
                    self.assertNotIn("257", arts)
                if doc_id == "ley_2445_2025":
                    self.assertNotIn("576A", arts)

    def test_fuentes_restringidas_por_hash_y_entidades(self):
        for doc_id in ["sentencia_csj_sc5191_2020", "sentencia_cc_c264_2026", "co_ley_1551_2012"]:
            self.assertFalse(evaluar_fuente(self.documentos[doc_id])["apta_para_busqueda"])
        d = self.documentos["sentencia_csj_sc5191_2020"]
        reemplazo = {**d, "archivos_raw": [{**a, "sha256": "0" * 64} for a in d["archivos_raw"]]}
        self.assertTrue(evaluar_fuente(reemplazo)["apta_para_busqueda"])
        texto = (SALIDA / d["texto_archivo"]).read_text(encoding="utf-8")
        fragmentos = segmentar_documento({**d, **evaluar_fuente(d)}, texto)
        self.assertTrue(all(f["apta_para_busqueda"] is False for f in fragmentos))
        self.assertTrue(evaluar_fuente(self.documentos["ley_1032_2006"])["apta_para_busqueda"])
        d = self.documentos["ley_2191_2022"]
        a = d["archivos_raw"][0]
        contenido = (RAIZ / "data/raw" / a["archivo"]).read_bytes()
        paginas, info = extraer_html(contenido, a)
        self.assertIn("DESCONEXIÓN LABORAL", paginas[0]["texto"])
        self.assertNotIn("&OacuteN", paginas[0]["texto"])
        self.assertFalse(re.search(r"&[AEIOUaeiou](?:acute|uml)|&[Nn]tilde", paginas[0]["texto"]))
        self.assertGreater(info["entidades_reparadas"], 0)

    def test_indice_conservado_fuera_de_busqueda(self):
        texto, fs, us = self.segmentar("auto_cc_a841_2025")
        indices = [u for u in us if u["tipo_unidad"] == "indice"]
        self.assertTrue(indices)
        self.assertTrue(all(u["apta_para_busqueda"] is False for u in indices))
        self.assertIn("La norma acusada", indices[0]["texto"])

    def test_jerarquias_y_versiones_anteriores(self):
        texto, fs, us = self.segmentar("co_decreto_1083_2015")
        for numero in ["2.1.1.1", "2.1.1.2", "2.2.1.1.1", "2.2.1.1.2", "2.2.5.7.1"]:
            self.assertTrue(any(u["articulo"] == numero for u in us), numero)
        self.assertTrue(any(u["articulo"] == "2.2.1.3.6" and u["version_fuente"] == "anterior_segun_fuente" for u in us))

    def test_notas_de_reforma_no_crean_articulos(self):
        texto, fs, us = self.segmentar("ley_1098_2006")
        art56 = [u for u in us if u["articulo"] == "56"]
        self.assertEqual(len(art56), 1)
        self.assertIn("artículo 2 de la Ley 1878 de 2018", art56[0]["texto"])

    def test_titulo_judicial_partido(self):
        texto, fs, us = self.segmentar("sentencia_cc_c029_2009")
        self.assertFalse(any(u["texto"].strip() == "2. El" for u in us))
        self.assertTrue(any(u["seccion"] == "2. El problema jurídico" for u in us))

    def test_jurisprudencia_anexos_y_numerales_partidos(self):
        for doc_id in ["sentencia_cc_c055_2022", "sentencia_cc_c1011_2008", "sentencia_cc_su337_1999", "sentencia_csj_sc5191_2020"]:
            with self.subTest(doc_id=doc_id):
                texto, fs, us = self.segmentar(doc_id)
                self.assertTrue(all(f["articulo"] is None for f in fs))
                self.assertTrue(all(len(u["texto"]) <= MAX_BLOQUE_JUDICIAL for u in us if u["tipo_unidad"] != "indice"))
                if doc_id == "sentencia_cc_c055_2022":
                    self.assertTrue(any((u["seccion"] or "").upper() == "ANEXO 1" for u in us))
                if doc_id == "sentencia_csj_sc5191_2020":
                    self.assertTrue(any("ACLARACI" in (u["seccion"] or "").upper() for u in us))

    def test_continuacion_no_es_nueva_seccion(self):
        texto, fs, us = self.segmentar("sentencia_cc_c748_2011")
        inicio = texto.index("Consideraciones\nsimilares deben realizarse")
        self.assertFalse(any(u["inicio"] == inicio for u in us))

    def test_pdf_cabecera_editorial_trazable(self):
        ruta = RAIZ / "data/raw/sentencia_ce_00367_2018/000.pdf"
        contenido = ruta.read_bytes()
        huella = hashlib.sha256(contenido).hexdigest()
        paginas, info = extraer_pdf(contenido)
        self.assertEqual(len(paginas), 11)
        self.assertEqual(len(info["cabeceras_retiradas"]), 11)
        self.assertTrue(all("EVA - Gestor Normativo" not in p["texto"][:180] for p in paginas))
        self.assertEqual(hashlib.sha256(ruta.read_bytes()).hexdigest(), huella)

    def test_resultado_determinista(self):
        texto, fs, us = self.segmentar("co_decreto_306_1992")
        self.assertEqual(fs, segmentar_documento(self.documentos["co_decreto_306_1992"], texto))


class TablasHTML(unittest.TestCase):
    def extraer(self, html):
        paginas, info = extraer_html(html.encode("utf-8"), {})
        return paginas[0]["texto"], info

    def test_tabla_simple_conserva_columnas_y_encabezado(self):
        texto, info = self.extraer(
            "<table><tr><th>Nombre</th><th>Valor</th></tr>"
            "<tr><td>uno</td><td>5</td></tr><tr><td>dos</td><td>10</td></tr></table>"
        )
        self.assertEqual(texto, "| Nombre | Valor |\n| --- | --- |\n| uno | 5 |\n| dos | 10 |")
        self.assertEqual(info["tablas_estructuradas"], 1)
        self.assertEqual(info["tablas_en_texto"], 0)

    def test_datos_no_se_convierten_en_encabezado(self):
        texto, _ = self.extraer(
            "<table><tr><td>uno</td><td>5</td></tr>"
            "<tr><td>dos</td><td>10</td></tr></table>"
        )
        self.assertNotIn("---", texto)
        self.assertEqual(texto.splitlines(), ["| uno | 5 |", "| dos | 10 |"])

    def test_bloques_y_tachados_no_se_pierden(self):
        texto, _ = self.extraer(
            '<table><tr><td><p>uno</p><p>dos</p></td><td>5<br>10</td></tr>'
            '<tr style="text-decoration: line-through"><td><s>viejo</s></td>'
            '<td>a|b<!-- oculto --></td></tr></table>'
        )
        self.assertIn("| uno dos | 5 10 |", texto)
        self.assertIn("[TEXTO TACHADO EN LA FUENTE: viejo]", texto)
        self.assertIn("[TEXTO TACHADO EN LA FUENTE: a\\|b]", texto)
        self.assertNotIn("oculto", texto)

    def test_tachado_de_tabla_completa_se_conserva(self):
        texto, _ = self.extraer(
            '<table style="text-decoration:line-through"><tr><td>uno</td><td>5</td></tr>'
            '<tr><td>dos</td><td>10</td></tr></table>'
        )
        self.assertTrue(texto.startswith("[TEXTO TACHADO EN LA FUENTE:"))
        self.assertTrue(texto.endswith("]"))

    def test_celdas_combinadas_conservan_recorrido_original(self):
        texto, info = self.extraer(
            '<table><tr><th colspan="2">Rango</th><th rowspan="2">Valor</th></tr>'
            '<tr><th>Desde</th><th>Hasta</th></tr>'
            '<tr><td>0</td><td>5</td><td>10</td></tr></table>'
        )
        self.assertEqual(info["tablas_estructuradas"], 0)
        self.assertEqual(info["tablas_en_texto"], 1)
        self.assertEqual(texto, "Rango | Valor |\nDesde | Hasta |\n0 | 5 | 10 |")

    def test_anidadas_y_maquetacion_conservan_texto_una_vez(self):
        for html in [
            '<table><tr><td><p>inicio</p><table><tr><td>interior</td></tr></table>'
            '<p>final</p></td></tr></table>',
            '<table role="presentation"><tr><td><p>inicio</p><p>interior</p></td><td>x</td></tr>'
            '<tr><td>final</td><td>y</td></tr></table>',
        ]:
            with self.subTest(html=html):
                texto, info = self.extraer(html)
                self.assertEqual(info["tablas_estructuradas"], 0)
                for palabra in ("inicio", "interior", "final"):
                    self.assertEqual(texto.count(palabra), 1)
                self.assertLess(texto.index("inicio"), texto.index("interior"))
                self.assertLess(texto.index("interior"), texto.index("final"))

    def test_articulos_en_tabla_siguen_segmentandose(self):
        texto, info = self.extraer(
            '<table><tr><td><p>ARTÍCULO 1. xxx</p>'
            '<p>(Ver Ley 388 de 1997; Art. 1.; Art. 6.)</p></td><td>a</td></tr>'
            '<tr><td><p>ARTÍCULO 2. yyy</p></td><td>b</td></tr></table>'
        )
        self.assertEqual(info["tablas_estructuradas"], 0)
        self.assertEqual([c["numero"] for c in candidatos_articulo(texto)], ["1", "2"])

    def test_articulos_con_ordinales_conservan_cabeceras(self):
        texto, info = self.extraer(
            '<table><tr><td>ARTÍCULO PRIMERO. xxx</td><td>a</td></tr>'
            '<tr><td>ARTÍCULO SEGUNDO. yyy</td><td>b</td></tr></table>'
        )
        self.assertEqual(info["tablas_estructuradas"], 0)
        self.assertTrue(texto.startswith("ARTÍCULO PRIMERO."))
        self.assertIn("\nARTÍCULO SEGUNDO.", texto)

    def test_celdas_fuera_de_fila_no_se_pierden(self):
        texto, info = self.extraer(
            '<table><td>fuera</td><tr><td>a</td><td>b</td></tr>'
            '<tr><td>c</td><td>d</td></tr></table>'
        )
        self.assertEqual(info["tablas_estructuradas"], 0)
        self.assertEqual(texto.count("fuera"), 1)


class MecanicaOffsets(unittest.TestCase):
    def test_fronteras_y_solapamiento_sin_contenido_juridico(self):
        # Son marcadores y caracteres repetidos. Nunca se exportan ni indexan.
        texto = "ARTÍCULO\n1º-\n" + "x " * 2400 + ".ARTÍCULO 2º-\n" + "y " * 2400
        documento = {"doc_id": "fixture_mecanico", "tipo": "ley", "avisos": [],
                     "partes": [{"inicio": 0, "fin": len(texto), "archivo": "fixture", "url": None, "pagina": None}]}
        fs = segmentar_documento(documento, texto, 1800, 200)
        self.assertEqual({f["articulo"] for f in fs if f["articulo"]}, {"1", "2"})
        self.assertTrue(all(f["texto"] == texto[f["inicio"]:f["fin"]] for f in fs))
        self.assertTrue(all(len(f["texto"]) <= 1800 for f in fs))
        for a, b in zip(fs, fs[1:]):
            if a["unidad_id"] != b["unidad_id"]:
                self.assertLessEqual(a["fin"], b["inicio"])


if __name__ == "__main__":
    unittest.main()
