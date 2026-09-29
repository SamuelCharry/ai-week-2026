"""Regresiones de la ampliación del corpus: remisiones, cabeceras y URLs de fuente."""
import json
import sys
import unittest
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from legalrag.preprocessing.ingesta import candidatos_articulo  # noqa: E402
from legalrag.citations.evidencia import EvidenciaV04  # noqa: E402
from legalrag.ingestion import ampliar as ampliar_corpus  # noqa: E402

CORPUS = RAIZ / "data/processed/corpus"


class RemisionesEditoriales(unittest.TestCase):
    """Las notas "(Ver Ley X; Art. N)" de la fuente no son cabeceras de artículo."""

    def test_remision_entre_parentesis_no_es_cabecera(self):
        texto = ("ARTÍCULO 1. Colombia es un Estado social de derecho.\n"
                 "(Ver Ley 388 de 1997; Art. 1.; Art. 6.; Art. 10)\n"
                 "ARTÍCULO 2. Son fines esenciales del Estado servir a la comunidad.\n")
        self.assertEqual([c["numero"] for c in candidatos_articulo(texto)], ["1", "2"])

    def test_cabecera_pegada_a_la_oracion_anterior_se_conserva(self):
        # El descarte mira los paréntesis sin cerrar, no cualquier prefijo: una
        # cabecera pegada al final de la oración anterior sigue siendo cabecera.
        texto = "El presente decreto rige desde su publicación. ARTÍCULO 5. El juez decide.\n"
        self.assertEqual([c["numero"] for c in candidatos_articulo(texto)], ["5"])

    def test_remision_con_parentesis_cerrado_dentro_si_se_descarta(self):
        texto = "(Ver Ley 80 de 1993; Art. 2, numeral 3 (literal a); Art. 41)\n"
        self.assertEqual(candidatos_articulo(texto), [])

    def test_constitucion_completa_con_los_380_articulos(self):
        ruta = CORPUS / "textos/co_constitucion_1_1991.txt"
        if not ruta.is_file():
            raise unittest.SkipTest("Se necesita el corpus procesado local")
        from legalrag.preprocessing.ingesta import _unidades_norma

        unidades = _unidades_norma(ruta.read_text(encoding="utf-8"))
        propios = {u["articulo"] for u in unidades
                   if u["tipo"] == "articulo" and (u["articulo"] or "").isdigit()}
        self.assertEqual([n for n in range(1, 381) if str(n) not in propios], [])


class Cabeceras(unittest.TestCase):
    """Toda norma identificable debe poder nombrarse de forma que el evaluador reconozca."""

    @classmethod
    def setUpClass(cls):
        if not (CORPUS / "documentos.jsonl").is_file():
            raise unittest.SkipTest("Se necesita el corpus procesado local")
        cls.evidencia = EvidenciaV04(RAIZ, CORPUS)

    def test_tramo_literal_une_lineas_partidas_por_la_fuente(self):
        # El Diario Oficial deja "LEY 890" y "DE 2004" en líneas seguidas.
        cabecera = self.evidencia.cabecera("ley_890_2004")
        self.assertIsNotNone(cabecera)
        self.assertEqual(cabecera["tipo_evidencia"], "cabecera_fuente")
        texto = self.evidencia.texto("ley_890_2004")
        self.assertEqual(texto[cabecera["inicio"]:cabecera["fin"]], cabecera["texto"])

    def test_cabecera_normalizada_la_reconoce_el_extractor_oficial(self):
        cabecera = self.evidencia.cabecera_normalizada("ley_1581_2012")
        self.assertIsNotNone(cabecera)
        citas = self.evidencia.citas
        cuerpos = citas.bodies(citas.extract(cabecera["texto"]))
        self.assertTrue(self.evidencia.identidad("ley_1581_2012") <= cuerpos)

    def test_cabecera_normalizada_no_declara_offsets(self):
        cabecera = self.evidencia.cabecera_normalizada("ley_1581_2012")
        self.assertNotIn("inicio", cabecera)
        self.assertNotIn("fin", cabecera)
        # verificar() comprueba offsets literales y debe dejarla pasar igual.
        self.evidencia.verificar([cabecera])

    def test_ningun_documento_identificable_queda_sin_cabecera(self):
        sin_nombre = [d for d in self.evidencia.documentos
                      if self.evidencia.identidad(d)
                      and not (self.evidencia.cabecera(d) or self.evidencia.cabecera_normalizada(d))]
        self.assertEqual(sin_nombre, [])


class FuentesDeDescarga(unittest.TestCase):
    """Patrones de URL y encadenado de las páginas partidas de Senado."""

    def url(self, canonico):
        _, url, _, _ = ampliar_corpus.plan_fuente({"areas": ["Derecho constitucional"]}, canonico)
        return url

    def test_las_su_van_sin_guion_y_las_c_y_t_con_guion(self):
        self.assertTrue(self.url(["jurisprudencia", "SU-16", "2020"]).endswith("/2020/su016-20.htm"))
        self.assertTrue(self.url(["jurisprudencia", "C-332", "2025"]).endswith("/2025/c-332-25.htm"))
        self.assertTrue(self.url(["jurisprudencia", "T-243", "2018"]).endswith("/2018/t-243-18.htm"))

    def test_sigue_la_cadena_de_paginas_de_senado(self):
        paginas = {
            "ley_0906_2004.html": b'x' * 2000 + b'<a href="ley_0906_2004_pr001.html">Siguiente</a>',
            "ley_0906_2004_pr001.html": b'y' * 2000 + b'<a href="ley_0906_2004_pr002.html">Siguiente</a>'
                                        b'<a href="ley_0906_2004.html">Anterior</a>',
            "ley_0906_2004_pr002.html": b'z' * 2000,
        }
        llamadas = []

        def falso(url, reintentos=2):
            llamadas.append(url)
            return 200, url, "text/html", paginas[url.rsplit("/", 1)[1]]

        original, ampliar_corpus.descargar = ampliar_corpus.descargar, falso
        espera, ampliar_corpus.time.sleep = ampliar_corpus.time.sleep, lambda s: None
        try:
            base = "http://www.secretariasenado.gov.co/senado/basedoc/ley_0906_2004.html"
            partes = ampliar_corpus.continuaciones_senado(base, paginas["ley_0906_2004.html"])
        finally:
            ampliar_corpus.descargar = original
            ampliar_corpus.time.sleep = espera
        self.assertEqual([p["url"].rsplit("/", 1)[1] for p in partes],
                         ["ley_0906_2004_pr001.html", "ley_0906_2004_pr002.html"])
        self.assertEqual(len(llamadas), 2)

    def test_el_pdf_se_valida_por_su_texto_extraido(self):
        ok, detalle = ampliar_corpus.validar(b"%PDF-1.4" + b"\x00" * 6000, "application/pdf",
                                            [r"\bsl\s*-?\s*3385\s*(-|de|/)\s*2022\b"])
        self.assertFalse(ok)
        self.assertNotIn("tras la ingesta", detalle)


class ManifiestoDeAmpliacion(unittest.TestCase):
    def test_registrar_guarda_un_original_por_tramo(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            raiz = Path(tmp)
            ficha = {"doc_id": "ley_1_2000", "titulo": "Ley 1 de 2000"}
            partes = [{"url": "http://x/a.html", "url_final": "http://x/a.html",
                       "tipo_contenido": "text/html", "contenido": b"<html>uno</html>"},
                      {"url": "http://x/a_pr001.html", "url_final": "http://x/a_pr001.html",
                       "tipo_contenido": "text/html", "contenido": b"<html>dos</html>"}]
            manifiesto = []
            ampliar_corpus.registrar(raiz, ficha, partes, manifiesto)
        entrada = manifiesto[0]
        self.assertEqual([a["archivo"] for a in entrada["archivos_raw"]],
                         ["ley_1_2000/000.html", "ley_1_2000/001.html"])
        self.assertEqual(entrada["url"], "http://x/a.html")
        self.assertTrue(json.dumps(entrada))


if __name__ == "__main__":
    unittest.main()
