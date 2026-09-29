"""Regresiones de formatos y procedencia del corpus recibido, sin modificar raw."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.corpus.ingesta import (
    candidatos_articulo, diagnosticar_continuidad, ejecutar_ingesta, extraer_docx,
    extraer_html, normalizar, procesar_documento, segmentar_documento, sha256,
    unidades_de_fragmentos,
)


class IngestaCorpusNuevo(unittest.TestCase):
    def setUp(self):
        self.temporal = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporal.cleanup)
        self.raw = Path(self.temporal.name)

    def archivo(self, nombre, contenido):
        ruta = self.raw / nombre
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(contenido)
        return {"archivo": nombre, "bytes": len(contenido), "sha256": sha256(contenido),
                "url": "https://fuente.example/documento"}

    def documento(self, archivo, tipo="sentencia"):
        return {"doc_id": "fixture", "tipo": tipo, "archivos_raw": [archivo],
                "redistribuir_raw": True}

    def docx(self, xml, notas=None):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as z:
            z.writestr("word/document.xml", xml)
            if notas:
                z.writestr("word/footnotes.xml", notas)
        return buffer.getvalue()

    def test_derivado_word_verificado_conserva_procedencia_y_offsets(self):
        original = self.archivo("caso/000.doc", b"\xd0\xcf\x11\xe0binary-not-text")
        derivado = self.archivo("caso/000.txt", "ANTECEDENTES\nTexto jurídico extraído.".encode())
        original["texto_derivado"] = {**derivado, "metodo": "antiword", "motivo": "Word"}
        registro, texto = procesar_documento(self.documento(original), self.raw)
        fs = segmentar_documento(registro, texto)
        self.assertEqual(texto, "ANTECEDENTES\nTexto jurídico extraído.")
        self.assertEqual(registro["extraccion"][0]["sha256_texto_derivado"], derivado["sha256"])
        self.assertTrue(all(f["texto"] == texto[f["inicio"]:f["fin"]] for f in fs))
        self.assertTrue(all(o["archivo"] == "caso/000.doc" and o["archivo_texto_derivado"] == "caso/000.txt"
                            for f in fs for o in f["origenes"]))
        self.assertEqual((self.raw / original["archivo"]).read_bytes(), b"\xd0\xcf\x11\xe0binary-not-text")

    def test_rechaza_hash_del_derivado_sin_fallback_silencioso(self):
        original = self.archivo("caso/000.doc", b"binary")
        td = self.archivo("caso/000.txt", b"uno")
        original["texto_derivado"] = td
        (self.raw / td["archivo"]).write_bytes(b"dos")
        with self.assertRaisesRegex(ValueError, "no coincide"):
            procesar_documento(self.documento(original), self.raw)

    def test_derivado_no_permite_ignorar_hash_original(self):
        original = self.archivo("caso/000.doc", b"binary")
        original["texto_derivado"] = self.archivo("caso/000.txt", b"texto")
        (self.raw / original["archivo"]).write_bytes(b"cambio")
        with self.assertRaisesRegex(ValueError, "no coincide"):
            procesar_documento(self.documento(original), self.raw)

    def test_rechaza_derivado_fuera_de_raw(self):
        original = self.archivo("caso/000.doc", b"binary")
        original["texto_derivado"] = {"archivo": "../escape.txt", "bytes": 1, "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "sale del directorio"):
            procesar_documento(self.documento(original), self.raw)

    def test_doc_sin_derivado_no_se_decodifica(self):
        original = self.archivo("caso/000.doc", b"aparenta texto pero es un DOC")
        with self.assertRaisesRegex(ValueError, "Word binario"):
            procesar_documento(self.documento(original), self.raw)

    def test_ocr_usa_derivado_y_formfeed_con_paginacion_verificada(self):
        original = self.archivo("caso/000.pdf", b"%PDF-1.4\ncontenido solo para prueba")
        original["texto_derivado"] = {**self.archivo("caso/000.ocr.txt", "Página primera.\fPágina segunda.\f".encode()),
                                      "metodo": "tesseract"}
        with patch("scripts.corpus.ingesta.pdfplumber.open") as abrir, patch(
                "scripts.corpus.ingesta.extraer_pdf", side_effect=AssertionError("No usar capa PDF")):
            abrir.return_value.__enter__.return_value.pages = [object(), object()]
            registro, texto = procesar_documento(self.documento(original), self.raw)
        self.assertTrue(registro["extraccion"][0]["mapeo_paginas_verificado"])
        self.assertEqual([p["pagina"] for p in registro["partes"]], [1, 2])
        self.assertEqual([texto[p["inicio"]:p["fin"]] for p in registro["partes"]],
                         ["Página primera.", "Página segunda."])
        self.assertIn("revisar_calidad_OCR", registro["avisos"])

    def test_ocr_con_paginas_discrepantes_no_inventa_numeros(self):
        original = self.archivo("caso/000.pdf", b"%PDF-1.4\nprueba")
        original["texto_derivado"] = self.archivo("caso/000.ocr.txt", b"primera\fsegunda")
        with patch("scripts.corpus.ingesta.pdfplumber.open") as abrir:
            abrir.return_value.__enter__.return_value.pages = [object()] * 3
            registro, texto = procesar_documento(self.documento(original), self.raw)
        self.assertTrue(all(p["pagina"] is None for p in registro["partes"]))
        self.assertIn("paginacion_derivado_no_verificada", registro["avisos"])
        self.assertIn("primera", texto)
        self.assertIn("segunda", texto)

    def test_docx_con_tablas_notas_saltos_y_sin_paginacion_inventada(self):
        ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        xml = f'<w:document {ns}><w:body><w:p><w:r><w:t>Título</w:t><w:br/><w:t>Texto</w:t>' \
              '<w:footnoteReference w:id="1"/></w:r></w:p><w:tbl><w:tr><w:tc><w:p><w:r><w:t>Columna A</w:t>' \
              '</w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Columna B</w:t></w:r></w:p></w:tc></w:tr></w:tbl>' \
              '</w:body></w:document>'
        notas = f'<w:footnotes {ns}><w:footnote w:id="1"><w:p><w:r><w:t>Nota conservada</w:t>' \
                '</w:r></w:p></w:footnote></w:footnotes>'
        original = self.archivo("caso/000.docx", self.docx(xml, notas))
        registro, texto = procesar_documento(self.documento(original), self.raw)
        self.assertIn("Título\nTexto [NOTA 1]", texto)
        self.assertIn("Columna A | Columna B", texto)
        self.assertIn("Nota conservada", texto)
        self.assertIsNone(registro["partes"][0]["pagina"])

    def test_docx_rechaza_declaraciones_de_entidad(self):
        xml = '<!DOCTYPE a [<!ENTITY texto "expansion">]><a>&texto;</a>'
        with self.assertRaisesRegex(ValueError, "DTD o ENTITY"):
            extraer_docx(self.docx(xml))

    def test_docx_nativo_recupera_notas_omitidas_por_sidecar_y_verifica_ambos(self):
        ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
        xml = f'<w:document {ns}><w:body><w:p><w:r><w:t>Cuerpo.</w:t>' \
              '<w:footnoteReference w:id="1"/></w:r></w:p></w:body></w:document>'
        notas = f'<w:footnotes {ns}><w:footnote w:id="1"><w:p><w:r><w:t>Fundamento de la nota.</w:t>' \
                '</w:r></w:p></w:footnote></w:footnotes>'
        original = self.archivo("caso/000.docx", self.docx(xml, notas))
        derivado = {**self.archivo("caso/000.txt", b"Cuerpo."), "metodo": "solo word/document.xml"}
        original["texto_derivado"] = derivado
        registro, texto = procesar_documento(self.documento(original), self.raw)
        self.assertIn("Fundamento de la nota.", texto)
        info = registro["extraccion"][0]
        self.assertEqual(info["selector"], "docx_xml_stdlib")
        self.assertEqual(info["derivado_conservado_no_utilizado"]["sha256"], derivado["sha256"])
        self.assertIsNone(registro["partes"][0]["archivo_texto_derivado"])
        (self.raw / derivado["archivo"]).write_bytes(b"Cambios")
        with self.assertRaisesRegex(ValueError, "no coincide"):
            procesar_documento(self.documento(original), self.raw)

    def test_senado_retira_controles_y_pie_conserva_titulo_y_vigencia(self):
        cuerpo = "<p>ARTÍCULO 1. " + "Texto legal. " * 25 + "</p>"
        html = '<div id="aj_data"><a class="hlk_inicio">Inicio</a><div id="selector_aj">Selector</div>' \
               '<header><h1>LEY 100 DE 1993</h1></header>' + cuerpo + \
               '<div class="caja_vja">Notas de vigencia: modificación.</div>' \
               '<a class="antsig">Siguiente</a><div id="logo_aj">Disposiciones analizadas por editorial</div></div>'
        bloques, info = extraer_html(html.encode(), {})
        texto = bloques[0]["texto"]
        for conservar in ["LEY 100 DE 1993", "ARTÍCULO 1.", "Notas de vigencia"]:
            self.assertIn(conservar, texto)
        for retirar in ["Inicio", "Selector", "Siguiente", "Disposiciones analizadas"]:
            self.assertNotIn(retirar, texto)
        self.assertEqual(len(info["elementos_html_retirados"]), 4)

    def test_word_css_aplanado_acotado_antes_del_primer_parrafo(self):
        html = '<div class="descripcion-contenido">BIBIANA\nNormal\n/* Style Definitions */\ntable.MsoNormalTable { x:y; }' \
               '<p>LEY 27 DE 1977</p><p>ARTÍCULO 1. ' + "Texto legal. " * 25 + '</p></div>'
        bloques, info = extraer_html(html.encode(), {})
        self.assertTrue(bloques[0]["texto"].startswith("LEY 27 DE 1977"))
        self.assertNotIn("BIBIANA", bloques[0]["texto"])
        self.assertEqual(info["elementos_html_retirados"][0]["motivo"], "metadatos_Word_y_CSS_aplanados")
        html = html.replace("BIBIANA", "ARTÍCULO 9 texto legal antes del CSS")
        bloques, _ = extraer_html(html.encode(), {})
        self.assertIn("ARTÍCULO 9", bloques[0]["texto"])

    def test_conceptos_circulares_y_compendios_no_reciben_articulos_propios(self):
        texto = "ARTÍCULO 1. Texto citado.\nARTÍCULO 2. Otro texto."
        for tipo in ("concepto", "circular", "compendio", "desconocido", "sentencia", "auto"):
            with self.subTest(tipo=tipo):
                documento = {"doc_id": "x", "tipo": tipo, "partes": [
                    {"archivo": "x", "url": None, "pagina": None, "inicio": 0, "fin": len(texto)}]}
                fs = segmentar_documento(documento, texto)
                us = unidades_de_fragmentos(documento, texto, fs)
                self.assertTrue(all(f["articulo"] is None for f in fs))
                self.assertEqual(diagnosticar_continuidad(documento, texto, us)["candidatos"], 0)

    def test_variantes_reales_cabecera_articular(self):
        texto = "ARTICULO. 1º—Texto\nArtículo 2o\n. Creación."
        self.assertEqual([c["numero"] for c in candidatos_articulo(texto)], ["1", "2"])
        self.assertEqual(normalizar("fin\finicio"), "fin\ninicio")

    def test_charset_cp1252_declarado_no_corrompe_enes_por_byte_indefinido(self):
        contenido = b'<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">' \
                    b'<p>Lo se\xf1alado en el a\xf1o. Byte no asignado: \x81.</p>'
        bloques, info = extraer_html(contenido, {"encoding": "auto"})
        self.assertIn("señalado en el año", bloques[0]["texto"])
        self.assertIn("\ufffd", bloques[0]["texto"])
        self.assertEqual(info["encoding"], "windows-1252")
        self.assertEqual(info["n_errores_decodificacion"], 1)
        original = self.archivo("caso/000.html", contenido)
        registro, _ = procesar_documento(self.documento(original), self.raw)
        self.assertIn("caracteres_de_reemplazo", registro["avisos"])
        self.assertIn("bytes_no_decodificables_en_charset_declarado", registro["avisos"])

    def test_utf8_valido_tiene_prioridad_sobre_meta_legacy_incorrecta(self):
        contenido = '<meta charset="windows-1252"><p>año, niño y acción.</p>'.encode("utf8")
        bloques, info = extraer_html(contenido, {})
        self.assertEqual(bloques[0]["texto"], "año, niño y acción.")
        self.assertEqual(info["encoding_origen"], "utf8_comprobado")
        self.assertEqual(info["n_errores_decodificacion"], 0)

    def test_metadatos_corpus_nuevo_se_conservan_en_fragmentos(self):
        original = self.archivo("caso/000.html", b"<p>ARTICULO 1. Texto de prueba.</p>")
        documento = {**self.documento(original, "ley"), "nivel": "nucleo",
                     "vigencia_fuente": "sin_marca", "origen_ampliacion": "grafo_normativo"}
        registro, texto = procesar_documento(documento, self.raw)
        for fragmento in segmentar_documento(registro, texto):
            for campo in ("nivel", "vigencia_fuente", "origen_ampliacion"):
                self.assertEqual(fragmento[campo], documento[campo])

    def test_ejecutar_ingesta_acepta_raw_explicito(self):
        original = self.archivo("caso/000.html", b"<p>ARTICULO 1. Texto de prueba.</p>")
        (self.raw / "manifest.json").write_text(json.dumps([self.documento(original, "ley")]), encoding="utf8")
        resultado = ejecutar_ingesta(self.raw / "otra_raiz", raw_dir=self.raw)
        self.assertEqual(resultado["resumen"]["textos_guardados"], 1)


if __name__ == "__main__":
    unittest.main()
