"""Regresiones del parche DOCX AlternateContent; ejecutar tras aplicar el parche."""
import io
import unittest
import zipfile

from scripts.corpus.ingesta import extraer_docx


class DocxAlternateContentTests(unittest.TestCase):
    NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
          'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
          'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"')

    def extraer(self, body, notas=None):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as z:
            z.writestr("word/document.xml", f"<w:document {self.NS}><w:body>{body}</w:body></w:document>")
            if notas:
                z.writestr("word/footnotes.xml", f"<w:footnotes {self.NS}>{notas}</w:footnotes>")
        bloques, info = extraer_docx(stream.getvalue())
        return bloques[0]["texto"], info

    @staticmethod
    def parrafo(texto):
        return f"<w:p><w:r><w:t>{texto}</w:t></w:r></w:p>"

    @staticmethod
    def alternativas(choice, fallback, segundo_choice=""):
        return ("<mc:AlternateContent><mc:Choice Requires='wps'>" + choice + "</mc:Choice>" +
                ("<mc:Choice Requires='otra'>" + segundo_choice + "</mc:Choice>" if segundo_choice else "") +
                "<mc:Fallback>" + fallback + "</mc:Fallback></mc:AlternateContent>")

    def test_textbox_tabla_identica_se_extrae_una_vez_con_filas_y_columnas(self):
        tabla = ("<w:tbl><w:tr><w:tc>" + self.parrafo("Juzgado") + "</w:tc><w:tc>" +
                 self.parrafo("Fecha") + "</w:tc></w:tr><w:tr><w:tc>" + self.parrafo("Primero") +
                 "</w:tc><w:tc>" + self.parrafo("2017") + "</w:tc></w:tr></w:tbl>")
        caja = "<w:drawing><w:txbxContent>" + self.parrafo("Resumen") + tabla + "</w:txbxContent></w:drawing>"
        body = "<w:p><w:r><w:t>Antes</w:t>" + self.alternativas(caja, caja) + "<w:t>Después</w:t></w:r></w:p>"
        texto, info = self.extraer(body)
        self.assertEqual(texto, "Antes\nResumen\nJuzgado | Fecha\nPrimero | 2017\nDespués")
        self.assertEqual(info["alternativas_docx"], {"choice": 1, "fallback": 0, "sin_rama_textual": 0})

    def test_choice_sin_texto_wordml_usa_fallback(self):
        choice = "<m:oMath><m:r><m:t>FORMATO_NO_SOPORTADO</m:t></m:r></m:oMath>"
        fallback = "<w:r><w:t>Representación textual</w:t></w:r>"
        texto, info = self.extraer("<w:p>" + self.alternativas(choice, fallback) + "</w:p>")
        self.assertEqual(texto, "Representación textual")
        self.assertEqual(info["alternativas_docx"]["fallback"], 1)

    def test_multiple_choice_selecciona_solo_primero_textual(self):
        choice = "<w:r><w:t>PRIMERO</w:t></w:r>"
        segundo = "<w:r><w:t>SEGUNDO</w:t></w:r>"
        fallback = "<w:r><w:t>FALLBACK</w:t></w:r>"
        texto, info = self.extraer("<w:p>" + self.alternativas(choice, fallback, segundo) + "</w:p>")
        self.assertEqual(texto, "PRIMERO")

    def test_alternativa_como_bloque_preserva_parrafos(self):
        rama = self.parrafo("Primera línea") + self.parrafo("Segunda línea")
        texto, info = self.extraer(self.alternativas(rama, rama))
        self.assertEqual(texto, "Primera línea\nSegunda línea")
        self.assertEqual(info["alternativas_docx"]["choice"], 1)

    def test_notas_y_referencias_se_conservan_sin_duplicar_alternativa(self):
        rama = "<w:r><w:t>Contenido único</w:t></w:r>"
        nota = '<w:footnote w:id="7"><w:p>' + self.alternativas(rama, rama) + "</w:p></w:footnote>"
        body = '<w:p><w:r><w:t>Cuerpo</w:t><w:footnoteReference w:id="7"/></w:r></w:p>'
        texto, info = self.extraer(body, nota)
        self.assertEqual(texto.count("[NOTA 7]"), 2)
        self.assertEqual(texto.count("Contenido único"), 1)
        self.assertEqual(info["alternativas_docx"]["choice"], 1)

    def test_documento_sin_alternativas_no_cambia_texto_ni_notas(self):
        body = '<w:p><w:r><w:t>Título</w:t><w:br/><w:t>Cuerpo</w:t><w:footnoteReference w:id="1"/></w:r></w:p>'
        nota = '<w:footnote w:id="1">' + self.parrafo("Nota preservada") + "</w:footnote>"
        texto, info = self.extraer(body, nota)
        self.assertEqual(texto, "Título\nCuerpo [NOTA 1]\n[NOTA 1]\nNota preservada")
        self.assertEqual(info["alternativas_docx"], {"choice": 0, "fallback": 0, "sin_rama_textual": 0})


if __name__ == "__main__":
    unittest.main()

