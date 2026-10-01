"""Extracción de HTML con el marcado que usa Función Pública (fragmento real del
artículo 10 de la Ley 1581 de 2012, con el HTML de la página)."""
from legalrag.extract import clean_text, extract_html

HTML = """<html><head><meta charset="utf-8"><script>var x=1;</script></head><body>
<nav><li>Menú</li></nav>
<div class="descripcion-contenido">
<p>LEY ESTATUTARIA 1581 DE 2012</p>
<p><strong>Artículo <a id="sp10" name="10"></a>10.</strong> <em><strong>Casos en que no es necesaria la autorización.</strong> </em>La autorización del Titular no será necesaria cuando se trate de:</p>
<p>a) Información requerida por una entidad pública o administrativa en ejercicio de sus funciones legales o por orden judicial;</p>
<p>b) Datos de naturaleza pública;</p>
</div>
<footer><p>Carrera 6 # 12-62</p></footer></body></html>"""


def test_selector_toma_solo_la_norma():
    texto = clean_text(extract_html(HTML.encode("utf-8"), "div.descripcion-contenido"))
    assert texto == (
        "LEY ESTATUTARIA 1581 DE 2012\n\n"
        "Artículo 10. Casos en que no es necesaria la autorización. La autorización del Titular no será "
        "necesaria cuando se trate de:\n\n"
        "a) Información requerida por una entidad pública o administrativa en ejercicio de sus funciones "
        "legales o por orden judicial;\n\n"
        "b) Datos de naturaleza pública;\n"
    )


def test_respeta_el_charset_declarado():
    html = '<html><head><meta charset="windows-1252"></head><body><p>ARTÍCULO 1o. Protección</p></body></html>'
    assert extract_html(html.encode("cp1252")) == "ARTÍCULO 1o. Protección"
