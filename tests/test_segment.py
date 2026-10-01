"""Segmentación por artículo sobre una norma real (Ley 1581 de 2012) y casos de
formato de encabezado que aparecen en otras normas colombianas."""
from legalrag.extract import clean_text
from legalrag.segment import normalize_article_number, segment_document

DOC = {"doc_id": "x", "norma": "Ley 1 de 2000"}


def arts(pasajes):
    return [p["articulo"] for p in pasajes if p["parte"] == 1]


def by_art(pasajes, n):
    return [p for p in pasajes if p["articulo"] == n]


def test_detecta_los_30_articulos_en_orden(pasajes):
    assert arts(pasajes) == [str(i) for i in range(1, 31)]


def test_offsets_reconstruyen_el_texto_exacto(pasajes, ley_texto):
    for p in pasajes:
        assert ley_texto[p["inicio"]:p["fin"]] == p["texto"]
        assert p["doc_id"] == "ley_1581_2012" and p["url"].startswith("https://")


def test_cada_pasaje_empieza_con_su_articulo(pasajes):
    for p in pasajes:
        if p["parte"] == 1:
            assert p["texto"].startswith(f"Artículo {p['articulo']}")


def test_incisos_y_paragrafos_quedan_en_su_articulo(pasajes):
    art2 = "\n\n".join(p["texto"] for p in by_art(pasajes, "2"))
    assert "f) A las bases de datos y archivos regulados por la Ley 79 de 1993." in art2
    assert "Parágrafo. Los principios sobre protección de datos" in art2
    art19 = by_art(pasajes, "19")
    assert len(art19) == 1
    assert art19[0]["paragrafos"] == ["Parágrafo 1°", "Parágrafo 2°"]
    art15 = by_art(pasajes, "15")[0]["texto"]
    assert "3. El término máximo para atender el reclamo será de quince (15) días hábiles" in art15


def test_no_mezcla_encabezados_ni_firmas(pasajes):
    todo = "\n".join(p["texto"] for p in pasajes)
    assert "TÍTULO" not in todo and "CAPÍTULO" not in todo
    assert "El Presidente del honorable Senado" not in todo
    assert by_art(pasajes, "30")[0]["texto"] == \
        "Artículo 30. Vigencia. La presente ley rige a partir de su promulgación."


def test_seccion_registra_titulo_y_capitulo(pasajes):
    s = by_art(pasajes, "19")[0]["seccion"]
    assert s.startswith("TÍTULO VII DE LOS MECANISMOS DE VIGILANCIA Y SANCIÓN")
    assert "CAPÍTULO I De la autoridad de protección de datos" in s
    # un TÍTULO nuevo reemplaza el capítulo anterior
    assert by_art(pasajes, "26")[0]["seccion"] == "TÍTULO VIII TRANSFERENCIA DE DATOS A TERCEROS PAÍSES"


def test_articulo_largo_se_divide_sin_perder_la_referencia(ley_texto):
    ps = segment_document(ley_texto, {"doc_id": "ley_1581_2012", "norma": "Ley 1581 de 2012"}, max_chars=800)
    art17 = by_art(ps, "17")
    assert len(art17) > 1
    assert [p["parte"] for p in art17] == list(range(1, len(art17) + 1))
    assert all(p["partes"] == len(art17) for p in art17)
    assert len({p["chunk_id"] for p in art17}) == len(art17)
    # las partes son consecutivas y cubren el artículo completo
    completo = by_art(segment_document(ley_texto, {"doc_id": "d", "norma": "n"}, max_chars=10**6), "17")[0]
    assert art17[0]["inicio"] == completo["inicio"] and art17[-1]["fin"] == completo["fin"]
    for a, b in zip(art17, art17[1:]):
        assert ley_texto[a["fin"]:b["inicio"]].strip() == ""
    assert all(len(p["texto"]) <= 800 for p in art17)


def test_clean_text_no_altera_la_norma(ley_texto):
    assert clean_text(ley_texto) == ley_texto


def test_variantes_de_encabezado():
    texto = "\n\n".join([
        "ARTÍCULO 42. DEBERES DEL JUEZ. Son deberes del juez:",
        "1. Dirigir el proceso.",
        "PARÁGRAFO. Texto del parágrafo.",
        "ARTICULO 1o. Colombia es un Estado social de derecho.",
        "ARTÍCULO 2.2.1.1.1. Objeto. Texto de decreto único.",
        "ARTÍCULO 20-A. Texto de artículo adicionado.",
        "ARTÍCULO 20 BIS. Otro adicionado.",
        "ARTÍCULO TRANSITORIO 1o. Texto transitorio.",
        "ARTÍCULO PRIMERO. Texto de resolución.",
        "Artículo 15 de la Constitución Política: esta línea es una referencia, no un artículo.",
    ])
    ps = segment_document(texto, DOC)
    assert arts(ps) == ["42", "1", "2.2.1.1.1", "20A", "20bis", "transitorio 1", "primero"]
    assert "PARÁGRAFO. Texto del parágrafo." in by_art(ps, "42")[0]["texto"]
    assert by_art(ps, "42")[0]["paragrafos"] == ["PARÁGRAFO"]
    assert "referencia, no un artículo" in by_art(ps, "primero")[0]["texto"]


def test_numero_repetido_no_se_pierde():
    texto = "Artículo 5. Versión vigente.\n\nArtículo 5. Otra versión en la misma página."
    ps = segment_document(texto, DOC)
    assert [p["chunk_id"] for p in ps] == ["x:art5", "x:art5#2"]
    assert [p["repeticion"] for p in ps] == [1, 2]


def test_normalizacion_de_numeros():
    assert normalize_article_number("1°") == "1"
    assert normalize_article_number("5o") == "5"
    assert normalize_article_number("20 - A") == "20A"
