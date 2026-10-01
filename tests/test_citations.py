"""Trazabilidad de citas: cada norma citada se compara con los pasajes recuperados."""
from legalrag.citations import extract_citations, format_citation, is_supported, norma_key

ALIASES = {"codigo general del proceso": "ley 1564 de 2012", "ley de proteccion de datos personales": "ley 1581 de 2012"}


def test_norma_key():
    assert norma_key("Ley Estatutaria 1581 de 2012") == "ley 1581 de 2012"
    assert norma_key("Decreto 1377 de 2013") == "decreto 1377 de 2013"
    assert norma_key("Decreto-Ley 019 de 2012") == "decreto ley 019 de 2012"


def test_extrae_articulo_y_norma_en_ambos_ordenes():
    t = "Según el artículo 10 de la Ley 1581 de 2012 y la Ley 1266 de 2008, artículo 3, ..."
    assert extract_citations(t) == [
        {"norma": "ley 1581 de 2012", "articulo": "10"},
        {"norma": "ley 1266 de 2008", "articulo": "3"},
    ]


def test_extrae_listas_y_alias():
    t = "Los artículos 14 y 15 de la Ley 1581 de 2012. El art. 42 del Código General del Proceso."
    assert extract_citations(t, ALIASES) == [
        {"norma": "ley 1581 de 2012", "articulo": "14"},
        {"norma": "ley 1581 de 2012", "articulo": "15"},
        {"norma": "ley 1564 de 2012", "articulo": "42"},
    ]


def test_articulo_sin_norma():
    assert extract_citations("como dice el artículo 2°") == [{"norma": None, "articulo": "2"}]


def test_respaldo_contra_pasajes_recuperados(pasajes):
    recuperados = [p for p in pasajes if p["articulo"] in ("10", "15")]
    assert is_supported({"norma": "ley 1581 de 2012", "articulo": "10"}, recuperados)
    # mismo artículo, otra norma: no está respaldado
    assert not is_supported({"norma": "ley 1564 de 2012", "articulo": "10"}, recuperados)
    # artículo de la norma que no fue recuperado
    assert not is_supported({"norma": "ley 1581 de 2012", "articulo": "26"}, recuperados)


def test_cada_cita_respaldada_se_rastrea_hasta_la_fuente(pasajes, ley_texto):
    cita = {"norma": "ley 1581 de 2012", "articulo": "26"}
    p = next(p for p in pasajes if p["articulo"] == "26")
    assert is_supported(cita, [p])
    assert ley_texto[p["inicio"]:p["fin"]] == p["texto"]
    assert p["url"] == "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981"


def test_formato_de_cita():
    nombres = {"ley 1581 de 2012": "Ley 1581 de 2012"}
    assert format_citation({"norma": "ley 1581 de 2012", "articulo": "10"}, nombres) == "Ley 1581 de 2012, artículo 10"
