"""Los tres formatos JSON del Anexo A, la verificación de citas y la abstención.

El modelo de lenguaje se reemplaza por una función que devuelve una salida fija
(doble de prueba): así se prueba el posprocesamiento sin GPU ni Ollama. La
ejecución con el modelo real es `python cli.py demo`.
"""
from legalrag.answer import CAMPOS, answer, validate
from legalrag.retrieve import Retriever

ALIASES = {"ley 1581 de 2012": "ley 1581 de 2012"}
NOMBRES = {"ley 1581 de 2012": "Ley 1581 de 2012"}


def fake_llm(salida):
    calls = []

    def llm(system, user, schema):
        calls.append({"user": user, "schema": schema})
        return salida
    llm.calls = calls
    return llm


def run(item, salida, pasajes):
    r = Retriever(pasajes, aliases=ALIASES)
    return answer(item, r, fake_llm(salida), ALIASES, NOMBRES, k=10)


def test_semi_open(pasajes):
    item = {"id": 512, "formato": "semi_open",
            "pregunta": "¿En qué casos no es necesaria la autorización del Titular?"}
    salida = {"abstencion": False,
              "respuesta": "No se requiere autorización para datos de naturaleza pública, entre otros casos. "
                           "Así lo dispone el artículo 10 de la Ley 1581 de 2012.",
              "palabras_clave": ["autorización", "datos públicos"],
              "referencia_legal": "Artículo 10 de la Ley 1581 de 2012; artículo 99 de la Ley 1581 de 2012"}
    out, traza = run(item, salida, pasajes)
    assert validate(out) == []
    assert list(out) == ["id", "formato", "abstencion", *CAMPOS["semi_open"], "pasajes_recuperados"]
    # la cita inexistente (art. 99) no llega a referencia_legal
    assert out["referencia_legal"] == "Ley 1581 de 2012, artículo 10"
    assert {"norma": "ley 1581 de 2012", "articulo": "99"} in traza["citas_sin_respaldo"]
    assert 0 < len(out["pasajes_recuperados"]) <= 10
    assert set(out["pasajes_recuperados"][0]) == {"doc_id", "inicio", "fin", "texto", "score"}
    assert traza["pasajes"][0]["articulo"] == "10"


def test_multiple_choice(pasajes):
    item = {"id": 1, "formato": "multiple_choice",
            "pregunta": "¿Cuál es el término máximo para atender un reclamo?",
            "opciones": {"A": "5 días", "B": "10 días", "C": "15 días hábiles", "D": "30 días"}}
    salida = {"abstencion": False, "respuesta_correcta": "c",
              "justificacion": "El artículo 15 de la Ley 1581 de 2012 fija quince días hábiles.",
              "descarte_opciones": {"A": "no", "B": "es el de consultas", "C": "x", "D": "no", "E": "no existe"}}
    out, _ = run(item, salida, pasajes)
    assert validate(out) == []
    assert out["respuesta_correcta"] == "C"
    assert set(out["descarte_opciones"]) == {"A", "B", "D"}


def test_open_ended(pasajes):
    item = {"id": 7, "formato": "open_ended",
            "pregunta": "¿Se pueden transferir datos personales a un país sin nivel adecuado de protección?"}
    salida = {"abstencion": False, "marco_normativo": "Artículo 26 de la Ley 1581 de 2012.",
              "analisis": "La regla general es la prohibición.", "jurisprudencia": "",
              "conclusion": "Solo en las excepciones del artículo 26 de la Ley 1581 de 2012."}
    out, traza = run(item, salida, pasajes)
    assert validate(out) == []
    assert out["abstencion"] is False
    assert traza["citas_sin_respaldo"] == []


def test_abstencion_si_el_modelo_la_pide(pasajes):
    item = {"id": 987, "formato": "open_ended", "pregunta": "¿Qué dice la ley sobre el derecho de superficie?"}
    out, traza = run(item, {"abstencion": True}, pasajes)
    assert out == {"id": 987, "formato": "open_ended", "abstencion": True, "marco_normativo": "",
                   "analisis": "", "jurisprudencia": "", "conclusion": "", "pasajes_recuperados": []}
    assert validate(out) == []


def test_abstencion_si_ninguna_cita_esta_respaldada(pasajes):
    item = {"id": 3, "formato": "semi_open", "pregunta": "¿Cuál es el término para atender un reclamo?"}
    salida = {"abstencion": False, "respuesta": "Según el artículo 391 del Código General del Proceso, diez días.",
              "palabras_clave": ["término"], "referencia_legal": "artículo 391 de la Ley 1564 de 2012"}
    out, traza = run(item, salida, pasajes)
    assert out["abstencion"] is True
    assert traza["motivo_abstencion"] == "ninguna norma citada figura en los pasajes recuperados"


def test_abstencion_sin_pasajes(pasajes):
    item = {"id": 4, "formato": "semi_open", "pregunta": "zzzz qqqq"}
    out, traza = run(item, {"abstencion": False, "respuesta": "x"}, pasajes)
    assert out["abstencion"] is True and traza["motivo_abstencion"] == "sin pasajes recuperados"


def test_prompt_lleva_norma_y_articulo_de_cada_pasaje(pasajes):
    item = {"id": 1, "formato": "semi_open", "pregunta": "¿Qué es un dato sensible?"}
    llm = fake_llm({"abstencion": True})
    answer(item, Retriever(pasajes, aliases=ALIASES), llm, ALIASES, NOMBRES)
    assert "[1] Ley 1581 de 2012, artículo 5" in llm.calls[0]["user"]
    assert llm.calls[0]["schema"]["required"] == ["abstencion", *CAMPOS["semi_open"]]


def test_respuesta_semiabierta_se_limita_a_150_palabras(pasajes):
    item = {"id": 5, "formato": "semi_open", "pregunta": "¿Qué es un dato sensible?"}
    larga = ("Los datos sensibles están definidos en el artículo 5 de la Ley 1581 de 2012. " * 20).strip()
    out, _ = run(item, {"abstencion": False, "respuesta": larga, "palabras_clave": [], "referencia_legal": ""}, pasajes)
    assert len(out["respuesta"].split()) <= 150 and out["respuesta"].endswith(".")
