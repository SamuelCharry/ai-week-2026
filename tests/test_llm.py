"""El sistema rechaza modelos de más de 8.000 millones de parámetros (sin Ollama:
se simula la respuesta de /api/show)."""
import pytest

from legalrag import llm


class FakeResponse:
    def __init__(self, size):
        self.size = size
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"details": {"parameter_size": self.size}}


@pytest.mark.parametrize("size,ok", [("7.6B", True), ("8.0B", True), ("494.03M", True), ("8.2B", False), ("70.6B", False)])
def test_limite_de_parametros(monkeypatch, size, ok):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(size))
    if ok:
        assert llm.check_model("m") <= 8.0
    else:
        with pytest.raises(RuntimeError, match="máximo"):
            llm.check_model("m")
