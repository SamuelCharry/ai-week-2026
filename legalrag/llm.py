"""Cliente mínimo para un modelo abierto servido localmente con Ollama.

Temperatura 0 y semilla fija: la misma entrada produce la misma salida.
Antes de usar un modelo se verifica que no supere 8.000 millones de parámetros.
"""
import json
import os
import re

import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("LEGALRAG_MODEL", "qwen2.5:7b-instruct")
MAX_PARAMS_B = 8.0


def parameter_size_b(model: str) -> float:
    """Tamaño declarado por Ollama ('7.6B', '494.03M') en miles de millones."""
    try:
        r = requests.post(f"{OLLAMA_URL}/api/show", json={"model": model}, timeout=60)
    except requests.ConnectionError as e:
        raise RuntimeError(f"No responde Ollama en {OLLAMA_URL}: ábralo o ejecute `ollama serve`") from e
    if r.status_code == 404:
        raise RuntimeError(f"Ollama no tiene el modelo {model}: ejecute `ollama pull {model}`")
    r.raise_for_status()
    size = r.json().get("details", {}).get("parameter_size", "")
    m = re.fullmatch(r"([\d\.]+)\s*([BM])", size.strip(), re.I)
    if not m:
        raise RuntimeError(f"Ollama no informa el tamaño de {model} (parameter_size={size!r})")
    value = float(m.group(1))
    return value if m.group(2).upper() == "B" else value / 1000


def check_model(model: str) -> float:
    size = parameter_size_b(model)
    if size > MAX_PARAMS_B:
        raise RuntimeError(f"{model} tiene {size}B parámetros; el reto permite máximo {MAX_PARAMS_B}B")
    return size


def make_ollama_llm(model: str = DEFAULT_MODEL, num_ctx: int = 8192):
    """Devuelve llm(system, user, schema) -> dict."""
    check_model(model)

    def llm(system: str, user: str, schema: dict) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": schema,
            "options": {"temperature": 0, "seed": 0, "num_ctx": num_ctx},
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
        r.raise_for_status()
        return json.loads(r.json()["message"]["content"])

    llm.model = model
    return llm
