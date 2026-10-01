"""Query decomposition for weak retrieval.

When the initial retrieval returns low-confidence results (top reranker < 0.4),
decompose the question into 2-3 sub-queries and merge results via RRF.
Uses the same decoder (Qwen3-8B) with a short generation budget.
"""
from __future__ import annotations

import json
import re


def decompose(question_text: str, generator) -> list[str]:
    """Generate 2-3 sub-queries from the original question using the LLM."""
    messages = [
        {"role": "system", "content": (
            "Eres un asistente que descompone preguntas juridicas colombianas en sub-consultas "
            "de busqueda. Dada una pregunta, genera exactamente 3 sub-consultas cortas y "
            "complementarias que ayuden a recuperar los fragmentos normativos relevantes. "
            "Devuelve SOLO un array JSON de 3 strings, sin texto adicional."
        )},
        {"role": "user", "content": question_text},
    ]
    try:
        raw = generator.complete(messages, max_new_tokens=120, enable_thinking=False)
        raw = raw.strip()
        if raw.startswith("["):
            queries = json.loads(raw)
            if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
                return [q.strip() for q in queries if q.strip()][:3]
        matches = re.findall(r'"([^"]{10,})"', raw)
        return matches[:3]
    except Exception:
        return []
