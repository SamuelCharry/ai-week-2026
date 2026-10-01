"""Aproximación local y gratuita del componente RAGAS del evaluador oficial (30 puntos), para comparar variantes.

El evaluador oficial (data/oficial/scripts/evaluate.py, score_ragas) usa RAGAS *answer correctness* contra la
`respuesta_esperada`, con un juez LLM de OpenRouter (créditos limitados):

    answer_correctness = 0,75 · corrección factual + 0,25 · similitud semántica
    corrección factual = TP / (TP + 0,5 · (FP + FN))     afirmaciones que coinciden, sobran o faltan

Aquí:
    similitud semántica  el mismo encoder del evaluador (intfloat/multilingual-e5-large), coseno de los textos
    corrección factual   con el NLI multilingüe (mDeBERTa-xnli): TP ≈ oraciones de la respuesta que la esperada
                         implica; FP ≈ las que no; FN ≈ oraciones de la esperada que la respuesta no implica

No reemplaza al juez: sirve para ordenar variantes de redacción (concisión, afirmaciones de más) antes de gastar
una corrida de RAGAS, que es la que calibra esta aproximación. Las abstenciones valen 0, como en el oficial.
"""
from legalrag.generation.politica import _ORACION

ENCODER = {"repo_id": "intfloat/multilingual-e5-large", "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"}


def texto_ragas(respuesta):
    """Mismo texto que manda el evaluador oficial al juez (evaluate.ragas_text)."""
    if respuesta.get("formato") == "semi_open":
        return str(respuesta.get("respuesta") or "")
    return " ".join(str(respuesta.get(k) or "") for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion"))


def oraciones(texto):
    return [o.strip() for o in _ORACION.split(texto or "") if len(o.strip()) > 15]


class RagasLocal:
    def __init__(self, ficha_nli, dispositivo="cuda"):
        self.ficha_nli, self.dispositivo = ficha_nli, dispositivo

    def abrir(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from legalrag.citations.respaldo_nli import VerificadorNLI

        self.nli = VerificadorNLI({"modelo": self.ficha_nli, "max_tokens": 512}, self.dispositivo)
        self.nli.abrir()
        self.tok = AutoTokenizer.from_pretrained(ENCODER["repo_id"], revision=ENCODER["revision"])
        self.enc = AutoModel.from_pretrained(ENCODER["repo_id"], revision=ENCODER["revision"],
                                             dtype=torch.float32).to(self.dispositivo).eval()

    def cerrar(self):
        self.nli.cerrar()
        self.enc = None

    def _vector(self, texto):
        import torch

        t = self.tok([texto], truncation=True, max_length=512, return_tensors="pt").to(self.dispositivo)
        with torch.inference_mode():
            oculto = self.enc(**t).last_hidden_state
            mascara = t["attention_mask"].unsqueeze(-1)
            vector = (oculto * mascara).sum(1) / mascara.sum(1)
        return torch.nn.functional.normalize(vector, dim=-1)[0]

    def puntuar(self, respuesta_texto, esperada):
        """(answer_correctness aproximada, detalle)."""
        similitud = float(self._vector(respuesta_texto) @ self._vector(esperada))
        propias, esperadas = oraciones(respuesta_texto), oraciones(esperada)
        if not propias or not esperadas:
            return 0.25 * similitud, {"similitud": round(similitud, 3), "factual": 0.0}
        tp = sum(implica >= 0.5 for implica, _, _ in self.nli.puntuar(propias, [esperada]))
        cubiertas = sum(implica >= 0.5 for implica, _, _ in self.nli.puntuar(esperadas, [respuesta_texto]))
        fp, fn = len(propias) - tp, len(esperadas) - cubiertas
        factual = tp / (tp + 0.5 * (fp + fn)) if tp else 0.0
        return 0.75 * factual + 0.25 * similitud, {"similitud": round(similitud, 3), "factual": round(factual, 3),
                                                     "tp": tp, "fp": fp, "fn": fn}

    def evaluar(self, entregas, muestra):
        """Promedio sobre las preguntas de texto libre de la muestra (abstención = 0) y detalle por id."""
        puntajes, detalle = [], {}
        for pregunta in muestra:
            if pregunta["formato"] == "multiple_choice":
                continue
            s = entregas.get(pregunta["id"])
            if not s or s.get("abstencion"):
                puntajes.append(0.0)
                continue
            puntaje, d = self.puntuar(texto_ragas(s), pregunta.get("respuesta_esperada") or "")
            puntajes.append(puntaje)
            detalle[pregunta["id"]] = {**d, "aprox": round(puntaje, 3)}
        promedio = sum(puntajes) / len(puntajes) if puntajes else 0.0
        return {"correctness_aprox": round(promedio, 4), "puntos_aprox": round(30 * promedio, 2), "detalle": detalle}
