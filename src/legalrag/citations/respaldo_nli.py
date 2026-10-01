"""Verificación de respaldo por inferencia textual (NLI), sin generar texto.

Cada oración de una respuesta de texto libre se compara con los pasajes entregados mediante un modelo
NLI multilingüe (mDeBERTa-v3-base-xnli, licencia MIT, 279 M parámetros): ¿algún pasaje la implica,
la contradice o no dice nada de ella? Es la idea de MiniCheck (Tang et al., EMNLP 2024): un verificador
pequeño alcanza a los grandes en revisar respaldo documental. MiniCheck solo cubre inglés; mDeBERTa,
más de 100 idiomas. Reemplaza a CoVe (tres generaciones, ~3 s) por unos cientos de milisegundos.

    modo "registrar"            solo anota cada oración con sus puntajes (para calibrar el umbral)
    modo "quitar_contradichas"  quita las oraciones que algún pasaje contradice (probabilidad >= umbral)
                                y que ningún pasaje implica. Nunca deja un campo vacío ni toca oraciones
                                que citan normas: esas ya pasan por la verificación de citas.

Las citas de normas siguen verificándose aparte (citations.verificacion); esto revisa el resto del texto,
que es lo que mide el juez RAGAS.
"""
from legalrag.generation.politica import _ORACION

CAMPOS_LIBRES = {"semi_open": ("respuesta",),
                 "open_ended": ("marco_normativo", "analisis", "jurisprudencia", "conclusion")}


class VerificadorNLI:
    def __init__(self, config, dispositivo="cuda"):
        self.config, self.dispositivo = config, dispositivo
        self.tokenizer = self.modelo = None

    def abrir(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        ficha = self.config["modelo"]
        self.tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], revision=ficha["revision"])
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            ficha["repo_id"], revision=ficha["revision"], dtype=torch.float32).to(self.dispositivo).eval()
        etiquetas = {str(v).lower(): int(k) for k, v in self.modelo.config.id2label.items()}
        self.implica, self.contradice = etiquetas["entailment"], etiquetas["contradiction"]

    def cerrar(self):
        self.modelo = None

    def puntuar(self, oraciones, premisas):
        """[(implica, contradice, pasaje)] por oración: la mayor probabilidad de cada una entre los pasajes."""
        import torch

        pares = [(p, o) for o in oraciones for p in premisas]
        probabilidades, lote = [], self.config.get("lote", 32)
        for i in range(0, len(pares), lote):
            premisa, hipotesis = zip(*pares[i:i + lote])
            entrada = self.tokenizer(list(premisa), list(hipotesis), truncation="only_first", padding=True,
                                     max_length=self.config.get("max_tokens", 512), return_tensors="pt").to(self.dispositivo)
            with torch.inference_mode():
                probabilidades += torch.softmax(self.modelo(**entrada).logits.float(), dim=-1).cpu().tolist()
        resultado = []
        for n in range(len(oraciones)):
            filas = probabilidades[n * len(premisas):(n + 1) * len(premisas)]
            mejor = max(range(len(filas)), key=lambda j: filas[j][self.implica])
            resultado.append((filas[mejor][self.implica], max(f[self.contradice] for f in filas), mejor + 1))
        return resultado

    def verificar(self, respuesta, pasajes, citas):
        """Registro de la verificación; en modo "quitar_contradichas" también corrige `respuesta`."""
        from legalrag.citations.normas import EvidenciaCorpus

        campos = CAMPOS_LIBRES.get(respuesta.get("formato"), ())
        oraciones = [(campo, o) for campo in campos for o in _ORACION.split(respuesta.get(campo) or "") if o.strip()]
        premisas = [EvidenciaCorpus.texto_entregado(p) for p in pasajes[:10]]
        if not oraciones or not premisas:
            return {"oraciones": 0}
        puntajes = self.puntuar([o for _, o in oraciones], premisas)
        umbral_implica = self.config.get("umbral_implica", 0.5)
        umbral_contradice = self.config.get("umbral_contradice", 0.9)
        detalle, quitar = [], set()
        for i, ((campo, oracion), (implica, contradice, pasaje)) in enumerate(zip(oraciones, puntajes)):
            cita = bool(citas.extract(oracion))
            detalle.append({"campo": campo, "oracion": oracion[:160], "implica": round(implica, 3),
                            "contradice": round(contradice, 3), "pasaje": pasaje, "cita_norma": cita})
            if contradice >= umbral_contradice and implica < umbral_implica and not cita:
                quitar.add(i)
        quitadas = []
        if self.config.get("modo") == "quitar_contradichas" and quitar:
            for campo in campos:
                indices = [i for i, (c, _) in enumerate(oraciones) if c == campo]
                conservar = [oraciones[i][1] for i in indices if i not in quitar]
                if conservar and len(conservar) < len(indices):
                    respuesta[campo] = " ".join(conservar)
                    quitadas += [oraciones[i][1][:160] for i in indices if i in quitar]
        return {"oraciones": len(oraciones), "respaldadas": sum(d["implica"] >= umbral_implica for d in detalle),
                "contradichas": len(quitar), "quitadas": quitadas, "detalle": detalle}
