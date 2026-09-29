"""Recuperación de la entrega (opción A): BM25 + BGE-M3 fusionados con RRF y reranker BGE.

Lee los artefactos que construye E06 (`legalrag.experimentos.corpus_definitivo`):
`chunks.sqlite`, con la tabla de fragmentos y su índice FTS5, y el índice FAISS de
BGE-M3 sobre `texto_busqueda`. La posición i del índice denso es el fragmento i + 1.
El encoder de consultas replica el de la indexación: pooling CLS, vectores
normalizados y como máximo 1.024 tokens. Los modelos se cargan una vez en `abrir`.
"""
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path


def consulta(entrada):
    """Texto de búsqueda: la pregunta y, en cerradas, sus opciones."""
    opciones = entrada.get("opciones") or {}
    return entrada["pregunta"] + "\n" + "\n".join(f"{k}. {v}" for k, v in opciones.items())


def rrf(*rankings, k=100, constante=60):
    fusion = defaultdict(float)
    for ranking in rankings:
        for puesto, (fragmento, _) in enumerate(ranking, 1):
            fusion[fragmento] += 1 / (constante + puesto)
    return sorted(fusion.items(), key=lambda x: (-x[1], x[0]))[:k]


class Fragmentos:
    """Acceso de solo lectura a chunks.sqlite."""

    def __init__(self, ruta):
        self.ruta = Path(ruta)
        if not self.ruta.is_file():
            raise FileNotFoundError(self.ruta)

    def _conexion(self):
        conexion = sqlite3.connect(f"file:{self.ruta}?mode=ro", uri=True)
        conexion.row_factory = sqlite3.Row
        return conexion

    def total(self):
        with closing(self._conexion()) as c:
            return c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def bm25(self, texto, k):
        terminos = dict.fromkeys(re.findall(r"\w+", texto.casefold()))
        expresion = " OR ".join('"' + t.replace('"', "") + '"' for t in terminos)
        if not expresion:
            return []
        with closing(self._conexion()) as c:
            return [(int(r[0]), float(-r[1])) for r in c.execute(
                "SELECT rowid, bm25(fts) FROM fts WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
                (expresion, k))]

    def filas(self, ids):
        if not ids:
            return []
        with closing(self._conexion()) as c:
            filas = {r["id"]: dict(r) for r in c.execute(
                f"SELECT * FROM chunks WHERE id IN ({','.join('?' for _ in ids)})", ids)}
        return [filas[i] for i in ids]


class EncoderConsultas:
    def __init__(self, ficha, dtype, max_tokens):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], revision=ficha["revision"])
        self.modelo = AutoModel.from_pretrained(ficha["repo_id"], revision=ficha["revision"],
                                                torch_dtype=getattr(torch, dtype)).to("cuda").eval()

    def codificar(self, texto):
        import torch

        tokens = self.tokenizer([texto], padding=True, truncation=True, max_length=self.max_tokens,
                                return_tensors="pt").to("cuda")
        with torch.inference_mode():
            cls = self.modelo(**tokens).last_hidden_state.float()[:, 0]
            return torch.nn.functional.normalize(cls, p=2, dim=1).cpu().numpy().astype("float32")


class Reordenador:
    def __init__(self, ficha, dtype, max_tokens, lote):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_tokens, self.lote = max_tokens, lote
        self.tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], revision=ficha["revision"])
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            ficha["repo_id"], revision=ficha["revision"], torch_dtype=getattr(torch, dtype)).to("cuda").eval()

    def puntuar(self, texto, candidatos):
        import torch

        puntajes, i = [], 0
        while i < len(candidatos):
            pares = [(texto, c) for c in candidatos[i:i + self.lote]]
            try:
                entrada = self.tokenizer(pares, padding=True, truncation=True, max_length=self.max_tokens,
                                         return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    puntajes.extend(self.modelo(**entrada).logits.float().view(-1).cpu().tolist())
                i += len(pares)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.lote == 1:
                    raise
                self.lote //= 2
        return puntajes


class RecuperadorHibrido:
    def __init__(self, raiz, config):
        self.raiz, self.config = Path(raiz), config
        self.fragmentos = self.indice = self.encoder = self.reordenador = self.citaciones = None

    def _ruta(self, clave):
        return self.raiz / self.config[clave]

    def abrir(self):
        import faiss

        from legalrag.evaluation.oficial import cargar_citaciones

        c = self.config
        self.fragmentos = Fragmentos(self._ruta("fragmentos"))
        meta = json.loads(self._ruta("indice_meta").read_text(encoding="utf-8"))
        if meta["revision"] != c["encoder"]["revision"] or meta["modelo"] != c["encoder"]["repo_id"]:
            raise ValueError("El índice denso se construyó con otro encoder o revisión")
        self.indice = faiss.read_index(str(self._ruta("indice_denso")))
        if self.indice.ntotal != self.fragmentos.total():
            raise ValueError("El índice denso y chunks.sqlite no tienen los mismos fragmentos")
        self.encoder = EncoderConsultas(c["encoder"], c["dtype"], c["max_tokens_encoder"])
        self.reordenador = Reordenador(c["reranker"], c["dtype"], c["max_tokens_reranker"], c["lote_reranker"])
        self.citaciones = cargar_citaciones(self.raiz)

    def cerrar(self):
        import gc

        self.encoder = self.reordenador = self.indice = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def ranking(self, entrada):
        c = self.config
        texto = consulta(entrada)
        lexico = self.fragmentos.bm25(texto, c["bm25_top"])
        puntajes, posiciones = self.indice.search(self.encoder.codificar(texto), c["denso_top"])
        denso = [(int(i) + 1, float(s)) for i, s in zip(posiciones[0], puntajes[0]) if i >= 0]
        candidatos = rrf(lexico, denso, k=max(c["bm25_top"], c["denso_top"]), constante=c["rrf_k"])[:c["rerank_top"]]
        filas = self.fragmentos.filas([f for f, _ in candidatos])
        puntajes = self.reordenador.puntuar(texto, [f["texto_busqueda"] for f in filas])
        return sorted(((f["id"], float(p)) for f, p in zip(filas, puntajes)), key=lambda x: (-x[1], x[0]))

    def _texto(self, doc_id):
        return (self._ruta("textos") / f"{doc_id}.txt").read_text(encoding="utf-8")

    def pasajes(self, ranking):
        """Hasta `max_pasajes` unidades literales, una por unidad, con offsets del texto fuente.

        Como en E06, reserva hasta dos plazas para la cabecera literal de la norma cuando el
        pasaje no la trae: sin ella el evaluador no liga la cita con la evidencia.
        """
        c = self.config
        maximo = c["max_pasajes"]
        elegidos, vistas, puntajes = [], set(), dict(ranking)
        for fila in self.fragmentos.filas([f for f, _ in ranking]):
            if fila["unidad_id"] in vistas:
                continue
            vistas.add(fila["unidad_id"])
            a, b = fila["inicio"], fila["fin"]
            if fila["unidad_fin"] - fila["unidad_inicio"] <= c["max_caracteres_pasaje"]:
                a, b = fila["unidad_inicio"], fila["unidad_fin"]
            elegidos.append({"doc_id": fila["doc_id"], "inicio": a, "fin": b, "texto": self._texto(fila["doc_id"])[a:b],
                             "score": puntajes[fila["id"]], "titulo": fila["titulo"], "articulo": fila["articulo"],
                             "unidad_id": fila["unidad_id"], "avisos": json.loads(fila["avisos"])})
            if len(elegidos) >= maximo:
                break
        extraer, cuerpos = self.citaciones.extract, self.citaciones.bodies
        cabeceras, docs = [], set()
        for p in elegidos[:4]:
            if p["doc_id"] in docs:
                continue
            docs.add(p["doc_id"])
            inicio = self._texto(p["doc_id"])[:800]
            if not extraer(inicio) or cuerpos(extraer(inicio)) <= cuerpos(extraer(p["texto"])):
                continue
            cabeceras.append({"doc_id": p["doc_id"], "inicio": 0, "fin": len(inicio), "texto": inicio,
                              "score": p["score"], "titulo": p["titulo"], "articulo": None,
                              "unidad_id": p["doc_id"] + "__cabecera_literal", "avisos": ["cabecera_literal_de_fuente"]})
            if len(cabeceras) == 2:
                break
        if cabeceras:
            elegidos = elegidos[:1] + cabeceras + elegidos[1:maximo - len(cabeceras)]
        return elegidos

    def buscar(self, entrada):
        return self.pasajes(self.ranking(entrada))
