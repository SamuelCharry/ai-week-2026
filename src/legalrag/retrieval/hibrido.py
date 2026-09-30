"""Recuperación de la entrega: BM25 + encoder denso fusionados con RRF y reranker BGE.

Lee `chunks.sqlite` (fragmentos + FTS5) y el índice FAISS del encoder que construye E06
(`legalrag.experimentos.corpus_definitivo`) o `indexing.r03`. La posición i del índice denso es
el fragmento i + 1. El encoder de consultas replica el de la indexación (pooling y prefijos de
cada modelo). Los modelos se cargan una vez en `abrir`.

Sobre el recorrido base (BM25 top N + denso top N -> RRF -> reranker) hay tres ajustes, cada uno
activable en `configs/sistema.json` para poder medirlo:

    enrutar_normas     si la pregunta nombra una norma que está en el corpus ("artículo 369 del
                       Código General del Proceso"), se agrega una búsqueda BM25 dentro de esa
                       norma a la fusión y se fija el artículo exacto al comienzo de la evidencia.
    consulta_con_tema  el campo `tema` de la pregunta (entrada, no respuesta) se suma a la consulta.
    encabezado_norma   cada pasaje entregado empieza con el nombre de su norma
                       (citations.normas); reemplaza las "cabeceras literales" de E06.
"""
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

INSTRUCCION_QWEN = "Instruct: Recupera pasajes jurídicos colombianos pertinentes a la pregunta\nQuery: "


def consulta(entrada, con_tema=False):
    """Texto de búsqueda: la pregunta, en cerradas sus opciones y, si se pide, el tema."""
    opciones = entrada.get("opciones") or {}
    partes = [entrada["pregunta"], *(f"{k}. {v}" for k, v in opciones.items())]
    if con_tema and entrada.get("tema"):
        partes.insert(0, str(entrada["tema"]).strip())
    return "\n".join(partes)


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

    @staticmethod
    def _expresion(texto):
        terminos = dict.fromkeys(re.findall(r"\w+", texto.casefold()))
        return " OR ".join('"' + t.replace('"', "") + '"' for t in terminos)

    def bm25(self, texto, k, doc_ids=None):
        expresion = self._expresion(texto)
        if not expresion:
            return []
        sql, parametros = "SELECT fts.rowid, bm25(fts) FROM fts", [expresion]
        if doc_ids:
            sql += f" JOIN chunks ON chunks.id = fts.rowid WHERE fts MATCH ? AND chunks.doc_id IN " \
                   f"({','.join('?' for _ in doc_ids)})"
            parametros += list(doc_ids)
        else:
            sql += " WHERE fts MATCH ?"
        with closing(self._conexion()) as c:
            return [(int(r[0]), float(-r[1])) for r in c.execute(sql + " ORDER BY bm25(fts) LIMIT ?", (*parametros, k))]

    def por_articulo(self, doc_id, articulo, k=3):
        with closing(self._conexion()) as c:
            return [int(r[0]) for r in c.execute(
                "SELECT id FROM chunks WHERE doc_id = ? AND articulo = ? ORDER BY id LIMIT ?", (doc_id, str(articulo), k))]

    def filas(self, ids):
        if not ids:
            return []
        with closing(self._conexion()) as c:
            filas = {r["id"]: dict(r) for r in c.execute(
                f"SELECT * FROM chunks WHERE id IN ({','.join('?' for _ in ids)})", ids)}
        return [filas[i] for i in ids]


class EncoderConsultas:
    """Pooling y prefijo de cada familia: BGE-M3 (CLS), E5 (media, "query: "), Qwen3 (último token)."""

    def __init__(self, ficha, dtype, max_tokens, dispositivo="cuda"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.max_tokens, self.dispositivo = max_tokens, dispositivo
        repo = ficha["repo_id"]
        self.pooling = ficha.get("pooling") or ("media" if "multilingual-e5" in repo else
                                                "ultimo" if "Qwen3-Embedding" in repo else "cls")
        self.prefijo = ficha.get("prefijo_consulta", "query: " if self.pooling == "media" else
                                 INSTRUCCION_QWEN if self.pooling == "ultimo" else "")
        self.tokenizer = AutoTokenizer.from_pretrained(repo, revision=ficha["revision"])
        self.modelo = AutoModel.from_pretrained(repo, revision=ficha["revision"],
                                                dtype=getattr(torch, dtype)).to(dispositivo).eval()

    def codificar(self, texto):
        import torch

        tokens = self.tokenizer([self.prefijo + texto], padding=True, truncation=True,
                                max_length=self.max_tokens, return_tensors="pt").to(self.dispositivo)
        with torch.inference_mode():
            oculto = self.modelo(**tokens).last_hidden_state.float()
            if self.pooling == "media":
                mascara = tokens["attention_mask"].unsqueeze(-1)
                vector = (oculto * mascara).sum(1) / mascara.sum(1)
            elif self.pooling == "ultimo":
                vector = oculto[torch.arange(oculto.shape[0], device=oculto.device), tokens["attention_mask"].sum(1) - 1]
            else:
                vector = oculto[:, 0]
            return torch.nn.functional.normalize(vector, p=2, dim=1).cpu().numpy().astype("float32")


class Reordenador:
    def __init__(self, ficha, dtype, max_tokens, lote, dispositivo="cuda"):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.max_tokens, self.lote, self.dispositivo = max_tokens, lote, dispositivo
        self.tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], revision=ficha["revision"])
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            ficha["repo_id"], revision=ficha["revision"], dtype=getattr(torch, dtype)).to(dispositivo).eval()

    def puntuar(self, texto, candidatos):
        import torch

        puntajes, i = [], 0
        while i < len(candidatos):
            pares = [(texto, c) for c in candidatos[i:i + self.lote]]
            try:
                entrada = self.tokenizer(pares, padding=True, truncation=True, max_length=self.max_tokens,
                                         return_tensors="pt").to(self.dispositivo)
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
        self.evidencia = None
        self.ultima_traza = {}

    def _ruta(self, clave):
        return self.raiz / self.config[clave]

    def abrir(self, modelos=True):
        """modelos=False abre solo lo léxico (sin GPU): útil para medir BM25 y el enrutamiento."""
        from legalrag.citations.normas import EvidenciaCorpus
        from legalrag.evaluation.oficial import cargar_citaciones

        c = self.config
        self.fragmentos = Fragmentos(self._ruta("fragmentos"))
        self.citaciones = cargar_citaciones(self.raiz)
        manifiesto = json.loads(self._ruta("manifiesto").read_text(encoding="utf-8"))
        self.evidencia = EvidenciaCorpus(self.citaciones, manifiesto)
        if not modelos:
            return
        import faiss

        meta = json.loads(self._ruta("indice_meta").read_text(encoding="utf-8"))
        if meta["revision"] != c["encoder"]["revision"] or meta["modelo"] != c["encoder"]["repo_id"]:
            raise ValueError("El índice denso se construyó con otro encoder o revisión")
        self.indice = faiss.read_index(str(self._ruta("indice_denso")))
        if self.indice.ntotal != self.fragmentos.total():
            raise ValueError("El índice denso y chunks.sqlite no tienen los mismos fragmentos")
        dispositivo = c.get("dispositivo", "cuda")
        self.encoder = EncoderConsultas(c["encoder"], c["dtype"], c["max_tokens_encoder"], dispositivo)
        if c.get("usar_reranker", True):
            self.reordenador = Reordenador(c["reranker"], c["dtype"], c["max_tokens_reranker"], c["lote_reranker"],
                                           dispositivo)

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
        texto = consulta(entrada, c.get("consulta_con_tema", False))
        rankings = []
        if c.get("usar_bm25", True):
            rankings.append(self.fragmentos.bm25(texto, c["bm25_top"]))
        if c.get("usar_denso", True) and self.indice is not None:
            puntajes, posiciones = self.indice.search(self.encoder.codificar(texto), c["denso_top"])
            rankings.append([(int(i) + 1, float(s)) for i, s in zip(posiciones[0], puntajes[0]) if i >= 0])
        fijos = []
        if c.get("enrutar_normas", False):
            nombradas = self.evidencia.normas_de(entrada["pregunta"] + " " + " ".join((entrada.get("opciones") or {}).values()))
            if nombradas:
                rankings.append(self.fragmentos.bm25(texto, c["bm25_top"], doc_ids=list(nombradas)))
                fijos = [f for doc_id, articulos in nombradas.items() for a in sorted(articulos)
                         for f in self.fragmentos.por_articulo(doc_id, a, 2)][:c.get("max_fijos", 3)]
            self.ultima_traza = {"normas_nombradas": {k: sorted(v) for k, v in nombradas.items()}, "fijos": fijos}
        candidatos = [f for f, _ in rrf(*rankings, k=max(c["bm25_top"], c["denso_top"]), constante=c["rrf_k"])]
        candidatos = list(dict.fromkeys(fijos + candidatos))[:c["rerank_top"] + len(fijos)]
        if self.reordenador is None or not c.get("usar_reranker", True):
            return [(f, 1.0 / (i + 1)) for i, f in enumerate(candidatos)]
        filas = self.fragmentos.filas(candidatos)
        puntajes = self.reordenador.puntuar(texto, [f["texto_busqueda"] for f in filas])
        orden = sorted(((f["id"], float(p)) for f, p in zip(filas, puntajes)), key=lambda x: (-x[1], x[0]))
        # El artículo que la pregunta nombra va primero aunque el reranker lo ponga más abajo.
        return [x for x in orden if x[0] in fijos] + [x for x in orden if x[0] not in fijos]

    def _texto(self, doc_id):
        return (self._ruta("textos") / f"{doc_id}.txt").read_text(encoding="utf-8")

    def pasajes(self, ranking):
        """Hasta `max_pasajes` unidades literales, una por unidad, con offsets del texto fuente.

        Con `encabezado_norma` cada pasaje lleva `encabezado` (nombre de la norma y artículo), que
        se antepone al texto entregado. Sin él, como en E06, se reservan hasta dos plazas para la
        cabecera literal de la norma cuando el pasaje no la trae.
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
            pasaje = {"doc_id": fila["doc_id"], "inicio": a, "fin": b, "texto": self._texto(fila["doc_id"])[a:b],
                      "score": puntajes[fila["id"]], "titulo": fila["titulo"], "articulo": fila["articulo"],
                      "unidad_id": fila["unidad_id"], "avisos": json.loads(fila["avisos"])}
            if c.get("encabezado_norma", True):
                pasaje["encabezado"] = self.evidencia.encabezado(pasaje)
            elegidos.append(pasaje)
            if len(elegidos) >= maximo:
                break
        if c.get("encabezado_norma", True):
            return elegidos
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
