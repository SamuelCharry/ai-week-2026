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
    reservar_nombradas N lugares de la evidencia para los mejores pasajes de la norma que la
                       pregunta nombra sin artículo (p. ej. "reorganización ley 1116 de 2006").
    max_por_documento  tope de pasajes de un mismo documento (las sentencias "se parecen" a todo).
    min_normativos     pasajes mínimos de leyes, decretos, códigos o Constitución cuando hay candidatos.
    ajustes_solo_texto_libre  los tres anteriores solo en semiabiertas y abiertas (las cerradas conservan su
                       evidencia).
    recuperar_por_opcion en cerradas, una búsqueda BM25 + densa más por opción (pregunta + opción).

El reranker es BGE-v2-m3 o Qwen3-Reranker (0.6B/4B), según `reranker.repo_id`.
"""
import json
import re
import sqlite3
from collections import defaultdict
from contextlib import closing
from pathlib import Path

NORMATIVOS = {"ley", "decreto", "decreto_ley", "constitucion", "acto_legislativo", "codigo", "resolucion",
              "acuerdo", "decision", "circular"}
INSTRUCCION_QWEN = "Instruct: Recupera pasajes jurídicos colombianos pertinentes a la pregunta\nQuery: "


def consulta(entrada, con_tema=False):
    """Texto de búsqueda: la pregunta, en cerradas sus opciones y, si se pide, el tema."""
    opciones = entrada.get("opciones") or {}
    partes = [entrada["pregunta"], *(f"{k}. {v}" for k, v in opciones.items())]
    if con_tema and entrada.get("tema"):
        partes.insert(0, str(entrada["tema"]).strip())
    return "\n".join(partes)


def es_normativo(tipo):
    import unicodedata
    tipo = "".join(c for c in unicodedata.normalize("NFD", str(tipo or "").lower()) if unicodedata.category(c) != "Mn")
    return tipo.strip().replace(" ", "_").replace("-", "_") in NORMATIVOS


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


class ReordenadorQwen:
    """Qwen3-Reranker (0.6B/4B): "yes"/"no" con la plantilla publicada por Qwen (la misma de E06).

    El puntaje es logit(yes) - logit(no), en la misma escala que los logits de BGE (0 = 50 %), así que
    los umbrales de evidencia débil significan lo mismo con cualquiera de los dos rerankers."""

    INSTRUCCION = "Dada una pregunta jurídica colombiana, identifica pasajes que permitan responderla con fundamento literal."

    def __init__(self, ficha, dtype, max_tokens, lote, dispositivo="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.max_tokens, self.lote, self.dispositivo = max_tokens, lote, dispositivo
        self.tokenizer = AutoTokenizer.from_pretrained(ficha["repo_id"], revision=ficha["revision"], padding_side="left")
        self.modelo = AutoModelForCausalLM.from_pretrained(
            ficha["repo_id"], revision=ficha["revision"], dtype=getattr(torch, dtype)).to(dispositivo).eval()
        self.si, self.no = self.tokenizer.convert_tokens_to_ids("yes"), self.tokenizer.convert_tokens_to_ids("no")
        codificar = lambda t: self.tokenizer.encode(t, add_special_tokens=False)  # noqa: E731
        self.prefijo = codificar('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
                                 'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
                                 '<|im_end|>\n<|im_start|>user\n')
        self.sufijo = codificar('<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')

    def puntuar(self, texto, candidatos):
        import torch

        ids = [self.prefijo + self.tokenizer.encode(f"<Instruct>: {self.INSTRUCCION}\n<Query>: {texto}\n<Document>: {c}",
                                                    add_special_tokens=False)[:self.max_tokens] + self.sufijo
               for c in candidatos]
        puntajes, i = [], 0
        while i < len(ids):
            try:
                entrada = self.tokenizer.pad({"input_ids": ids[i:i + self.lote]}, padding=True,
                                             return_tensors="pt").to(self.dispositivo)
                with torch.inference_mode():
                    logits = self.modelo(**entrada).logits[:, -1, :].float()
                puntajes.extend((logits[:, self.si] - logits[:, self.no]).cpu().tolist())
                i += len(ids[i:i + self.lote])
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.lote == 1:
                    raise
                self.lote //= 2
        return puntajes


def crear_reordenador(ficha, dtype, max_tokens, lote, dispositivo="cuda"):
    clase = ReordenadorQwen if "Qwen3-Reranker" in ficha["repo_id"] else Reordenador
    return clase(ficha, dtype, max_tokens, lote, dispositivo)


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
            self.reordenador = crear_reordenador(c["reranker"], c["dtype"], c["max_tokens_reranker"], c["lote_reranker"],
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

    def ranking(self, entrada, expansion=None):
        """`expansion`: texto extra de búsqueda (hipótesis del decoder, generation.pasos). Suma sus
        rankings BM25 y denso a la fusión, y una búsqueda dentro de las normas que nombra; el reranker
        sigue puntuando contra la pregunta."""
        c = self.config
        texto = consulta(entrada, c.get("consulta_con_tema", False))
        rankings = []
        busquedas = [texto] + ([expansion] if expansion else [])
        if c.get("recuperar_por_opcion") and entrada.get("opciones"):
            # Options-aware retrieval (2025): una búsqueda por opción, con la pregunta, para traer la evidencia que
            # distingue entre las opciones; el reranker sigue puntuando contra la pregunta completa.
            busquedas += [entrada["pregunta"].strip() + "\n" + opcion for opcion in entrada["opciones"].values()]
        for busqueda in busquedas:
            if c.get("usar_bm25", True):
                rankings.append(self.fragmentos.bm25(busqueda, c["bm25_top"]))
            if c.get("usar_denso", True) and self.indice is not None:
                puntajes, posiciones = self.indice.search(self.encoder.codificar(busqueda), c["denso_top"])
                rankings.append([(int(i) + 1, float(s)) for i, s in zip(posiciones[0], puntajes[0]) if i >= 0])
        fijos, ruta, nombradas = [], [], {}
        if c.get("enrutar_normas", False):
            nombradas = self.evidencia.normas_de(entrada["pregunta"] + " " + " ".join((entrada.get("opciones") or {}).values()))
            if nombradas:
                ruta = self.fragmentos.bm25(texto, c["bm25_top"], doc_ids=list(nombradas))
                rankings.append(ruta)
                fijos = [f for doc_id, articulos in nombradas.items() for a in sorted(articulos)
                         for f in self.fragmentos.por_articulo(doc_id, a, 2)][:c.get("max_fijos", 3)]
        de_expansion, de_hipotesis, ruta_agente = [], [], []
        if expansion:
            # Normas que solo nombra la hipótesis o el agente reformulador.
            normas_expansion = self.evidencia.normas_de(expansion)
            de_hipotesis = [d for d in normas_expansion if d not in nombradas]
            if de_hipotesis:
                ruta_agente = self.fragmentos.bm25(texto + " " + expansion, c["bm25_top"], doc_ids=de_hipotesis)
                rankings.append(ruta_agente)
            if c.get("expansion_articulos"):
                # Los artículos que nombra el agente entran directo al reranker (no como fijos): si el agente se
                # equivoca, el reranker los deja fuera. La auditoría mostró que en la 58 y la 247 el reranker sí
                # aceptaría la norma correcta; lo que faltaba era que llegara a los candidatos.
                de_expansion = [f for doc_id, articulos in normas_expansion.items() for a in sorted(articulos)
                                for f in self.fragmentos.por_articulo(doc_id, a, 2)][:c.get("max_articulos_expansion", 6)]
        reservar = c.get("reservar_nombradas", 0) if nombradas else 0
        # Normas que nombra el agente reformulador: la misma reserva que las que nombra la pregunta (el ciclo del
        # enunciado: reformular con el nombre de la norma y recuperar de ella). Una norma equivocada no resta en
        # citas: solo restan las citas que no están en los pasajes.
        reservar_agente = c.get("reservar_expansion", 0) if de_hipotesis else 0
        # Con reserva, los mejores de la búsqueda dentro de la norma nombrada entran siempre al reranker.
        de_ruta = [f for f, _ in ruta[:max(5, 2 * reservar)]] if reservar else []
        de_ruta += [f for f, _ in ruta_agente[:max(5, 2 * reservar_agente)]] if reservar_agente else []
        candidatos = [f for f, _ in rrf(*rankings, k=max(c["bm25_top"], c["denso_top"]), constante=c["rrf_k"])]
        candidatos = list(dict.fromkeys(fijos + de_ruta + de_expansion + candidatos))[
            :c["rerank_top"] + len(fijos) + len(de_ruta) + len(de_expansion)]
        if self.reordenador is None or not c.get("usar_reranker", True):
            orden = [(f, 1.0 / (i + 1)) for i, f in enumerate(candidatos)]
            docs = {f["id"]: f["doc_id"] for f in self.fragmentos.filas(candidatos)} if reservar or reservar_agente else {}
        else:
            filas = self.fragmentos.filas(candidatos)
            puntajes = self.reordenador.puntuar(texto, [f["texto_busqueda"] for f in filas])
            orden = sorted(((f["id"], float(p)) for f, p in zip(filas, puntajes)), key=lambda x: (-x[1], x[0]))
            docs = {f["id"]: f["doc_id"] for f in filas}
        # El artículo nombrado va primero; después, si se pide, los mejores pasajes de la norma nombrada.
        reservados = [x[0] for x in orden if x[0] not in fijos and docs.get(x[0]) in nombradas][:reservar]
        reservados += [x[0] for x in orden if x[0] not in fijos and x[0] not in reservados
                       and docs.get(x[0]) in de_hipotesis][:reservar_agente]
        if c.get("enrutar_normas", False) or expansion:
            self.ultima_traza = {"normas_nombradas": {k: sorted(v) for k, v in nombradas.items()}, "fijos": fijos,
                                 "reservados": reservados, "expansion": expansion}
        primeros = fijos + reservados
        return [x for f in primeros for x in orden if x[0] == f] + [x for x in orden if x[0] not in primeros]

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
        tope, minimo = c.get("max_por_documento"), c.get("min_normativos", 0)
        elegidos, vistas, puntajes, por_doc = [], set(), dict(ranking), {}
        reserva_normativa = []  # unidades normativas que no alcanzaron lugar, por si hay que asegurar el mínimo
        for fila in self.fragmentos.filas([f for f, _ in ranking]):
            if fila["unidad_id"] in vistas:
                continue
            vistas.add(fila["unidad_id"])
            if len(elegidos) >= maximo or (tope and por_doc.get(fila["doc_id"], 0) >= tope):
                if len(elegidos) >= maximo and minimo and es_normativo(fila["tipo"]) and len(reserva_normativa) < minimo:
                    reserva_normativa.append(fila)
                if len(elegidos) >= maximo and (not minimo or len(reserva_normativa) >= minimo):
                    break
                continue
            por_doc[fila["doc_id"]] = por_doc.get(fila["doc_id"], 0) + 1
            elegidos.append(self._pasaje(fila, puntajes))
        # Si faltan pasajes normativos, reemplazan a los últimos no normativos (nunca a los primeros tres).
        faltan = minimo - sum(es_normativo(p["tipo"]) for p in elegidos)
        for fila in reserva_normativa[:max(0, faltan)]:
            cambiable = [i for i in range(len(elegidos) - 1, 2, -1) if not es_normativo(elegidos[i]["tipo"])]
            if not cambiable:
                break
            elegidos[cambiable[0]] = self._pasaje(fila, puntajes)
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

    def _pasaje(self, fila, puntajes):
        c = self.config
        a, b = fila["inicio"], fila["fin"]
        if fila["unidad_fin"] - fila["unidad_inicio"] <= c["max_caracteres_pasaje"]:
            a, b = fila["unidad_inicio"], fila["unidad_fin"]
        pasaje = {"doc_id": fila["doc_id"], "inicio": a, "fin": b, "texto": self._texto(fila["doc_id"])[a:b],
                  "score": puntajes[fila["id"]], "titulo": fila["titulo"], "articulo": fila["articulo"],
                  "tipo": fila["tipo"], "unidad_id": fila["unidad_id"], "avisos": json.loads(fila["avisos"])}
        if c.get("encabezado_norma", True):
            pasaje["encabezado"] = self.evidencia.encabezado(pasaje)
        return pasaje

    def buscar(self, entrada, expansion=None):
        c = self.config
        if c.get("ajustes_solo_texto_libre") and entrada.get("formato") == "multiple_choice":
            # En la 4090 los ajustes subieron citas pero cambiaron la evidencia de una cerrada que se acertaba.
            self.config = {**c, "reservar_nombradas": 0, "max_por_documento": None, "min_normativos": 0}
        try:
            return self.pasajes(self.ranking(entrada, expansion))
        finally:
            self.config = c
