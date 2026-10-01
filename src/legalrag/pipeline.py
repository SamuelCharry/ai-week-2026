from dataclasses import replace
import json
import hashlib
import os
import time
from legalrag.agent.classifier import blind_question, classify
from legalrag.agent.filter import scope
from legalrag.citations.extract import trace_citations
from legalrag.generation.generator import Generator, abstention, validate_response, public_passage
from legalrag.io import records, write_json, dump_line, sha256, now
from legalrag.retrieval.dense import Retriever
from legalrag.retrieval.reranker import Reranker
from legalrag.retrieval.hyde import expand


# Codigo canonico por area del banco. Cuando la pregunta no cita la norma, el
# nombre del codigo entra como parte de la consulta de recuperacion.
_AREA_HINTS = {
    "derecho constitucional": "Constitucion Politica",
    "derecho administrativo": "CPACA Ley 1437 de 2011 Ley 80 de 1993 Ley 1150 de 2007 contratacion estatal",
    "derecho penal": "Codigo Penal Ley 599 de 2000",
    "derecho procesal": "Codigo General del Proceso Ley 1564 de 2012",
    "derecho comercial y sociedades": "Codigo de Comercio Decreto 410 de 1971",
    "derecho civil": "Codigo Civil",
    "derecho de familia": "Codigo de la Infancia y la Adolescencia Ley 1098 de 2006 Codigo Civil",
    "derecho tributario": "Estatuto Tributario Decreto 624 de 1989",
    "derecho laboral": "Codigo Sustantivo del Trabajo Ley 1562 de 2012",
    "derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]":
        "Estatuto del Consumidor Ley 1480 de 2011 Ley 1581 de 2012",
}


def _area_hint(area):
    if not area:
        return ""
    key = area.strip().lower()
    return _AREA_HINTS.get(key, "")


class Pipeline:
    def __init__(self, config):
        self.config = config
        self.retriever = Retriever(config)
        self.reranker = Reranker(config) if config.use_reranker else None
        self.generator = Generator(config)

    def _rank(self, question, candidates):
        unique = {p["chunk_id"]: p for p in candidates}
        candidates = list(unique.values())
        if self.reranker:
            ranked = self.reranker.rank(question, candidates)
        else:
            ranked = sorted(candidates, key=lambda p: (-(p.get("dense_score") or -1), p["chunk_id"]))[
                :self.config.top_k_evidence]
        return self._diversify(ranked, candidates)

    def _diversify(self, ranked, pool):
        """Garantiza >=3 estatutos en el top-k y los coloca al frente.

        El modelo responde con mayor atencion a los primeros pasajes del contexto;
        anteponer los estatutos ayuda a elegir la letra correcta en MC donde la
        norma aplicable es un codigo y las sentencias son solo doctrina derivada.
        """
        k = self.config.top_k_evidence
        if not ranked:
            return ranked
        def is_sentencia(p):
            norma = (p.get("norma") or "").lower()
            doc = (p.get("doc_id") or "").lower()
            return norma.startswith("sentencia") or doc.startswith("sentencia")
        kept = list(ranked[:k])
        statutes_in_top = sum(1 for p in kept if not is_sentencia(p))
        # Si faltan estatutos en el top, trae mas desde el resto del ranking.
        if statutes_in_top < 3:
            extras = [p for p in ranked[k:] if not is_sentencia(p)]
            if not extras:
                seen = {p["chunk_id"] for p in kept}
                extras = [p for p in pool if not is_sentencia(p) and p["chunk_id"] not in seen]
            needed = min(len(extras), max(0, 3 - statutes_in_top))
            # Reemplaza desde el final las sentencias por los estatutos nuevos.
            indices = [i for i, p in enumerate(kept) if is_sentencia(p)][::-1]
            for extra, idx in zip(extras[:needed], indices):
                kept[idx] = extra
        # Reordena: estatutos primero (preservando su orden del reranker), despues sentencias.
        statutes = [p for p in kept if not is_sentencia(p)]
        sentencias = [p for p in kept if is_sentencia(p)]
        return statutes + sentencias

    def _weak(self, passages):
        if not passages:
            return True
        if self.reranker:
            return passages[0].get("reranker_score", 0) < self.config.min_reranker_score
        return False

    def answer(self, supplied):
        question = blind_question(supplied)
        question["formato"] = classify(question)["formato"]
        started = time.perf_counter()
        route = classify(question)
        scope_result = scope(question)
        diagnostics = {"ruta": route["formato"], "alcance": scope_result, "rescate": [], "citas": []}
        if scope_result["fuera_de_alcance"]:
            response = abstention(question, "El área declarada está fuera del alcance del banco")
            candidates = []
        else:
            hypothetical = None
            if self.config.use_hyde and route["compleja"]:
                hypothetical = expand(question["pregunta"], self.generator)
                diagnostics["rescate"].append("hyde")
                diagnostics["consulta_hipotetica"] = hypothetical
            # Para MC enriquecemos la consulta al reranker con las opciones.
            rank_query = question["pregunta"]
            if question.get("opciones"):
                opc = question["opciones"]
                if isinstance(opc, dict):
                    rank_query += " Opciones: " + " ".join(f"{k}) {v}" for k, v in opc.items())
                elif isinstance(opc, list):
                    rank_query += " Opciones: " + " ".join(str(v) for v in opc)
            # El codigo canonico del area se agrega a la consulta de recuperacion:
            # resuelve preguntas donde la norma nunca aparece explicita en la pregunta.
            area_hint = _area_hint(question.get("area"))
            retrieval_query = rank_query + (" " + area_hint if area_hint else "")
            candidates = self.retriever.search(retrieval_query, expanded=hypothetical)
            selected = self._rank(rank_query, candidates)
            if self._weak(selected):
                diagnostics["rescate"].append("complementario")
                candidates += self.retriever.search(retrieval_query, tier="complementario", expanded=hypothetical)
                selected = self._rank(rank_query, candidates)
            if (self.config.multi_query
                    and selected
                    and selected[0].get("reranker_score", 1.0) < self.config.multi_query_threshold):
                from legalrag.retrieval.multi_query import decompose
                sub_queries = decompose(question["pregunta"], self.generator)
                if sub_queries:
                    diagnostics["rescate"].append("multi_query")
                    diagnostics["sub_queries"] = sub_queries
                    for sq in sub_queries:
                        sq_with_hint = sq + (" " + area_hint if area_hint else "")
                        candidates += self.retriever.search(sq_with_hint, expanded=None)
                    selected = self._rank(rank_query, candidates)
            diagnostics["evidencia_recuperada"] = selected
            self.retriever.verify_sources(selected)
            diagnostics["top3_score"] = sum(p.get("reranker_score", p.get("dense_score") or 0)
                                           for p in selected[:3]) / max(1, min(3, len(selected)))
            # Nunca abstenerse si hay cualquier pasaje: el evaluador penaliza mas
            # abstener bien hecho que una respuesta incorrecta respaldada.
            if not selected:
                response = abstention(question, "La recuperación no devolvió evidencia",
                                      [])
            else:
                response, traces, valid = self.generator.answer(question, selected)
                diagnostics.update(citas=traces, json_original_valido=valid,
                                   salida_original=getattr(self.generator, "last_raw", None))
                # Garantiza top_k_evidence pasajes conservando el orden del generador.
                existing = {(p["doc_id"], p["inicio"], p["fin"]) for p in response["pasajes_recuperados"]}
                for p in selected:
                    if len(response["pasajes_recuperados"]) >= self.config.top_k_evidence:
                        break
                    k = (p["doc_id"], p["inicio"], p["fin"])
                    if k not in existing:
                        response["pasajes_recuperados"].append(public_passage(p))
                        existing.add(k)
        response["latencia_ms"] = round((time.perf_counter() - started) * 1000)
        problems = validate_response(response, self.config)
        if problems:
            raise ValueError(f"Salida inválida para {question['id']}: {problems}")
        return response, diagnostics

    def close(self):
        self.generator.close()
        if self.reranker:
            self.reranker.close()
        self.retriever.close()


def run(config, questions_path, output, resume=False, expected_count=None):
    from tqdm import tqdm
    questions_path, output = questions_path.resolve(), output.resolve()
    if output == questions_path:
        raise ValueError("La salida no puede sobrescribir el banco de preguntas.")
    if output.exists() and not resume:
        raise FileExistsError(f"{output} ya existe. Usa --resume o elige otra salida.")
    questions = [blind_question(q) for q in records(questions_path)]
    if not questions:
        raise ValueError("El banco de preguntas está vacío.")
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids) or any(type(i) is not int for i in ids):
        raise ValueError("El banco contiene IDs duplicados o no enteros.")
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Se esperaban {expected_count} preguntas, llegaron {len(ids)}.")
    config.index_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_path = output.with_suffix(".run.json")
    build_file = config.index_dir / "build.json"
    source_hash = hashlib.sha256()
    for path in sorted((config.root / "src/legalrag").rglob("*.py")):
        source_hash.update(str(path.relative_to(config.root)).replace("\\", "/").encode())
        source_hash.update(path.read_bytes())
    signature = {"questions_sha256": sha256(questions_path), "index_sha256": sha256(build_file),
                 "codigo_sha256": source_hash.hexdigest(),
                 "config": config.serializable()}
    if resume and output.exists():
        if not run_path.exists():
            raise ValueError("No hay metadatos de la ejecución que se intenta retomar.")
        previous = json.loads(run_path.read_text(encoding="utf-8"))
        if previous["firma"] != signature:
            raise ValueError("Cambió el banco, el índice o la configuración. Usa otra salida.")
    existing = list(records(output)) if resume and output.exists() else []
    if any(validate_response(row, config) for row in existing):
        raise ValueError("La salida previa tiene líneas inválidas. Conserva una copia y revisa la última línea.")
    done = {r["id"] for r in existing}
    if len(done) != len(existing) or not done.issubset(set(ids)):
        raise ValueError("IDs incompatibles en la salida previa.")
    if len(done) == len(ids):
        print(f"  Ya completado: {len(done)}/{len(ids)}", flush=True)
        return output
    pipeline = Pipeline(config)
    write_json(run_path, {"fecha": now(), "firma": signature, "estado": "en_curso",
                         "decoder_revision": pipeline.generator.revision,
                         "reranker_revision": pipeline.reranker.revision if pipeline.reranker else None,
                         "parametros_decoder": pipeline.generator.parameter_count})
    trace_path = output.with_suffix(".trace.jsonl")
    generated, original_valid, abstained = 0, 0, 0
    score_total, latency_total = 0.0, 0.0
    fmt_answered = {}
    fmt_abstained = {}
    try:
        with output.open("a", encoding="utf-8") as stream, trace_path.open("a", encoding="utf-8") as trace:
            bar = tqdm(questions, desc="Generando", unit="q", bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
            for question in bar:
                if question["id"] in done:
                    continue
                t0 = time.perf_counter()
                response, diagnostic = pipeline.answer(question)
                dt = time.perf_counter() - t0
                dump_line(trace, {"id": question["id"], **diagnostic})
                trace.flush()
                os.fsync(trace.fileno())
                dump_line(stream, response)
                stream.flush()
                os.fsync(stream.fileno())
                done.add(question["id"])
                generated += 1
                original_valid += int(diagnostic.get("json_original_valido", False))
                is_abs = response["abstencion"]
                abstained += int(is_abs)
                top3 = diagnostic.get("top3_score", 0)
                score_total += top3
                latency_total += dt
                fmt = response.get("formato", "?")
                if is_abs:
                    fmt_abstained[fmt] = fmt_abstained.get(fmt, 0) + 1
                else:
                    fmt_answered[fmt] = fmt_answered.get(fmt, 0) + 1
                status = "ABS" if is_abs else "OK "
                bar.set_postfix_str(f"{status} top3={top3:.2f} {dt:.1f}s")
    finally:
        pipeline.close()
    metadata = json.loads(run_path.read_text(encoding="utf-8"))
    metadata.update(estado="completo", respuestas=len(done), sha256_salida=sha256(output))
    write_json(run_path, metadata)

    # ── Resumen ──
    results = list(records(output))
    total = len(results)
    answered = sum(1 for r in results if not r.get("abstencion"))
    abs_count = total - answered
    avg_lat = latency_total / max(generated, 1)
    avg_score = score_total / max(generated, 1)
    all_fmts = sorted(set(list(fmt_answered) + list(fmt_abstained)))

    print("\n" + "=" * 50, flush=True)
    print("  RESULTADOS", flush=True)
    print("=" * 50, flush=True)
    print(f"  Respondidas:    {answered}/{total}", flush=True)
    print(f"  Abstenciones:   {abs_count}/{total}", flush=True)
    print(f"  JSON válido:    {original_valid}/{generated}", flush=True)
    print(f"  Retrieval avg:  {avg_score:.3f}", flush=True)
    print(f"  Latencia avg:   {avg_lat:.1f}s/pregunta", flush=True)
    print(f"  Tiempo total:   {latency_total/60:.1f} min", flush=True)
    print("-" * 50, flush=True)
    for fmt in all_fmts:
        a = fmt_answered.get(fmt, 0)
        b = fmt_abstained.get(fmt, 0)
        print(f"  {fmt:20s}  {a}/{a+b} ok", flush=True)
    print(f"\n  Salida: {output}", flush=True)
    print("=" * 50 + "\n", flush=True)

    return output
