from collections import defaultdict
from pathlib import Path
import subprocess
import sys
from legalrag.generation.generator import validate_response
from legalrag.io import records, corpus_records, source_path, write_json


def evaluate(config, predictions, gold=None, official=False, ragas=False):
    if ragas and not official:
        raise ValueError("--ragas requiere --official")
    if official and (gold is None or Path(gold).resolve() != config.questions_file.resolve()):
        raise ValueError("El adaptador oficial usa sample_50.jsonl. Para test se necesita la clave oficial.")
    rows = list(records(predictions))
    reference = {q["id"]: q for q in records(gold)} if gold else {}
    docs = {d["doc_id"]: d for d in corpus_records(config)}
    seen, answered = set(), set()
    problems = []
    correct = unsupported = total_passages = abstained = 0
    latency = []
    total_closed = sum(q["formato"] == "multiple_choice" and bool(q.get("respuesta_correcta"))
                       for q in reference.values())
    by_area = defaultdict(lambda: {"preguntas": 0, "cerradas": 0, "aciertos_cerradas": 0,
                                  "abstenciones": 0, "errores_formato": 0})
    for q in reference.values():
        stats = by_area[q.get("area", "sin_area")]
        stats["preguntas"] += 1
        stats["cerradas"] += int(q["formato"] == "multiple_choice" and bool(q.get("respuesta_correcta")))
    for i, row in enumerate(rows, 1):
        errors = validate_response(row, config)
        qid = row.get("id") if isinstance(row, dict) else None
        if type(qid) is not int:
            problems.append({"linea": i, "errores": errors or ["ID no entero"]})
            continue
        if qid in seen:
            problems.append({"id": qid, "errores": ["ID duplicado"]})
            continue
        seen.add(qid)
        q = reference.get(qid, {})
        if reference and not q:
            errors.append("ID fuera del banco")
        if q and q["formato"] != row.get("formato"):
            errors.append("Formato distinto del banco")
        stats = by_area[q.get("area", "sin_area")]
        if not reference:
            stats["preguntas"] += 1
        if errors:
            stats["errores_formato"] += 1
            problems.append({"id": qid, "errores": errors})
            continue
        stats["abstenciones"] += int(row["abstencion"])
        abstained += int(row["abstencion"])
        latency.append(row.get("latencia_ms", 0))
        cache = {}
        for p in row["pasajes_recuperados"]:
            total_passages += 1
            doc = docs.get(p["doc_id"])
            if doc is None:
                unsupported += 1
                errors.append(f"doc_id inexistente: {p['doc_id']}")
                continue
            try:
                if p["doc_id"] not in cache:
                    cache[p["doc_id"]] = source_path(config, doc).read_text(encoding="utf-8")
                text = cache[p["doc_id"]]
            except (OSError, UnicodeError) as exc:
                unsupported += 1
                errors.append(f"Fuente no disponible: {exc}")
                continue
            literal = (text[p["inicio"]:p["fin"]] == p["texto"]
                       if "inicio" in p and "fin" in p else p["texto"] in text)
            if not literal:
                unsupported += 1
                errors.append(f"Pasaje no literal: {p['doc_id']}")
        if not row["abstencion"] and not errors:
            answered.add(qid)
            if q.get("formato") == "multiple_choice" and q.get("respuesta_correcta"):
                okay = row["respuesta_correcta"] == q["respuesta_correcta"]
                correct += int(okay)
                stats["aciertos_cerradas"] += int(okay)
        if errors:
            problems.append({"id": qid, "errores": errors})
        if i % 10 == 0:
            print(f"[evaluation] {i}/{len(rows)} — aciertos cerradas: {correct}/{total_closed} — "
                  f"pasajes no verificables: {unsupported}/{total_passages}", flush=True)
    missing = set(reference) - seen if reference else set()
    report = {
        "respuestas": len(rows), "faltantes": sorted(missing), "errores": problems,
        "accuracy_cerradas_local": correct / total_closed if total_closed else None,
        "coverage": len(answered) / max(1, len(reference) or len(seen)),
        "abstenciones": abstained, "pasajes_no_verificables": unsupported, "pasajes": total_passages,
        "latencia_promedio_ms": sum(latency) / max(1, len(latency)), "por_area": dict(by_area),
        "nota": "La accuracy local incluye todas las cerradas. El evaluador oficial excluye sus IDs defectuosos. "
                "Corrección semántica, citas jurídicas y abstención calibrada: consultar reporte oficial.",
    }
    config.reports.mkdir(parents=True, exist_ok=True)
    write_json(config.reports / "evaluacion_local.json", report)
    if reference:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        areas = sorted(by_area)
        errors_pct = [100 * (1 - by_area[a]["aciertos_cerradas"] / by_area[a]["cerradas"])
                      if by_area[a]["cerradas"] else float("nan") for a in areas]
        fig, ax = plt.subplots(figsize=(11, max(4, len(areas) * .5)))
        ax.barh(areas, errors_pct, color="#527c9c")
        ax.set(xlabel="Cerradas incorrectas, omitidas o abstenciones (%)", xlim=(0, 100),
               title="Errores por área — muestra con respuesta conocida")
        fig.tight_layout()
        fig.savefig(config.reports / "errores_por_area.png", dpi=160)
        plt.close(fig)
    if official:
        evaluator = config.data_dir / "oficial/scripts/evaluate.py"
        command = [sys.executable, str(evaluator), "--submission", str(Path(predictions).resolve()),
                   "--split", "sample", "--out", str(config.reports / "evaluacion_oficial.json")]
        if ragas:
            command.append("--ragas")
        subprocess.run(command, check=True, cwd=evaluator.parent)
    print(f"[evaluation] accuracy: {report['accuracy_cerradas_local']} — coverage: {report['coverage']:.3f} "
          f"— avg latency: {report['latencia_promedio_ms']/1000:.2f}s/q", flush=True)
    return report

