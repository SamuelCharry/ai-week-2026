"""Evaluador oficial de la hackathon.

Es el mismo evaluador que ejecuta el jurado. Los equipos pueden correrlo sobre
sample_50.jsonl durante toda la semana para conocer con exactitud su puntaje.

Uso, contra la muestra publica:
    python scripts/evaluate.py --submission mi_equipo.jsonl --split sample

Con el juez de texto libre:
    python scripts/evaluate.py --submission mi_equipo.jsonl --split sample --ragas

Uso interno del jurado, contra la clave del test:
    python scripts/evaluate.py --submission equipo.jsonl --split test --ragas

Componentes (80 pts automaticos del total de 100):
    20  accuracy en cerradas
    30  RAGAS answer correctness en texto libre   (requiere --ragas y API key)
    20  calidad de citacion normativa             (determinista, sin API)
    10  abstencion calibrada                      (determinista, sin API)

Sin --ragas el reporte se emite igual, con la seccion de correccion marcada como
pendiente: sirve para iterar rapido y sin consumo de API.

El juez de texto libre corre sobre OpenRouter con el modelo z-ai/glm-5.3-flash.
La llave se entrega en la sesion inaugural. Se lee de la variable de entorno
OPENROUTER_API_KEY o de un archivo scripts/.env con la linea
OPENROUTER_API_KEY=... Ese archivo no debe versionarse. El componente de similitud semantica de la metrica usa un
encoder abierto que se ejecuta localmente, sin llamadas de red adicionales.

Dependencias del juez: pip install -r scripts/requirements-evaluador.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import citations
from common import DATA, FLAWED_IDS, read_jsonl

PTS = {"cerradas": 20.0, "ragas": 30.0, "citas": 20.0, "abstencion": 10.0}

BASELINE = {"cerradas": 0.905, "ragas": 0.451}

JUEZ_MODELO = "z-ai/glm-5.3-flash"
JUEZ_BASE_URL = "https://openrouter.ai/api/v1"
JUEZ_ENCODER = "intfloat/multilingual-e5-large"
JUEZ_MAX_TOKENS = 16384
JUEZ_REASONING = {"effort": "minimal"}


def answer_text(sub: dict) -> str:
    """Texto de la respuesta del que se extraen las citas normativas."""
    f = sub.get("formato")
    if f == "multiple_choice":
        return sub.get("justificacion") or ""
    if f == "semi_open":
        return " ".join(str(sub.get(k) or "") for k in ("respuesta", "referencia_legal"))
    return " ".join(str(sub.get(k) or "")
                    for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion"))


def ragas_text(sub: dict) -> str:
    """Texto que se manda al juez: solo el contenido sustantivo, sin metadatos."""
    f = sub.get("formato")
    if f == "semi_open":
        return str(sub.get("respuesta") or "")
    return " ".join(str(sub.get(k) or "")
                    for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion"))


def validate(subs: list[dict], expected_ids: set[int]) -> list[str]:
    """Errores de forma. Una entrega invalida no se descalifica: los items mal
    formados cuentan como fallo, que es lo mismo que no haber respondido."""
    problems: list[str] = []
    seen: set[int] = set()
    required = {
        "multiple_choice": ("respuesta_correcta", "justificacion", "descarte_opciones"),
        "semi_open": ("respuesta", "palabras_clave", "referencia_legal"),
        "open_ended": ("marco_normativo", "analisis", "jurisprudencia", "conclusion"),
    }
    for i, s in enumerate(subs):
        if not isinstance(s.get("id"), int):
            problems.append(f"linea {i+1}: 'id' ausente o no entero")
            continue
        if s["id"] in seen:
            problems.append(f"item {s['id']}: duplicado")
        seen.add(s["id"])
        if s["id"] not in expected_ids:
            problems.append(f"item {s['id']}: no pertenece al split evaluado")
        if s.get("formato") not in required:
            problems.append(f"item {s['id']}: 'formato' invalido")
            continue
        if not s.get("abstencion"):
            faltan = [k for k in required[s["formato"]] if s.get(k) in (None, "", [], {})]
            if faltan:
                problems.append(f"item {s['id']}: faltan campos {faltan}")
            if not s.get("pasajes_recuperados"):
                problems.append(f"item {s['id']}: sin pasajes_recuperados")
    faltantes = expected_ids - seen
    if faltantes:
        problems.append(f"{len(faltantes)} items del split sin respuesta "
                        f"(cuentan como fallo): {sorted(faltantes)[:10]}...")
    return problems


MAX_PASAJES_EVIDENCIA = 10


def citas_respaldadas(sub: dict) -> set[tuple]:
    """Citas presentes en los pasajes que el sistema recupero para este item.

    El universo de verificacion es el corpus del propio equipo, sin limite
    superior: toda norma que el equipo incorpore y recupere queda disponible
    como respaldo de una citacion.

    Solo cuentan los primeros MAX_PASAJES_EVIDENCIA pasajes, para que el respaldo
    no se reduzca a adjuntar el corpus completo en cada respuesta.
    """
    cites: set[tuple] = set()
    for p in (sub.get("pasajes_recuperados") or [])[:MAX_PASAJES_EVIDENCIA]:
        cites |= citations.extract(str(p.get("texto") or ""))
    return cites


def score_closed(subs: dict[int, dict], key: dict[int, dict], ids: list[int]) -> dict:
    """Exactitud sobre los items cerrados calificados. Devuelve conteos y puntos."""
    ok = 0
    for qid in ids:
        s = subs.get(qid)
        if s and not s.get("abstencion") and s.get("respuesta_correcta") == key[qid]["respuesta_correcta"]:
            ok += 1
    acc = ok / len(ids) if ids else 0.0
    return {"n": len(ids), "aciertos": ok, "accuracy": acc,
            "referencia": BASELINE["cerradas"],
            "puntos": round(PTS["cerradas"] * acc, 2)}


def score_citations(subs: dict[int, dict], key: dict[int, dict]) -> dict:
    """Recall de citas penalizado por las citas sin respaldo en la evidencia.

    Solo entran los items cuyo fundamento de referencia tiene al menos una cita
    extraible (84% del banco); en el resto el legal_basis es doctrina o prosa y
    no hay contra que comparar.
    """
    agg = defaultdict(int)
    for f in ("n_ref", "n_citadas", "aciertos", "aciertos_respaldados",
              "aciertos_sin_respaldo", "incorrectas", "citas_sin_respaldo"):
        agg[f] = 0
    evaluados = 0
    for qid, k in key.items():
        ref = citations.extract(k.get("legal_basis") or "")
        if not ref:
            continue
        evaluados += 1
        s = subs.get(qid)
        if not s or s.get("abstencion"):
            agg["n_ref"] += len(citations.bodies(ref))
            continue
        r = citations.score(answer_text(s), k.get("legal_basis") or "",
                            citas_respaldadas(s))
        for f in ("n_ref", "n_citadas", "aciertos", "aciertos_respaldados",
                  "aciertos_sin_respaldo", "incorrectas", "citas_sin_respaldo"):
            agg[f] += r[f]

    W_SIN_RESPALDO = 0.5
    aciertos_ponderados = agg["aciertos_respaldados"] + W_SIN_RESPALDO * agg["aciertos_sin_respaldo"]
    recall = aciertos_ponderados / agg["n_ref"] if agg["n_ref"] else 0.0
    tasa_sin_respaldo = agg["citas_sin_respaldo"] / agg["n_citadas"] if agg["n_citadas"] else 0.0
    bruto = max(0.0, recall - 2 * tasa_sin_respaldo)
    return {"items_evaluados": evaluados, **dict(agg),
            "recall_citas_ponderado": round(recall, 4),
            "tasa_sin_respaldo": round(tasa_sin_respaldo, 4),
            "indice": round(bruto, 4),
            "puntos": round(PTS["citas"] * bruto, 2)}


def score_abstention(subs: dict[int, dict], key: dict[int, dict],
                     closed_ids: set[int]) -> dict:
    """Regla de puntuacion selectiva: acertar vale 1, abstenerse vale 0.5,
    equivocarse vale 0.

    Es la version clasica de selective prediction, y fija el incentivo correcto:
    una respuesta bien fundada es mejor que callar, y callar es mejor que
    inventar una norma. Abstenerse en todo da como maximo la mitad de este
    componente (5 de 100 en el ranking) y cero en accuracy y en citas, asi que
    la estrategia degenerada no es competitiva.

    Referencia de acierto: en cerradas, la opcion correcta; en texto libre,
    recuperar al menos una de las normas del fundamento de referencia.
    """
    W_ABSTENCION = 0.5
    tp = fp = fn = tn = 0
    for qid, k in key.items():
        ref = citations.extract(k.get("legal_basis") or "")
        if qid not in closed_ids and not ref:
            continue
        s = subs.get(qid)
        abst = bool(s and s.get("abstencion"))
        if s is None:
            fn += 1
            continue
        if qid in closed_ids:
            acerto = s.get("respuesta_correcta") == k.get("respuesta_correcta")
        else:
            got = citations.extract(answer_text(s))
            acerto = bool(citations.bodies(ref) & citations.bodies(got))
        if abst and not acerto:
            tp += 1
        elif abst and acerto:
            fp += 1
        elif not abst and acerto:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    calibracion = (tn + W_ABSTENCION * (tp + fp)) / total if total else 0.0
    return {"items_evaluados": total, "abstuvo_bien": tp, "abstuvo_de_mas": fp,
            "respondio_bien": tn, "respondio_mal": fn,
            "calibracion": round(calibracion, 4),
            "puntos": round(PTS["abstencion"] * calibracion, 2)}


def cargar_env() -> None:
    """Carga las variables de scripts/.env al entorno, si el archivo existe.

    Acepta lineas KEY=VALOR, con comillas opcionales y comentarios con #. No
    sobrescribe una variable que ya venga definida en el entorno, de modo que
    una llave exportada en la terminal tiene prioridad sobre el archivo.
    """
    for ruta in (Path(__file__).resolve().parent / ".env", DATA.parent / ".env"):
        if not ruta.is_file():
            continue
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            linea = linea.strip()
            if not linea or linea.startswith("#") or "=" not in linea:
                continue
            clave, valor = linea.split("=", 1)
            clave = clave.replace("export", "", 1).strip()
            valor = valor.strip().strip('"').strip("'")
            if clave and not os.environ.get(clave):
                os.environ[clave] = valor


def api_key_juez(base_url: str) -> str:
    """Devuelve la llave del juez segun el proveedor, o termina con un mensaje claro.

    base_url: endpoint del juez. Determina que variable de entorno se consulta.
    La llave puede venir del entorno o de un archivo .env junto a este script.
    """
    cargar_env()
    if "openrouter" in base_url:
        var, donde = "OPENROUTER_API_KEY", "la sesion inaugural"
    else:
        var, donde = "OPENAI_API_KEY", "su proveedor"
    llave = os.environ.get(var, "").strip()
    if not llave:
        sys.exit(
            "Falta la llave %s, necesaria para el juez de texto libre.\n"
            "Se entrega en %s. Puede declararla de dos formas:\n"
            "  1. Archivo scripts/.env con la linea:  %s=...\n"
            "  2. Variable de entorno:\n"
            "       PowerShell : $env:%s = \"...\"\n"
            "       bash       : export %s=...\n"
            "Sin ella, corra sin --ragas para obtener los 50 puntos deterministas."
            % (var, donde, var, var, var))
    return llave


def score_ragas(subs: dict[int, dict], key: dict[int, dict], judged: list[int],
                modelo: str = JUEZ_MODELO, base_url: str = JUEZ_BASE_URL) -> dict:
    """RAGAS answer correctness sobre el subconjunto estratificado de texto libre.

    judged: ids de los items de texto libre que entran en el componente.
    modelo: identificador del modelo juez en el proveedor.
    base_url: endpoint compatible con la API de OpenAI.

    La metrica combina correccion factual juzgada por el LLM con similitud
    semantica calculada por un encoder abierto que corre localmente, porque el
    proveedor del juez no expone endpoint de embeddings. Las abstenciones no
    entran al juez y computan como cero, por lo que el denominador es el total
    de items juzgados.

    Devuelve los conteos, el puntaje y la configuracion con que se calculo.
    """
    llave = api_key_juez(base_url)
    try:
        from datasets import Dataset
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from ragas import evaluate as ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_correctness
    except ImportError as exc:
        sys.exit(
            "Falta una dependencia del juez de texto libre (%s).\n"
            "  pip install -r scripts/requirements-evaluador.txt\n"
            "Sin ellas, corra sin --ragas para obtener los 50 puntos deterministas."
            % exc.name)

    rows = {"question": [], "answer": [], "ground_truth": []}
    usados: list[int] = []
    for qid in judged:
        s = subs.get(qid)
        if not s or s.get("abstencion"):
            continue
        rows["question"].append(key[qid]["pregunta"])
        rows["answer"].append(ragas_text(s))
        rows["ground_truth"].append(key[qid]["respuesta_esperada"])
        usados.append(qid)

    if not usados:
        return {"n_juzgados": len(judged), "n_respondidos": 0, "correctness": 0.0,
                "modelo_juez": modelo, "encoder": JUEZ_ENCODER,
                "referencia": BASELINE["ragas"], "puntos": 0.0}

    llm = LangchainLLMWrapper(ChatOpenAI(
        model=modelo, base_url=base_url, api_key=llave, temperature=0,
        max_tokens=JUEZ_MAX_TOKENS,
        extra_body={"reasoning": JUEZ_REASONING}))
    emb = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=JUEZ_ENCODER))

    res = ragas_evaluate(Dataset.from_dict(rows), metrics=[answer_correctness],
                         llm=llm, embeddings=emb)
    serie = res.to_pandas()["answer_correctness"]
    fallidos = int(serie.isna().sum())
    vals = list(serie.fillna(0.0))
    correctness = sum(vals) / len(judged)
    reporte = {"n_juzgados": len(judged), "n_respondidos": len(usados),
               "n_fallidos": fallidos,
               "correctness": round(correctness, 4),
               "modelo_juez": modelo, "encoder": JUEZ_ENCODER,
               "referencia": BASELINE["ragas"],
               "puntos": round(PTS["ragas"] * correctness, 2)}
    if fallidos:
        reporte["aviso"] = (
            "El juez no devolvio veredicto en %d de %d items; computan como cero. "
            "Revise la corrida antes de publicar el puntaje." % (fallidos, len(usados)))
        print(reporte["aviso"], file=sys.stderr)
    return reporte


def main() -> int:
    """Punto de entrada: lee la entrega, la valida, la califica y emite el reporte JSON."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission", required=True, type=Path)
    ap.add_argument("--split", choices=("sample", "test"), default="sample")
    ap.add_argument("--ragas", action="store_true",
                    help="corre el juez de texto libre (requiere OPENROUTER_API_KEY)")
    ap.add_argument("--judge-model", default=JUEZ_MODELO,
                    help="modelo juez (por defecto %s)" % JUEZ_MODELO)
    ap.add_argument("--judge-base-url", default=JUEZ_BASE_URL,
                    help="endpoint compatible con la API de OpenAI "
                         "(por defecto %s)" % JUEZ_BASE_URL)
    ap.add_argument("--out", type=Path, default=None, help="reporte JSON de salida")
    args = ap.parse_args()

    if args.split == "sample":
        rows = read_jsonl(DATA / "sample_50.jsonl")
        key = {r["id"]: {"respuesta_correcta": r.get("respuesta_correcta"),
                         "respuesta_esperada": r.get("respuesta_esperada"),
                         "legal_basis": r.get("legal_basis"),
                         "pregunta": r["pregunta"],
                         "formato": r["formato"]} for r in rows}
        judged = [q for q, k in key.items() if k["formato"] != "multiple_choice"]
        closed_ids = [q for q, k in key.items()
                      if k["formato"] == "multiple_choice" and q not in FLAWED_IDS]
    else:
        akey = {r["id"]: r for r in read_jsonl(DATA / "answer_key_992.jsonl")}
        by_id = {r["id"]: r for r in read_jsonl(DATA / "test_992.jsonl")}
        subset = json.loads((DATA / "scoring_subset.json").read_text(encoding="utf-8"))
        key = {q: {"respuesta_correcta": r.get("respuesta_correcta"),
                   "respuesta_esperada": r.get("respuesta_esperada"),
                   "legal_basis": r.get("legal_basis"),
                   "pregunta": by_id[q]["pregunta"],
                   "formato": r["formato"]} for q, r in akey.items()}
        judged = subset["free_text_judged_ids"]
        closed_ids = subset["closed_scored_ids"]

    subs_list = read_jsonl(args.submission)
    problems = validate(subs_list, set(key))
    subs = {s["id"]: s for s in subs_list if isinstance(s.get("id"), int)}

    report: dict = {
        "equipo": args.submission.stem,
        "split": args.split,
        "validacion": {"errores": len(problems), "detalle": problems[:40]},
        "cerradas": score_closed(subs, key, closed_ids),
        "citas": score_citations(subs, key),
        "abstencion": score_abstention(subs, key, set(closed_ids)),
    }
    if args.ragas:
        report["correccion_ragas"] = score_ragas(
            subs, key, judged, args.judge_model, args.judge_base_url)
    else:
        report["correccion_ragas"] = {"estado": "pendiente (correr con --ragas)",
                                      "puntos": None}

    puntos = [report[k].get("puntos") for k in ("cerradas", "correccion_ragas", "citas", "abstencion")]
    report["total_automatico"] = {
        "obtenidos": round(sum(p for p in puntos if p is not None), 2),
        "posibles": sum(PTS[k] for k, p in zip(("cerradas", "ragas", "citas", "abstencion"), puntos)
                        if p is not None),
        "de_100_del_ranking": 80.0,
        "nota": "los 20 pts restantes son interfaz grafica, bitacora de corpus, video y reproducibilidad",
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
