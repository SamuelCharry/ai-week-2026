"""00 — Verifica qué normas de `seed_targets.json` NO están en el corpus actual.

Lee:
  data/corpus_manifest.json        (lo que ya tenemos)
  data/oficial/data/seed_targets.json   (lo sugerido por los organizadores)
  reports/brechas_corpus.json      (si existe; brechas adicionales detectadas por audit)

Escribe:
  process/00_gaps.json             lista ordenada por prioridad (items_del_banco desc)
  process/00_gaps.md               resumen humano
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import load_config, paths, setup_logging, write_json

log = setup_logging("verify_gaps")


# Alias de códigos a (tipo, numero): si el canonico es un alias de código, también
# cuenta como presente cualquier doc con la ley/decreto correspondiente.
CODE_TO_LAW = {
    "codigo_general_proceso": ("ley", "1564", None),           # Ley 1564 2012
    "codigo_penal": ("ley", "599", None),                      # Ley 599 2000
    "codigo_procedimiento_penal": ("ley", "906", None),        # Ley 906 2004
    "cpaca": ("ley", "1437", None),                            # Ley 1437 2011
    "codigo_infancia": ("ley", "1098", None),                  # Ley 1098 2006
    "codigo_nacional_policia": ("ley", "1801", None),          # Ley 1801 2016
    "codigo_disciplinario": ("ley", "1952", None),             # Ley 1952 2019
    "estatuto_tributario": ("decreto", "624", None),           # Decreto 624 1989
    "estatuto_consumidor": ("ley", "1480", None),              # Ley 1480 2011
    "codigo_civil": ("ley", "84", None),                       # Ley 84 1873
    "codigo_comercio": ("decreto", "410", None),               # Decreto 410 1971
    "codigo_sustantivo_trabajo": ("decreto", "2663", None),    # Decreto 2663 1950
    "codigo_procesal_trabajo": ("decreto", "2158", None),      # Decreto 2158 1948
    "decision_andina_486": ("decision", "486", None),
}


def _norm_id(canon: list) -> list[str]:
    """Devuelve 1-N llaves de matching contra tokens del manifest."""
    kind = (canon[0] if canon else "").lower()
    num = str(canon[1]).replace(".", "").lstrip("0") if len(canon) > 1 and canon[1] else ""
    yr = str(canon[2]) if len(canon) > 2 and canon[2] else ""
    keys = []
    if "constitucion" in kind:
        keys.append("constitucion_1991")
    elif kind == "jurisprudencia" and num:
        keys.append(f"sentencia:{num.lstrip('0').split('-')[-1]}:{yr}")
        # también con número padded
        m = re.fullmatch(r"([a-z]+)-?(\d+)", num.lower())
        if m:
            keys.append(f"sentencia:{int(m.group(2))}:{yr}")
    elif kind in CODE_TO_LAW:
        alias_kind, alias_num, _ = CODE_TO_LAW[kind]
        # Match cualquier año si no está especificado
        keys.append(f"{alias_kind}:{alias_num}:*")
    elif kind and num and yr:
        keys.append(f"{kind}:{num}:{yr}")
    return keys


def _doc_id_tokens(doc: dict) -> set[str]:
    """Tokens útiles para matching aproximado contra un canonico."""
    tokens = set()
    tipo = (doc.get("tipo") or "").lower().replace(" ", "_")
    num = str(doc.get("numero") or "").replace(".", "").lstrip("0")
    yr = str(doc.get("anio") or "")
    doc_id = (doc.get("doc_id") or "").lower()
    if tipo and num and yr:
        tokens.add(f"{tipo}:{num}:{yr}")
        tokens.add(f"{tipo}:{num}:*")   # para match con alias sin año
    if "constitucion" in tipo or "constitucion" in doc_id:
        tokens.add("constitucion_1991")
    m = re.search(r"sentencia_[a-z]+_(\d+)_(\d{4})", doc_id)
    if m:
        tokens.add(f"sentencia:{int(m.group(1))}:{m.group(2)}")
    m = re.search(r"(c|t|su|sl|sc|sp|stc|stl|ac|au)[-_]?(\d+)[_-](\d{4})", doc_id)
    if m:
        tokens.add(f"sentencia:{int(m.group(2))}:{m.group(3)}")
    return tokens


def main() -> int:
    cfg = load_config()
    p = paths(cfg)

    manifest_path = p["input"] / "corpus_manifest.json"
    seeds_path = p["input"] / "data" / "oficial" / "data" / "seed_targets.json"
    if not manifest_path.is_file():
        log.error("No existe %s. Monta el repositorio en data/.", manifest_path)
        return 2
    if not seeds_path.is_file():
        log.error("No existe %s.", seeds_path)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    documentos = manifest.get("documentos", [])
    seed_docs = seeds.get("documentos", seeds if isinstance(seeds, list) else [])

    have: set[str] = set()
    for d in documentos:
        have |= _doc_id_tokens(d)

    missing = []
    present = []
    for s in seed_docs:
        canon = s.get("canonico") or []
        keys = _norm_id(canon)
        if any(k in have for k in keys):
            present.append(s)
            continue
        missing.append({
            "canonico": canon,
            "keys": keys,
            "norma": s.get("norma", ""),
            "items_del_banco": int(s.get("items_del_banco", 0)),
            "areas": s.get("areas", []),
            "donde_buscar": s.get("donde_buscar", ""),
            "fuente": _detect_source(s.get("donde_buscar", "")),
        })

    missing.sort(key=lambda m: -m["items_del_banco"])
    out = {
        "resumen": {
            "corpus_actual": len(documentos),
            "seeds_total": len(seed_docs),
            "seeds_presentes": len(present),
            "seeds_faltantes": len(missing),
            "items_del_banco_en_juego_si_cerramos_brecha": sum(m["items_del_banco"] for m in missing),
        },
        "por_fuente": _count_by_source(missing),
        "por_area": _count_by_area(missing),
        "missing": missing,
    }

    out_path = p["output"] / "00_gaps.json"
    write_json(out_path, out)
    log.info("Escribí %s", out_path)
    _write_markdown(p["output"] / "00_gaps.md", out)
    log.info("Resumen: %s / %s normas faltantes, %s items del banco en juego",
             out["resumen"]["seeds_faltantes"], out["resumen"]["seeds_total"],
             out["resumen"]["items_del_banco_en_juego_si_cerramos_brecha"])
    return 0


def _detect_source(url: str) -> str:
    low = url.lower()
    if "corteconstitucional" in low:
        return "corte_constitucional"
    if "suin" in low:
        return "suin"
    if "secretariasenado" in low:
        return "senado"
    if "dian" in low or "normograma" in low:
        return "dian"
    if "sic.gov.co" in low:
        return "sic"
    if "corte-suprema" in low or "cortesuprema" in low:
        return "corte_suprema"
    if "consejodeestado" in low:
        return "consejo_estado"
    return "otro"


def _count_by_source(missing: list[dict]) -> dict:
    counts: dict = {}
    for m in missing:
        counts.setdefault(m["fuente"], {"n": 0, "items": 0})
        counts[m["fuente"]]["n"] += 1
        counts[m["fuente"]]["items"] += m["items_del_banco"]
    return counts


def _count_by_area(missing: list[dict]) -> dict:
    counts: dict = {}
    for m in missing:
        for a in m["areas"]:
            counts.setdefault(a, 0)
            counts[a] += m["items_del_banco"]
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _write_markdown(path: Path, out: dict) -> None:
    r = out["resumen"]
    lines = [
        "# Brechas del corpus (vs seed_targets)\n",
        f"- Corpus actual: **{r['corpus_actual']}** documentos",
        f"- Seeds totales: **{r['seeds_total']}**",
        f"- Seeds presentes: **{r['seeds_presentes']}**",
        f"- **Seeds faltantes: {r['seeds_faltantes']}**",
        f"- **Items del banco en juego si cerramos la brecha: {r['items_del_banco_en_juego_si_cerramos_brecha']}**\n",
        "## Por fuente\n",
        "| Fuente | Normas faltantes | Items del banco |",
        "|---|---:|---:|",
    ]
    for fuente, v in sorted(out["por_fuente"].items(), key=lambda kv: -kv[1]["items"]):
        lines.append(f"| {fuente} | {v['n']} | {v['items']} |")
    lines.append("\n## Por área\n")
    lines.append("| Área | Items en juego |")
    lines.append("|---|---:|")
    for area, items in out["por_area"].items():
        lines.append(f"| {area} | {items} |")
    lines.append("\n## Top 30 por prioridad\n")
    lines.append("| Canonico | Items | Fuente | URL |")
    lines.append("|---|---:|---|---|")
    for m in out["missing"][:30]:
        url = (m["donde_buscar"][:60] + "…") if len(m["donde_buscar"]) > 60 else m["donde_buscar"]
        lines.append(f"| {m['canonico']} | {m['items_del_banco']} | {m['fuente']} | {url} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
