"""Utilidades compartidas por los scripts de la hackathon.

Define las rutas de trabajo, la lista canonica de areas del banco, los items
excluidos de la calificacion y las funciones de lectura y escritura de JSONL que
usan los demas scripts.
"""
from __future__ import annotations

import json
from pathlib import Path

HACK = Path(__file__).resolve().parents[1]
DATA = HACK / "data"
BANK = DATA / "questions_export.json"

FLAWED_IDS = {374}

AREAS = [
    "Derecho constitucional",
    "Derecho administrativo",
    "Derecho penal",
    "Derecho procesal",
    "Derecho comercial y sociedades",
    "Derecho civil",
    "Derecho de familia",
    "Derecho tributario",
    "Derecho laboral",
    "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]",
]

AREA_SLUG = {
    "Derecho constitucional": "constitucional",
    "Derecho administrativo": "administrativo",
    "Derecho penal": "penal",
    "Derecho procesal": "procesal",
    "Derecho comercial y sociedades": "comercial",
    "Derecho civil": "civil",
    "Derecho de familia": "familia",
    "Derecho tributario": "tributario",
    "Derecho laboral": "laboral",
    "Derecho de los mercados [competencia, consumidor, datos personales y propiedad intelectual]": "mercados",
}


def load_bank(path: Path | None = None, include_archived: bool = False) -> list[dict]:
    """Carga el banco de preguntas y normaliza el campo 'format'.

    path: archivo JSON del banco. Por defecto, BANK.
    include_archived: si es False, descarta los items que no estan aprobados.

    Devuelve la lista de items ordenada por (format, id). 'format' es uno de
    'multiple_choice', 'semi_open' u 'open_ended'. Las cerradas conservan
    'options' (lista de {text, is_correct}) y las de texto libre conservan
    'expected_answer'. Todos los formatos conservan 'legal_basis'.
    """
    raw = json.loads((path or BANK).read_text(encoding="utf-8"))
    items: list[dict] = []
    for it in raw["multiple_choice_questions"]:
        it = dict(it)
        it["format"] = "multiple_choice"
        items.append(it)
    for it in raw["text_questions"]:
        it = dict(it)
        it["format"] = it["type"]
        items.append(it)
    if not include_archived:
        items = [it for it in items if it["state"] == "approved"]
    items.sort(key=lambda it: (it["format"], it["id"]))
    return items


def correct_option(item: dict) -> str:
    """Devuelve el texto de la unica opcion correcta de un item cerrado."""
    return next(o["text"] for o in item["options"] if o["is_correct"])


def read_jsonl(path: Path) -> list[dict]:
    """Lee un archivo JSONL y devuelve la lista de objetos, ignorando lineas vacias."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Escribe rows como JSONL en path, creando los directorios necesarios."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
