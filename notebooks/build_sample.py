"""Construye el corpus de laboratorio a partir del fixture local de Ley 1581."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legalrag.segment import segment_document  # noqa: E402

QUESTIONS = [
    "¿Cuál es el objeto de la Ley 1581 de 2012?",
    "¿A qué bases de datos se aplica la Ley 1581 y cuáles están exceptuadas?",
    "¿Qué significa dato personal y quién es el encargado del tratamiento?",
    "¿Cuáles son los principios de finalidad y libertad en el tratamiento de datos?",
    "¿Qué datos se consideran sensibles, incluidos los biométricos?",
    "¿En qué casos se permite tratar datos sensibles?",
    "¿Cómo protege la ley los datos de niños, niñas y adolescentes?",
    "¿Qué derechos tiene el titular para conocer, actualizar y rectificar sus datos?",
    "¿Qué características debe tener la autorización del titular?",
    "¿Cuándo no es necesaria la autorización del titular?",
    "¿En qué forma se puede suministrar la información solicitada?",
    "¿Qué debe informar el responsable al titular al pedir autorización?",
    "¿A quiénes se les puede suministrar información personal?",
    "¿Cómo se tramitan las consultas y cuál es su plazo de respuesta?",
    "¿Cómo se presentan los reclamos de corrección o supresión de datos?",
    "¿Cuál es el requisito antes de acudir a la Superintendencia por una queja?",
    "¿Cuáles son los deberes del responsable del tratamiento?",
    "¿Cuáles son los deberes del encargado del tratamiento?",
    "¿Qué autoridad ejerce la vigilancia de protección de datos?",
    "¿Con qué recursos cuenta la Superintendencia para ejercer sus funciones?",
]
TEST_ARTICLES = {3, 5, 8, 10, 13, 15, 18, 20}
EXTRA_QUESTIONS = [
    ("¿Puede quedar fuera del régimen una agenda de contactos usada solo en casa?", [2], "test"),
    ("Si una empresa almacena huellas digitales, ¿maneja información especialmente protegida?", [5], "dev"),
    ("¿Existe una excepción para analizar información médica con fines científicos sin identificar personas?", [6], "dev"),
    ("Encontré un dato equivocado sobre mí en una base; ¿qué facultad tengo?", [8], "test"),
    ("Una autoridad solicita información para ejercer una función legal; ¿se exige permiso del titular?", [10], "dev"),
    ("¿Cómo puedo acceder a mis datos y cuánto pueden tardar en contestar?", [14], "dev"),
    ("¿Qué procedimiento sigo para pedir que corrijan un registro inexacto?", [15], "test"),
    ("¿Qué obligaciones tienen quien decide sobre la base y quien la procesa por su cuenta?", [17, 18], "dev"),
    ("¿Cómo se llama la persona que trata datos por cuenta de quien decide sobre la base?", [3], "test"),
    ("¿Qué personas pueden recibir los datos de un titular?", [13], "dev"),
    ("¿Cuál entidad puede vigilar el cumplimiento de las reglas sobre privacidad?", [19], "test"),
    ("¿Qué debe hacer el proveedor que administra una base cuando recibe novedades sobre un dato?", [18], "dev"),
]


def build() -> dict:
    source = ROOT / "tests/fixtures/ley_1581_2012.txt"
    raw = source.read_text(encoding="utf-8")
    articles = segment_document(raw, {"doc_id": "ley_1581_2012", "norma": "Ley 1581 de 2012"}, max_chars=100_000)
    docs = []
    for a in articles[:20]:
        n = int(a["articulo"])
        docs.append({
            "doc_id": f"art{n:02d}", "article": n, "title": a["texto"].split(".", 2)[1].strip() if "." in a["texto"] else f"Artículo {n}",
            "section": a["seccion"], "text": a["texto"],
            "source": "Ley 1581 de 2012", "source_file": "tests/fixtures/ley_1581_2012.txt",
            "source_start": a["inicio"], "source_end": a["fin"],
        })
    queries = [{"query_id": f"q{n:02d}", "question": q, "relevant_doc_ids": [f"art{n:02d}"],
                "split": "test" if n in TEST_ARTICLES else "dev"}
               for n, q in enumerate(QUESTIONS, 1)]
    queries.extend({"query_id": f"q{n:02d}", "question": text,
                    "relevant_doc_ids": [f"art{article:02d}" for article in relevant], "split": split}
                   for n, (text, relevant, split) in enumerate(EXTRA_QUESTIONS, len(QUESTIONS) + 1))
    return {"description": "20 artículos reales del fixture local, con 32 consultas y juicios de relevancia escritos para el laboratorio; no son un benchmark externo.",
            "documents": docs, "queries": queries}


if __name__ == "__main__":
    out = Path(__file__).with_name("sample_ley1581.json")
    out.write_text(json.dumps(build(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{out}: 20 documentos y 32 consultas")
