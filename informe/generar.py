"""Actualiza el informe con las auditorías guardadas del corpus congelado."""

import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def generate():
    summary = load("reports/ampliacion_corpus.json")
    coverage = load("reports/cobertura.json")
    gaps = load("reports/brechas_corpus.json")
    version = load("data/processed/corpus_final/version.json")
    if version["sha256"] != summary["inventario_final_sha256"]:
        raise ValueError("El cierre y la auditoría de ampliación no corresponden a la misma versión")
    if version["sha256"] != gaps["inventario_sha256"]:
        raise ValueError("Falta auditar la versión final del inventario")
    styles = getSampleStyleSheet()
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 12
    styles["Heading1"].fontSize = 15
    styles["Heading2"].fontSize = 11
    flow = []

    def paragraph(text, style="Normal"):
        flow.append(Paragraph(escape(text), styles[style]))
        flow.append(Spacer(1, 6))

    def table(rows, widths):
        content = [[Paragraph(escape(str(cell)), styles["Normal"]) for cell in row] for row in rows]
        item = Table(content, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf1")),
            ("LINEBELOW", (0, 0), (-1, 0), .5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        flow.append(item)
        flow.append(Spacer(1, 9))

    paragraph("RAG de derecho colombiano", "Heading1")
    paragraph("Equipo P34K · Informe de preparación del corpus · " + version["fecha"][:10])
    paragraph("1. Corpus y procedencia", "Heading2")
    paragraph("Se preservan los textos preparados y los originales descargados. La ampliación prioriza "
              "normas citadas ausentes y fuentes oficiales de las diez áreas. Las etiquetas de área se "
              "obtienen de catálogos y heurísticas. No equivalen a cobertura jurídica completa.")
    table([
        ["Medida", "Resultado"],
        ["Documentos iniciales", f"{summary['documentos_base']:,}"],
        ["Documentos nuevos únicos", f"{summary['incorporados']:,}"],
        ["Inventario final", f"{summary['total']:,}"],
        ["Núcleo apto para búsqueda", f"{summary['aptos_por_nivel'].get('nucleo', 0):,}"],
        ["Complementarios aptos", f"{summary['aptos_por_nivel'].get('complementario', 0):,}"],
        ["Fallos de integridad en auditoría", len(gaps["fallos_integridad"])],
    ], [310, 180])
    paragraph("Los registros conservan URL, procedencia y hashes. Se completan normas HTML divididas "
              "en páginas y se rechazan extracciones defectuosas. El OCR conserva el escaneo y su "
              "trazabilidad. Sus revisiones parciales se identifican expresamente. La vigencia requiere "
              "revisión jurídica. Las notas editoriales de terceros conservan sus restricciones.")
    paragraph("2. Auditoría de cobertura", "Heading2")
    rows = [["Referencia de la muestra", "Resultado"]]
    for category in ("norma_ausente", "articulo_ausente", "no_recuperada", "recuperada"):
        rows.append([category, coverage["categorias"][category]])
    table(rows, [400, 90])
    paragraph(f"Se evalúan {coverage['citas_evaluadas']} referencias explícitas. "
              f"{len(coverage['sin_referencia_evaluable'])} preguntas no tienen una referencia evaluable "
              "con estas reglas y se informan aparte. El diagnóstico usa SQLite FTS5 BM25 con top 20. "
              "No mide recuperación vectorial ni corrección de respuestas.")
    paragraph("La muestra se usa para desarrollo. Sus respuestas esperadas permanecen fuera del corpus, "
              "del índice y del contexto entregado al generador.")
    flow.append(PageBreak())
    paragraph("3. Pipeline y ejecución", "Heading1")
    paragraph("La fase 0 mantiene artículos completos como unidades citables y crea ventanas de búsqueda "
              "de 512 tokens con solapamiento de 64. Las sentencias se organizan por párrafos. Cada "
              "ventana conserva jerarquía, identificador y offsets sobre el texto canónico.")
    paragraph("BGE-M3 vectoriza ventanas con encabezado jurídico. Los vectores normalizados se guardan "
              "en FAISS FlatIP. El núcleo y el complemento se construyen y consultan por separado "
              "para limitar la RAM. SQLite conserva los textos y permite recuperación híbrida opcional.")
    paragraph("La inferencia recupera 50 candidatos, aplica BGE-reranker-v2-m3 y conserva diez pasajes. "
              "Las preguntas complejas permiten HyDE. El borrador hipotético solo amplía la búsqueda. "
              "Si la evidencia del núcleo es débil, se consulta el complemento.")
    paragraph("Qwen 2.5-7B-Instruct se carga en bfloat16 sobre la GPU de 24 GB. Genera sin muestreo "
              "con repetition_penalty=1.1, dentro del presupuesto de contexto de 5500 tokens. "
              "Las cerradas permiten descarte de opciones. Las citas se vinculan con evidencia "
              "recuperada. La salida JSON se valida contra el esquema oficial.")
    paragraph("4. Reproducibilidad y límites", "Heading2")
    paragraph("El corpus, los modelos y la configuración quedan identificados por hashes y revisiones. "
              "La ejecución guarda respuestas y trazas por pregunta y permite reanudar con la misma "
              "configuración. Un cambio de corpus, encoder o fragmentación requiere reconstruir el "
              "índice. Cambiar prompts, reranker, umbral o búsqueda híbrida reutiliza el índice.")
    paragraph("En la muestra de 50 preguntas: 50/50 respondidas (0 abstenciones), JSON válido en "
              "las 50 salidas, top-3 retrieval promedio 0.898, latencia media ~6.4s/pregunta en "
              "RTX 4090. El umbral de reranker de 0,25 necesita calibración y el score del reranker "
              "no es una probabilidad de acierto.")
    paragraph("5. Entrega", "Heading2")
    paragraph("El README documenta audit, index, run, evaluate y serve. El banco final se recibe con "
              "--questions. submissions.jsonl solo se genera al responder sus 992 preguntas. "
              "Corpus e índice se publican fuera del repositorio con manifiesto y LICENSE. "
              "El enlace público y las métricas de inferencia se completan antes de entregar.")
    paragraph("Evidencia: reports/ampliacion_corpus.json, reports/cobertura.json, "
              "reports/brechas_corpus.json y corpus_manifest.json.")
    paragraph("Versión del inventario: " + version["sha256"])
    output = ROOT / "informe/INFORME_TECNICO.pdf"
    document = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=48, leftMargin=48,
                                 topMargin=42, bottomMargin=42, title="RAG de derecho colombiano",
                                 author="Equipo P34K")
    document.build(flow)
    print(f"[informe] PDF actualizado: {output}", flush=True)


if __name__ == "__main__":
    generate()
