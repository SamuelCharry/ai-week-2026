"""OCR por página para originales escaneados, con hashes y offsets."""

from pathlib import Path
import argparse
import json

from legalrag.config import CONFIG
from legalrag.io import sha256, write_json, now, dump_line


def ocr_document(config, metadata, raw_path, tessdata, dpi=300):
    import pymupdf
    from legalrag.ingestion.acquire import areas_from_text
    raw_path, tessdata = Path(raw_path).resolve(), Path(tessdata).resolve()
    if not raw_path.is_relative_to(config.root):
        raise ValueError("El original debe estar dentro del proyecto")
    if not (tessdata / "spa.traineddata").is_file():
        raise FileNotFoundError("Falta spa.traineddata para Tesseract")
    parts, pages, offset = [], [], 0
    with pymupdf.open(raw_path) as document:
        total = len(document)
        for index, page in enumerate(document):
            textpage = page.get_textpage_ocr(language="spa", dpi=dpi, full=True, tessdata=str(tessdata))
            text = page.get_text(textpage=textpage).strip()
            if len(text) < 50:
                raise ValueError(f"La página {index + 1} requiere revisión manual")
            if parts:
                parts.append("\n\f\n")
                offset += 3
            pages.append({"pagina": index + 1, "inicio": offset, "fin": offset + len(text)})
            parts.append(text)
            offset += len(text)
            print(f"[ocr] {index + 1}/{total} páginas — {offset} caracteres", flush=True)
    text_path = raw_path.with_name("texto_ocr.txt")
    text = "".join(parts)
    text_path.write_text(text, encoding="utf-8", newline="\n")
    areas, scores = areas_from_text(text, metadata.get("areas", []))
    doc = dict(metadata)
    doc.update(archivo_raw=raw_path.relative_to(config.root).as_posix(),
               sha256_raw=sha256(raw_path), bytes_raw=raw_path.stat().st_size,
               texto_archivo=text_path.relative_to(config.root).as_posix(),
               sha256_texto=sha256(text_path), caracteres=len(text),
               metodo_extraccion="ocr_tesseract", ocr={"idioma": "spa", "dpi": dpi,
               "pymupdf": pymupdf.VersionBind, "idioma_sha256": sha256(tessdata / "spa.traineddata"),
               "paginas": pages}, areas=areas, areas_puntajes=scores,
               apta_para_busqueda=True, fecha_descarga=now(), fecha_consulta=now()[:10],
               vigencia="por_verificar", revision_juridica="pendiente",
               avisos=["texto_obtenido_por_OCR", "vigencia_por_verificar"],
               nivel=metadata.get("nivel", "nucleo"), redistribuir_raw=True,
               licencia_fuente="Texto oficial. Se conservan licencia de origen y excepciones de data/raw/LICENSE.")
    write_json(raw_path.with_name("procedencia_ocr.json"), doc)
    return doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    doc = ocr_document(CONFIG, json.loads(args.metadata.read_text(encoding="utf-8")), args.source, args.tessdata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as stream:
        dump_line(stream, doc)
