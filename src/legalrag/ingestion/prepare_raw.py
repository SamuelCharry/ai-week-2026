"""Reconstruye el corpus base desde sus originales y su catálogo de fuentes."""

from collections import Counter
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree
import json
import os
import shutil
import subprocess
import zipfile

from legalrag.io import atomic_text, dump_line, records, safe_path, sha256, write_json
from legalrag.preprocessing.clean import clean_text
from legalrag.preprocessing.pdf import extract_pdf


BAD_SOURCE_HASHES = {
    "f3708f6b39baf84a413956d444a62858a16f5f99ea295568bf5a582f128b5808": "OCR defectuoso confirmado",
    "ba6b8388c4690612026f9953b5236da8ace97f16a069b0c04d78c1963508cfaf": "Codificación defectuosa confirmada",
    "7f6b5f8d6c0137064111ad0e5b74a45e205ed5fbaa2ddd3c818cf47db0ab0d2b": "Fuente excluida en revisión inicial",
    "39c754f18852167e1c99d619b6606718ef0079ae9ac2fc4ec33ce37cf0dc0c0b": "Fuente excluida en revisión inicial",
    "d6872dbbd1d6d0084e597e11252066c26ee69c63454f79fcfa818a025ed61aac": "Fuente excluida en revisión inicial",
}


def source_catalog(config):
    catalog = config.root / "data/raw/base_manifest.json"
    if not catalog.is_file():
        raise FileNotFoundError(f"Falta el catálogo de fuentes originales: {catalog}")
    base = json.loads(catalog.read_text(encoding="utf-8"))
    extras = Path(__file__).with_name("base_additions.jsonl")
    documents = base + list(records(extras))
    ids = [doc["doc_id"] for doc in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("El catálogo original contiene identificadores repetidos")
    return sorted(documents, key=lambda doc: doc["doc_id"]), {"catalogo_base": sha256(catalog),
                                                                "altas_catalogo": sha256(extras)}


def doc_converter():
    binary = os.environ.get("LEGALRAG_DOC_CONVERTER") or shutil.which("antiword") or shutil.which("soffice")
    if not binary:
        raise RuntimeError("Los originales .doc requieren antiword o LibreOffice en PATH. "
                           "También se puede definir LEGALRAG_DOC_CONVERTER.")
    result = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=20)
    return binary, (result.stdout or result.stderr).strip()[:200]


def extract_doc(path, converter):
    binary = converter[0]
    if "antiword" in Path(binary).stem.casefold():
        result = subprocess.run([binary, "-m", "UTF-8.txt", str(path)], capture_output=True, timeout=120)
        if result.returncode or not result.stdout.strip():
            raise ValueError(f"antiword no pudo extraer {path.name}: {result.stderr[:250]!r}")
        return result.stdout.decode("utf-8", errors="replace")
    with TemporaryDirectory() as temp:
        result = subprocess.run([binary, "--headless", "--convert-to", "txt:Text (encoded):UTF8",
                                 "--outdir", temp, str(path)], capture_output=True, timeout=120)
        output = Path(temp) / (path.stem + ".txt")
        if result.returncode or not output.is_file():
            raise ValueError(f"LibreOffice no pudo extraer {path.name}: {result.stderr[:250]!r}")
        return output.read_text(encoding="utf-8-sig")


def extract_docx(raw):
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        tree = ElementTree.fromstring(archive.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs = []
    for para in tree.iter(ns + "p"):
        value = "".join(node.text or "" for node in para.iter(ns + "t")).strip()
        if value:
            paragraphs.append(value)
    return "\n\n".join(paragraphs)


def extract_original(path, raw, converter):
    extension = path.suffix.casefold()
    if extension in {".html", ".htm"}:
        from legalrag.ingestion.acquire import html_text
        return html_text(raw)[1]
    if extension == ".pdf":
        return extract_pdf(raw)
    if extension == ".docx":
        return extract_docx(raw)
    if extension == ".doc":
        return extract_doc(path, converter)
    raise ValueError(f"Formato de original no reconocido: {extension}")


def _read_derived_text(config, item):
    """Devuelve el texto pre-OCR del catálogo si existe y su SHA-256 coincide."""
    derived = item.get("texto_derivado")
    if not derived:
        return None
    text_path = safe_path(config.root / "data/raw/base", derived["archivo"])
    if not text_path.is_file():
        return None
    if sha256(text_path) != derived["sha256"]:
        return None
    return text_path.read_text(encoding="utf-8")


def parser_fingerprint(converter):
    from legalrag.ingestion import acquire
    from legalrag.preprocessing import clean, pdf
    files = [Path(__file__), Path(acquire.__file__), Path(clean.__file__), Path(pdf.__file__)]
    return {"archivos": {str(path.name): sha256(path) for path in files}, "convertidor_doc": converter[1]}


def prepare_raw(config, replace=False):
    """Verifica hashes de los originales y escribe los textos derivados en data/processed."""
    from hashlib import sha256 as digest

    documents, catalog_hashes = source_catalog(config)
    destination = config.prepared
    inventory = destination / "documentos.jsonl"
    if inventory.exists() and not replace:
        raise FileExistsError("El corpus preparado ya existe. Usa prepare --replace para reconstruirlo desde raw.")
    converter = doc_converter()
    parser = parser_fingerprint(converter)
    fingerprint = digest(json.dumps({**catalog_hashes, **parser}, sort_keys=True).encode()).hexdigest()
    text_dir = destination / "textos"
    state_dir = destination / "registros"
    text_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    failures = []
    with atomic_text(inventory) as output:
        for number, source in enumerate(documents, 1):
            doc_id = source["doc_id"]
            originals = source.get("archivos_raw") or []
            if not originals:
                raise ValueError(f"{doc_id}: catálogo sin originales")
            paths = [safe_path(config.root / "data/raw/base", item["archivo"]) for item in originals]
            for path, item in zip(paths, originals):
                if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                    raise ValueError(f"{doc_id}: falta o cambió el original {item['archivo']}")
            text_path = text_dir / f"{doc_id}.txt"
            state_path = state_dir / f"{doc_id}.json"
            cache = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            cached = (cache.get("parser_sha256") == fingerprint and text_path.is_file()
                      and cache.get("sha256_texto") == sha256(text_path))
            if cached:
                doc = cache["documento"]
                counts["reutilizados"] += 1
            else:
                try:
                    pieces = []
                    for path, item in zip(paths, originals):
                        derived = _read_derived_text(config, item)
                        if derived is not None:
                            pieces.append(derived)
                        else:
                            pieces.append(extract_original(path, path.read_bytes(), converter))
                    text = clean_text("\n\n".join(piece for piece in pieces if piece.strip()))
                    if len(text) < 80:
                        raise ValueError("Texto insuficiente")
                    encoded = text.encode("utf-8")
                    with atomic_text(text_path) as stream:
                        stream.write(text)
                    excluded = [BAD_SOURCE_HASHES[item["sha256"]] for item in originals
                                if item["sha256"] in BAD_SOURCE_HASHES]
                    if doc_id == "sentencia_csj_sp1945_2019":
                        excluded.append("OCR de columnas defectuoso en revisión textual")
                    doc = dict(source)
                    doc.update(texto_archivo=text_path.relative_to(config.root).as_posix(),
                               sha256_texto=digest(encoded).hexdigest(), caracteres=len(text),
                               apta_para_busqueda=not excluded,
                               estado_extraccion="extraido", version_ingesta="raw-1",
                               revision_juridica="pendiente", avisos=excluded)
                    write_json(state_path, {"parser_sha256": fingerprint, "sha256_texto": doc["sha256_texto"],
                                            "documento": doc})
                    counts["extraidos"] += 1
                except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
                    failures.append({"doc_id": doc_id, "error": str(exc)})
                    doc = dict(source)
                    doc.update(texto_archivo=text_path.relative_to(config.root).as_posix(), sha256_texto=None,
                               caracteres=0, apta_para_busqueda=False, estado_extraccion="error",
                               avisos=["extraccion_fallida"], error=str(exc))
                    counts["errores"] += 1
            dump_line(output, doc)
            if number % 50 == 0 or number == len(documents):
                print(f"[prepare] {number}/{len(documents)} originales — "
                      f"extraídos: {counts['extraidos']} — reutilizados: {counts['reutilizados']} "
                      f"— errores: {counts['errores']}", flush=True)
    report = {"documentos": len(documents), "conteos": dict(counts), "errores": failures,
              "catalogos_sha256": catalog_hashes, "parser": parser,
              "inventario_sha256": sha256(inventory)}
    write_json(destination / "preparacion.json", report)
    if failures:
        raise RuntimeError(f"La preparación dejó {len(failures)} documentos sin extraer. "
                           "Revisar data/processed/corpus_preparado/preparacion.json y repetir prepare --replace.")
    print(f"[prepare] {len(documents)} textos derivados de originales verificados", flush=True)
    return report
