from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os


def records(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: JSON inválido") from exc


@contextmanager
def atomic_text(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        yield stream
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_json(path, value):
    with atomic_text(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def dump_line(stream, value):
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def safe_path(base, relative):
    base = Path(base).resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Ruta fuera del corpus: {relative}")
    return path


def inventory_path(config):
    frozen = config.root / "data/processed/corpus_final/documentos.jsonl"
    if frozen.is_file() and (frozen.parent / "version.json").is_file():
        return frozen
    prepared = config.prepared / "documentos.jsonl"
    if prepared.is_file():
        return prepared
    published = config.root / "corpus_manifest.json"
    if published.is_file():
        return published
    raise FileNotFoundError(f"Falta el inventario preparado: {prepared}")


def corpus_records(config):
    source = inventory_path(config)
    seen = set()
    if source.suffix == ".jsonl":
        base = records(source)
    else:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        base = iter(data["documentos"])
    for doc in base:
        if doc["doc_id"] in seen:
            raise ValueError(f"doc_id duplicado en inventario: {doc['doc_id']}")
        seen.add(doc["doc_id"])
        yield doc
    if source.parent.name == "corpus_final":
        return
    for extra in sorted((config.root / "data/raw/ampliacion").glob("documentos*.jsonl")):
        for doc in records(extra):
            if doc["doc_id"] not in seen:
                seen.add(doc["doc_id"])
                yield doc


def source_path(config, doc):
    relative = doc.get("texto_archivo", "")
    candidates = []
    if relative:
        candidates.extend([safe_path(config.prepared, relative), safe_path(config.root, relative)])
    candidates.append(safe_path(config.root / "corpus", doc["doc_id"] + ".txt"))
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"{doc['doc_id']}: falta texto canónico")


def partition(doc):
    return "nucleo" if doc.get("nivel") in ("nucleo", "núcleo", "core") else "complementario"
