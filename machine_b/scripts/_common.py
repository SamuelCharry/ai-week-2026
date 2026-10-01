"""Utilidades compartidas: carga de config, rutas, logging, hashing atomico."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def paths(config: dict) -> dict:
    out = {}
    for key, value in config["paths"].items():
        p = Path(value)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        p.mkdir(parents=True, exist_ok=True)
        out[key] = p
    return out


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    """Escribe en un temp del mismo directorio y hace rename atómico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, prefix=".tmp_") as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_json(path: Path, data: dict | list) -> None:
    write_atomic(path, json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    write_atomic(path, lines.encode("utf-8"))


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
