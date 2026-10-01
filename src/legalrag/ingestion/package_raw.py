"""Empaqueta solo originales, catálogos de procedencia y material oficial."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


DERIVED_NAMES = {"texto.txt", "texto_completo.txt", "texto_ocr.txt", "texto_revisado.txt"}


def raw_inputs(config):
    for name in ("raw", "oficial"):
        root = config.data_dir / name
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            if name == "raw" and "ampliacion" in path.parts and path.name in DERIVED_NAMES:
                continue
            yield path, path.relative_to(config.root).as_posix()


def package_raw(config):
    output = config.data_dir / "data_raw_oficial.zip"
    temp = output.with_suffix(".zip.tmp")
    files = list(raw_inputs(config))
    if not files:
        raise RuntimeError("No hay originales para empaquetar")
    names = [name for _, name in files]
    if len(names) != len(set(names)):
        raise ValueError("Hay rutas duplicadas en la selección del ZIP")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        for number, (path, name) in enumerate(files, 1):
            archive.write(path, name)
            if number % 1000 == 0 or number == len(files):
                print(f"[package] {number}/{len(files)} archivos — "
                      f"{temp.stat().st_size / 1024**3:.2f} GiB", flush=True)
    with ZipFile(temp) as archive:
        if set(archive.namelist()) != set(names):
            raise ValueError("El ZIP no contiene todas las rutas seleccionadas")
        corrupted = archive.testzip()
        if corrupted:
            raise ValueError(f"Entrada dañada en el ZIP: {corrupted}")
    temp.replace(output)
    print(f"[package] ZIP verificado: {output}", flush=True)
    return output
