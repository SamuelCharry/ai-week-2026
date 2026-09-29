"""Comando único de reproducción: de un clon limpio al puntaje sobre la muestra.

1. Comprueba el material oficial (data/oficial).
2. Descarga y verifica el corpus procesado y el índice congelado (configs/sistema.json,
   corpus_indice), salvo que ya estén en disco.
3. Responde las 50 preguntas de muestra con el sistema entregado.
4. Ejecuta el evaluador oficial sobre esa entrega.

Uso: python -m scripts.sistema.reproducir [--ragas]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from scripts.evaluacion.entrega import comprobar_oficiales  # noqa: E402
from scripts.sistema.pipeline import CONFIG, leer_config, responder_lote  # noqa: E402


def _sha256(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(8 * 1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def asegurar_corpus_indice(raiz, config):
    """Deja el corpus procesado y el índice en disco. Devuelve las rutas requeridas."""
    datos = config["corpus_indice"]
    requeridos = [raiz / r for r in datos["requeridos"]]
    if all(r.exists() for r in requeridos):
        return requeridos
    if not datos.get("url") or not datos.get("sha256"):
        raise RuntimeError("Faltan el corpus o el índice y configs/sistema.json no declara corpus_indice.url "
                           "y corpus_indice.sha256. Publicar el ZIP y declarar el enlace.")
    archivo = raiz / datos["archivo"]
    if not archivo.is_file() or _sha256(archivo) != datos["sha256"]:
        archivo.parent.mkdir(parents=True, exist_ok=True)
        temporal = archivo.with_name(archivo.name + ".part")
        print("Descargando corpus e índice:", datos["url"], flush=True)
        urllib.request.urlretrieve(datos["url"], temporal)
        if _sha256(temporal) != datos["sha256"]:
            temporal.unlink()
            raise ValueError("El ZIP descargado no coincide con corpus_indice.sha256")
        temporal.replace(archivo)
    destino = (raiz / datos["extraer_en"]).resolve()
    with zipfile.ZipFile(archivo) as paquete:
        for nombre in paquete.namelist():
            if not (destino / nombre).resolve().is_relative_to(destino):
                raise ValueError(f"Ruta insegura dentro del ZIP: {nombre}")
        paquete.extractall(destino)
    faltantes = [str(r) for r in requeridos if not r.exists()]
    if faltantes:
        raise FileNotFoundError("El ZIP no trae: " + ", ".join(faltantes))
    return requeridos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--ragas", action="store_true", help="incluye el juez de texto libre (OPENROUTER_API_KEY)")
    args = ap.parse_args()

    config = leer_config(args.config)
    oficiales = comprobar_oficiales(RAIZ / config["oficial"])
    asegurar_corpus_indice(RAIZ, config)
    salida = RAIZ / config["salidas"]["sample"]
    resumen = responder_lote(RAIZ, RAIZ / config["entradas"]["sample"], salida, config)
    print(json.dumps({k: v for k, v in resumen.items() if k != "errores_esquema"}, ensure_ascii=False, indent=2))
    reporte = salida.with_name(salida.stem + "_reporte.json")
    comando = [sys.executable, str(oficiales["evaluador"]), "--submission", str(salida),
               "--split", "sample", "--out", str(reporte)] + (["--ragas"] if args.ragas else [])
    subprocess.run(comando, check=True)
    print("Reporte del evaluador oficial:", reporte)


if __name__ == "__main__":
    main()
