import argparse
import concurrent.futures
import hashlib
import http.client
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile


RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
CATALOGO = RAIZ / "configs/modelos.json"
REGISTRO = RAIZ / "data/modelos/descargas.json"


def sha256(ruta):
    h = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(8 * 1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def git_blob(ruta):
    h = hashlib.sha1(f"blob {ruta.stat().st_size}\0".encode())
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(archivo):
    ruta = RAIZ / archivo["ruta"]
    ruta.parent.mkdir(parents=True, exist_ok=True)
    esperado = archivo.get("sha256")
    if ruta.exists() and (not archivo.get("bytes") or ruta.stat().st_size == archivo["bytes"]):
        actual = sha256(ruta)
        valido = esperado is None or actual == esperado
        if archivo.get("git_blob"):
            valido = valido and git_blob(ruta) == archivo["git_blob"]
        if valido:
            print(f"Disponible: {ruta.name}", flush=True)
            return {"ruta": archivo["ruta"], "sha256": actual, "bytes": ruta.stat().st_size}
    parcial = ruta.with_name(ruta.name + ".part")
    for intento in range(5):
        try:
            inicio = parcial.stat().st_size if parcial.exists() else 0
            cabeceras = {"User-Agent": "descarga-modelos/1.0"}
            if inicio:
                cabeceras["Range"] = f"bytes={inicio}-"
            url = archivo["url"]
            url += ("&" if "?" in url else "?") + f"descarga={time.time_ns()}"
            solicitud = urllib.request.Request(url, headers=cabeceras)
            with urllib.request.urlopen(solicitud, timeout=90) as respuesta:
                if inicio and respuesta.status == 206:
                    rango = respuesta.headers.get("Content-Range", "")
                    if not rango.startswith(f"bytes {inicio}-"):
                        raise ValueError(f"Rango inesperado: {rango}")
                    modo = "ab"
                else:
                    inicio = 0
                    modo = "wb"
                recibido = inicio
                ultimo = time.monotonic()
                with parcial.open(modo) as salida:
                    while bloque := respuesta.read(1024 * 1024):
                        salida.write(bloque)
                        recibido += len(bloque)
                        if time.monotonic() - ultimo >= 20:
                            total = archivo.get("bytes", 0)
                            detalle = f"{100 * recibido / total:.1f}%" if total else f"{recibido / 1e6:.0f} MB"
                            print(f"{ruta.name}: {detalle}", flush=True)
                            ultimo = time.monotonic()
            if archivo.get("bytes") and parcial.stat().st_size != archivo["bytes"]:
                raise ValueError(f"Tamaño incompleto: {ruta.name}")
            actual = sha256(parcial)
            if esperado and actual != esperado:
                parcial.replace(parcial.with_name(parcial.name + ".hash_incorrecto"))
                raise ValueError(f"Hash incorrecto: {ruta.name}")
            if archivo.get("git_blob"):
                if git_blob(parcial) != archivo["git_blob"]:
                    raise ValueError(f"Contenido distinto al snapshot: {ruta.name}")
            parcial.replace(ruta)
            print(f"Verificado: {ruta.name}", flush=True)
            return {"ruta": archivo["ruta"], "sha256": actual, "bytes": ruta.stat().st_size}
        except (OSError, ValueError, urllib.error.URLError, http.client.IncompleteRead) as error:
            print(f"Intento {intento + 1}/5 en {ruta.name}: {error}", flush=True)
            if intento == 4:
                raise
            time.sleep(min(2 ** intento, 8))


def descargar_lista(archivos):
    previos = json.loads(REGISTRO.read_text(encoding="utf-8")) if REGISTRO.exists() else []
    por_ruta = {x["ruta"]: x for x in previos}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ejecutor:
        futuros = [ejecutor.submit(descargar, archivo) for archivo in archivos]
        for futuro in concurrent.futures.as_completed(futuros):
            registro = futuro.result()
            por_ruta[registro["ruta"]] = registro
            REGISTRO.parent.mkdir(parents=True, exist_ok=True)
            REGISTRO.write_text(json.dumps(list(por_ruta.values()), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extraer(archivo, destino):
    destino = destino.resolve()
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archivo) as paquete:
        for nombre in paquete.namelist():
            (destino / nombre).resolve().relative_to(destino)
        paquete.extractall(destino)


def preparar_runtime(config):
    archivos = config["runtime"]["archivos"]
    descargar_lista(archivos)
    destino = RAIZ / config["runtime"]["ruta"]
    for archivo in archivos:
        extraer(RAIZ / archivo["ruta"], destino)
    print(f"Runtime: {destino}", flush=True)


def preparar_fuente(config):
    commit = config["runtime"]["commit"]
    codigo = RAIZ / "data/runtime/fuente" / f"llama.cpp-{commit}"
    url = f"https://api.github.com/repos/ggml-org/llama.cpp/git/trees/{commit}?recursive=1"
    for intento in range(5):
        try:
            peticion = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
            with urllib.request.urlopen(peticion, timeout=60) as respuesta:
                contenido = respuesta.read()
                if contenido.startswith(b"\x1f\x8b"):
                    contenido = gzip.decompress(contenido)
                arbol = json.loads(contenido)
            break
        except (OSError, ValueError, http.client.IncompleteRead):
            if intento == 4:
                raise
            time.sleep(2)
    archivos = []
    for item in arbol["tree"]:
        nombre = item["path"]
        necesario = nombre == "convert_hf_to_gguf.py" or (
            nombre.endswith(".py") and nombre.startswith(("conversion/", "gguf-py/gguf/")))
        if item["type"] == "blob" and necesario:
            archivos.append({"ruta": (codigo / nombre).relative_to(RAIZ).as_posix(),
                             "url": f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{commit}/{nombre}",
                             "bytes": item["size"], "git_blob": item["sha"]})
    descargar_lista(archivos)
    return codigo


def convertir_salamandra(config):
    modelo = next(x for x in config["decoders"] if x["nombre"] == "salamandra-7b-instruct")
    destino = RAIZ / modelo["ruta"]
    registro = destino.parent / "conversion.json"
    if destino.exists() and registro.exists():
        anterior = json.loads(registro.read_text(encoding="utf-8"))
        if sha256(destino) == anterior["sha256"] and anterior["revision_modelo"] == modelo["revision"]:
            print("Salamandra Q4_K_M disponible", flush=True)
            return
    codigo = preparar_fuente(config)
    entorno = dict(os.environ)
    entorno["PYTHONPATH"] = str(codigo / "gguf-py") + os.pathsep + entorno.get("PYTHONPATH", "")
    entorno["PYTHONIOENCODING"] = "utf-8"
    snapshot = RAIZ / modelo["snapshot"]
    intermedio = destino.parent / "salamandra-7b-instruct-F16.gguf"
    temporal = destino.with_name(destino.name + ".part")
    comandos = [
        [sys.executable, str(codigo / "convert_hf_to_gguf.py"), str(snapshot), "--outfile", str(intermedio), "--outtype", "f16", "--use-temp-file"],
        [str(RAIZ / config["runtime"]["ruta"] / "llama-quantize.exe"), str(intermedio), str(temporal), "Q4_K_M", "8"],
    ]
    for comando in comandos:
        print("Ejecutando: " + " ".join(comando), flush=True)
        subprocess.run(comando, cwd=codigo, env=entorno, check=True)
    temporal.replace(destino)
    registro.write_text(json.dumps({
        "repo_id": modelo["repo_id"],
        "revision_modelo": modelo["revision"],
        "revision_runtime": config["runtime"]["commit"],
        "cuantizacion": "Q4_K_M",
        "sha256": sha256(destino),
        "bytes": destino.stat().st_size,
        "comandos": comandos,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Salamandra convertida y verificada", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solo", choices=["todo", "salamandra", "runtime", "conversion", "qwen", "encoders"], default="todo")
    argumentos = parser.parse_args()
    config = json.loads(CATALOGO.read_text(encoding="utf-8"))
    if argumentos.solo in {"todo", "salamandra"}:
        modelo = next(x for x in config["decoders"] if x["nombre"] == "salamandra-7b-instruct")
        descargar_lista(modelo["archivos"])
    if argumentos.solo in {"todo", "runtime"}:
        preparar_runtime(config)
    if argumentos.solo in {"todo", "conversion"}:
        convertir_salamandra(config)
    if argumentos.solo in {"todo", "qwen"}:
        for modelo in config["decoders"]:
            if modelo["nombre"] != "salamandra-7b-instruct":
                descargar_lista(modelo["archivos"])
    if argumentos.solo == "encoders":
        from huggingface_hub import snapshot_download
        for modelo in config["encoders"]:
            ruta = snapshot_download(repo_id=modelo["repo_id"], revision=modelo["revision"], allow_patterns=["*.json", "*.safetensors", "*.bin", "*.model", "*.py", "*.txt"])
            print(f"{modelo['repo_id']}: {ruta}", flush=True)


if __name__ == "__main__":
    main()
