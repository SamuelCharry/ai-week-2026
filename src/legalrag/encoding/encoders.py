import json
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))
from huggingface_hub.constants import HF_HUB_CACHE
from legalrag.modelos import descargar

catalogo = json.loads(Path("configs/modelos.json").read_text(encoding="utf-8"))
registros = []
for ficha in sorted(catalogo["encoders"], key=lambda f: f["prioridad"]):
    print("Descargando " + ficha["nombre"], flush=True)
    ruta = Path(HF_HUB_CACHE) / ("models--" + ficha["repo_id"].replace("/", "--")) / "snapshots" / ficha["revision"]
    ruta.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/api/models/{ficha['repo_id']}/revision/{ficha['revision']}?blobs=true"
    with urllib.request.urlopen(url, timeout=60) as respuesta:
        datos = json.load(respuesta)
    for meta in datos["siblings"]:
        nombre_meta = meta["rfilename"]
        if not nombre_meta.endswith((".json", ".model", ".txt")):
            continue
        destino = ruta / nombre_meta
        if not destino.resolve().is_relative_to(ruta.resolve()):
            raise ValueError("Archivo fuera del snapshot")
        destino.parent.mkdir(parents=True, exist_ok=True)
        url_meta = f"https://huggingface.co/{ficha['repo_id']}/resolve/{ficha['revision']}/{nombre_meta}"
        if meta.get("lfs"):
            descargar({"ruta": str(destino), "bytes": meta["size"], "sha256": meta["lfs"]["sha256"], "url": url_meta})
            continue
        contenido = destino.read_bytes() if destino.exists() else b""
        huella_git = lambda b: hashlib.sha1(b"blob " + str(len(b)).encode() + b"\0" + b).hexdigest()
        if huella_git(contenido) != meta["blobId"]:
            with urllib.request.urlopen(url_meta, timeout=60) as respuesta:
                contenido = respuesta.read()
            if huella_git(contenido) != meta["blobId"]:
                raise ValueError("No coincide el hash de " + nombre_meta)
            destino.write_bytes(contenido)
    disponibles = {f["rfilename"]: f for f in datos["siblings"]}
    nombre = "model.safetensors" if "model.safetensors" in disponibles else "pytorch_model.bin"
    pesos = disponibles[nombre]
    archivo = {"ruta": str(ruta / nombre), "bytes": pesos["size"],
               "sha256": pesos["lfs"]["sha256"],
               "url": f"https://huggingface.co/{ficha['repo_id']}/resolve/{ficha['revision']}/{nombre}"}
    descargar(archivo)
    registros.append({"modelo": ficha["repo_id"], "revision": ficha["revision"], **archivo})
    Path("data/experimentos").mkdir(parents=True, exist_ok=True)
    Path("data/experimentos/encoders_descarga.json").write_text(json.dumps(registros, indent=2), encoding="utf-8")
    print("Disponible " + ficha["nombre"], flush=True)
