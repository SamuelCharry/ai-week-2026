import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from collections import defaultdict

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ))

from scripts.evaluacion.entrega import auditar_citas
from scripts.generacion.cliente import CAMPOS, ServidorLocal, generar
from scripts.indice.recuperacion import cargar_corpus, leer_jsonl, sha256


def crear_sondas(unidades, cantidad=1):
    normas = defaultdict(list)
    for unidad in unidades:
        if unidad.get("articulo") is not None and 160 <= len(unidad["texto"]) <= 1200:
            normas[unidad["doc_id"]].append(unidad)
    candidatas = []
    for doc_id, articulos in sorted(normas.items()):
        if len(articulos) >= 4:
            articulos.sort(key=lambda u: hashlib.sha256(u["unidad_id"].encode()).hexdigest())
            candidatas.append((articulos[0], articulos[:4]))
    candidatas.sort(key=lambda par: hashlib.sha256(par[0]["unidad_id"].encode()).hexdigest())
    sondas = []
    for unidad, opciones in candidatas[:cantidad]:
        opciones = sorted(opciones, key=lambda u: u["inicio"])
        for formato in CAMPOS:
            entrada = {"id": unidad["unidad_id"] + "_" + formato,
                       "formato": formato,
                       "pregunta": f"artículo {unidad['articulo']} de {unidad['titulo']}"}
            if formato == "multiple_choice":
                entrada["opciones"] = {letra: opcion["texto"] for letra, opcion in zip("ABCD", opciones)}
            contextos = [True] if formato == "multiple_choice" else [True, False]
            for con_contexto in contextos:
                pregunta = dict(entrada)
                pregunta["id"] += "_con_contexto" if con_contexto else "_sin_contexto"
                sondas.append({"entrada": pregunta, "pasajes": [unidad] if con_contexto else [],
                               "con_contexto": con_contexto, "unidad_id": unidad["unidad_id"],
                               "doc_id": unidad["doc_id"], "areas": unidad.get("areas", [])})
    if not sondas:
        raise ValueError("No hay cuatro artículos íntegros de una misma norma para construir las sondas")
    return sondas


class Medidor:
    def __init__(self, pid):
        self.pid = pid
        self.fin = threading.Event()
        self.ram = []
        self.vram = []
        self.hilo = threading.Thread(target=self.medir, daemon=True)

    def medir(self):
        import psutil

        ultima_gpu = 0
        while not self.fin.is_set():
            try:
                proceso = psutil.Process(self.pid)
                familia = [proceso] + proceso.children(recursive=True)
                self.ram.append(sum(p.memory_info().rss for p in familia) / 2**20)
                if time.monotonic() - ultima_gpu > 1:
                    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    valores = [float(x) for x in r.stdout.strip().splitlines() if x.strip().isdigit()]
                    if valores:
                        self.vram.append(sum(valores))
                    ultima_gpu = time.monotonic()
            except (OSError, ValueError, psutil.Error, subprocess.TimeoutExpired):
                pass
            self.fin.wait(0.1)

    def __enter__(self):
        self.hilo.start()
        return self

    def __exit__(self, tipo, valor, traza):
        self.fin.set()
        self.hilo.join(timeout=7)

    def resultado(self):
        return {"ram_rss_pico_mib": max(self.ram) if self.ram else None,
                "vram_total_pico_mib": max(self.vram) if self.vram else None,
                "muestras_ram": len(self.ram), "muestras_vram": len(self.vram)}


def medir_formato(salida, documentos):
    if salida is None:
        return {"abstencion": None, "citas_detectadas": 0, "citas_sin_respaldo": None,
                "citas_respaldadas": None, "citas_indeterminadas": None,
                "auditoria_disponible": False, "citas_revision_manual": True,
                "extension_local_valida": False}
    citas = auditar_citas(salida, documentos)
    formato = salida["formato"]
    campo = "respuesta" if formato == "semi_open" else "analisis"
    texto = salida.get(campo, "")
    oraciones = len([x for x in re.split(r"[.!?]+(?:\s+|$)", texto) if x.strip()])
    palabras = len(texto.split())
    extension = True
    if not salida["abstencion"]:
        if formato == "semi_open":
            extension = 3 <= oraciones <= 5 and palabras <= 150
        if formato == "open_ended":
            extension = 5 <= oraciones <= 8
    return {"abstencion": salida["abstencion"], "citas_detectadas": len(citas["citas"]),
            "citas_sin_respaldo": citas["sin_respaldo"], "citas_revision_manual": citas["revision_manual"],
            "citas_respaldadas": sum(c.get("respaldada") is True for c in citas["citas"]),
            "citas_indeterminadas": sum(c.get("respaldada") is None for c in citas["citas"]),
            "auditoria_disponible": True,
            "citas": citas["citas"], "extension_local_valida": extension,
            "oraciones_estimadas": oraciones, "palabras_campo_principal": palabras}


def resumen(resultados):
    grupos = defaultdict(list)
    for fila in resultados:
        grupos[(fila["modelo"], fila["gramatica"])].append(fila)
    filas = []
    for (modelo, gramatica), casos in grupos.items():
        segundos = [c["segundos"] for c in casos if c.get("segundos") is not None]
        sin_contexto = [c for c in casos if not c["con_contexto"]]
        con_contexto = [c for c in casos if c["con_contexto"]]
        repetidos = [c for c in casos if c["corrida"] == 2]
        n_citas = sum(c.get("citas_detectadas", 0) for c in casos)
        sin_respaldo = sum(c.get("citas_sin_respaldo") or 0 for c in casos)
        respaldadas = sum(c.get("citas_respaldadas") or 0 for c in casos)
        indeterminadas = sum(c.get("citas_indeterminadas") or 0 for c in casos)
        no_auditables = sum(not c.get("auditoria_disponible", False) for c in casos)
        ram = [c["ram_rss_pico_mib"] for c in casos if c.get("ram_rss_pico_mib") is not None]
        vram = [c["vram_total_pico_mib"] for c in casos if c.get("vram_total_pico_mib") is not None]
        filas.append({"modelo": modelo, "gramatica": gramatica, "ejecuciones": len(casos),
                      "json_validos": sum(c.get("json_valido", False) for c in casos),
                      "claves_validas": sum(c.get("claves_validas", False) for c in casos),
                      "abstenciones_sin_contexto": sum(c.get("abstencion") is True for c in sin_contexto),
                      "casos_sin_contexto": len(sin_contexto),
                      "abstenciones_con_contexto": sum(c.get("abstencion") is True for c in con_contexto),
                      "casos_con_contexto": len(con_contexto),
                      "citas_detectadas": n_citas, "citas_sin_respaldo": sin_respaldo,
                      "citas_respaldadas": respaldadas, "citas_indeterminadas": indeterminadas,
                      "respuestas_no_auditables": no_auditables,
                      "citas_respaldadas_proporcion": respaldadas / n_citas if n_citas and not no_auditables else None,
                      "repeticiones_identicas": sum(c.get("determinista") is True for c in repetidos),
                      "pares": len(repetidos), "mediana_s": statistics.median(segundos) if segundos else None,
                      "maximo_s": max(segundos) if segundos else None,
                      "dentro_22_s": sum(c.get("segundos", float("inf")) <= 22 for c in casos),
                      "ram_rss_pico_mib": max(ram) if ram else None,
                      "vram_total_pico_mib": max(vram) if vram else None})
    return filas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalogo", default="configs/modelos.json")
    parser.add_argument("--corpus", default="data/processed/corpus")
    parser.add_argument("--salida", default="data/processed/comparacion_decoders")
    parser.add_argument("--modelos", nargs="+")
    parser.add_argument("--unidades", type=int, default=1)
    parser.add_argument("--gramatica", choices=["sin", "con", "ambas"], default="con")
    parser.add_argument("--contexto", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    argumentos = parser.parse_args()
    catalogo = json.loads((RAIZ / argumentos.catalogo).read_text(encoding="utf-8"))
    base_salida = RAIZ / argumentos.salida
    nombre_ejecucion = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    carpeta = base_salida / nombre_ejecucion
    carpeta.mkdir(parents=True, exist_ok=True)
    corpus = RAIZ / argumentos.corpus
    unidades, exclusiones = cargar_corpus(corpus)
    documentos = leer_jsonl(corpus / "documentos.jsonl")
    sondas = crear_sondas(unidades, argumentos.unidades)
    (carpeta / "sondas.json").write_text(json.dumps(sondas, ensure_ascii=False, indent=2), encoding="utf-8")
    modos = [False, True] if argumentos.gramatica == "ambas" else [argumentos.gramatica == "con"]
    configuracion = {"sha256_unidades": sha256(corpus / "unidades.jsonl"),
                    "sha256_documentos": sha256(corpus / "documentos.jsonl"),
                    "contexto": argumentos.contexto, "max_tokens": argumentos.max_tokens,
                    "unidades": argumentos.unidades, "gramatica": argumentos.gramatica,
                    "exclusiones_corpus": exclusiones, "temperatura": 0, "semilla": 0,
                    "cache_prompt": False, "corridas": 2, "runtime": catalogo["runtime"]["commit"],
                    "endpoint_generacion": "/completion", "plantilla": "/apply-template",
                    "repeat_penalty": 1.0, "frequency_penalty": 0, "presence_penalty": 0,
                    "sha256_catalogo": sha256(RAIZ / argumentos.catalogo),
                    "sha256_generacion": sha256(RAIZ / "scripts/generacion/cliente.py"),
                    "sha256_evaluacion": sha256(RAIZ / "scripts/evaluacion/entrega.py"),
                    "sha256_recuperacion": sha256(RAIZ / "scripts/indice/recuperacion.py"),
                    "sha256_ejecutor_notebook": sha256(RAIZ / "scripts/entorno/ejecutar_notebook.py"),
                    "python": sys.version, "plataforma": sys.platform,
                    "sha256_comparador": sha256(Path(__file__))}
    disponibles = {m["nombre"] for m in catalogo["decoders"]}
    if argumentos.modelos and not set(argumentos.modelos) <= disponibles:
        raise ValueError("Hay modelos que no figuran entre los decoders admitidos")
    nombre_servidor = "llama-server.exe" if os.name == "nt" else "llama-server"
    ejecutable = Path(os.environ.get("LLAMA_SERVER") or RAIZ / catalogo["runtime"]["ruta"] / nombre_servidor)
    archivos_runtime = sorted(ejecutable.parent.glob("*.dll")) + sorted(ejecutable.parent.glob("*.so*")) + [ejecutable]
    configuracion["sha256_runtime"] = {p.name: sha256(p) for p in archivos_runtime if p.is_file()}
    if not ejecutable.is_file():
        raise FileNotFoundError(f"Falta {ejecutable}")
    huella = hashlib.sha256(json.dumps(configuracion, sort_keys=True).encode()).hexdigest()
    (carpeta / "configuracion.json").write_text(json.dumps(configuracion, indent=2, ensure_ascii=False), encoding="utf-8")
    estados = []
    for modelo in catalogo["decoders"]:
        if argumentos.modelos and modelo["nombre"] not in argumentos.modelos:
            continue
        if modelo["parametros"] > 8000000000:
            raise ValueError("Modelo fuera del límite de parámetros")
        destino = carpeta / modelo["nombre"]
        destino.mkdir(exist_ok=True)
        primeras = {}
        resultados = []
        try:
            hashes_modelo = {}
            archivos = modelo["archivos"]
            if modelo["nombre"] == "salamandra-7b-instruct":
                conversion = json.loads((RAIZ / modelo["ruta"]).with_name("conversion.json").read_text(encoding="utf-8"))
                if conversion["revision_modelo"] != modelo["revision"]:
                    raise ValueError("La conversión no corresponde al snapshot del catálogo")
                archivos = [{"ruta": modelo["ruta"], "sha256": conversion["sha256"]}]
            for archivo in archivos:
                actual = sha256(RAIZ / archivo["ruta"])
                if actual != archivo["sha256"]:
                    raise ValueError("Cambió un archivo del modelo: " + archivo["ruta"])
                hashes_modelo[archivo["ruta"]] = actual
            with ServidorLocal(ejecutable, RAIZ / modelo["ruta"],
                               destino, contexto=argumentos.contexto) as servidor:
                info = {"configuracion_sha256": huella, "modelo": modelo,
                        "sha256_modelo": hashes_modelo, "propiedades_servidor": servidor.propiedades,
                        "capas_gpu": servidor.capas_gpu,
                        "comando_servidor": servidor.comando, "pid": servidor.proceso.pid,
                        "memoria": "RSS del proceso y memoria total de GPU muestreados. Incluye uso gráfico externo."}
                (destino / "configuracion.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
                for gramatica in modos:
                    for corrida in [1, 2]:
                        for sonda in sondas:
                            entrada = sonda["entrada"]
                            raw = destino / ("con_gramatica" if gramatica else "sin_gramatica") / f"corrida_{corrida}" / entrada["id"]
                            fila = {"modelo": modelo["nombre"], "gramatica": gramatica, "corrida": corrida,
                                    "con_contexto": sonda["con_contexto"], "id": entrada["id"],
                                    "formato": entrada["formato"], "unidad_id": sonda["unidad_id"],
                                    "areas": sonda["areas"], "configuracion_sha256": huella}
                            salida = None
                            try:
                                with Medidor(servidor.proceso.pid) as memoria:
                                    salida, registro = generar(servidor.cliente, entrada, sonda["pasajes"],
                                                              contexto=argumentos.contexto, max_tokens=argumentos.max_tokens,
                                                              gramatica=gramatica, abstencion_automatica=False, carpeta_raw=raw)
                                fila.update(registro, **memoria.resultado(), **medir_formato(salida, documentos))
                                clave = (gramatica, entrada["id"])
                                if corrida == 1:
                                    primeras[clave] = (registro.get("contenido"), salida)
                                else:
                                    fila["determinista"] = primeras.get(clave) == (registro.get("contenido"), salida)
                                if salida is not None:
                                    (raw / "salida.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
                            except (OSError, ValueError, RuntimeError, KeyError) as error:
                                fila.update(error=str(error), json_valido=False, claves_validas=False, determinista=False)
                            resultados.append(fila)
                            (destino / "resultados.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in resultados) + "\n", encoding="utf-8")
                            print(f"{modelo['nombre']} | {entrada['formato']} | contexto={sonda['con_contexto']} | gramatica={gramatica} | corrida={corrida} | JSON={fila.get('json_valido')}", flush=True)
                estados.append({"modelo": modelo["nombre"], "estado": "ejecutado", "respuestas": len(resultados)})
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            (destino / "error.txt").write_text(str(error), encoding="utf-8")
            print(f"{modelo['nombre']}: {error}", flush=True)
            estados.append({"modelo": modelo["nombre"], "estado": "error", "detalle": str(error)})
    todos = []
    for modelo in catalogo["decoders"]:
        ruta = carpeta / modelo["nombre"] / "resultados.jsonl"
        if ruta.exists():
            todos.extend(r for r in leer_jsonl(ruta) if r.get("configuracion_sha256") == huella)
    tabla = resumen(todos)
    (carpeta / "resumen.json").write_text(json.dumps(tabla, indent=2, ensure_ascii=False), encoding="utf-8")
    (carpeta / "estado_modelos.json").write_text(json.dumps(estados, indent=2, ensure_ascii=False), encoding="utf-8")
    (base_salida / "ultima_ejecucion.json").write_text(json.dumps({"ruta": carpeta.relative_to(RAIZ).as_posix()}, indent=2), encoding="utf-8")
    reporte = RAIZ / "reports/f_decoders.csv"
    reporte.parent.mkdir(parents=True, exist_ok=True)
    with reporte.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(tabla[0]) if tabla else ["modelo", "gramatica", "ejecuciones"])
        escritor.writeheader()
        escritor.writerows(tabla)
    print(json.dumps(tabla, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
