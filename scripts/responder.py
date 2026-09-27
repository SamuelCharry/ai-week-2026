import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from scripts.auxiliares.generacion import ClienteLocal, controlar_citas, generar
from scripts.auxiliares.recuperacion import cargar_indice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indice", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--salida-raw", default="data/processed/respuestas_raw")
    parser.add_argument("--contexto", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sin-gramatica", action="store_true")
    argumentos = parser.parse_args()
    entrada = json.load(sys.stdin)
    recuperador = cargar_indice(RAIZ / argumentos.indice, device="cpu")
    pasajes = recuperador.buscar(entrada["pregunta"], k=10)
    for pasaje in pasajes:
        original = recuperador.unidades[pasaje["unidad_id"]]
        if any(pasaje[k] != original[k] for k in ["doc_id", "articulo", "inicio", "fin", "texto"]):
            raise ValueError("El pasaje recuperado no coincide con la unidad completa")
    clave = hashlib.sha256(json.dumps(entrada, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]
    ejecucion = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    carpeta = RAIZ / argumentos.salida_raw / clave / ejecucion
    cliente = ClienteLocal(argumentos.endpoint)
    propiedades, _ = cliente.pedir("/props")
    efectivo = propiedades.get("default_generation_settings", {}).get("n_ctx", propiedades.get("n_ctx"))
    if efectivo is None or efectivo < argumentos.contexto:
        raise ValueError("No se confirmó el contexto del servidor")
    salida, registro = generar(cliente, entrada, pasajes, contexto=argumentos.contexto,
                              max_tokens=argumentos.max_tokens, gramatica=not argumentos.sin_gramatica,
                              carpeta_raw=carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    if salida is not None:
        if not registro["abstencion_programatica"]:
            (carpeta / "salida_modelo.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")
        documentos = list({u["doc_id"]: u for u in recuperador.unidades.values()}.values())
        salida, control = controlar_citas(salida, documentos)
        registro.update(control)
    (carpeta / "registro.json").write_text(json.dumps(registro, ensure_ascii=False, indent=2), encoding="utf-8")
    if salida is None:
        raise ValueError("El modelo no produjo el formato requerido. La respuesta original quedó guardada.")
    print(json.dumps(salida, ensure_ascii=False))


if __name__ == "__main__":
    main()
