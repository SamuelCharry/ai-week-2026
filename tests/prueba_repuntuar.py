"""Prueba sin GPU de repuntuar_03 y comparar con una carpeta 03 simulada.

Uso: python -m tests.prueba_repuntuar
"""
import json
import sys
import tempfile
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))

from legalrag.experimentos import v04  # noqa: E402
from legalrag.generation import politica as gen  # noqa: E402


def main():
    preguntas, entradas = v04.muestra(RAIZ)
    ev = v04.evidencia(RAIZ)
    claves = {q["id"]: q for q in preguntas}
    # Pasajes simulados: primeras unidades del Código Civil con la cabecera vieja del 03.
    from legalrag.evaluation.oficial import Evidencia
    vieja = Evidencia(RAIZ, RAIZ / "data/processed/corpus")
    unidades = []
    with (RAIZ / "data/processed/corpus/unidades.jsonl").open(encoding="utf-8") as archivo:
        for linea in archivo:
            if '"ley_84_1873"' in linea:
                u = json.loads(linea)
                if u.get("articulo"):
                    unidades.append(u | {"score": 1.0})
                if len(unidades) == 5:
                    break
    pasajes_03, _ = vieja.preparar(unidades)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        recup, ejec = tmp / "recuperacion", tmp / "ejecucion"
        (recup / "recuperaciones").mkdir(parents=True)
        (recup / "recuperaciones" / "densa.json").write_text(json.dumps([{"id": e["id"], "pasajes": pasajes_03} for e in entradas]))
        diag = []
        for n, e in enumerate(entradas):
            letras = list((e.get("opciones") or {}).keys())
            if e["formato"] == "multiple_choice":
                contenido = json.dumps({"respuesta_correcta": claves[e["id"]]["respuesta_correcta"] if n % 2 else None,
                                        "justificacion": "Según el artículo 1 del Código Civil.", "descarte_opciones": {},
                                        "abstencion": True})
            elif n % 3 == 0:
                contenido = '{"respuesta": "El Código Civil regula esto y esto y esto y esto'  # truncado
            else:
                contenido = json.dumps({"respuesta": "La Ley 9999 de 2020 dice algo. El Código Civil regula el caso.",
                                        "palabras_clave": ["x"], "referencia_legal": "Código Civil",
                                        "marco_normativo": "Código Civil", "analisis": "a", "jurisprudencia": "b",
                                        "conclusion": "c", "abstencion": True})
            respuesta_03 = {"id": e["id"], "formato": e["formato"], **gen.abstencion(e["formato"], pasajes_03)}
            destino = ejec / "densa" / "respuestas" / f"{e['id']}.json"
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(json.dumps({"respuesta": respuesta_03, "registro": {"contenido": contenido}}))
            diag.append({"id": e["id"], "abstencion": True, "acierto_cerrada": False if e["formato"] == "multiple_choice" else None,
                         "citas": {"aciertos": 0}})
        (ejec / "densa" / "diagnostico_por_pregunta.json").write_text(json.dumps(diag))
        salida = tmp / "salida"
        resumen = v04.repuntuar_03(RAIZ, ejec, recup, ["densa"], salida)
        print(json.dumps(resumen, indent=1))
        tabla = v04.comparar(RAIZ, ejec, salida, "densa")
        print(tabla["cambio"].value_counts().to_dict())
        notas = json.loads((salida / "densa" / "notas.json").read_text())
        print({m: sum(n["motivo"] == m for n in notas) for m in {n["motivo"] for n in notas}})
        r = resumen["densa"]
        assert r["errores_validacion_oficial"] == 0 and r["tasa_sin_respaldo"] == 0, r
        sub = [json.loads(l) for l in (salida / "densa" / "submissions.jsonl").read_text().splitlines()]
        # Las cerradas sin letra conservan la respuesta (y la cabecera) del 03.
        civil = [p for s in sub if not s["abstencion"] for p in s["pasajes_recuperados"]
                 if p.get("tipo_evidencia") == "cabecera_fuente"]
        assert civil and all("CÓDIGO CIVIL" in p["texto"] for p in civil), civil[:1]
        print("OK: repuntuar_03 y comparar funcionan; cabecera del Código Civil corregida")


if __name__ == "__main__":
    main()
