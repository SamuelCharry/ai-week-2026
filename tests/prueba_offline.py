"""Prueba de humo sin GPU: arma respuestas sintéticas, las post-procesa y las
pasa por el evaluador oficial y el esquema. Verifica la tubería, no la calidad.

Uso: python -m tests.prueba_offline
"""
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

RAIZ = next(p for p in Path(__file__).resolve().parents if (p / "configs/experimentos.json").is_file())
sys.path.insert(0, str(RAIZ / "src"))

from legalrag.generation import politica as gen  # noqa: E402
from legalrag.experimentos.v04 import evaluar, evidencia, muestra  # noqa: E402


def main():
    preguntas, entradas = muestra(RAIZ)
    ev = evidencia(RAIZ)
    citas = ev.citas
    por_cuerpo = defaultdict(list)
    for doc_id in ev.documentos:
        for cuerpo in ev.identidad(doc_id):
            por_cuerpo[cuerpo].append(doc_id)
    unidades = defaultdict(list)
    objetivo = {d for ds in por_cuerpo.values() for d in ds}
    with (RAIZ / "data/processed/corpus/unidades.jsonl").open(encoding="utf-8") as archivo:
        for linea in archivo:
            u = json.loads(linea)
            if u["doc_id"] in objetivo and u.get("articulo") and len(unidades[u["doc_id"]]) < 3:
                unidades[u["doc_id"]].append({k: u[k] for k in ("doc_id", "unidad_id", "inicio", "fin", "texto", "articulo")}
                                             | {"score": 1.0})
    claves = {q["id"]: q for q in preguntas}
    respuestas, fallos = [], []
    for entrada in entradas:
        q = claves[entrada["id"]]
        ref = citas.bodies(citas.extract(q.get("legal_basis") or ""))
        docs = [d for c in ref for d in por_cuerpo.get(c, [])][:2] or ["co_constitucion_1_1991"]
        crudas = [u for d in docs for u in unidades[d]][:5]
        pasajes, _ = ev.reconstruir(crudas)
        # Salida simulada con una cita inventada que debe eliminarse.
        base = "La respuesta se funda en los pasajes. La Ley 9999 de 2020 lo confirma expresamente."
        letras = list((entrada.get("opciones") or {}).keys()) or ["A"]
        crudo = {"multiple_choice": {"justificacion": base, "respuesta_correcta": q.get("respuesta_correcta") or letras[0],
                                     "descarte_opciones": {l: "No aplica." for l in letras}},
                 "semi_open": {"respuesta": base, "palabras_clave": ["prueba"], "referencia_legal": "Ley 9999 de 2020"},
                 "open_ended": {k: base for k in ("marco_normativo", "analisis", "jurisprudencia", "conclusion")}}[entrada["formato"]]
        respuesta, registro = gen.postprocesar(entrada, crudo, pasajes, ev, {"citar_evidencia": "todas", "abstener_libre": "sin_evidencia"})
        if "9999" in json.dumps({k: v for k, v in respuesta.items() if k != "pasajes_recuperados"}, ensure_ascii=False):
            fallos.append((entrada["id"], "la cita inventada sobrevivió"))
        respuestas.append(respuesta)
    # generar() con un cliente simulado: prompt, recorte de contexto y lectura de la salida.
    class ClienteFalso:
        def aplicar_plantilla(self, conversacion):
            return "\n".join(m["content"] for m in conversacion)

        def contar(self, conversacion):
            return len(self.aplicar_plantilla(conversacion)) // 3

        def pedir(self, ruta, datos):
            assert "json_schema" in datos and datos["repeat_penalty"] > 1
            return {"content": '{"justificacion": "Por el artículo 1. Se repite se repite se re',
                    "stop_type": "limit", "tokens_predicted": 900}, b""

    config = {"contexto": 3000, "max_tokens": 900, "repeat_penalty": 1.1, "repeat_last_n": 256,
              "longitudes": True, "max_caracteres_pasaje": 1800}
    entrada = next(e for e in entradas if e["formato"] == "multiple_choice")
    crudo, usados, registro = gen.generar(ClienteFalso(), entrada, respuestas[0]["pasajes_recuperados"], ev, config)
    if not crudo or registro["tokens_prompt"] > 3000 - 900:
        fallos.append(("generar", registro.get("tokens_prompt"), crudo))
    final, _ = gen.postprocesar(entrada, crudo, usados, ev, {"citar_evidencia": "todas"})
    if final["respuesta_correcta"] not in entrada["opciones"] or final["abstencion"]:
        fallos.append(("postprocesar truncado", final["respuesta_correcta"]))
    print("Prompt de ejemplo (recortado a", registro["tokens_prompt"], "tokens aprox.):")
    print(ClienteFalso().aplicar_plantilla(gen.mensajes(entrada, usados, ev))[:1500])
    # JSON truncado
    reparado, _ = gen.reparar_json('{"justificacion": "texto que se repite se repite se re')
    if not reparado or "justificacion" not in reparado:
        fallos.append(("reparar_json", reparado))
    with tempfile.TemporaryDirectory() as carpeta:
        resumen, resultado, _ = evaluar(RAIZ, Path(carpeta), respuestas)
    print(json.dumps(resumen, indent=1))
    assert resumen["errores_validacion_oficial"] == 0, resultado["validacion"]
    assert resumen["errores_esquema"] in (0, None), resumen
    assert resumen["tasa_sin_respaldo"] == 0, resultado["citas"]
    assert not fallos, fallos
    print("OK: tubería v04 sin errores de validación ni citas sin respaldo")


if __name__ == "__main__":
    main()
