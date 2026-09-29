"""Servicio HTTP de la interfaz: POST /preguntar.

Contrato en interfaz/README.md. Atiende una consulta a la vez porque el decoder local
tampoco admite concurrencia.

Uso: python -m legalrag serve [--puerto 8000]
"""
import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from legalrag.evaluation.entrega import preparar_entrada
from legalrag.agent.componentes import cargar
from legalrag.config import CONFIG, RAIZ, leer_config

FORMATOS = {"multiple_choice", "semi_open", "open_ended"}


def consultar(sistema, cuerpo):
    """Responde una consulta de la interfaz con el objeto de entrega y su traza."""
    pregunta = str(cuerpo.get("pregunta", "")).strip()
    formato = cuerpo.get("formato", "semi_open")
    if not pregunta or formato not in FORMATOS:
        raise ValueError("Se requiere `pregunta` y un `formato` válido")
    entrada = {"id": 0, "formato": formato, "pregunta": pregunta}
    if formato == "multiple_choice":
        entrada["opciones"] = cuerpo.get("opciones") or {}
    entrada = preparar_entrada(entrada)
    inicio = time.perf_counter()
    pasajes = sistema.recuperar(entrada)
    respuesta = sistema.responder(entrada, pasajes)
    traza = {"segundos": round(time.perf_counter() - inicio, 2), "k": len(respuesta.get("pasajes_recuperados", []))}
    return {"respuesta": respuesta, "traza": traza}


def manejador(sistema):
    class Manejador(BaseHTTPRequestHandler):
        def _enviar(self, estado, contenido):
            datos = json.dumps(contenido, ensure_ascii=False).encode("utf-8")
            self.send_response(estado)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(datos)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(datos)

        def do_OPTIONS(self):
            self._enviar(204, {})

        def do_GET(self):
            if self.path == "/salud":
                self._enviar(200, {"estado": "ok"})
            else:
                self._enviar(404, {"error": "Ruta desconocida"})

        def do_POST(self):
            if self.path != "/preguntar":
                self._enviar(404, {"error": "Ruta desconocida"})
                return
            try:
                cuerpo = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                self._enviar(200, consultar(sistema, cuerpo))
            except (ValueError, json.JSONDecodeError) as error:
                self._enviar(400, {"error": str(error)})
            except Exception as error:
                self._enviar(500, {"error": f"{type(error).__name__}: {error}"})

    return Manejador


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puerto", type=int, default=8000)
    ap.add_argument("--config", type=Path, default=CONFIG)
    args = ap.parse_args()

    with cargar(RAIZ, leer_config(args.config)) as sistema:
        servidor = HTTPServer(("127.0.0.1", args.puerto), manejador(sistema))
        print(f"Servicio en http://127.0.0.1:{args.puerto}/preguntar", flush=True)
        try:
            servidor.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            servidor.server_close()


if __name__ == "__main__":
    main()
