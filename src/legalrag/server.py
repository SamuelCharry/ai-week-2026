from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
import re
from legalrag.pipeline import Pipeline
from legalrag.io import safe_path


def serve(config, port):
    pipeline = Pipeline(config)
    frontend = config.root / "interfaz"

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, code, value):
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json(200, {"estado": "listo", "modelo": config.llm_model})
                return
            relative = self.path.split("?", 1)[0].lstrip("/") or "index.html"
            try:
                path = safe_path(frontend, relative)
                data = path.read_bytes()
            except (ValueError, OSError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/preguntar":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 24000:
                    self.send_json(400, {"error": "Pregunta demasiado larga o vacía"})
                    return
                payload = json.loads(self.rfile.read(length))
                question = {"id": 0, "pregunta": str(payload.get("pregunta", "")).strip(),
                            "formato": payload.get("formato") or "semi_open"}
                if not question["pregunta"]:
                    raise ValueError("Escribe una pregunta.")
                if question["formato"] == "multiple_choice":
                    options = payload.get("opciones")
                    if not options:
                        matches = re.findall(r"(?ms)^\s*([ABCD])[).:]\s*(.*?)(?=^\s*[ABCD][).:]|\Z)",
                                             question["pregunta"])
                        options = dict(matches)
                    if set(options or {}) != set("ABCD"):
                        raise ValueError("Incluye las opciones A), B), C) y D), cada una en una línea.")
                    question["opciones"] = options
                response, diagnostic = pipeline.answer(question)
                passages = diagnostic.get("evidencia_recuperada", [])
                citations = diagnostic.get("citas", [])
                corpus = {p["doc_id"]: {"doc_id": p["doc_id"], "norma_key": p["norma"],
                          "norma": p["encabezado"], "url": p.get("url")} for p in passages}
                self.send_json(200, {"respuesta": response, "traza": {
                    "modelo": config.llm_model, "k": config.top_k_retrieval,
                    "segundos": response["latencia_ms"] / 1000,
                    "citas_respaldadas": [c for c in citations if c["verificada"]],
                    "citas_sin_respaldo": [c for c in citations if not c["verificada"]],
                    "corpus": list(corpus.values()),
                    "pasajes": [{"norma": p["doc_id"], "articulo": None,
                                 "url": corpus.get(p["doc_id"], {}).get("url")}
                                for p in response["pasajes_recuperados"]]
                }})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {"error": str(exc)})
            except Exception as exc:
                self.log_error("%s", exc)
                self.send_json(500, {"error": "No se pudo completar la consulta. Revisa la consola."})

    server = HTTPServer(("127.0.0.1", port), Handler)
    server.timeout = 300
    print(f"[interfaz] http://127.0.0.1:{port} — consultas seriales para limitar VRAM", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[interfaz] cierre solicitado", flush=True)
    finally:
        server.server_close()
        pipeline.close()
