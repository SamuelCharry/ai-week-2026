"""Línea de comandos del sistema.

  python cli.py fetch                 descarga y limpia las normas de corpus/sources.json
  python cli.py index                 segmenta por artículo y construye el índice en indice/
  python cli.py ask "pregunta" [...]  responde una pregunta y muestra los pasajes
  python cli.py run --entrada X.jsonl --salida Y.jsonl
  python cli.py demo                  fetch + index + preguntas de demo/ (prueba completa)
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from legalrag import corpus
from legalrag.index import write_jsonl

INDEX_DIR = "indice"
SALIDAS = Path("salidas")


class Log:
    """Imprime y, si se indica, guarda lo impreso en un archivo."""
    def __init__(self, path: Path | None = None):
        self.f = None
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "w", encoding="utf-8")

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        print(line)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()


def cmd_fetch(args, log=print):
    docs = corpus.load_sources()
    manifest = []
    for d in docs:
        rec = corpus.fetch(d)
        log(f"[fetch] {d['doc_id']}: {rec['num_articulos']} artículos, consultado {rec['fecha_consulta']} ({d['url']})")
        manifest.append(rec)
    path = corpus.CORPUS / "corpus_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"[fetch] manifiesto: {path}")


def cmd_index(args, log=print):
    from legalrag.index import build_index
    docs = corpus.load_sources()
    pasajes = corpus.all_passages(docs)
    for d in docs:
        ps = [p for p in pasajes if p["doc_id"] == d["doc_id"]]
        arts = Counter(p["articulo"] for p in ps if p["parte"] == 1)
        partidos = sorted({p["articulo"] for p in ps if p["partes"] > 1})
        repetidos = sorted(a for a, n in arts.items() if n > 1)
        log(f"[index] {d['doc_id']}: {len(arts)} artículos, {len(ps)} pasajes; "
            f"divididos: {partidos or 'ninguno'}; números repetidos: {repetidos or 'ninguno'}")
    build_index(pasajes, INDEX_DIR)
    log(f"[index] {len(pasajes)} pasajes indexados en {INDEX_DIR}/")


def _answerer(args):
    from legalrag.answer import answer
    from legalrag.llm import make_ollama_llm
    from legalrag.retrieve import Retriever
    docs = corpus.load_sources()
    aliases = corpus.aliases(docs)
    nombres = {d["norma_key"]: d["norma"] for d in docs}
    retriever = Retriever.from_dir(INDEX_DIR, aliases=aliases)
    llm = make_ollama_llm(args.modelo) if args.modelo else make_ollama_llm()
    return lambda item: answer(item, retriever, llm, aliases, nombres, k=args.k), llm.model


def _show(out, traza, log):
    log(f"\n=== id {out['id']} ({out['formato']}) — {traza['segundos']} s ===")
    log(json.dumps({k: v for k, v in out.items() if k != "pasajes_recuperados"}, ensure_ascii=False, indent=2))
    log("Pasajes recuperados (los que respaldan la respuesta):")
    for p in traza["pasajes"]:
        log(f"  {p['rank']:>2}. {p['norma']}, artículo {p['articulo']} (parte {p['parte']}) "
            f"score={p['score']}  [{p['chunk_id']}]")
    if out["abstencion"] and out["pasajes_recuperados"] == []:
        log(f"  (abstención: {traza.get('motivo_abstencion')}; en la entrega pasajes_recuperados queda vacío)")
    log(f"Citas respaldadas: {traza.get('citas_respaldadas', [])}")
    log(f"Citas SIN respaldo: {traza.get('citas_sin_respaldo', [])}")


def _run_items(items, args, salida: Path, log):
    from legalrag.answer import validate
    ask, modelo = _answerer(args)
    log(f"[run] modelo {modelo}, temperatura 0, k={args.k}")
    outs, trazas = [], []
    for item in items:
        out, traza = ask(item)
        errores = validate(out)
        if errores:
            log(f"[run] id {item['id']}: formato inválido {errores}")
        _show(out, traza, log)
        outs.append(out)
        trazas.append({"id": item["id"], **traza})
    write_jsonl(salida, outs)
    write_jsonl(salida.with_name(salida.stem + "_traza.jsonl"), trazas)
    log(f"\n[run] {len(outs)} respuestas en {salida} (traza en {salida.stem}_traza.jsonl)")


def cmd_ask(args, log=print):
    item = {"id": args.id, "formato": args.formato, "pregunta": args.pregunta}
    if args.opciones:
        item["opciones"] = json.loads(args.opciones)
    ask, _ = _answerer(args)
    out, traza = ask(item)
    _show(out, traza, log)
    log("\nJSON de entrega:")
    log(json.dumps(out, ensure_ascii=False))


def cmd_run(args, log=print):
    with open(args.entrada, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    _run_items(items, args, Path(args.salida), log)


def cmd_demo(args):
    log = Log(SALIDAS / "demo_log.txt")
    cmd_fetch(args, log)
    cmd_index(args, log)
    with open("demo/preguntas_demo.jsonl", encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    _run_items(items, args, SALIDAS / "demo.jsonl", log)
    log(f"[demo] log completo en {SALIDAS / 'demo_log.txt'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    sub.add_parser("index")
    for name in ("ask", "run", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--modelo", default=None, help="modelo de Ollama (por defecto qwen2.5:7b-instruct)")
        p.add_argument("--k", type=int, default=10, help="pasajes recuperados por pregunta")
        if name == "ask":
            p.add_argument("pregunta")
            p.add_argument("--formato", default="semi_open",
                           choices=["multiple_choice", "semi_open", "open_ended"])
            p.add_argument("--opciones", help='JSON, p. ej. \'{"A": "...", "B": "..."}\'')
            p.add_argument("--id", type=int, default=0)
        if name == "run":
            p.add_argument("--entrada", required=True)
            p.add_argument("--salida", required=True)
    args = ap.parse_args(argv)
    {"fetch": cmd_fetch, "index": cmd_index, "ask": cmd_ask, "run": cmd_run, "demo": cmd_demo}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
