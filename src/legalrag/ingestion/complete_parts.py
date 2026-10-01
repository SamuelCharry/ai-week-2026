"""Conserva todas las páginas de normas HTML divididas por el publicador."""

import os
import re
from pathlib import Path
from urllib.parse import urljoin, urlsplit, urldefrag

from legalrag.config import CONFIG
from legalrag.ingestion.acquire import CLIENT, html_text
from legalrag.io import records, sha256, write_json, atomic_text, dump_line, now


def part_links(raw, base_url):
    name = Path(urlsplit(base_url).path).stem
    name = re.sub(r"_pr\d+$", "", name)
    pattern = re.compile(re.escape(name) + r"_pr\d+\.html?$", re.I)
    result = set()
    for link in re.findall(rb"href=[\"']([^\"']+)[\"']", raw, re.I):
        url = urldefrag(urljoin(base_url, link.decode("utf-8", errors="replace")))[0]
        if pattern.fullmatch(Path(urlsplit(url).path).name):
            result.add(url)
    return sorted(result)


def complete_parts(config):
    folder = config.root / "data/raw/ampliacion"
    inventory = folder / "documentos_completados.jsonl"
    state = folder / "estado_partes.json"
    write_json(state, {"estado": "en_curso", "pid": os.getpid(), "inicio": now()})
    previous = {d["doc_id"]: d for d in records(inventory)} if inventory.exists() else {}
    candidates, seen = [], set()
    for source in sorted(folder.glob("documentos*.jsonl")):
        if source == inventory:
            continue
        for doc in records(source):
            if doc["doc_id"] in seen or doc["doc_id"] in previous:
                continue
            seen.add(doc["doc_id"])
            raw = config.root / doc["archivo_raw"]
            if raw.suffix == ".html" and part_links(raw.read_bytes(), doc["url"]):
                candidates.append(doc)
    failures = []
    for i, original in enumerate(candidates, 1):
        doc = dict(original)
        directory = folder / "normas_multipagina" / doc["doc_id"]
        directory.mkdir(parents=True, exist_ok=True)
        main = config.root / doc["archivo_raw"]
        parts = [{"url": doc["url"], "archivo": doc["archivo_raw"], "sha256": sha256(main)}]
        pending = part_links(main.read_bytes(), doc["url"])
        visited = {doc["url"]}
        try:
            while pending:
                url = pending.pop(0)
                if url in visited:
                    continue
                visited.add(url)
                response = CLIENT.request(url)
                if "html" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError(f"La página no devolvió HTML: {url}")
                title, text = html_text(response.content)
                if len(text) < 300 or re.search(r"not found|access denied|just a moment", title, re.I):
                    raise ValueError(f"Página incompleta: {url}")
                path = directory / Path(urlsplit(url).path).name
                path.write_bytes(response.content)
                parts.append({"url": response.url, "archivo": path.relative_to(config.root).as_posix(),
                              "sha256": sha256(path)})
                pending.extend(u for u in part_links(response.content, response.url) if u not in visited)
            parts[1:] = sorted(parts[1:], key=lambda p: int(re.search(r"_pr(\d+)", p["url"])[1]))
            text = "\n\n".join(html_text((config.root / p["archivo"]).read_bytes())[1].strip() for p in parts)
            path = directory / "texto_completo.txt"
            path.write_text(text, encoding="utf-8", newline="\n")
            doc.update(partes_html=parts, metodo_extraccion="html_multipagina",
                       texto_archivo=path.relative_to(config.root).as_posix(),
                       sha256_texto=sha256(path), caracteres=len(text), fecha_completacion=now())
            previous[doc["doc_id"]] = doc
            write_json(directory / "procedencia.json", doc)
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({"doc_id": doc["doc_id"], "error": str(exc)})
        print(f"[ingestion] partes {i}/{len(candidates)} — completados: {len(previous)} — fallos: {len(failures)}", flush=True)
    with atomic_text(inventory) as stream:
        for doc in previous.values():
            dump_line(stream, doc)
    write_json(config.reports / "completacion_partes.json", {"fecha": now(), "completados": len(previous), "fallos": failures})
    write_json(state, {"estado": "terminado", "fin": now(), "pid": os.getpid()})
    return previous


if __name__ == "__main__":
    complete_parts(CONFIG)
