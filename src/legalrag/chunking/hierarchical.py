import re
from collections import Counter
from legalrag.citations.extract import fold, norm_identity, article_number

HEADING = re.compile(
    r"(?im)^[ \t]*(?:[«“\"']\s*)?"
    r"(?:(?P<level>LIBRO|T[ÍI]TULO|CAP[ÍI]TULO)\s+(?P<label>[^\n]{1,130})"
    r"|ART[ÍI]CULO\s+(?P<article>\d+(?:\s*(?:BIS|TER|[A-Za-z](?![A-Za-z])))?"
    r"|PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO|SEXTO|S[ÉE]PTIMO|OCTAVO|NOVENO|D[ÉE]CIMO|[ÚU]NICO)"
    r"(?:\s*[°.º]\s*[.\-:–]?|\s*[.\-:–]|\s*$))"
)


def parents(doc, text):
    identity = norm_identity(doc)
    # Las citas de artículos dentro de una sentencia no son artículos propios.
    judgment = fold(doc.get("tipo", "")) in {"sentencia", "auto", "concepto", "compendio"} or identity.startswith("sentencia_")
    hierarchy = {"libro": "", "titulo": "", "capitulo": ""}
    starts = []
    if not judgment:
        for match in HEADING.finditer(text):
            if match["level"]:
                level = fold(match["level"])
                hierarchy[level] = match["label"].strip()
                if level == "libro":
                    hierarchy["titulo"] = hierarchy["capitulo"] = ""
                elif level == "titulo":
                    hierarchy["capitulo"] = ""
            else:
                # Un artículo citado dentro del texto de una reforma no inicia otra unidad propia.
                prefix = text[max(0, match.start() - 240):match.start()]
                if starts and (
                    match.group(0).lstrip().startswith(("«", "“", '"', "'"))
                    or re.search(r"(?:quedar[aá]n?\s+as[ií]|siguiente\s+texto)\s*[:.]?\s*$", prefix, re.I)
                ):
                    continue
                starts.append((match.start(), article_number(match["article"]), dict(hierarchy)))
    if not starts:
        # Agrupar párrafos contiguos evita indexar encabezados o frases aisladas.
        paragraphs = [0] + [m.start() for m in re.finditer(r"(?m)(?<=\n)\n(?=\S)", text)]
        group_start = 0
        for position, start in enumerate(paragraphs):
            end = paragraphs[position + 1] if position + 1 < len(paragraphs) else len(text)
            opening = fold(text[start:min(end, start + 160)].strip())
            section = re.match(r"(?:[ivx\d]+[.)]\s*)?(?:hechos|antecedentes|consideraciones|fundamentos|resuelve|decision)\b", opening)
            if start > group_start and (end - group_start > 2400 or section):
                starts.append((group_start, None, dict(hierarchy)))
                group_start = start
        starts.append((group_start, None, dict(hierarchy)))
    elif starts[0][0] > 0:
        starts.insert(0, (0, None, {"libro": "", "titulo": "", "capitulo": ""}))
    article_counts = Counter(article for _, article, _ in starts if article)
    for position, (start, article, hierarchy) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start == end:
            continue
        yield {
            "parent_id": f"{doc['doc_id']}:{start}:{end}", "doc_id": doc["doc_id"],
            "norma": identity, "numero_articulo": article, **hierarchy,
            "inicio": start, "fin": end,
            "atribucion_ambigua": article is not None and article_counts[article] > 1,
        }


def header(doc, parent):
    parts = [doc.get("titulo") or parent["norma"]]
    parts.extend(f"{key.capitalize()} {parent[key]}" for key in ("libro", "titulo", "capitulo") if parent[key])
    if parent["numero_articulo"]:
        parts.append(f"Art. {parent['numero_articulo']}")
    return "[" + " — ".join(parts) + "]"


def windows(doc, text, tokenizer, config):
    for parent in parents(doc, text):
        body = text[parent["inicio"]:parent["fin"]]
        tokens = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True,
                           truncation=False, verbose=False)["offset_mapping"]
        heading = header(doc, parent)
        heading_ids = tokenizer.encode(heading, add_special_tokens=False)
        heading = tokenizer.decode(heading_ids[:128], skip_special_tokens=True)
        step = config.window_tokens - config.overlap_tokens
        if step <= 0:
            raise ValueError("El solapamiento debe ser menor que la ventana.")
        for offset in range(0, len(tokens), step):
            end_token = min(offset + config.window_tokens, len(tokens))
            start = parent["inicio"] + tokens[offset][0]
            end = parent["inicio"] + tokens[end_token - 1][1]
            if end > start:
                yield {**parent, "parent_inicio": parent["inicio"], "parent_fin": parent["fin"],
                       "chunk_id": f"{doc['doc_id']}:{start}:{end}",
                       "inicio": start, "fin": end, "encabezado": heading, "texto": text[start:end]}
            if end_token == len(tokens):
                break
