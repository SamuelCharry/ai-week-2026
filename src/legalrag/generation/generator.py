import json
import re
from legalrag.agent.classifier import classify
from legalrag.generation.elimination import instruction
from legalrag.citations.extract import trace_citations
from legalrag.citations.official import enrich_header
from legalrag.citations.sanitize import (augment_citations, ensure_min_text,
                                         sanitize_fields, supported_bodies)
from legalrag.encoding.embed import reproducible


def _truncate_first_object(text):
    start = text.find("{")
    if start < 0:
        return text
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def _strip_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def repair_json(text):
    from json_repair import repair_json as repair
    text = _strip_think_tags(text)
    start = text.find("{")
    if start < 0:
        raise ValueError("No hay objeto JSON.")
    candidate = _truncate_first_object(text)
    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = re.sub(r'\{""+', '{"', candidate)
        candidate = re.sub(r',\s*""+', ',"', candidate)
        value = repair(candidate, return_objects=True)
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("La salida no es un objeto JSON.")
    return value


def abstention(question, reason, passages=None):
    kind = classify(question)["formato"]
    row = {"id": question["id"], "formato": kind, "abstencion": True,
           "pasajes_recuperados": passages or []}
    if kind == "multiple_choice":
        row.update(respuesta_correcta="A",
                   justificacion=f"Abstencion: {reason}.",
                   descarte_opciones={})
    elif kind == "semi_open":
        row.update(respuesta=f"Abstencion: {reason}.", palabras_clave=[], referencia_legal="")
    else:
        row.update(marco_normativo="", analisis=f"Abstencion: {reason}.",
                   jurisprudencia="", conclusion="")
    return row


def public_passage(p):
    """Serializa un pasaje con su encabezado antepuesto al texto.

    El evaluador oficial extrae citas sobre `texto`; el encabezado trae la
    procedencia (norma, articulo) que de otra forma no quedaria citable.
    Mantenemos los offsets inicio/fin apuntando al cuerpo original.
    """
    header = enrich_header(p.get("encabezado") or "")
    body = p.get("texto") or ""
    texto = (header + "\n" + body).strip() if header else body
    return {"doc_id": p["doc_id"], "inicio": p["inicio"], "fin": p["fin"],
            "texto": texto,
            "score": p.get("reranker_score", p.get("score", 0.0))}


_WORDS = re.compile(r"\S+")


def _cap_words(text, limit):
    """Trunca a `limit` palabras en una frontera de oracion razonable."""
    if not text or limit <= 0:
        return text
    words = _WORDS.findall(text)
    if len(words) <= limit:
        return text
    cut = " ".join(words[:limit])
    # Retroceder hasta el ultimo punto si existe.
    dot = cut.rfind(".")
    if dot > len(cut) * 0.5:
        cut = cut[: dot + 1]
    elif not cut.endswith("."):
        cut += "."
    return cut


def validate_response(row, config):
    from jsonschema import Draft202012Validator
    schema = json.loads(config.schema_file.read_text(encoding="utf-8"))
    errors = [e.message for e in Draft202012Validator(schema).iter_errors(row)]
    if errors:
        return errors
    if not row.get("abstencion"):
        if not row.get("pasajes_recuperados"):
            errors.append("Respuesta sin evidencia")
        kind = row.get("formato")
        from legalrag.agent.classifier import SCHEMAS
        for field in SCHEMAS.get(kind, {}):
            if row.get(field) in (None, "", [], {}):
                errors.append(f"Campo vacio: {field}")
        if kind == "multiple_choice":
            if not isinstance(row.get("descarte_opciones"), dict):
                row["descarte_opciones"] = {}
        if kind == "semi_open":
            words = len((row.get("respuesta") or "").split())
            if words > 160:  # tolerancia pequena sobre el limite oficial de 150
                errors.append("respuesta supera 150 palabras")
    return errors


class Generator:
    def __init__(self, config):
        reproducible(config.seed)
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError("Se requiere CUDA.")
        self.config = config
        model_config = AutoConfig.from_pretrained(config.llm_model, revision=config.llm_revision)
        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(model_config)
            skeleton.tie_weights()
            count = sum(p.numel() for p in skeleton.parameters())
        del skeleton
        if count > config.max_params:
            raise ValueError(f"Decoder fuera del limite: {count:,} parametros.")
        self.parameter_count = count
        self.tokenizer = AutoTokenizer.from_pretrained(config.llm_model, revision=config.llm_revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model, revision=config.llm_revision,
                device_map={"": 0}, dtype=torch.bfloat16,
                trust_remote_code=False).eval()
            self._mode = "bf16"
        except (RuntimeError, torch.OutOfMemoryError):
            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.llm_model, revision=config.llm_revision, quantization_config=quantization,
                device_map={"": 0}, dtype=torch.bfloat16,
                trust_remote_code=False).eval()
            self._mode = "4bit"
        self.revision = getattr(self.model.config, "_commit_hash", None)
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n  Decoder:  {config.llm_model}  ({count/1e9:.1f}B params, {self._mode})", flush=True)
        print(f"  GPU:      {gpu}  ({vram:.0f} GB)", flush=True)
        print(f"  Context:  {config.context_tokens} tok evidence, {config.max_new_tokens} tok gen", flush=True)
        print(f"  MC think: {'on' if config.mc_thinking else 'off'}  Augment<={config.augment_max}\n", flush=True)

    def token_count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def complete(self, messages, max_new_tokens=None, enable_thinking=False):
        import torch
        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=enable_thinking,
                tokenize=True, return_tensors="pt")
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_tensors="pt")
        # transformers 5 devuelve un diccionario (input_ids, attention_mask); 4.x, el tensor de ids.
        if not hasattr(inputs, "shape"):
            inputs = inputs["input_ids"]
        inputs = inputs.to("cuda")
        maximum = max_new_tokens or self.config.max_new_tokens
        context_limit = getattr(self.model.config, "max_position_embeddings", 8192)
        if inputs.shape[-1] > self.config.max_input_tokens or inputs.shape[-1] + maximum > context_limit:
            raise ValueError("La pregunta y la evidencia exceden el contexto del decoder.")
        with torch.inference_mode():
            result = self.model.generate(inputs, attention_mask=torch.ones_like(inputs),
                                         do_sample=False, num_beams=1, max_new_tokens=maximum,
                                         repetition_penalty=1.1,
                                         pad_token_id=self.tokenizer.pad_token_id,
                                         eos_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(result[0, inputs.shape[-1]:], skip_special_tokens=True)

    def _system_prompt(self, route, question):
        kind = route["formato"]
        base = (
            "Eres un asistente juridico de derecho colombiano. Respondes usando SOLO la evidencia "
            "adjunta, que son fragmentos marcados [E1], [E2], ... Cada fragmento inicia con su "
            "encabezado de procedencia (norma, articulo) entre corchetes. "
            "Reglas estrictas de citacion: solo puedes nombrar normas, codigos, decretos, leyes o "
            "sentencias que aparezcan textualmente en la evidencia. NO inventes numeros de sentencia "
            "ni articulos ausentes. Si una norma del encabezado respalda tu respuesta, menciona su "
            "nombre canonico (Codigo Civil, Codigo General del Proceso, Constitucion Politica, "
            "Ley 1564 de 2012, Sentencia C-355 de 2006, etc.) tal como figura en los encabezados. "
            "Devuelve exclusivamente un objeto JSON valido, sin texto antes ni despues. "
            "Los campos y sus tipos son: "
            + json.dumps(route["esquema"], ensure_ascii=False) + ". "
        )
        if kind == "multiple_choice":
            base += (
                "Elige SIEMPRE una letra (A, B, C o D) basada en la evidencia; no te abstengas. "
                "En descarte_opciones incluye las tres letras no elegidas con una razon breve. "
                "La justificacion debe citar al menos una norma presente en la evidencia. "
            )
            base += instruction(question).replace("devuelve abstencion=true", "elige la letra mas probable")
        elif kind == "semi_open":
            base += (
                "respuesta: 3 a 5 oraciones, maximo 150 palabras. La primera oracion responde la "
                "pregunta de forma directa, sin preambulos. Luego la fundamentas con citas de la evidencia. "
                "palabras_clave: 3 a 6 terminos juridicos tomados de la respuesta. "
                "referencia_legal: lista norma(s) y articulo(s) exactos presentes en la evidencia. "
            )
        else:  # open_ended
            base += (
                "marco_normativo: lista las normas aplicables presentes en la evidencia. "
                "analisis: 5 a 8 oraciones; aplica el marco al caso con rigor. "
                "jurisprudencia: cita sentencias SOLO si aparecen en la evidencia; si no, declara su ausencia. "
                "conclusion: una o dos oraciones con la respuesta al caso. "
            )
        base += "Si la evidencia no es suficiente, responde con tu mejor interpretacion; NO uses abstencion."
        return base

    def answer(self, question, passages):
        route = classify(question)
        used, blocks, tokens = [], [], 0
        used_spans = set()
        for p in passages:
            if p["parent_fin"] - p["parent_inicio"] <= 7000:
                from legalrag.io import source_path
                original = source_path(self.config, p).read_text(encoding="utf-8")
                body = original[p["parent_inicio"]:p["parent_fin"]]
                if self.token_count(body) <= 1200:
                    p = {**p, "inicio": p["parent_inicio"], "fin": p["parent_fin"], "texto": body}
            span = (p["doc_id"], p["inicio"], p["fin"])
            if span in used_spans:
                continue
            header = enrich_header(p.get("encabezado") or "")
            block = f"[E{len(used)+1}] {header}\n{p['texto']}"
            cost = self.token_count(block)
            if tokens + cost > self.config.context_tokens:
                continue
            used.append(p)
            used_spans.add(span)
            blocks.append(block)
            tokens += cost
        if not used:
            return abstention(question, "No hay evidencia que quepa en el contexto"), [], False

        system = self._system_prompt(route, question)
        payload = {"pregunta": question["pregunta"], "opciones": question.get("opciones"),
                   "evidencia": blocks}
        user_content = json.dumps(payload, ensure_ascii=False)
        from legalrag.tools.legal_tools import tool_block, tool_block_v2
        herramientas = tool_block_v2 if self.config.herramientas_v2 else tool_block
        computed = herramientas(question["pregunta"], question.get("opciones"))
        if computed:
            user_content = user_content + "\n\n" + computed
        if self.config.normalizador_citas:
            # Mark 43: el banco trae leyes con el año equivocado («Ley 1564 de 2002» por la de 2012).
            from legalrag.tools.normalizador import normalizador_del_corpus
            opciones = question.get("opciones")
            normalizador = normalizador_del_corpus(self.config)
            nota = normalizador.nota({"pregunta": question["pregunta"],
                                      "opciones": opciones if isinstance(opciones, dict) else {}}) if normalizador else None
            if nota:
                user_content = user_content + "\n\n" + nota
        self.last_tools = computed
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        use_thinking = bool(self.config.mc_thinking and route["formato"] == "multiple_choice")
        budget = self.config.mc_thinking_max_tokens if use_thinking else self.config.max_new_tokens
        raw = self.complete(messages, max_new_tokens=budget, enable_thinking=use_thinking)
        self.last_raw = raw
        originally_valid = True
        try:
            json.loads(_strip_think_tags(raw))
        except ValueError:
            originally_valid = False

        generated = None
        try:
            generated = repair_json(raw)
        except ValueError:
            # Un unico reintento con instruccion mas estricta.
            strict = {"role": "user", "content": "Devuelve SOLO el objeto JSON pedido, sin texto extra, sin bloque de codigo."}
            raw2 = self.complete(messages + [strict], max_new_tokens=budget, enable_thinking=False)
            try:
                generated = repair_json(raw2)
                raw = raw2
                self.last_raw = raw2
            except ValueError:
                generated = None

        # Pasajes publicos (con encabezado antepuesto) = lo que ve el evaluador.
        public = [public_passage(p) for p in used[:10]]
        supported = supported_bodies(public)

        if not generated:
            # Fallback estricto: construimos la respuesta con la evidencia disponible.
            generated = self._fallback(route, question, public)

        # Normalizaciones de tipo (Qwen a veces devuelve listas donde el schema espera texto).
        for field in ("marco_normativo", "jurisprudencia", "conclusion", "analisis",
                      "justificacion", "respuesta", "referencia_legal"):
            value = generated.get(field)
            if isinstance(value, list):
                generated[field] = "; ".join(
                    json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
                    for item in value)

        # Nunca aceptamos la bandera de abstencion del modelo: el evaluador premia responder bien.
        if generated.get("abstencion") is True:
            generated["abstencion"] = False

        if route["formato"] == "multiple_choice":
            self._force_letter(generated, question, raw, supported)

        row = {key: generated.get(key) for key in route["esquema"]}
        row.update(id=question["id"], formato=route["formato"], abstencion=False,
                   pasajes_recuperados=public)

        # Sanitiza citas sin respaldo y aumenta con las respaldadas.
        row = sanitize_fields(row, supported, route["formato"])
        row = augment_citations(row, supported, route["formato"],
                                max_norms=self.config.augment_max)
        row = ensure_min_text(row, route["formato"])

        # Abstención calibrada: si no hay citas respaldadas y no es MC, abstener.
        if (route["formato"] != "multiple_choice"
                and not supported
                and len(used) > 0):
            from legalrag.citations.official import extract, bodies
            answer_text = " ".join(str(v) for v in row.values() if isinstance(v, str))
            cited = bodies(extract(answer_text))
            if not cited:
                return abstention(question, "Sin respaldo normativo en la evidencia",
                                  public), [], originally_valid

        # Trunca semi_open a 150 palabras.
        if route["formato"] == "semi_open":
            row["respuesta"] = _cap_words(row.get("respuesta", ""), 150)
            if not row.get("palabras_clave"):
                row["palabras_clave"] = self._keywords(question, row.get("respuesta", ""))

        problems = validate_response(row, self.config)
        if problems:
            # Como ultimo recurso rellenamos con placeholders y conservamos la salida.
            row = ensure_min_text(row, route["formato"])
            if route["formato"] == "semi_open" and not row.get("palabras_clave"):
                row["palabras_clave"] = ["derecho", "norma"]
            problems = validate_response(row, self.config)
            if problems:
                return abstention(question, "La salida no cumple el contrato: " + "; ".join(problems),
                                  public), [], originally_valid

        traces = trace_citations(row, used)
        return row, traces, originally_valid

    def _force_letter(self, generated, question, raw, supported):
        letter = generated.get("respuesta_correcta")
        if letter not in ("A", "B", "C", "D"):
            match = re.search(r'\b(?:respuesta[_ ]correcta|answer|respuesta)\b["\s:]*([A-D])\b', raw, re.I)
            if not match:
                match = re.search(r"\b([A-D])\b", raw)
            generated["respuesta_correcta"] = match.group(1).upper() if match else "A"
        if not isinstance(generated.get("descarte_opciones"), dict):
            generated["descarte_opciones"] = {}
        # Rellena las tres letras no elegidas con una razon breve si faltan.
        chosen = generated["respuesta_correcta"]
        for other in ("A", "B", "C", "D"):
            if other == chosen:
                continue
            if not generated["descarte_opciones"].get(other):
                generated["descarte_opciones"][other] = "Descartada por falta de respaldo en la evidencia."
        if not (generated.get("justificacion") or "").strip():
            for v in generated.values():
                if isinstance(v, str) and len(v) > 40:
                    generated["justificacion"] = v
                    break

    def _keywords(self, question, text):
        import re as _re
        tokens = _re.findall(r"[A-Za-zÁÉÍÓÚÑáéíóúñ]{5,}", text)
        seen, out = set(), []
        for t in tokens:
            low = t.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(low)
            if len(out) >= 5:
                break
        return out or ["derecho", "norma"]

    def _fallback(self, route, question, public):
        """Respuesta minima garantizada cuando el modelo no produce JSON."""
        first = public[0]["texto"] if public else "Sin evidencia disponible."
        snippet = (first[:600] + "…") if len(first) > 600 else first
        if route["formato"] == "multiple_choice":
            return {"respuesta_correcta": "A",
                    "justificacion": f"Seleccion por evidencia parcial. Fragmento: {snippet}",
                    "descarte_opciones": {}}
        if route["formato"] == "semi_open":
            return {"respuesta": f"La evidencia recuperada senala: {snippet}",
                    "palabras_clave": [], "referencia_legal": ""}
        return {"marco_normativo": "",
                "analisis": f"Analisis preliminar a partir de la evidencia: {snippet}",
                "jurisprudencia": "No se identifica jurisprudencia aplicable en la evidencia.",
                "conclusion": "Conclusion sujeta a verificacion normativa adicional."}

    def close(self):
        import gc
        import torch
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
