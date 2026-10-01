import { initParticles } from "./particles.js";

const API = (window.P34K_API || location.origin).replace(/\/$/, "");

const DEMO = new URLSearchParams(location.search).get("demo") === "1";

function sinTildes(texto) {
  return texto
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}

function esPrueba(pregunta) {
  return sinTildes(pregunta)
    .split(/[^a-z0-9]+/)
    .includes("prueba");
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null && text !== "") node.textContent = text;
  return node;
}

function block(label, text) {
  const wrap = el("div", "block");
  wrap.append(el("p", "block__label", label), el("p", "block__text", text));
  return wrap;
}

function plural(n, singular, plural_) {
  return `${n} ${n === 1 ? singular : plural_}`;
}

function citationLabel(cita) {
  const art = cita.articulo ? `Artículo ${cita.articulo}` : "Referencia";
  return cita.norma ? `${art} · ${cita.norma}` : `${art} · norma no identificada`;
}

function renderAnswer(slot, data) {
  slot.replaceChildren();

  if (data.abstencion === true) {
    const box = el("div", "abstention");
    box.append(
      el("p", "abstention__title", "El sistema se abstiene"),
      el(
        "p",
        "abstention__text",
        "Los pasajes recuperados no permiten sustentar una respuesta, o las " +
          "normas que se citaron no figuran en ellos. Se prefiere no afirmar.",
      ),
    );
    slot.append(box);
    return;
  }

  if (data.formato === "multiple_choice") {
    const head = el("p", "choice");
    head.append(el("span", "choice__letter", data.respuesta_correcta || "—"));
    head.append(el("span", "choice__label", "Opcion correcta"));
    slot.append(head);
    if (data.justificacion) slot.append(block("Justificacion", data.justificacion));

    const descartes = Object.entries(data.descarte_opciones || {});
    if (descartes.length) {
      const wrap = el("div", "block");
      wrap.append(el("p", "block__label", "Opciones descartadas"));
      const list = el("ul", "discards");
      for (const [letra, motivo] of descartes) {
        const li = el("li", "discards__item");
        li.append(el("b", null, `${letra})`), document.createTextNode(` ${motivo}`));
        list.append(li);
      }
      wrap.append(list);
      slot.append(wrap);
    }
    return;
  }

  if (data.formato === "semi_open") {
    slot.append(el("p", "answer__lead", data.respuesta));
    if (data.referencia_legal) {
      slot.append(block("Referencia legal", data.referencia_legal));
    }
    const claves = data.palabras_clave || [];
    if (claves.length) {
      const wrap = el("div", "block");
      wrap.append(el("p", "block__label", "Palabras clave"));
      const chips = el("ul", "chips");
      for (const k of claves) chips.append(el("li", "chip", k));
      wrap.append(chips);
      slot.append(wrap);
    }
    return;
  }

  if (data.formato === "open_ended") {
    if (data.marco_normativo) slot.append(block("Marco normativo", data.marco_normativo));
    if (data.analisis) slot.append(block("Analisis", data.analisis));

    if (data.jurisprudencia) slot.append(block("Jurisprudencia", data.jurisprudencia));
    if (data.conclusion) slot.append(block("Conclusion", data.conclusion));
    return;
  }

  slot.append(el("p", "block__text", `Formato no reconocido: ${data.formato}`));
}

const ESTADOS = {
  ok: { tag: "respaldada", nota: "Un pasaje recuperado sustenta esta cita." },
  weak: {
    tag: "sin respaldo",
    nota: "La norma esta en el corpus, pero ningun pasaje recuperado la sustenta.",
  },
  missing: {
    tag: "no existe",
    nota: "Esta norma no esta en el corpus consultado: la respuesta la cito sin fuente.",
  },
  unknown: {
    tag: "sin comprobar",
    nota: "Ningun pasaje la sustenta. Sin la lista del corpus no se puede decir si la norma existe.",
  },
};

function citeRow(cita, estado) {
  const { tag, nota } = ESTADOS[estado];
  const li = el("li", `check check--${estado}`);

  const head = el("div", "check__head");
  head.append(el("span", "check__name", citationLabel(cita)), el("span", "check__tag", tag));
  li.append(head, el("p", "check__note", nota));
  return li;
}

function renderChecks(slot, countSlot, data, traza) {
  slot.replaceChildren();

  if (!traza) {
    countSlot.textContent = "";
    slot.append(
      el(
        "p",
        "muted",
        "El backend no devolvio traza, asi que no se puede comprobar ninguna cita.",
      ),
    );
    return;
  }

  const ok = traza.citas_respaldadas || [];
  const bad = traza.citas_sin_respaldo || [];
  const total = ok.length + bad.length;
  countSlot.textContent = total ? `${ok.length} / ${total}` : "";

  if (!total) {
    slot.append(
      el(
        "p",
        "muted",
        data.abstencion
          ? "El sistema se abstuvo, asi que no cito ninguna norma."
          : "La respuesta no cita ninguna norma.",
      ),
    );
    return;
  }

  const corpus = traza.corpus || null;
  const claves = null;

  const list = el("ul", "checks");
  for (const c of ok) list.append(citeRow(c, "ok"));
  for (const c of bad) {
    let estado = "unknown";
    if (claves) {
      estado = c.norma && claves.has(String(c.norma).toLowerCase()) ? "weak" : "missing";
    }
    list.append(citeRow(c, estado));
  }
  slot.append(list);

  const inventadas = claves
    ? bad.filter((c) => !c.norma || !claves.has(String(c.norma).toLowerCase())).length
    : 0;
  const veredicto = el(
    "p",
    inventadas ? "verdict verdict--bad" : "verdict verdict--ok",
    inventadas
      ? `${inventadas} de ${plural(total, "norma citada", "normas citadas")} ` +
          `${inventadas === 1 ? "no existe" : "no existen"} en el corpus.`
      : total === 1
        ? (bad.length ? "La referencia no se pudo vincular con los pasajes." : "Referencia vinculada con la evidencia.")
        : `${ok.length} de ${total} referencias vinculadas con la evidencia.`,
  );
  slot.append(veredicto);
}

function renderCorpus(slot, countSlot, data, traza) {
  slot.replaceChildren();

  const corpus = (traza && traza.corpus) || null;
  if (!corpus || !corpus.length) {
    countSlot.textContent = "";
    slot.append(
      el(
        "p",
        "muted",
        "No hay documentos recuperados para esta consulta.",
      ),
    );
    return;
  }

  const usados = new Set((data.pasajes_recuperados || []).map((p) => p.doc_id));
  countSlot.textContent = `${usados.size} / ${corpus.length}`;

  const list = el("ul", "corpus");
  for (const d of corpus) {
    const usada = usados.has(d.doc_id);
    const li = el("li", usada ? "corpus__item corpus__item--hit" : "corpus__item");

    const nombre = el("span", "corpus__name");
    if (d.url) {
      const a = el("a", null, d.norma);
      a.href = d.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      nombre.append(a);
    } else {
      nombre.textContent = d.norma;
    }

    li.append(nombre, el("span", "corpus__tag", usada ? "en esta respuesta" : "no recuperada"));
    list.append(li);
  }
  slot.append(list);
}

function renderRun(slot, data, traza) {
  slot.replaceChildren();
  const filas = [
    ["Formato", data.abstencion ? `${data.formato} · abstencion` : data.formato],
    ["Pasajes", String((data.pasajes_recuperados || []).length)],
  ];
  if (traza && traza.k) filas.push(["k solicitado", String(traza.k)]);
  if (traza && traza.modelo) filas.push(["Modelo", String(traza.modelo)]);
  if (traza && traza.segundos) filas.push(["Tiempo", `${traza.segundos} s`]);
  filas.push(["Decodificacion", "T = 0, semilla fija"]);

  for (const [k, v] of filas) {
    const div = el("div", "run__item");
    div.append(el("dt", null, k), el("dd", null, v));
    slot.append(div);
  }
}

function renderPassages(slot, countSlot, data, traza) {
  slot.replaceChildren();

  const pasajes = data.pasajes_recuperados || [];
  const meta = (traza && traza.pasajes) || [];
  countSlot.textContent = String(pasajes.length);

  if (!pasajes.length) {
    slot.append(
      el(
        "p",
        "muted",
        data.abstencion
          ? "En una abstencion el objeto de entrega va con pasajes_recuperados vacio."
          : "No se recupero ningun pasaje para esta pregunta.",
      ),
    );
    return;
  }

  const list = el("ol", "passages");
  pasajes.forEach((p, i) => {
    const m = meta[i] || {};
    const li = el("li", "passage");

    const head = el("div", "passage__head");
    const titulo = m.norma ? `${m.norma}, articulo ${m.articulo}` : p.doc_id;
    head.append(el("p", "passage__title", titulo));

    head.append(el("span", "passage__score", `score ${p.score}`));
    li.append(head);

    li.append(el("p", "passage__text", p.texto));

    const pie = el("p", "passage__meta");
    pie.append(el("span", null, p.doc_id));
    if (typeof p.inicio === "number" && typeof p.fin === "number") {
      pie.append(el("span", null, `caracteres ${p.inicio}–${p.fin}`));
    }
    if (m.url) {
      const a = el("a", null, "fuente");
      a.href = m.url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      pie.append(a);
    }
    li.append(pie);
    list.append(li);
  });
  slot.append(list);
}

function initEvidenceView(getQuestion) {
  const view = document.querySelector("#evidence-view");
  const open = document.querySelector("#open-evidence");
  const close = document.querySelector("#close-evidence");
  if (!view || !open || !close) return;

  open.addEventListener("click", () => {
    const titulo = view.querySelector("#proof-title");

    if (titulo) titulo.textContent = getQuestion();
    view.showModal();
  });

  close.addEventListener("click", () => view.close());

  view.addEventListener("click", (event) => {
    if (!event.target.closest(".proof__inner")) view.close();
  });
}

function initQueryForm() {
  const form = document.querySelector("#query-form");
  if (!form) return;

  const input = form.querySelector("#question");
  const formato = form.querySelector("#formato");
  const submit = form.querySelector("#submit");
  const status = document.querySelector("#form-status");
  const results = document.querySelector("#results");
  const answerSlot = document.querySelector("#answer-slot");
  const answerFormat = document.querySelector("#answer-format");
  const passagesSlot = document.querySelector("#passages-slot");
  const passagesCount = document.querySelector("#passages-count");
  const checkSlot = document.querySelector("#check-slot");
  const checkCount = document.querySelector("#check-count");
  const corpusSlot = document.querySelector("#corpus-slot");
  const corpusCount = document.querySelector("#corpus-count");
  const runSlot = document.querySelector("#run-slot");

  function say(message, tone) {
    status.textContent = message;
    status.dataset.tone = tone || "";
  }

  function autogrow() {
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 260)}px`;
  }
  input.addEventListener("input", autogrow);
  autogrow();

  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(autogrow);
  }
  window.addEventListener("resize", autogrow);

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  input.addEventListener("input", () => {
    if (input.getAttribute("aria-invalid") === "true") {
      input.removeAttribute("aria-invalid");
      say("");
    }
  });

  let inFlight = false;

  const banner = document.querySelector("#demo-banner");

  let lastQuestion = "";
  initEvidenceView(() => lastQuestion);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (inFlight) return;

    const pregunta = input.value.trim();
    if (!pregunta) {
      say("Escribe una pregunta juridica antes de consultar.", "error");
      input.setAttribute("aria-invalid", "true");
      input.focus();
      return;
    }
    input.removeAttribute("aria-invalid");

    const simulada = DEMO;
    banner.hidden = !simulada;

    lastQuestion = pregunta;
    inFlight = true;
    submit.disabled = true;
    submit.textContent = "Consultando";
    document.documentElement.dataset.thinking = "true";
    say("Recuperando pasajes y generando la respuesta…");

    try {
      let payload;

      if (simulada) {
        const { demoAnswer } = await import("./demo.js");
        await new Promise((r) => setTimeout(r, 600));
        payload = demoAnswer(pregunta, formato.value);
      } else {
        const res = await fetch(`${API}/preguntar`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pregunta, formato: formato.value }),
        });
        if (!res.ok) {
          const error = await res.json();
          throw new Error(error.error || `el servidor respondió ${res.status}`);
        }
        payload = await res.json();
      }

      const data = payload.respuesta || payload;
      const traza = payload.traza || null;
      if (!data || !data.formato) throw new Error("respuesta sin el campo formato");

      answerFormat.textContent = data.abstencion ? "abstencion" : data.formato;
      renderAnswer(answerSlot, data);
      renderPassages(passagesSlot, passagesCount, data, traza);
      renderChecks(checkSlot, checkCount, data, traza);
      renderCorpus(corpusSlot, corpusCount, data, traza);
      renderRun(runSlot, data, traza);

      results.hidden = false;

      const nPas = (data.pasajes_recuperados || []).length;
      const nCit = traza ? (traza.citas_respaldadas || []).length : 0;
      say(
        data.abstencion
          ? "El sistema se abstuvo. Abajo esta el detalle."
          : `Respuesta lista: ${plural(nPas, "pasaje recuperado", "pasajes recuperados")}, ` +
            `${plural(nCit, "norma citada", "normas citadas")} con respaldo.`,
      );
    } catch (error) {

      say(
        `No se pudo consultar (${error.message}). La pregunta sigue en el ` +
          `campo. Revisa que el servicio este corriendo en ${API}.`,
        "error",
      );
    } finally {
      inFlight = false;
      submit.disabled = false;
      submit.textContent = "Consultar";
      delete document.documentElement.dataset.thinking;
    }
  });
}

function initDemoBanner() {
  const banner = document.querySelector("#demo-banner");
  if (banner) banner.hidden = !DEMO;
}

function boot() {
  initDemoBanner();
  initParticles(document.querySelector("#field-canvas"));
  initQueryForm();

  const year = document.querySelector("#year");
  if (year) year.textContent = String(new Date().getFullYear());
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
