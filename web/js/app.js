import { initParticles } from "./particles.js";

const API = (typeof window.P34K_API === "string" ? window.P34K_API : "http://127.0.0.1:8000").replace(/\/$/, "");

const DEMO = new URLSearchParams(location.search).get("demo") === "1";

const FORMATOS = {
  semi_open: "Semiabierta",
  open_ended: "Abierta",
  multiple_choice: "Selección múltiple",
};
const TIEMPO_LIMITE = 90000;

function esObjeto(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function enlaceSeguro(value) {
  if (typeof value !== "string") return null;
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function validarRespuesta(payload, formato) {
  const invalida = () => { throw new Error("El servicio devolvió una respuesta incompleta o con un formato inválido."); };
  if (!esObjeto(payload)) invalida();
  const data = esObjeto(payload.respuesta) ? payload.respuesta : payload;
  const traza = payload.traza ?? null;
  if (!esObjeto(data) || !Object.hasOwn(FORMATOS, data.formato) || data.formato !== formato || typeof data.abstencion !== "boolean") invalida();
  const texto = (v) => typeof v === "string" && v.trim().length > 0;
  if (!data.abstencion) {
    if (formato === "semi_open" && !texto(data.respuesta)) invalida();
    if (formato === "multiple_choice" && (!texto(data.respuesta_correcta) || !texto(data.justificacion))) invalida();
    if (formato === "open_ended" && ![data.marco_normativo, data.analisis, data.conclusion].some(texto)) invalida();
  }
  for (const campo of ["referencia_legal", "marco_normativo", "analisis", "jurisprudencia", "conclusion", "justificacion"]) {
    if (data[campo] != null && typeof data[campo] !== "string") invalida();
  }
  if (data.palabras_clave != null && (!Array.isArray(data.palabras_clave) || !data.palabras_clave.every(texto))) invalida();
  if (data.descarte_opciones != null && (!esObjeto(data.descarte_opciones) || !Object.values(data.descarte_opciones).every(texto))) invalida();
  if (data.pasajes_recuperados != null && (!Array.isArray(data.pasajes_recuperados) || !data.pasajes_recuperados.every(p => esObjeto(p) && texto(p.doc_id) && texto(p.texto)))) invalida();
  if (traza !== null) {
    if (!esObjeto(traza)) invalida();
    for (const campo of ["citas_respaldadas", "citas_sin_respaldo", "corpus", "pasajes"]) {
      if (traza[campo] != null && (!Array.isArray(traza[campo]) || !traza[campo].every(esObjeto))) invalida();
    }
  }
  for (const cita of [...(traza?.citas_respaldadas || []), ...(traza?.citas_sin_respaldo || [])]) {
    if (!texto(cita.norma) || (cita.articulo != null && !["string", "number"].includes(typeof cita.articulo))) invalida();
  }
  for (const doc of traza?.corpus || []) {
    if (!texto(doc.doc_id) || !texto(doc.norma || doc.norma_key)) invalida();
  }
  return { data, traza };
}

function metadataPasaje(pasaje, posicion, metadatos) {
  const unica = (filas) => filas.length === 1 ? filas[0] : null;
  const coherentes = metadatos.filter(m =>
    (!m.doc_id || m.doc_id === pasaje.doc_id) &&
    (pasaje.articulo == null || m.articulo == null || String(m.articulo) === String(pasaje.articulo))
  );
  if (pasaje.fragmento_id) {
    const encontrada = unica(coherentes.filter(m => m.fragmento_id === pasaje.fragmento_id));
    if (encontrada) return encontrada;
  }
  const compatibles = coherentes.filter(m => !m.fragmento_id || m.fragmento_id === pasaje.fragmento_id);
  const porDocumento = compatibles.filter(m => m.doc_id === pasaje.doc_id);
  const porArticulo = pasaje.articulo != null
    ? unica(porDocumento.filter(m => String(m.articulo) === String(pasaje.articulo)))
    : null;
  return porArticulo || unica(porDocumento) || unica(compatibles.filter(m =>
    m.rank === (pasaje.rank ?? posicion + 1) && (!m.doc_id || m.doc_id === pasaje.doc_id)
  )) || {};
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
  const partes = [];
  if (cita.articulo != null && cita.articulo !== "") partes.push(`Artículo ${cita.articulo}`);
  partes.push(cita.norma || "Referencia sin identificar");
  return partes.join(" · ");
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
        "No hay evidencia suficiente para responder con respaldo.",
      ),
    );
    slot.append(box);
    return;
  }

  if (data.formato === "multiple_choice") {
    const head = el("p", "choice");
    head.append(el("span", "choice__letter", data.respuesta_correcta || "—"));
    head.append(el("span", "choice__label", "Opción correcta"));
    slot.append(head);
    if (data.justificacion) slot.append(block("Justificación", data.justificacion));

    const descartes = Object.entries(data.descarte_opciones || {});
    if (descartes.length) {
      const wrap = el("div", "block");
      wrap.append(el("p", "block__label", "Opciones descartadas"));
      const list = el("ul", "discards");
      for (const [letra, motivo] of descartes) {
        const li = el("li", "discards__item");
        li.append(el("span", null, `${letra})`), document.createTextNode(` ${motivo}`));
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
    if (data.analisis) slot.append(block("Análisis", data.analisis));

    if (data.jurisprudencia) slot.append(block("Jurisprudencia", data.jurisprudencia));
    if (data.conclusion) slot.append(block("Conclusión", data.conclusion));
    return;
  }

  slot.append(el("p", "block__text", `Formato no reconocido: ${data.formato}`));
}

const ESTADOS = {
  ok: { tag: "con respaldo", nota: "La revisión recibida marca esta cita con respaldo." },
  weak: { tag: "sin respaldo", nota: "La referencia figura en el inventario, pero no tiene respaldo confirmado." },
  missing: { tag: "fuera del inventario", nota: "La referencia no figura en el inventario recibido." },
  unknown: { tag: "sin comprobar", nota: "Falta información para comprobar esta referencia." },
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
  countSlot.textContent = "";
  if (!Array.isArray(traza?.citas_respaldadas) || !Array.isArray(traza?.citas_sin_respaldo)) {
    slot.append(el("p", "muted", "No se recibió la revisión de citas."));
    return;
  }
  const ok = traza.citas_respaldadas;
  const bad = traza.citas_sin_respaldo;
  const total = ok.length + bad.length;
  if (!total) {
    slot.append(el("p", "muted", "La revisión recibida no incluye citas."));
    return;
  }
  countSlot.textContent = `${ok.length} / ${total}`;
  const clave = (v) => typeof v === "string" ? v.trim().normalize("NFC").toLocaleLowerCase("es") : "";
  const corpus = traza.corpus;
  const inventarioCompleto = Array.isArray(corpus) && corpus.every(d => clave(d.norma_key) || clave(d.norma));
  const claves = new Set((corpus || []).flatMap(d => [clave(d.norma_key), clave(d.norma)]).filter(Boolean));
  const list = el("ul", "checks");
  for (const c of ok) list.append(citeRow(c, "ok"));
  let ausentes = 0;
  for (const c of bad) {
    let estado = "unknown";
    if (clave(c.norma) && claves.has(clave(c.norma))) estado = "weak";
    else if (clave(c.norma) && inventarioCompleto) {
      estado = "missing";
      ausentes += 1;
    }
    list.append(citeRow(c, estado));
  }
  slot.append(list);
  const mensaje = ausentes
    ? `${plural(ausentes, "referencia no figura", "referencias no figuran")} en el inventario recibido.`
    : bad.length
      ? `${plural(bad.length, "cita pendiente", "citas pendientes")} de comprobar.`
      : `${plural(ok.length, "cita marcada", "citas marcadas")} con respaldo en la revisión recibida.`;
  slot.append(el("p", bad.length ? "verdict verdict--bad" : "verdict verdict--ok", mensaje));
}

function renderCorpus(slot, countSlot, data, traza) {
  slot.replaceChildren();
  countSlot.textContent = "";
  const corpus = traza?.corpus;
  if (!Array.isArray(corpus)) {
    slot.append(el("p", "muted", "No se recibió el inventario del corpus."));
    return;
  }
  if (!corpus.length) {
    slot.append(el("p", "muted", "El inventario recibido está vacío."));
    return;
  }
  const usados = new Set((data.pasajes_recuperados || []).map(p => p.doc_id));
  const recuperados = corpus.filter(d => d.doc_id && usados.has(d.doc_id)).length;
  countSlot.textContent = `${recuperados} / ${corpus.length}`;
  const list = el("ul", "corpus");
  for (const d of corpus) {
    const usada = d.doc_id && usados.has(d.doc_id);
    const li = el("li", usada ? "corpus__item corpus__item--hit" : "corpus__item");
    const nombre = el("span", "corpus__name");
    const url = enlaceSeguro(d.url);
    if (url) {
      const a = el("a", null, d.norma || d.doc_id || "Documento sin título");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      nombre.append(a);
    } else {
      nombre.textContent = d.norma || d.doc_id || "Documento sin título";
    }
    li.append(nombre, el("span", "corpus__tag", usada ? "recuperado" : "no recuperado"));
    list.append(li);
  }
  slot.append(list);
}

function renderRun(slot, data, traza) {
  slot.replaceChildren();
  const filas = [
    ["Formato", FORMATOS[data.formato]],
    ["Pasajes", String((data.pasajes_recuperados || []).length)],
  ];
  if (Number.isFinite(traza?.k)) filas.push(["Pasajes solicitados", String(traza.k)]);
  if (typeof traza?.modelo === "string") filas.push(["Modelo", traza.modelo]);
  if (Number.isFinite(traza?.segundos)) filas.push(["Tiempo", `${traza.segundos} s`]);
  if (Number.isFinite(traza?.temperatura)) filas.push(["Temperatura", String(traza.temperatura)]);
  if (Number.isFinite(traza?.semilla)) filas.push(["Semilla", String(traza.semilla)]);
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
          ? "No hay pasajes para mostrar."
          : "No se recibieron pasajes para esta pregunta.",
      ),
    );
    return;
  }

  const list = el("ol", "passages");
  pasajes.forEach((p, i) => {
    const m = { ...metadataPasaje(p, i, meta), ...p };
    const li = el("li", "passage");

    const head = el("div", "passage__head");
    const titulo = m.norma ? citationLabel(m) : p.doc_id;
    head.append(el("p", "passage__title", titulo));

    if (Number.isFinite(p.score)) head.append(el("span", "passage__score", `Recuperación: ${p.score}`));
    li.append(head);

    li.append(el("p", "passage__text", p.texto));

    const pie = el("p", "passage__meta");
    pie.append(el("span", null, p.doc_id));
    if (typeof p.inicio === "number" && typeof p.fin === "number") {
      pie.append(el("span", null, `caracteres ${p.inicio}–${p.fin}`));
    }
    const url = enlaceSeguro(m.url);
    if (url) {
      const a = el("a", null, "fuente");
      a.href = url;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      pie.append(a);
    }
    li.append(pie);
    list.append(li);
  });
  slot.append(list);
}

function initEvidenceView(getResponse) {
  const view = document.querySelector("#evidence-view");
  const open = document.querySelector("#open-evidence");
  const close = document.querySelector("#close-evidence");
  if (!view || !open || !close) return;

  open.addEventListener("click", () => {
    const titulo = view.querySelector("#proof-title");

    const respuesta = getResponse();
    if (!respuesta) return;
    if (titulo) titulo.textContent = respuesta.pregunta;
    renderPassages(view.querySelector("#proof-body"), el("span"), respuesta.data, respuesta.traza);
    view.showModal();
  });

  close.addEventListener("click", () => view.close());
  view.addEventListener("close", () => open.focus());

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
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
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

  function actualizarFormato() {
    input.placeholder = formato.value === "multiple_choice"
      ? "Escribe la pregunta y sus opciones de respuesta."
      : "¿Qué son los datos sensibles y cómo se protegen?";
    autogrow();
  }
  formato.addEventListener("change", actualizarFormato);
  actualizarFormato();

  let inFlight = false;

  const banner = document.querySelector("#demo-banner");

  let ultimaRespuesta = null;
  initEvidenceView(() => ultimaRespuesta);

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (inFlight) return;

    const pregunta = input.value.trim();
    if (!pregunta) {
      say("Escribe una pregunta jurídica antes de consultar.", "error");
      input.setAttribute("aria-invalid", "true");
      input.focus();
      return;
    }
    input.removeAttribute("aria-invalid");

    const simulada = DEMO;
    const formatoElegido = formato.value;
    banner.hidden = !simulada;

    results.hidden = true;
    ultimaRespuesta = null;
    input.readOnly = true;
    formato.disabled = true;
    form.setAttribute("aria-busy", "true");
    inFlight = true;
    submit.disabled = true;
    submit.textContent = "Consultando";
    document.documentElement.dataset.thinking = "true";
    say(simulada ? "Cargando el ejemplo…" : "Consultando…");
    const controller = new AbortController();
    const limite = setTimeout(() => controller.abort(), TIEMPO_LIMITE);

    try {
      let payload;

      if (simulada) {
        const { demoAnswer } = await import("./demo.js");
        await new Promise((r) => setTimeout(r, 600));
        payload = demoAnswer(pregunta, formatoElegido);
      } else {
        const res = await fetch(`${API}/preguntar`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pregunta, formato: formatoElegido }),
          signal: controller.signal,
        });
        if (!res.ok) {
          const mensajes = {
            400: "El servicio no aceptó la consulta. Revisa la pregunta e inténtalo de nuevo.",
            429: "Hay demasiadas consultas. Espera un momento e inténtalo de nuevo.",
          };
          throw new Error(mensajes[res.status] || "El servicio no pudo responder. Inténtalo de nuevo más tarde.");
        }
        payload = await res.json();
      }

      const { data, traza } = validarRespuesta(payload, formatoElegido);

      answerFormat.textContent = data.abstencion ? "Abstención" : FORMATOS[data.formato];
      renderAnswer(answerSlot, data);
      renderPassages(passagesSlot, passagesCount, data, traza);
      renderChecks(checkSlot, checkCount, data, traza);
      renderCorpus(corpusSlot, corpusCount, data, traza);
      renderRun(runSlot, data, traza);

      ultimaRespuesta = { pregunta, data, traza };
      document.querySelector("#results-question").textContent = pregunta;
      results.hidden = false;

      const nPas = (data.pasajes_recuperados || []).length;
      const nCit = traza?.citas_respaldadas?.length;
      say(
        data.abstencion
          ? "El sistema se abstuvo. Abajo está el detalle."
          : `Respuesta lista: ${plural(nPas, "pasaje recuperado", "pasajes recuperados")}, ` +
            (nCit === undefined ? "sin revisión de citas." : `${plural(nCit, "cita", "citas")} con respaldo reportado.`),
      );
    } catch (error) {
      const mensaje = error.name === "AbortError"
        ? "La consulta tardó demasiado. Inténtalo de nuevo."
        : error instanceof TypeError
          ? "No se pudo conectar con el servicio. Puedes probar la demo mientras se conecta el backend."
          : error instanceof SyntaxError
            ? "El servicio devolvió una respuesta que no se pudo leer. Inténtalo de nuevo."
            : error.message;
      say(mensaje, "error");
    } finally {
      clearTimeout(limite);
      inFlight = false;
      input.readOnly = false;
      formato.disabled = false;
      form.removeAttribute("aria-busy");
      submit.disabled = false;
      submit.textContent = "Consultar";
      delete document.documentElement.dataset.thinking;
    }
  });
}

function initDemoBanner() {
  const banner = document.querySelector("#demo-banner");
  if (banner) banner.hidden = !DEMO;
  const enlace = document.querySelector(".specline a");
  if (enlace && DEMO) {
    enlace.textContent = "Salir de la demo";
    enlace.href = location.pathname;
  }
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
