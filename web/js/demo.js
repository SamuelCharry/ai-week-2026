const PASAJES = [
  {
    fragmento_id: "demo_a_1",
    doc_id: "demo_documento_a",
    texto: "La ficha de este ejemplo lleva una etiqueta azul.",
    score: 0.84,
  },
  {
    fragmento_id: "demo_b_1",
    doc_id: "demo_documento_b",
    texto: "Este documento de ejemplo describe un archivo auxiliar. No indica el color de la ficha.",
    score: 0.51,
  },
].map((pasaje) => ({ ...pasaje, inicio: 0, fin: pasaje.texto.length }));

const TRAZA_PASAJES = PASAJES.map((pasaje, index) => ({
  fragmento_id: pasaje.fragmento_id,
  doc_id: pasaje.doc_id,
  rank: index + 1,
  norma: index === 0 ? "Documento A (ejemplo)" : "Documento B (ejemplo)",
  articulo: "1",
  score: pasaje.score,
}));

const CORPUS = [
  {
    norma_key: "documento a (ejemplo)",
    norma: "Documento A (ejemplo)",
    doc_id: "demo_documento_a",
  },
  {
    norma_key: "documento b (ejemplo)",
    norma: "Documento B (ejemplo)",
    doc_id: "demo_documento_b",
  },
];

const REFERENCIAS =
  "Documento A (ejemplo), artículo 1. Documento B (ejemplo), artículo 2. Documento C (ejemplo), artículo 1.";

const LIMITES =
  "El artículo 2 del Documento B (ejemplo) no aparece en los pasajes. El Documento C (ejemplo) no está en el corpus simulado.";

const TRAZA = {
  citas_respaldadas: [{ norma: "Documento A (ejemplo)", articulo: "1" }],
  citas_sin_respaldo: [
    { norma: "Documento B (ejemplo)", articulo: "2" },
    { norma: "Documento C (ejemplo)", articulo: "1" },
  ],
  pasajes: TRAZA_PASAJES,
  corpus: CORPUS,
  modelo: "Demo local, sin modelo",
  k: 2,
};

const RESPUESTAS = {
  semi_open: {
    formato: "semi_open",
    abstencion: false,
    respuesta:
      "La ficha del ejemplo lleva una etiqueta azul, según el Documento A (ejemplo), artículo 1. " + LIMITES,
    referencia_legal: REFERENCIAS,
    palabras_clave: ["ficha", "etiqueta azul", "ejemplo de interfaz"],
    pasajes_recuperados: PASAJES,
  },
  open_ended: {
    formato: "open_ended",
    abstencion: false,
    marco_normativo: "Referencias simuladas para revisar la interfaz. " + REFERENCIAS,
    analisis:
      "El Documento A (ejemplo), artículo 1, indica que la ficha lleva una etiqueta azul. " + LIMITES,
    jurisprudencia: "",
    conclusion: "Solo el color azul de la ficha tiene respaldo en los pasajes de esta demo.",
    pasajes_recuperados: PASAJES,
  },
  multiple_choice: {
    formato: "multiple_choice",
    abstencion: false,
    respuesta_correcta: "B",
    justificacion:
      "En este ejemplo, la opción B dice que la ficha lleva una etiqueta azul. Referencias simuladas: " +
      REFERENCIAS + " " + LIMITES,
    descarte_opciones: {
      A: "La opción A dice rojo. El Documento A (ejemplo), artículo 1, indica azul.",
      C: "La opción C dice verde. Los pasajes no indican ese color.",
      D: "La opción D dice que la ficha no tiene etiqueta. El pasaje indica que sí tiene.",
    },
    pasajes_recuperados: PASAJES,
  },
};

export function demoAnswer(pregunta, formato) {
  const elegido = Object.hasOwn(RESPUESTAS, formato) ? formato : "semi_open";
  const pideAbstencion = pregunta
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .split(/[^a-z0-9]+/)
    .includes("abstencion");

  if (pideAbstencion) {
    return {
      respuesta: {
        id: `demo-abstencion-${elegido}`,
        formato: elegido,
        abstencion: true,
        pasajes_recuperados: [],
      },
      traza: {
        citas_respaldadas: [],
        citas_sin_respaldo: [],
        pasajes: [],
        corpus: CORPUS,
        modelo: "Demo local, sin modelo",
        k: 2,
      },
    };
  }

  return {
    respuesta: { ...RESPUESTAS[elegido], id: `demo-${elegido}` },
    traza: TRAZA,
  };
}
