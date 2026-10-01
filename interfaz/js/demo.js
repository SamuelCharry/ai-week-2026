const PASAJES = [
  {
    doc_id: "ley_1581_2012",
    texto:
      "Para los propositos de la presente ley, se entiende por datos sensibles aquellos que afectan la intimidad del Titular o cuyo uso indebido puede generar su discriminacion, tales como aquellos que revelen el origen racial o etnico, la orientacion politica, las convicciones religiosas o filosoficas, la pertenencia a sindicatos, organizaciones sociales o de derechos humanos, asi como los datos relativos a la salud, a la vida sexual y los datos biometricos.",
    score: 0.8412,
    inicio: 10240,
    fin: 10658,
  },
  {
    doc_id: "decreto_1377_2013",
    texto:
      "El Tratamiento de los datos sensibles a que se refiere el articulo 5 de la Ley 1581 de 2012 esta prohibido, con excepcion de los casos expresamente senalados en el articulo 6 de la citada ley. En el Tratamiento de datos sensibles, cuando dicho Tratamiento sea posible conforme a lo establecido en el articulo 6, deberan cumplirse las siguientes obligaciones.",
    score: 0.7731,
    inicio: 4820,
    fin: 5168,
  },
  {
    doc_id: "ley_1266_2008",
    texto:
      "La administracion de datos personales de naturaleza financiera, crediticia, comercial, de servicios y la proveniente de terceros paises se sujeta a los principios de veracidad, finalidad, circulacion restringida, temporalidad, interpretacion integral de derechos constitucionales, seguridad y confidencialidad.",
    score: 0.5108,
    inicio: 2210,
    fin: 2512,
  },
];

const TRAZA_PASAJES = [
  {
    rank: 1,
    norma: "ley 1581 de 2012",
    articulo: "5",
    url: "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981",
    score: 0.8412,
  },
  {
    rank: 2,
    norma: "decreto 1377 de 2013",
    articulo: "6",
    url: "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=53646",
    score: 0.7731,
  },
  {
    rank: 3,
    norma: "ley 1266 de 2008",
    articulo: "4",
    url: "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=34488",
    score: 0.5108,
  },
];

const CORPUS = [
  {
    norma_key: "ley 1581 de 2012",
    norma: "Ley 1581 de 2012",
    doc_id: "ley_1581_2012",
    url: "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=49981",
  },
  {
    norma_key: "decreto 1377 de 2013",
    norma: "Decreto 1377 de 2013",
    doc_id: "decreto_1377_2013",
    url: "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=53646",
  },
  {
    norma_key: "ley 1266 de 2008",
    norma: "Ley 1266 de 2008",
    doc_id: "ley_1266_2008",
    url: "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=34488",
  },
  {
    norma_key: "constitucion politica de 1991",
    norma: "Constitucion Politica de 1991",
    doc_id: "cp_1991",
    url: "https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=4125",
  },
];

const TRAZA = {
  citas_respaldadas: [
    { norma: "ley 1581 de 2012", articulo: "5" },
    { norma: "decreto 1377 de 2013", articulo: "6" },
  ],
  citas_sin_respaldo: [
    { norma: "ley 1266 de 2008", articulo: "12" },
    { norma: "ley 2099 de 2029", articulo: "7" },
  ],
  pasajes: TRAZA_PASAJES,
  corpus: CORPUS,
  modelo: "qwen2.5-7b-instruct (demostracion)",
  k: 10,
  segundos: 3.4,
};

const RESPUESTAS = {
  semi_open: {
    formato: "semi_open",
    abstencion: false,
    respuesta:
      "Son datos sensibles aquellos que afectan la intimidad del titular o cuyo uso indebido puede generar su discriminacion: origen racial o etnico, orientacion politica, convicciones religiosas o filosoficas, pertenencia a sindicatos u organizaciones sociales, y los datos de salud, de la vida sexual y biometricos. Su tratamiento esta prohibido como regla general y solo procede en los casos tasados del articulo 6, entre ellos la autorizacion explicita del titular.",
    referencia_legal: "Ley 1581 de 2012, articulos 5 y 6; Decreto 1377 de 2013, articulo 6",
    palabras_clave: [
      "datos sensibles",
      "autorizacion explicita",
      "prohibicion de tratamiento",
      "habeas data",
      "discriminacion",
    ],
    pasajes_recuperados: PASAJES,
  },

  open_ended: {
    formato: "open_ended",
    abstencion: false,
    marco_normativo:
      "El regimen general de proteccion de datos personales esta en la Ley 1581 de 2012, cuyo articulo 5 define la categoria de datos sensibles y cuyo articulo 6 fija la regla de prohibicion. El Decreto 1377 de 2013 desarrolla los deberes de informacion previa y las condiciones del tratamiento.",
    analisis:
      "La prohibicion del articulo 6 no es absoluta: opera como regla, y las excepciones tasadas son de interpretacion restrictiva. La mas relevante en la practica es la autorizacion explicita del titular, que exige informarle previamente de que los datos son sensibles y de que no esta obligado a autorizar su tratamiento. La carga de acreditar que esa autorizacion se obtuvo, y en que terminos, recae sobre el responsable.",
    jurisprudencia:
      "La Corte Constitucional ha sostenido que el habeas data es un derecho autonomo y que el consentimiento del titular es la regla en el tratamiento de datos personales.",
    conclusion:
      "El tratamiento de datos sensibles solo procede con autorizacion explicita e informada del titular o en las demas excepciones del articulo 6, y su acreditacion corresponde al responsable.",
    pasajes_recuperados: PASAJES,
  },

  multiple_choice: {
    formato: "multiple_choice",
    abstencion: false,
    respuesta_correcta: "B",
    justificacion:
      "El articulo 6 de la Ley 1581 de 2012 prohibe el tratamiento de datos sensibles y admite excepciones tasadas, entre ellas la autorizacion explicita del titular. La prohibicion existe, pero no es absoluta.",
    descarte_opciones: {
      A: "Afirma una prohibicion absoluta; el articulo 6 enumera excepciones.",
      C: "Confunde el regimen de datos sensibles con el de datos de naturaleza publica.",
      D: "Atribuye la autorizacion al responsable del tratamiento y no al titular.",
    },
    pasajes_recuperados: PASAJES,
  },
};

const ABSTENCION = {
  formato: "semi_open",
  abstencion: true,
  pasajes_recuperados: [],
};

const TRAZA_ABSTENCION = {
  citas_respaldadas: [],
  citas_sin_respaldo: [],
  pasajes: [],
  corpus: CORPUS,
  modelo: "qwen2.5-7b-instruct (demostracion)",
  k: 10,
  segundos: 1.2,
};

export function demoAnswer(pregunta, formato) {
  const pide = pregunta
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");

  if (pide.includes("abstencion")) {
    return { respuesta: { ...ABSTENCION, id: "demo-abstencion" }, traza: TRAZA_ABSTENCION };
  }

  const base = RESPUESTAS[formato] || RESPUESTAS.semi_open;
  return { respuesta: { ...base, id: `demo-${formato}` }, traza: TRAZA };
}
