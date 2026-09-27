# Corpus

## Inventario

282 documentos, con 241 HTML y 41 PDF. La consulta de las fuentes se hizo el 26 de septiembre de 2026. corpus_manifest.json conserva las URLs, fechas y hashes de cada documento.

| Área | Documentos |
|---|---:|
| Constitucional | 74 |
| Administrativo | 44 |
| Penal | 40 |
| Procesal | 55 |
| Comercial y sociedades | 17 |
| Civil | 38 |
| Familia | 30 |
| Tributario | 20 |
| Laboral | 43 |
| Mercados | 25 |

Un documento puede pertenecer a varias áreas. Estas cantidades no indican cobertura de las preguntas del banco. La comparación con seed_targets.json queda pendiente de recibir el archivo oficial.

## Criterio

Se incluyen normas y providencias de fuentes públicas. La procedencia y las condiciones registradas se conservan por documento. Las notas y ediciones de terceros no se relicencian.

Las unidades con atribución ambigua quedan fuera del índice. También quedan fuera los originales auditados de SC5191 de 2020, C264 de 2026 y Ley 1551 de 2012 por defectos de texto confirmados. Los originales se conservan.

La revisión jurídica de vigencia sigue pendiente. Los metadatos separan las señales temporales de la fuente y la revisión del documento.

## Método

Los originales están en data/raw. La ingesta extrae texto, normaliza caracteres y conserva metadatos y procedencia. Las normas se separan por artículo. La jurisprudencia conserva párrafos, secciones o bloques identificados como divisiones técnicas.

Cada unidad guarda offsets sobre el texto normalizado. Las ventanas localizan la unidad y la búsqueda devuelve su texto completo. Las preguntas de evaluación y sus respuestas no entran al corpus.

La ejecución se realiza en notebooks/01_ingesta_normalizacion.ipynb. La indexación se reconstruye con scripts/indexar.py y configs/indice.json.

## Cambios

La ingesta 0.3.0 produjo 55449 unidades y 71402 ventanas. Reparó encabezados partidos o unidos, reconoció numeración jerárquica y separó versiones anteriores. También reparó 126 entidades HTML y retiró 164 cabeceras editoriales.

Pasaron 22 controles estructurales y 11 pruebas de ingesta. Persisten 5569 encabezados ambiguos y avisos de fuente. Los resultados y decisiones están en los notebooks 00 y 01.

No se reemplazó ninguna fuente en esta etapa. La publicación del corpus y su licencia de distribución quedan pendientes de revisar las condiciones de cada fuente.
