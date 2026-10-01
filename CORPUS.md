# Corpus colombiano para AI Week 2026 — bitácora

Corpus de derecho colombiano de fuentes públicas oficiales, con el que se construyó el índice de entrega
(BM25 en SQLite FTS5 + BGE-M3 en FAISS). Inventario fijado en `data/releases/corpus_eval_v1/`; corpus
procesado e índice publicados según la sección [Corpus e índice](README.md#corpus-e-índice) del README.

## 1. Inventario

<!-- cifras:inicio -->
Pendiente: se genera con `python3 src/inventario_corpus.py` sobre el corpus indexado de entrega.
<!-- cifras:fin -->

- **Por documento** (título, fuente, URL, fecha de consulta, artículos, fragmentos, áreas):
  `data/releases/corpus_eval_v1/inventario.csv`, generado por el mismo comando.
- **Manifiesto** con `doc_id`, `titulo`, `fuente`, `url`, `fecha_consulta`, `areas` (exigidos) más tipo, número,
  año, órgano, SHA-256 del texto, estado de extracción y restricciones:
  `data/releases/corpus_eval_v1/corpus_manifest.json`.
- **Originales** en `data/raw/<doc_id>/` con URL final, SHA-256 y bytes en `data/raw/manifest.json`.

Evolución del corpus:

| Ronda | Documentos | Qué se agregó |
|---|---:|---|
| Base curada | 388 | Constitución, códigos y leyes núcleo de las diez áreas |
| Rondas 1–4 | ~3.300 | Grafo normativo (normas citadas por el corpus), jurisprudencia de unificación, doctrina oficial DIAN y Función Pública |
| Ronda 5 | ~19.440 | Barrido de sentencias C y SU de la Corte Constitucional, leyes y decretos del Senado, Circular Única SIC, conceptos y circulares de superintendencias |
| Fuentes puntuales | +39 | Decretos del salario mínimo y del auxilio de transporte 2015–2026, resoluciones de la UVT 2015–2026, Resolución 0368 de 2014 (MinAmbiente), Decreto 46 de 2024 (`configs/corpus_puntuales.json`) |
| Reemplazos | 2 | Leyes 1581 de 2012 y 1712 de 2014 desde el Gestor Normativo (ver §3) |

## 2. Criterio

**Frente al banco.** El material oficial (`data/oficial/data/seed_targets.json`) lista 186 normas con el número de
ítems del banco que las usan como fundamento (Constitución 90, CGP 65, CST 37, Estatuto Tributario 35…).
Es un piso, no un techo: el examen puede pedir otras. `python3 src/cobertura_corpus.py` mide la cobertura:
antes de las correcciones de esta bitácora cubría 171 de 186 normas, 521 de 559 ítems (93,2 %). De las 15
faltantes, la Decisión Andina 486 (23 ítems) sí estaba en el corpus pero sin nombre de norma (corregido, §3);
la mayoría de las demás son errores de digitación del banco (Ley 11500 de 2007, Ley 116 de 2006, Ley 1150 de
2005, Ley 964 de 2006…), cuyas normas reales sí están, o sentencias de un ítem sin texto publicado.

**Más allá de la lista.** El grafo normativo del propio corpus: toda norma que los documentos citan por encima de
un umbral debe estar en el corpus o excluida con motivo (`python -m legalrag.ingestion.auditoria`, sección G;
exclusiones en `configs/corpus_exclusiones.json`).

**Valores que las preguntas dan por sabidos.** Las cuantías y topes se expresan en salarios mínimos o UVT: se
agregaron los decretos y resoluciones que los fijan (el corpus los citaba sin tenerlos).

**Fuera del alcance, con motivo.** Derecho ambiental e internacional (salvo normas que una pregunta del banco pide
leer, como la Resolución 0368 de 2014) y la doctrina masiva de superintendencias (~37.000 conceptos), por riesgo de
ruido en la recuperación. No se indexó el banco de preguntas, ni respuestas esperadas, ni datos generados por modelos.

## 3. Método

**Descarga** (`legalrag.ingestion.reconstruir`): por documento se prueban las fuentes oficiales candidatas (Senado,
Gestor Normativo de Función Pública, relatorías, compilaciones oficiales) y se elige la más completa (artículos
propios, huecos de numeración, parte resolutiva en sentencias), validando que el texto nombre la norma pedida.
Certificados TLS intermedios declarados en `configs/certificados/`, sin desactivar la verificación.

**Extracción** (`legalrag.preprocessing.ingesta`): HTML, PDF y Word con procedencia por archivo y página; OCR con
Tesseract (español, 300 ppp, un hilo, determinista) solo para PDF escaneados; retiro de navegación comprobada;
texto canónico con SHA-256 en `data/processed/corpus_preparado/textos/`.

**Segmentación** (`preprocessing.ingesta.segmentar_documento`, `experimentos.corpus_definitivo.chunk_rows`): normas por artículo,
sentencias por bloques; ventanas de 1.500 caracteres con 200 de solapamiento. El texto de búsqueda de cada fragmento
lleva un encabezado Markdown (`# norma — título`, `## sección`, `### Artículo N`); el pasaje entregado sigue siendo el
tramo literal del texto canónico, con sus posiciones.

**Metadatos**: tipo, número, año y órgano por documento; nombre oficial de la norma (`citations.normas.nombre_numerado`)
para el encabezado de cada pasaje, de modo que el extractor oficial de citas reconozca la norma.

**Correcciones verificadas sobre el corpus indexado** (`src/revisar_corpus.py`, `src/auditar_fallas.py`):

| Problema | Causa | Corrección |
|---|---|---|
| 270 normas con menos de la mitad de sus artículos detectados (DUR tributario 1625/2016: 713 de ~3.310) | La secuencia de artículos no admitía la numeración jerárquica de los decretos únicos (1.2.1.23.17 → 1.2.1.23.1.1), ni un primer artículo entre comillas o tras «ARTÍCULO ÚNICO» | `ingesta._siguiente`; DUR 1625/2016: 713 → 1.979; Ley 9/1989: 0 → 118 |
| Leyes 1581/2012 y 1712/2014 con 820.565 y 719.842 caracteres para ~30 artículos | La página del Senado de algunas leyes estatutarias trae pegada la sentencia de control | Reemplazo por la versión del Gestor Normativo (33.263 y 38.288 caracteres) |
| Decisión Andina 486 sin nombre de norma | El tipo `decision` no tenía nombre oficial | «Decisión 486 de la Comisión de la Comunidad Andina» |
| Sentencias de la Sala Penal leídas como la Constitución | «Sentencia CP-147 de 2014»: el extractor oficial lee «CP» como Constitución Política | Formato de la Corte para salas con sigla ambigua: «CP147-2014» |

**Reproducción**: `python -m legalrag.ingestion.reconstruir` (descarga), `python3 src/main.py --desde-raw --solo-preparar`
(texto, inventario e índice) y, para cambios de segmentación sobre textos ya preparados,
`python3 src/reconstruir_indice.py`. Fuentes puntuales y reemplazos: `python -m legalrag.ingestion.agregar_puntuales`.

## 4. Limitaciones y publicación

- La vigencia queda `por_verificar`: las notas de la fuente no certifican vigencia actual.
- Errores de digitación del banco (año o número equivocado) no pueden coincidir con ninguna norma; el sistema avisa al
  modelo cuando la ley citada existe con otro año (`citations.normalizador`).
- Normas sin artículos detectables (tratados aprobados por ley de tres artículos, decretos que adoptan planes o tablas)
  se recuperan por texto, sin fijación de artículo.
- Sitios oficiales cambiantes o retirados pueden impedir recuperar bytes idénticos desde las URL; por eso se publican
  los originales con sus hashes.
- Licencias por fuente en `licencia_fuente` y `redistribuir_raw` del manifiesto; el paquete publicado lleva `LICENSE`.
  Enlace, SHA-256 y vigencia de 30 días: sección [Corpus e índice](README.md#corpus-e-índice).
