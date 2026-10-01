# Corpus colombiano para AI Week 2026

Versión documental para experimentos: **corpus_eval_v1**, fijada el 29 de septiembre de 2026.
El [reporte de estructura y experimentos](docs/REPORTE_CORPUS_Y_EXPERIMENTOS.md) desarrolla las decisiones.

## 1. Inventario

Se recibieron 13.945 documentos y se incorporaron 22 fuentes oficiales verificadas.
El preparado completo contiene **13.967 documentos**, 15.166 archivos originales
y 1.060 derivados. Se verificaron existencia, tamaño y SHA-256, sin discrepancias.
Los textos suman 1.179.656.332 caracteres. Ningún original previo fue modificado.

La selección documental para evaluar contiene **13.962 documentos**; cinco
quedan excluidos por omisión declarada del texto o defecto confirmado de la fuente.
Los 130 seleccionados con restricciones específicas permanecen identificados;
no se atribuyen números de artículo, lista, página o vigencia que no estén comprobados.

- [Manifiesto seleccionado](data/releases/corpus_eval_v1/corpus_manifest.json):
  `doc_id`, título, fuente, URL, fecha de consulta, áreas heredadas, tipo, número,
  año, órgano, texto/hash, condiciones declaradas y restricciones.
- [Identidad de versión](data/releases/corpus_eval_v1/snapshot.json).
- [Fuentes excluidas](data/releases/corpus_eval_v1/excluidos.json).
- [Restricciones específicas](data/releases/corpus_eval_v1/restricciones.json).
- [Inventario preparado completo](data/processed/corpus_preparado/documentos.jsonl):
  procedencia por archivo/página cuando verificable, offsets y avisos.
- [Ydata del censo textual](reports/perfil_corpus_preparado/ydata_textos_preparados.html).

El número de artículos propios **no está certificado** por un conteo regex.
El inventario de unidades citables se añadirá al construir la segmentación por tipo;
las páginas originales sólo se asignan cuando la extracción permite comprobarlas.
Esta limitación no debe ocultarse para presentar el inventario como una entrega final completa.

## 2. Criterio de selección

Las diez áreas del enunciado orientan cobertura y evaluación; sus etiquetas en el
manifiesto son heredadas, pueden solaparse y no son una validación individual de
materia. Se preservan normas, jurisprudencia, conceptos, circulares y compilaciones.
Se amplió mediante fuentes oficiales y referencias del grafo; las semillas no
constituyen por sí solas un universo suficiente. No se añadió el banco de
preguntas, respuestas esperadas, resúmenes de soluciones ni datos generados por modelos cerrados.

Se excluyen Decreto 1497/1993, Decreto 1940/1992 y Decreto 543/1993 por omisión
declarada de articulado en la fuente; SC5191/2020 y C264/2026 por defectos confirmados
identificados por hash. Se conservan aparte para auditoría. No se excluye toda
fuente con notas editoriales ni se asume que una norma derogada carece de utilidad
para preguntas históricas.

**Fuentes puntuales** ([configs/corpus_puntuales.json](configs/corpus_puntuales.json),
`python -m legalrag.ingestion.agregar_puntuales`). El diagnóstico de la muestra mostró
que el corpus usaba valores que no traía: citaba el salario mínimo pero no tenía
ninguno de los decretos que lo fijan. Se agregan, de fuente oficial, los decretos del
salario mínimo y del auxilio de transporte de 2015 a 2026, incluidos el 1469/2025
(suspendido) y el transitorio 0159/2026, y las resoluciones de la UVT de la DIAN del
mismo periodo. Única excepción al alcance no ambiental: la Resolución 0368 de 2014 del
Ministerio de Ambiente, porque una pregunta del banco público pide leerla. Se agregan
al índice existente sin reconstruirlo, y `snapshot.json` registra la ampliación.

La justificación por áreas debe contrastarse con recall de documentos y unidades,
errores por área/formato y preguntas sin fundamento disponible. El volumen no
prueba cobertura de vigencia, excepciones, conflictos, precedentes o decisión judicial.

## 3. Método y reproducción

La ingesta verifica originales y derivados, extrae HTML/PDF/Word, conserva notas,
retira patrones de navegación comprobados y produce canónico con procedencia y
offsets. Los cambios verificables de metadatos y los respaldos están en
[docs/CORPUS_PREPARACION.md](docs/CORPUS_PREPARACION.md). No se infiere texto legal faltante.

```powershell
# Desde los originales disponibles localmente:
.venv\Scripts\python.exe -m legalrag.preprocessing.preparar_corpus --raw data/data/raw --output data/processed/corpus_preparado --workers 6
.venv\Scripts\python.exe -m legalrag.preprocessing.auditar_preparacion

# Antes de cada lote de evaluación, comprobar la versión ya fijada:
.venv\Scripts\python.exe -m legalrag.ingestion.fijar_corpus_evaluacion --version corpus_eval_v1 --verificar

# Perfiles reproducibles, en entorno separado:
.venv-profiling\Scripts\python.exe -m legalrag.preprocessing.perfil_preparado
.venv-profiling\Scripts\python.exe -m legalrag.preprocessing.perfil_estructura
```

El snapshot referencia los textos locales por rutas relativas a la raíz del
proyecto y hash; no duplica sus bytes ni los vuelve inmutables. Si un texto cambia,
la comprobación falla. Crear otra versión y reconstruir el índice correspondiente.
Los índices y resultados anteriores de 386 documentos no pertenecen a esta versión.

Las URLs, hashes y fechas permiten auditar cada descarga. **No se ha demostrado
todavía una reconstrucción completa desde URLs en un contenedor limpio**: el
comando anterior parte de los originales locales. Sitios cambiantes o fuentes
retiradas pueden impedir recuperar bytes idénticos; conservar los originales
permitidos y documentar incidencias. Tampoco se ha construido el índice nuevo.

## 4. Limitaciones, licencias y estado de entrega

La vigencia permanece `por_verificar`; las marcas de la fuente no certifican
vigencia actual. `areas` está poblado, pero `temas` está vacío. Hay 2.845 claves
prioritarias ausentes del grafo heredado, cuya identidad debe verificarse antes
de incorporarlas. OCR, numeración automática de Word y estructura normativa
presentan restricciones explícitas en los informes.

Las condiciones se conservan por fuente en `licencia_fuente` y
`redistribuir_raw`. No se aplica automáticamente una licencia propia a notas,
ediciones o materiales de terceros. El snapshot es local y **no se ha publicado**.
La entrega final aún necesita revisar redistribución, adjuntar LICENSE y avisos,
publicar corpus e índice durante 30 días y demostrar reconstrucción reproducible.

Este documento satisface la documentación del estado real del corpus; no declara
terminados los requisitos pendientes ni garantiza los cinco puntos de la rúbrica.
