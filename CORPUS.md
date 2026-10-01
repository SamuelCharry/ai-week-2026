# Corpus colombiano

## Entrada

`data/data_raw_oficial.zip` contiene `data/raw/` y `data/oficial/`. Es el paquete de entrada, no el corpus procesado de la entrega.

`data/raw/base/` conserva los originales del inventario inicial. `data/raw/base_manifest.json` identifica 13.945 documentos y `src/legalrag/ingestion/base_additions.jsonl` registra otras 22 fuentes originales. `data/raw/ampliacion/` conserva los originales descargados, sus catálogos y las páginas HTML de normas divididas. Los textos derivados no están en el ZIP.

Cada fuente conserva URL, identificador, fecha cuando se conoce y SHA-256 del archivo descargado. `prepare` verifica los originales antes de extraer. Los `.doc` requieren LibreOffice o antiword. La resolución que necesita OCR usa el modelo español guardado en `data/raw/ocr_model/` y correcciones registradas por página. El modelo proviene de [tessdata_fast](https://github.com/tesseract-ocr/tessdata_fast/blob/main/spa.traineddata) y conserva su licencia Apache 2.0 en esa carpeta.

## Construcción

Ejecutar `prepare`, `finalize`, `profile` y `audit --coverage` en ese orden, según el README. Las salidas se crean en `data/processed/`, `reports/` y `corpus_manifest.json`. Después se ejecuta `index` para fragmentar, generar embeddings y guardar FAISS.

`prepare` produce textos a partir de los originales base. `finalize` vuelve a extraer la ampliación desde sus originales, verifica hashes, resuelve duplicados y conserva núcleo y fuentes complementarias por separado. `profile` usa metadatos y medidas de calidad. `audit` construye el grafo de citas y clasifica las referencias de la muestra en norma ausente, artículo ausente, no recuperada y recuperada.

Los conteos, exclusiones y decisiones de la nueva ejecución se leen en `reports/ampliacion_corpus.json`, `reports/perfil_resumen.json`, `reports/brechas_corpus.json` y `reports/cobertura.json`. No se fija una cifra final antes de reconstruir el corpus desde raw.

## Criterios

Se conservan los originales y el texto citable. La limpieza para búsqueda no sustituye los pasajes originales. Las sentencias no reciben como artículo propio las normas que mencionan. Las marcas de vigencia y las anotaciones editoriales permanecen como metadatos; ninguna descarga certifica vigencia jurídica.

La selección y procesamiento del corpus se publica bajo `data/raw/LICENSE`, con sus excepciones para textos y ediciones de terceros. El código tiene la licencia del repositorio. El ZIP de entrada contiene también los términos del modelo de OCR.

El corpus procesado y el índice se publicarán en otro comprimido tras ejecutar y verificar el flujo. Ese entregable incluirá `LICENSE`, `corpus_manifest.json`, textos procesados e índice vectorial. El enlace público se declara en el README.
