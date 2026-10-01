# Machine B — Corpus expansion + Contextual Retrieval + indexado

Pipeline **standalone** que corre en la segunda máquina mientras la Machine A sigue
iterando el código del generador. Entrada: `data/`. Salida: `process/index_v2.faiss` y
`process/manifest_v2.json`, listos para empalmar en Machine A.

## Qué hace, en orden

1. **`00_verify_gaps.py`** — compara `data/corpus_manifest.json` + `data/oficial/data/seed_targets.json`
   contra lo que ya existe y produce `process/00_gaps.json`: lista exacta de normas por descargar,
   con su URL y los `items_del_banco` que cubren (prioridad).
2. **`01_download.py`** — baja los documentos faltantes de Corte Constitucional (relatoria),
   SUIN-Juriscol, Secretaría del Senado, DIAN y SIC. Guarda PDF/HTML crudo en
   `process/01_downloaded/<fuente>/<doc_id>.{pdf,html}`.
3. **`02_to_markdown.py`** — convierte cada PDF/HTML a Markdown con metadatos
   (`norma`, `articulo`, `fecha`), usando `pymupdf4llm` para PDFs y `trafilatura` para HTML.
   Salida: `process/02_markdown/<doc_id>.md`.
4. **`03_chunk.py`** — segmenta por artículo (regex de "ARTÍCULO N") con solapamiento de 64 tokens
   y máximo 512 tokens. Preserva encabezado jerárquico `[Norma — Libro — Título — Art. N]`.
   Salida: `process/03_chunks.jsonl`.
5. **`04_contextual_enrich.py`** — por cada chunk, llama a Qwen3-8B (greedy, 1 pasada, 80 tokens
   máx) para producir una línea de contexto que lo sitúa en el documento. Este contexto se
   antepone al chunk antes de embeberlo. Patrón **Contextual Retrieval** de Anthropic
   (sep 2024, reduce fallos de recuperación 49–67%).
   Salida: `process/04_chunks_enriched.jsonl`.
6. **`05_embed_index.py`** — embebe con BGE-M3 (mismo encoder que Machine A), construye
   `IndexFlatIP`, serializa a `process/05_index.faiss` + SQLite con metadatos.
7. **Manifest final** — `process/manifest_v2.json` con sha256 por archivo, revisiones
   de modelo y configuración.

## Cómo correrlo en Machine B

```bash
# 1. Montar o copiar el data/ del repo principal:
rsync -av machineA:/ruta/al/repo/data/  ./data/

# 2. Instalar deps:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Pipeline:
make all                       # ejecuta 00 → 05 secuencial
# o paso a paso:
make verify                    # 00_verify_gaps.py → process/00_gaps.json
make download                  # 01_download.py (horas; polite scraping con backoff)
make markdown                  # 02_to_markdown.py
make chunk                     # 03_chunk.py
make enrich                    # 04_contextual_enrich.py (horas de GPU)
make index                     # 05_embed_index.py (minutos)

# 4. Al terminar, empalma a Machine A:
rsync -av process/05_index.faiss process/05_metadata.sqlite process/manifest_v2.json \
     machineA:/ruta/al/repo/index/
```

## Presupuesto estimado (RTX 4090 o equivalente)

| Paso | Tiempo | Red / GPU |
|---|---|---|
| 00 verify | 10 s | — |
| 01 download | 2–4 h | red (120 docs, backoff 1 s) |
| 02 markdown | 15 min | CPU |
| 03 chunk | 5 min | CPU |
| 04 enrich (contextual) | **18–30 h** | GPU 24 GB, Qwen3-8B greedy, ~50k nuevos chunks + reproceso selectivo |
| 05 embed+index | 1–2 h | GPU + RAM |

Si no hay tiempo para `04 enrich` sobre el corpus completo, se corre solo sobre los
chunks nuevos y los top-50 "difíciles" identificados por `brechas_corpus.json`.

## Decisiones y edge cases

- **Determinismo**: `04_contextual_enrich.py` usa `temperature=0` + revision pin de Qwen3.
  Dos corridas producen el mismo contexto.
- **Reindexado parcial opcional**: `05_embed_index.py` acepta `--merge-with ../data/index`
  para unir el índice actual + los chunks nuevos sin tocar los existentes
  (sha256 por chunk evita duplicados).
- **Robustez de scraping**: cada scraper tiene retry exponencial (3 intentos), cache en disco,
  user-agent identificable y `robots.txt` respetado. Si una fuente cae, el pipeline continúa
  marcando el doc como `faltante` en el manifest.
- **Licencia**: todas las fuentes son de distribución pública (SUIN, Senado, Cortes, DIAN, SIC).
  El `process/LICENSE` queda como CC BY 4.0 como pide el enunciado.
- **Idempotencia**: cada script escribe atómicamente (write-rename). Reejecutar un paso
  reusa lo ya descargado/convertido si el sha256 coincide (ver `process/.cache/`).
- **Prohibición del banco**: ningún script lee `data/oficial/data/sample_50.jsonl` ni
  `test_992.jsonl` como fuente de documentos; solo lee `seed_targets.json` (que son
  normas, no respuestas).

## Compatibilidad con Machine A

El índice nuevo mantiene el mismo esquema de SQLite + FAISS. En Machine A basta con
apuntar `src/legalrag/config.py` → `index_dir = 'index_v2'` (o `rsync` el contenido).
No se tocan `prepare_raw.py`, `finalize.py` ni `audit`: Machine B produce directamente
el índice listo para `legalrag.cli run`.
