# P34K — Derecho colombiano con un modelo pequeño

Sistema RAG para la Hackathon 2026 de la AI Week (Universidad de los Andes). Responde preguntas de
derecho colombiano con un modelo abierto de menos de 8.000 millones de parámetros y un corpus
jurídico propio. Cada respuesta incluye los pasajes que la sustentan, y toda norma citada debe
aparecer en esos pasajes.

El enunciado está en [`docs/enunciado.pdf`](docs/enunciado.pdf) y el informe técnico en
[`informe/`](informe/INFORME_TECNICO.md).

## Reproducción con un solo comando

Requiere Docker, una GPU NVIDIA y NVIDIA Container Toolkit.

```bash
./reproducir.sh
```

El comando construye el contenedor y descarga el corpus procesado y el índice congelado (sección
[Corpus e índice](#corpus-e-índice)), además de los modelos. Luego responde las 50 preguntas de
muestra y ejecuta el evaluador oficial. La entrega y el reporte quedan en `data/reproduccion/`. Con
`./reproducir.sh --ragas` también se evalúa el texto libre; para eso hay que exportar antes
`OPENROUTER_API_KEY`.

Sin Docker, desde un entorno con las dependencias instaladas:

```bash
python -m legalrag run
```

## Arquitectura

El pipeline vive en `src/legalrag/`, con una carpeta por etapa:

```text
fuentes oficiales
   │  ingestion/       descarga, inventario, grafo normativo, auditoría y manifiesto
   ▼
data/raw ─► preprocessing/  extracción (HTML, PDF, OCR, Word), normalización, artículos y procedencia
   │
   ▼
data/processed/corpus ─► chunking/  ventanas de búsqueda con cabecera literal
                          encoding/  encoder abierto (BGE-M3), vectores normalizados
                          indexing/  índice FAISS congelado y verificado por hashes
   │
pregunta ─► retrieval/  denso, BM25, RRF y reranker
             citations/  identidad de normas, cabeceras y evidencia
             generation/ prompt, JSON por formato y llama.cpp con temperatura 0
             agent/      orquestación: Sistema, pipeline por lotes y servicio HTTP
             evaluation/ esquema, auditoría de citas y evaluador oficial
   │
   ├─► submissions.jsonl
   └─► interfaz/ (POST /preguntar)
```

| Paso del enunciado | Dónde está |
|---|---|
| 1. Ingesta y normalización | `ingestion/`, `preprocessing/`: cada fragmento identifica su norma y su artículo |
| 2. Indexación vectorial | `chunking/`, `encoding/`, `indexing/`: encoder abierto e índice reconstruible |
| 3. Generación | `generation/` (prompts, JSON por formato, llama.cpp) y `agent/componentes.py` |
| 4. Citas y abstención | `citations/`, `generation/politica.py`, `evaluation/entrega.py` (`auditar_citas`) |
| 5. Enriquecimiento del corpus | `CORPUS.md`, `corpus_manifest.json`, `ingestion/` |

| Componente | Configuración |
|---|---|
| Encoder | BAAI/bge-m3, 1.024 dimensiones |
| Índice | FAISS `IndexFlatIP`, vectores normalizados, ventanas de 320 tokens con solapamiento de 32 |
| Decoder | BSC-LT/salamandra-7b-instruct, Q4_K_M, 7.768.117.248 parámetros, llama.cpp |
| Generación | temperatura 0, semilla 0, contexto de 8.192 tokens |
| Recuperación y generación finales | pendientes en `configs/sistema.json` |

La configuración congelada está en `configs/sistema.json` y se lee con `legalrag.config`.
Revisiones, licencias y conteos de parámetros en `configs/modelos.json`. No se usa ningún modelo
cerrado en el sistema. El juez de OpenRouter solo interviene en la autoevaluación.

### Dónde se conecta la versión final

`src/legalrag/agent/componentes.py` define la clase `Sistema` con cuatro métodos: `abrir`,
`recuperar`, `responder` y `cerrar`. El pipeline, el servicio de la interfaz y la reproducción solo
dependen de esa clase. Su docstring lista las piezas ya disponibles: índice, BM25, reranker,
evidencia, política de generación y servidor local. También lista los requisitos del enunciado que
dependen de ella. Sus parámetros van en `recuperacion` y `generacion` de `configs/sistema.json`.

## Ejecución

```bash
python -m legalrag ingest     # descarga los originales desde las URL del inventario
python -m legalrag prepare    # texto canónico con procedencia
python -m legalrag index --config configs/indice.json --congelar
python -m legalrag answer --split sample   # 50 preguntas -> data/reproduccion/submissions_sample.jsonl
python -m legalrag answer --split test     # 992 preguntas -> submissions.jsonl
python -m legalrag evaluate --submission submissions.jsonl --split sample
python -m legalrag serve      # backend de la interfaz en http://127.0.0.1:8000
python -m http.server 8766 --directory interfaz --bind 127.0.0.1   # interfaz en http://127.0.0.1:8766
```

`python -m legalrag <comando> --help` muestra los argumentos de cada comando.

El pipeline guarda cada respuesta al terminarla y reanuda si se interrumpe. Si una pregunta falla,
la tanda continúa: la falla queda registrada y el proceso termina con código de error. Al final
valida la entrega con el esquema oficial. El archivo `*_resumen.json` informa los tiempos frente al
presupuesto de 22 s por pregunta. Las preguntas pasan por `preparar_entrada`, que descarta los
campos de respuesta y `legal_basis`, así que no llegan al sistema.

Para el sábado, copiar `test_992.jsonl` a `data/oficial/data/` antes de correr `--split test`.

## Corpus e índice

Corpus procesado, índice vectorial serializado y `LICENSE` (CC-BY-4.0):

**Enlace: pendiente.** Al publicarlo, declarar la URL y el SHA-256 del ZIP en
`configs/sistema.json` (`corpus_indice`) para que la reproducción lo descargue y lo verifique.

El índice se congela en el momento de la entrega. Se reconstruye desde el corpus con
`python -m legalrag index --config configs/indice.json --congelar`, y el corpus desde las URL
declaradas con `python -m legalrag ingest`, según el método de `CORPUS.md`.

### Estado del corpus

- **Entrega ampliada:** 13.967 documentos en `data/data/raw`. La preparación
  (`python -m legalrag prepare --raw data/data/raw --output data/processed/corpus_preparado --workers 6`)
  conserva los originales, verifica hashes, lee OCR y Word, y genera texto canónico con procedencia.
  Correcciones, comprobaciones y falencias en [docs/CORPUS_PREPARACION.md](docs/CORPUS_PREPARACION.md).
- **Versión de evaluación `corpus_eval_v1`:** 13.962 documentos seleccionados, cinco fuentes
  excluidas y restricciones por documento. Manifiesto en `data/releases/corpus_eval_v1`. Plan en
  `configs/plan_evaluacion_corpus_v1.json`; reporte en
  [docs/REPORTE_CORPUS_Y_EXPERIMENTOS.md](docs/REPORTE_CORPUS_Y_EXPERIMENTOS.md).
- Falta generar las unidades citables y el índice nuevo sobre esa versión.

## Dependencias

Python 3.12, PyTorch 2.6.0 con CUDA 12.4 y los paquetes de `requirements.txt`. El decoder corre en
llama.cpp, que `src/legalrag/generation/runtime.py` compila con CUDA en la revisión fijada en
`configs/modelos.json`. La primera conversión de Salamandra requiere 45 GiB libres.

```bash
python -m venv .venv
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt -r data/oficial/scripts/requirements-evaluador.txt
.venv/bin/pip install -e .     # instala el paquete legalrag desde src/
```

`requirements/` contiene los entornos auxiliares: `experimentos.txt` (Jupyter), `colab.txt`,
`notebook-definitivo.txt` (E06) y `profiling.txt`.

## Estructura

```text
README.md  LICENSE  requirements.txt  pyproject.toml  submissions.jsonl
CORPUS.md  corpus_manifest.json
informe/         INFORME_TECNICO.pdf (máximo 3 páginas) y su fuente
interfaz/        interfaz gráfica
src/legalrag/
  ingestion/     descarga, inventario, rondas de fuentes, grafo, exclusiones, auditoría, manifiesto
  preprocessing/ extracción, OCR, Word, normalización, artículos y preparación del corpus
  chunking/      ventanas de búsqueda
  encoding/      encoder y descarga de encoders
  indexing/      construcción y carga del índice FAISS
  retrieval/     recuperación densa, BM25, RRF y reranker
  generation/    cliente y runtime de llama.cpp, prompts y política de generación
  citations/     evidencia, identidad de normas y cabeceras
  agent/         Sistema (a completar), pipeline por lotes, servicio HTTP y reproducción
  evaluation/    esquema, evaluador oficial, sondas, diagnóstico y comparación de corridas
  experimentos/  comparaciones de encoders y decoders, experimentos 03 a 06
  config.py      configuración congelada
  cli.py         python -m legalrag {ingest|prepare|index|answer|run|evaluate|serve}
tests/
configs/         sistema.json (entrega), modelos.json, indice.json, experimentos.json, corpus_*.json
data/oficial/    material del reto: muestra, esquema y evaluador
docs/            enunciado, guía de E06, preparación del corpus y reporte de experimentos
notebooks/       basicos/ y experimentos/ (e01 a e06)
requirements/    entornos auxiliares
```

Los datos, los índices, los modelos y los resultados quedan fuera de Git. Solo se versiona
`data/oficial/`.

## Pruebas

```bash
python -m unittest discover -s tests -p 'test_*.py' -t .
```

## Experimentos en Colab

La evaluación del corpus definitivo usa [E06](notebooks/experimentos/e06_corpus_definitivo.ipynb)
y su [guía de ejecución](docs/GUIA_NOTEBOOK_DEFINITIVO.md) para RTX 4090, A100 o L4. E06 indexa
`corpus_eval_v1` (13.962 documentos) y puntúa con el paquete oficial de `data/oficial`. E01 a E05
son la línea experimental anterior, sobre el corpus de 386 documentos.

1. Subir `data/colab/ai-week.zip` (`python -m legalrag.experimentos.paquete`) a `Mi unidad/AIWEEK`, sin descomprimir.
2. Abrir el notebook en Colab y seleccionar L4 o A100.
3. Ejecutar las celdas en orden. Si la instalación pide reiniciar, reiniciar la sesión y empezar desde la primera celda.

Cada pregunta se guarda al terminar y `reanudar = "auto"` continúa la última ejecución. El modelo
convertido se conserva en `AIWEEK/modelos_cache`. En Colab, la llave del juez se guarda en Secretos
como `OPENROUTER_API_KEY`; solo se usa para evaluar y las llamadas están desactivadas por defecto.

## Entregables (enunciado, §9.2)

| N.º | Entregable | Estado |
|---|---|---|
| 2 | Repositorio con README: dependencias, arquitectura, comando único | este archivo |
| 3 | `submissions.jsonl` con las 992 respuestas | pendiente (sábado) |
| 4 | `CORPUS.md` y `corpus_manifest.json` | `CORPUS.md` en la raíz; el manifiesto se regenera con el corpus nuevo |
| 5 | Corpus e índice con licencia abierta, enlace en [Corpus e índice](#corpus-e-índice) | pendiente |
| 6 | Informe técnico de tres páginas como máximo | `informe/INFORME_TECNICO.md` (esqueleto); falta el PDF |
| 7 | Video de cinco minutos como máximo | pendiente |
| 8 | Interfaz gráfica | `interfaz/` + `python -m legalrag serve` |

## Licencia

El código se publica bajo [MIT](LICENSE). El corpus y el índice se publican aparte, bajo CC-BY-4.0.

## Integridad

Las respuestas esperadas solo se usan en la evaluación. No entran al índice ni al contexto del
modelo: `preparar_entrada` rechaza esos campos y la auditoría del corpus
(`src/legalrag/ingestion/auditoria.py`) busca fugas del banco en el índice.
