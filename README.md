# RAG de derecho colombiano

Equipo P34K. Preguntas jurídicas con fuentes del corpus, BGE-M3 y Qwen3-8B.

## Estado

`data/` contiene los originales y el material oficial. El ZIP `data/data_raw_oficial.zip` reúne esas dos carpetas. Los textos normalizados, el manifiesto final y el índice se crean al ejecutar el flujo desde raw.

`CORPUS.md` documenta los insumos y las decisiones. La extracción, la auditoría, la indexación y la evaluación quedan pendientes de ejecución desde estos originales. La implementación nueva aún no tiene una medición en la RTX.

## Instalación

Python 3.12 de 64 bits y RTX 4090 de 24 GB. Los originales `.doc` necesitan LibreOffice o antiword instalado. Ejecutar desde la raíz del repositorio en PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Comprobar la GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU no disponible')"
```

Debe aparecer `CUDA: True`. PyTorch usa CUDA 12.8. FAISS utiliza la RAM del computador. La primera ejecución descarga los modelos.

Si LibreOffice no está en `PATH`, indicar su ejecutable antes de preparar el corpus:

```powershell
$env:LEGALRAG_DOC_CONVERTER = 'C:\Program Files\LibreOffice\program\soffice.exe'
```

## Fase -1. Reconstruir desde raw

Descomprimir `data/data_raw_oficial.zip` en la raíz del proyecto si `data/raw/` y `data/oficial/` no están presentes. El catálogo y los originales base están en `data/raw/`; la ampliación descargada está en `data/raw/ampliacion/`. Las preguntas oficiales no se incorporan al corpus.

Ejecutar en este orden. `prepare` comprueba los hashes y extrae los originales base; `finalize` reextrae la ampliación desde sus originales. Los archivos `.txt` derivados que aparezcan después son salidas del flujo:

```powershell
.\.venv\Scripts\python.exe -m legalrag.cli prepare
.\.venv\Scripts\python.exe -m legalrag.cli finalize
.\.venv\Scripts\python.exe -m legalrag.cli profile
.\.venv\Scripts\python.exe -m legalrag.cli audit --coverage
```

Los resultados quedan en `data/processed/`, `corpus_manifest.json` y `reports/`. La cobertura distingue norma ausente, artículo ausente, no recuperada y recuperada. Usa BM25 para el diagnóstico, no mide todavía la recuperación vectorial.

Si se interrumpe `prepare`, repetirlo con `--replace`. Reutiliza únicamente textos cuya huella de parser y hashes sigan siendo válidos. `finalize` se puede repetir tras cerrar la preparación.

`profile` guarda el perfil de metadatos en `reports/perfil_corpus.html`. `audit` genera el grafo y la cobertura. Los originales no se alteran. El OCR de una resolución usa el modelo de idioma y la revisión documentada incluidos en `data/raw/`.

Después de la auditoría se puede generar el informe con los resultados de esa ejecución:

```powershell
.\.venv\Scripts\python.exe informe\generar.py
```

## Fase 0. Indexar una vez

Solo después de que `audit` termine sin errores:

```powershell
.\.venv\Scripts\python.exe -m legalrag.cli index
```

Conserva los artículos completos como unidades citables. Crea ventanas de búsqueda de 512 tokens con solapamiento de 64, añade el encabezado jurídico, calcula embeddings normalizados con BGE-M3 y guarda FAISS FlatIP y metadatos en `index/`.

El núcleo y las fuentes complementarias quedan separados. Se construye una partición a la vez para controlar la RAM.

## Fases 1 a 5. Responder y mejorar

```powershell
.\.venv\Scripts\python.exe -m legalrag.cli run --questions data/oficial/data/sample_50.jsonl --output salidas/experimento_01.jsonl --hybrid
```

El recorrido es clasificación → recuperación de 20 candidatos → reranking → 3 pasajes → Salamandra → citas y reparación del JSON. Las preguntas complejas añaden HyDE a la búsqueda. Su texto hipotético nunca se usa como evidencia. Si el núcleo no basta, se consulta el índice complementario.

Salamandra se carga en 4 bits. La generación es greedy, sin muestreo. Las cerradas incluyen descarte de opciones. Se permite abstención cuando no hay sustento. La reparación del formato es determinista.

Para retomar una ejecución interrumpida, repetir el comando con `--resume`. Deben coincidir banco, código, configuración e índice. Para otro experimento, usar otro archivo de salida.

La recuperación híbrida se activa con `--hybrid`. Para comparar sin reranker o sin HyDE, usar `--no-reranker` o `--no-hyde`. Estos cambios reutilizan el índice.

## Evaluar

```powershell
.\.venv\Scripts\python.exe -m legalrag.cli evaluate --predictions salidas/experimento_01.jsonl --gold data/oficial/data/sample_50.jsonl --official
```

La revisión local comprueba formato, citas literales, cerradas, abstención y latencia. Guarda `reports/evaluacion_local.json` y `reports/errores_por_area.png`. La gráfica local cubre las cerradas con clave conocida.

Para el juez de texto libre, instalar `data/oficial/scripts/requirements-evaluador.txt`, configurar la llave privada como indica el evaluador oficial y añadir `--ragas`. La llave solo se usa para evaluar y no se sube al repositorio. Sus dependencias pueden instalarse en otro entorno para conservar el entorno de inferencia.

El umbral del reranker parte de 0.25 y necesita calibración con la muestra. Su score no equivale a una probabilidad de respuesta correcta.

## Qué obliga a repetir la fase 0

| Cambio | Reconstruir el índice |
|---|---|
| Prompt, decoder, reranker, umbral, HyDE o búsqueda híbrida | No |
| Corpus, encoder, encabezados o fragmentación | Sí |

Si se necesita reconstruir, usar `index --replace`. Ampliar el corpus y procesarlo una vez permite fijar una versión para los experimentos. Incorporar fuentes después requiere una versión nueva.

## Interfaz y entrega

```powershell
.\.venv\Scripts\python.exe -m legalrag.cli serve
```

Abrir http://127.0.0.1:8000. Detener con `Ctrl + C` antes de lanzar otra ejecución que use los modelos.

Cuando llegue el banco final:

```powershell
.\.venv\Scripts\python.exe -m legalrag.cli run --questions data/oficial/data/test_992.jsonl --output submissions.jsonl --expected-count 992 --hybrid
```

Las preguntas de muestra y sus respuestas esperadas se usan para desarrollo. No se incorporan al corpus ni al índice. El informe técnico se actualiza con los resultados reales antes de entregar.

## Corpus e índice

Enlace público del comprimido: pendiente de publicación por el equipo.

El comprimido debe contener corpus procesado, índice serializado, manifiesto y `LICENSE`. El vínculo debe permitir descargar sin solicitar permisos y mantenerse activo 30 días después del evento. Corpus, índices y modelos no se versionan en Git.

El código usa MIT. El procesamiento del corpus conserva la licencia de `data/raw/LICENSE` y sus excepciones para contenido de terceros. Cada modelo mantiene su licencia. Se conservan fuentes, fechas, hashes y marcas de vigencia, sin certificar vigencia jurídica por una descarga.

El esquema oficial describe `null` para la abstención cerrada pero su enum solo admite A–D. Por compatibilidad se guarda A como valor técnico con `abstencion=true`, sin tratarla como respuesta elegida. Revisar esta decisión si cambia el esquema oficial.
