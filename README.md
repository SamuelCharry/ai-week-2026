# AI Week 2026

Sistema RAG para preguntas de derecho colombiano. Recupera artículos del corpus, responde con Salamandra y registra las fuentes utilizadas.

## Configuración actual

| Componente | Configuración |
|---|---|
| Encoder | BAAI/bge-m3 |
| Índice | FAISS IndexFlatIP, vectores normalizados |
| Segmentación | 320 tokens, solapamiento de 32 |
| Decoder | BSC-LT/salamandra-7b-instruct, Q4_K_M |
| Generación | Temperatura 0, semilla 0, contexto de 8192 tokens |
| Evidencia | Hasta 5 artículos completos y sus cabeceras literales |

Las revisiones y licencias están en `configs/modelos.json`. Salamandra tiene 7.768.117.248 parámetros y cumple el límite de 8.000 millones.

La siguiente comparación mantiene los modelos y cambia la recuperación: densa, híbrida y densa con reranker abierto. Su configuración está en `configs/experimentos.json`. La variante definitiva queda pendiente de los resultados.

## Estructura

```text
notebooks/
  basicos/       00_eda, 01_ingesta_normalizacion, 02_indexacion
  experimentos/  e01_encoders, e02_decoders, e03, e04, e05
scripts/
  corpus/        ingesta, ampliar, fuentes_csj, constitucion_senado, cobertura, auditar, manifiesto
  indice/        recuperacion, reordenamiento, sondas, construir, verificar, comparar_encoders
  generacion/    cliente, politica, evidencia, responder, comparar_decoders
  evaluacion/    oficial, entrega, comparar, diagnostico
  entorno/       runtime, modelos, encoders, paquete, ejecutar_notebook
  experimentos/  orquestador, experimentar, v04, v05
  pruebas/
configs/
data/
web/
```

Los notebooks de `basicos/` recorren el pipeline: exploración, ingesta e índice. Los de
`experimentos/` comparan una cosa a la vez y conservan sus salidas. El trabajo nuevo se
sigue desde `experimentos/e05.ipynb`.

Cada carpeta de `scripts/` es un paquete importable como `scripts.<carpeta>.<modulo>`, y
los ejecutables se corren con `python -m`. `configs/evolucion.json` guarda el puntaje de
cada corrida evaluada y alimenta la tabla de `CORPUS.md`.

## Colab

1. Subir `data/colab/ai-week.zip` a `Mi unidad/AIWEEK`, sin descomprimir.
2. Abrir `notebooks/experimentos/e05.ipynb` en Colab y seleccionar L4 o A100.
3. Ejecutar las celdas en orden. Si la instalación pide reiniciar, reiniciar la sesión y empezar desde la primera celda.

El notebook busca el índice BGE en `AIWEEK/resultados`, donde quedaron las comparaciones anteriores. Comprueba sus archivos, modelo, segmentación y corpus antes de reutilizarlo. Si no encuentra uno compatible, informa el motivo. `permitir_construir = True` habilita la construcción de un índice nuevo.

Los resultados nuevos quedan en `AIWEEK/experimentos`. `reanudar = "auto"` continúa la última ejecución. Para crear otra, usar `reanudar = ""`. Una configuración o un entorno incompatibles requieren una ejecución separada.

Cada pregunta se guarda al terminar. El modelo convertido se conserva en `AIWEEK/modelos_cache`. La primera preparación de Salamandra requiere 45 GiB libres. Una copia verificada evita repetir la conversión en sesiones posteriores.

El paquete anterior `ai-week-colab.zip` y sus resultados se conservan. El paquete nuevo contiene el código actualizado, el corpus procesado y el material oficial, sin claves ni pesos.

## Local

Python 3.12 y GPU NVIDIA con CUDA para la tanda completa.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements-experimentos.txt
.venv\Scripts\python.exe -m ipykernel install --user --name ai-week --display-name "AI Week"
.venv\Scripts\python.exe -m jupyter lab
```

Abrir `notebooks/experimentos/e05.ipynb` con el kernel AI Week. El entorno se detecta automáticamente. Los resultados quedan en `data/experimentos_sistema`.

Para actualizar el paquete de Colab:

```powershell
.venv\Scripts\python.exe -m scripts.entorno.paquete
```

Deja `data/colab/ai-week.zip`. Los paquetes anteriores (`ai-week-colab.zip`,
`ai-week-experimentos.zip`, `ai-week-v04.zip`) se conservan pero ya no se regeneran.

## Resultados y evaluación

Cada variante guarda respuestas originales, salidas evaluadas, configuración, tiempos y motivos de abstención. La ejecución produce `resumen.csv`, `por_pregunta.csv` y una entrega `submissions.jsonl` por variante.

El evaluador oficial sin llave cubre 50 puntos: preguntas cerradas, citas y abstención. Los 30 puntos de texto libre quedan pendientes hasta activar el juez oficial. En Colab, su llave se guarda en Secretos como `OPENROUTER_API_KEY`. Solo se utiliza para evaluación y las llamadas están desactivadas por defecto.

El recall de normas no demuestra que se haya encontrado el artículo necesario. Los controles de citas tampoco certifican la corrección jurídica. Se revisan los ejemplos y las omisiones de contexto junto con las métricas. Las latencias suman recuperación previamente medida y generación.

El esquema oficial rechaza `respuesta_correcta: null` aunque lo describe para abstenciones cerradas. Esa discrepancia se registra sin alterar el material oficial. Una ejecución con errores de esquema no se considera una entrega validada.

Las respuestas esperadas se usan únicamente para evaluación. No entran al índice ni al contexto del modelo.

## Corpus e índice

El corpus actual contiene 386 documentos. Los originales están en `data/raw` y los derivados en `data/processed/corpus`. Los datos, índices, modelos y resultados quedan fuera de Git.

La revisión de vigencia, el enriquecimiento documentado y el enlace público del corpus e índice están pendientes. El paquete final debe incluir licencia, manifiesto, corpus procesado e índice serializado.

## Interfaz

```powershell
python -m http.server 8766 --directory web --bind 127.0.0.1
```

Abrir [la interfaz](http://127.0.0.1:8766/) o [la demo](http://127.0.0.1:8766/?demo=1). El contrato del servicio está en [web/README.md](web/README.md). La conexión con el backend sigue pendiente.
