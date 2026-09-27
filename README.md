# AI Week 2026

Sistema RAG en desarrollo para responder preguntas de derecho colombiano con citas y abstención cuando la evidencia sea insuficiente.

## Estructura

```text
data/
  raw/          # Originales y manifest.json
  processed/    # Textos limpios y fragmentos
  index/        # Índices de recuperación
  oficial/      # Material oficial y evaluador
  colab/        # ZIP para trasladar el proyecto
  experimentos_oficiales/ # Resultados locales
notebooks/      # Exploración y experimentación
scripts/        # Auxiliares de los notebooks
  auxiliares/   # Funciones compartidas
  pruebas/      # Comprobaciones
configs/        # Modelos e indexación
reports/        # Salidas locales de los experimentos
web/            # Interfaz de consulta
```

## Etapas

0. EDA.
1. Ingesta y normalización.
   - 01.1. Comparación de encoders.
2. Indexación vectorial.
   - 02.1. Comparación de decoders.
3. Generación.
4. Citas y abstención.
5. Enriquecimiento del corpus.
6. Bono: técnicas admitidas.

## Datos

Corpus local de 282 documentos: 241 HTML y 41 PDF. Los originales se conservan sin modificaciones. Los derivados se generan en `data/processed/`.

Los datos y los índices no se incluyen en Git. Las carpetas vacías se conservan mediante `.gitkeep`. La revisión de vigencia y la distribución del corpus están pendientes.

## Entorno

Python 3.12. Para ingesta, indexación y experimentos en Windows con CUDA 12.4:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements-experimentos.txt
.venv\Scripts\python.exe -m ipykernel install --user --name ai-week --display-name "AI Week"
.venv\Scripts\python.exe -m jupyter lab
```

El kernel de los experimentos es AI Week. requirements.txt contiene solo las dependencias de ingesta y exploración.

El EDA requiere los originales y `data/raw/manifest.json`. Su revisión del texto usa las salidas de `01`, por lo que se repite después de la ingesta. Los notebooks se organizan de `00_eda` a `06_bono_tecnicas_admitidas`.

Los notebooks `00` y `01` incluyen tablas y gráficas de la ejecución. Los resultados y decisiones están escritos en celdas Markdown. La ingesta usa el corpus completo por defecto. El modo `muestra` permite una ejecución reducida.

El trabajo se sigue desde los notebooks. Las funciones compartidas están en scripts/auxiliares y los comandos de apoyo en scripts. Los notebooks los importan o llaman según la etapa.

Las salidas se guardan en `data/processed/<modo>/`: textos, metadatos, unidades completas, ventanas de búsqueda, reporte, comprobaciones y resumen. Las ventanas localizan la unidad. La evidencia conserva el artículo completo. Las atribuciones ambiguas y los avisos de extracción quedan señalados para revisión.

Las versiones de las bibliotecas de procesamiento corresponden a la ejecución guardada en los notebooks.

## Local y Colab

Los notebooks detectan el entorno. También se puede indicar `MODO = "local"` o `MODO = "colab"` en su primera celda.

Para Colab:

1. Subir `data/colab/ai-week-colab.zip` a `Mi unidad/AIWEEK` en Drive, sin descomprimir.
2. Subir el notebook a Colab y elegir GPU A100.
3. Ejecutar sus celdas en orden. Si la instalación solicita reiniciar, reiniciar la sesión y empezar desde la primera celda.

El ZIP incluye código, originales, corpus procesado y material oficial. Excluye claves, modelos y entornos. Los pesos se descargan en Colab. La preparación inicial de Salamandra requiere 45 GiB libres para convertir su revisión oficial.

Los resultados se guardan en `AIWEEK/resultados/<identificador del paquete>` mientras se ejecutan las preguntas. En local quedan en `data/experimentos_oficiales`. Para reanudar una comparación, escribir el nombre de su carpeta en `reanudar`. Si cambian la configuración o los archivos verificados, se necesita otra ejecución.


## Modelos

Las opciones recomendadas en el enunciado. El límite del decoder se aplica a 8000000000 parámetros reales. Las revisiones, licencias y exclusiones están en configs/modelos.json.

```powershell
.venv\Scripts\python.exe scripts/descargar_encoders.py
.venv\Scripts\python.exe scripts/descargar_modelos.py
.venv\Scripts\python.exe scripts/comparar_encoders.py
.venv\Scripts\python.exe scripts/comparar_decoders.py
```

Los pesos se guardan fuera de Git. Las comparaciones actuales se hacen desde los notebooks con las 50 preguntas oficiales. Las sondas anteriores quedan disponibles en los scripts como comprobación técnica.

Los encoders y decoders se comparan por separado para medir el consumo de cada uno. No conviene ejecutar ambas comparaciones al mismo tiempo en una GPU de 4 GB.

## Índice

```powershell
.venv\Scripts\python.exe scripts/indexar.py --config configs/indice.json
```

El encoder, tamaño y solapamiento están en la configuración. El script conserva hashes y rechaza la sobrescritura de un índice congelado. La reconstrucción se revisa en el notebook 02.

La configuración inicial usa multilingual-e5-large con 320 tokens, solapamiento de 32 y búsqueda híbrida. Los experimentos de selección están en los notebooks 01.1 y 02.1.

Para ejecutar un notebook y guardar sus salidas desde la terminal:

```powershell
.venv\Scripts\python.exe scripts/ejecutar_notebook.py notebooks/02_indexacion_vectorial.ipynb
```

El notebook 02 permite repetir la construcción y comparar sus hashes mediante `reconstruir_para_verificar`. Guarda los artículos completos y sus cabeceras literales para la evaluación.

En Windows, el ejecutor evita la suspensión automática mientras corre el notebook y libera esa condición al terminar. No cambia el plan de energía ni impide suspender el equipo manualmente.

## Evaluación

El paquete oficial va en data/oficial, conservando sus carpetas data, schema y scripts. Las preguntas, respuestas esperadas y fundamentos de referencia no entran al índice. El modelo recibe únicamente la entrada filtrada y los pasajes recuperados.

```powershell
.venv\Scripts\python.exe scripts/evaluar_lunes.py --check-only
.venv\Scripts\python.exe -m unittest discover -s scripts/pruebas -v
```

El primer comando indica qué material oficial falta. Las pruebas locales comprueban el funcionamiento del pipeline.

El notebook 02.1 ejecuta el evaluador sin API para medir 50 puntos. El juez RAGAS es opcional, consume la llave asignada para evaluación y añade los 30 puntos de texto libre. En Colab la llave se guarda en Secretos como `OPENROUTER_API_KEY`.

El esquema publicado describe `respuesta_correcta: null` al abstenerse, pero su enumeración lo rechaza. Se conserva `null` y se registra el conflicto en `validacion_esquema.json` hasta recibir aclaración. No se declara una entrega válida contra el esquema si quedan errores.

Las latencias de la evaluación suman recuperación previamente medida y generación. La latencia integrada de la aplicación y el puntaje final siguen pendientes. La A100 y el computador local se comparan por separado.

## Interfaz

Desde la raíz del repositorio:

```powershell
python -m http.server 8766 --directory web --bind 127.0.0.1
```

Abrir [la interfaz](http://127.0.0.1:8766/) o [el modo demo](http://127.0.0.1:8766/?demo=1). No requiere Node ni compilación.

El backend aún no está conectado. La demo usa datos fijos para revisar los formatos y la abstención. El contrato esperado del servicio está en [web/README.md](web/README.md).
