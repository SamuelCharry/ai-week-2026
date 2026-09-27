# AI Week 2026

Sistema RAG en desarrollo para responder preguntas de derecho colombiano con citas y abstención cuando la evidencia sea insuficiente.

## Estructura

```text
data/
  raw/          # Originales y manifest.json
  processed/    # Textos limpios y fragmentos
  index/        # Índices de recuperación
  oficial/      # Material del lunes
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

## Modelos

Las opciones recomendadas en el enunciado. El límite del decoder se aplica a 8000000000 parámetros reales. Las revisiones, licencias y exclusiones están en configs/modelos.json.

```powershell
.venv\Scripts\python.exe scripts/descargar_encoders.py
.venv\Scripts\python.exe scripts/descargar_modelos.py
.venv\Scripts\python.exe scripts/comparar_encoders.py
.venv\Scripts\python.exe scripts/comparar_decoders.py
```

Los pesos se guardan fuera de Git. La comparación previa al lunes usa referencias y extractos del corpus. Mide recuperación y funcionamiento, sin asignar exactitud jurídica.

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

El notebook 02 construye el índice dos veces en carpetas distintas y compara sus hashes. Después repite las sondas sobre el índice completo.

En Windows, el ejecutor evita la suspensión automática mientras corre el notebook y libera esa condición al terminar. No cambia el plan de energía ni impide suspender el equipo manualmente.

## Evaluación

El paquete del lunes va en data/oficial, conservando sus carpetas data, schema y scripts. Las preguntas y respuestas esperadas no entran al índice.

```powershell
.venv\Scripts\python.exe scripts/evaluar_lunes.py --check-only
.venv\Scripts\python.exe -m unittest discover -s scripts/pruebas -v
```

El primer comando indica qué material oficial falta. Las pruebas locales comprueban el funcionamiento del pipeline.

## Interfaz

Desde la raíz del repositorio:

```powershell
python -m http.server 8766 --directory web --bind 127.0.0.1
```

Abrir [la interfaz](http://127.0.0.1:8766/) o [el modo demo](http://127.0.0.1:8766/?demo=1). No requiere Node ni compilación.

El backend aún no está conectado. La demo usa datos fijos para revisar los formatos y la abstención. El contrato esperado del servicio está en [web/README.md](web/README.md).
