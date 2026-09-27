# AI Week 2026

Sistema RAG en desarrollo para responder preguntas de derecho colombiano con citas y abstención cuando la evidencia sea insuficiente.

## Estructura

```text
data/
  raw/          # Originales y manifest.json
  processed/    # Textos limpios y fragmentos
  index/        # Índices de recuperación
notebooks/      # Exploración y experimentación
src/            # Implementación en Python
```

## Etapas

0. EDA.
1. Ingesta y normalización.
2. Indexación vectorial.
3. Generación.
4. Citas y abstención.
5. Enriquecimiento del corpus.
6. Bono: técnicas admitidas.

## Datos

Corpus local de 282 documentos: 241 HTML y 41 PDF. Los originales se conservan sin modificaciones. Los derivados se generan en `data/processed/`.

Los datos y los índices no se incluyen en Git. Las carpetas vacías se conservan mediante `.gitkeep`. La revisión de vigencia y la distribución del corpus están pendientes.

## Entorno

```bash
python -m pip install -r requirements.txt
jupyter lab
```

El EDA requiere los originales y `data/raw/manifest.json`. Su revisión del texto usa las salidas de `01`, por lo que se repite después de la ingesta. Los notebooks se organizan de `00_eda` a `06_bono_tecnicas_admitidas`.

Los notebooks `00` y `01` incluyen tablas y gráficas de la ejecución. Los resultados y decisiones están escritos en celdas Markdown. La ingesta usa el corpus completo por defecto. El modo `muestra` permite una ejecución reducida.

Las salidas se guardan en `data/processed/<modo>/`: textos, metadatos, unidades completas, ventanas de búsqueda, reporte, comprobaciones y resumen. Las ventanas localizan la unidad. La evidencia conserva el artículo completo. Las atribuciones ambiguas y los avisos de extracción quedan señalados para revisión.

Las versiones de las bibliotecas de procesamiento corresponden a la ejecución guardada en los notebooks.

La revisión del texto detectó artículos unidos y unidades con cortes por corregir. La segmentación requiere ajustes antes de construir el índice.
