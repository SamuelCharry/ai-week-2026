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

Corpus local de 282 documentos: 241 HTML y 41 PDF. Los originales se conservan sin modificaciones; los derivados se generan en `data/processed/`.

Los datos y los índices no se incluyen en Git. Las carpetas vacías se conservan mediante `.gitkeep`. La revisión de vigencia y la distribución del corpus están pendientes.

## Entorno

```bash
python -m pip install -r requirements.txt
jupyter lab
```

El EDA requiere los originales y `data/raw/manifest.json`. Los notebooks se organizan de `00_eda` a `06_bono_tecnicas_admitidas`.
