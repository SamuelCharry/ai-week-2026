# Laboratorio de recuperación (2 notebooks)

1. `01_comparacion_encoders_chunking.ipynb`: extracción y aplanado, categorías, cinco estrategias de chunking, TF-IDF y encoder multilingüe E5; BGE-M3 es opcional. Compara cobertura, límites lógicos, redundancia, truncamiento real del tokenizador, Recall@k, MRR@5, nDCG@5, memoria y tiempos. Guarda la configuración elegida en `artefactos_lab/`.
2. `02_comparacion_recuperacion_faiss.ipynb`: compara FlatIP, FlatL2, IVFFlat y HNSW; BM25, RRF, filtro por categoría y cross-encoder activo sobre 10 candidatos. Mide recall de vecinos, calidad por artículo, latencia, memoria y fallos por consulta. Si no existe una ejecución completa del notebook 01, usa un arranque independiente con TF-IDF.

Desde la raíz del proyecto:

```powershell
pip install -r requirements.txt -r notebooks/requirements.txt
python notebooks/build_sample.py
jupyter lab notebooks
```

Ejecutar 01 y luego 02, de arriba abajo. La muestra `sample_ley1581.json` ya está incluida y contiene 20 artículos del fixture local y 32 consultas etiquetadas. `build_sample.py` permite reconstruirla. Las 19 consultas de desarrollo sirven para escoger; las 13 de prueba se abren solo al final. El corpus contiene una sola ley, así que las diferencias pequeñas no son concluyentes para otras áreas jurídicas. El cross-encoder está activo por defecto; la primera ejecución descarga sus pesos y puede tardar varios minutos en CPU.

Prueba rápida sin descarga neuronal:

```powershell
python notebooks/smoke_test.py
```

Esta prueba usa solo TF-IDF y crea resultados temporales en `artefactos_lab/`. Para comparar encoders, ejecutar el notebook 01 completo.
