# Informe técnico — equipo P34K

Hackathon 2026, AI Week, Universidad de los Andes. Máximo tres páginas (enunciado, §9.2, entregable 6). Se entrega como `informe/INFORME_TECNICO.pdf`.

> Esqueleto. Las secciones marcadas **Pendiente** se completan cuando la recuperación y la generación
> finales queden conectadas en `src/legalrag/agent/componentes.py` y se corra la muestra completa.

## 1. Arquitectura

```text
pregunta ─► recuperación ─► evidencia ─► decoder ─► verificación de citas ─► JSON de entrega
            (índice FAISS    (artículos    (temp. 0)   (normas citadas ⊂
             + opcional       completos y                pasajes recuperados;
             BM25/reranker)   cabeceras)                 si no, se suprime o
                                                          se declara abstención)
```

- **Corpus:** derecho colombiano de fuentes públicas oficiales. Inventario, criterio y método en `CORPUS.md`.
- **Índice:** encoder BGE-M3, `IndexFlatIP` con vectores normalizados, segmentación por artículo con ventanas
  de 320 tokens y solapamiento de 32. Se reconstruye con `python -m legalrag index --config configs/indice.json --congelar`.
- **Decoder:** servido con llama.cpp, temperatura 0 y semilla 0.
- **Recuperación y generación finales:** **Pendiente** (variante elegida, k, reranker, política de citas y abstención).

## 2. Selección de encoder y decoder

| Componente | Elegido | Alternativas medidas | Evidencia |
|---|---|---|---|
| Encoder | BAAI/bge-m3 (568 M, MIT) | multilingual-e5-large, jina-embeddings-v3, multilingual-e5-base | `notebooks/experimentos/e01_encoders.ipynb` |
| Decoder | BSC-LT/salamandra-7b-instruct, Q4_K_M (7.768 M ≤ 8.000 M) | Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct | `notebooks/experimentos/e02_decoders.ipynb` |

Revisiones, licencias y conteos de parámetros en `configs/modelos.json`. **Pendiente:** resumir en dos o tres
frases por qué ganó cada uno (métrica y margen).

## 3. Resultados sobre las 50 preguntas de muestra

Evaluador oficial (`data/oficial/scripts/evaluate.py --split sample`). Historial en `configs/evolucion.json`.

| Corrida | Documentos | Cerradas (20) | Texto libre (30) | Citas (20) | Abstención (10) | Total |
|---|---:|---:|---:|---:|---:|---:|
| 03 — corpus inicial | 282 | | | | | 11,0 |
| 04 — reglas de citas y abstención | 282 | | | | | 30,0 |
| Final | **Pendiente** | | | | | |

Tiempo por pregunta frente al presupuesto de 22 s: **Pendiente** (`*_resumen.json` del pipeline).

## 4. Limitaciones

**Pendiente.** El enunciado valora identificarlas con precisión. Puntos de partida conocidos:

- El recall de normas no demuestra que se haya recuperado el artículo necesario.
- El control de citas compara norma y artículo; no certifica la corrección jurídica ni la vigencia.
- Doctrina masiva (conceptos de superintendencias) fuera del corpus por riesgo de ruido.
- El esquema oficial rechaza `respuesta_correcta: null` en abstenciones cerradas aunque lo describe.
