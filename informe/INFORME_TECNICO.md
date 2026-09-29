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
- **Segmentación e índice:** artículo para normas y bloques para providencias, ventanas de 1.500 caracteres con
  200 de solapamiento (1.304.984 fragmentos). BM25 en SQLite FTS5 y BGE-M3 en FAISS `IndexFlatIP`.
- **Recuperación (opción A):** BM25 top 100 + denso top 100, RRF (k = 60), reranker BGE-v2-m3 sobre 50, hasta 10 pasajes.
- **Decoder:** Qwen2.5-7B-Instruct en bfloat16, greedy (temperatura 0), contexto de 6.144 tokens.
- **Citas y abstención:** JSON validado; abstención si una cita no está en la evidencia o el JSON no es válido.

## 2. Selección de encoder y decoder

| Componente | Elegido | Alternativas medidas | Evidencia |
|---|---|---|---|
| Encoder | BAAI/bge-m3 (568 M, MIT) + reranker bge-reranker-v2-m3 | multilingual-e5-large, jina-embeddings-v3, multilingual-e5-base, Qwen3-Embedding-0.6B | `e01_encoders.ipynb`, `e06_corpus_definitivo.ipynb` (R00–R05) |
| Decoder | Qwen/Qwen2.5-7B-Instruct, bfloat16 (7.616 M ≤ 8.000 M) | Salamandra-7B-Instruct, Qwen3-4B-Instruct-2507, Qwen2.5-1.5B | `notebooks/experimentos/e02_decoders.ipynb`, `e06_corpus_definitivo.ipynb` |

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
