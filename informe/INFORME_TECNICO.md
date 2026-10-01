# Informe técnico — equipo P34K

Hackathon 2026, AI Week, Universidad de los Andes. Máximo tres páginas (enunciado §9.2, entregable 6); se entrega
como `informe/INFORME_TECNICO.pdf`. Las cifras marcadas «pendiente» se completan con la corrida final.

## 1. Arquitectura

Un ciclo agéntico de tres agentes alrededor de un decoder abierto, que sigue el ciclo que propone el enunciado
(anexo B): recuperar, comprobar, reformular con el nombre de la norma probable, redactar solo con la evidencia y
verificar cada cita.

```text
pregunta ─► recuperación híbrida ─► agente reformulador ─► herramientas ─► redactor ─► verificación ─► JSON
            BM25 + BGE-M3, RRF,      normas y artículos     calculadora     Qwen3-8B     citas ⊂ evidencia,
            norma nombrada,          aplicables → nuevos    SMMLV/UVT ·     temp. 0      esquema oficial,
            reranker, 10 pasajes     candidatos al reranker normalizador                 reintento de JSON
```

- **Recuperación.** BM25 (SQLite FTS5) y BGE-M3 (FAISS) top 100 cada uno, fusionados con RRF (k = 60). Si la pregunta
  nombra una norma del corpus, se busca dentro de ella y el artículo nombrado se fija primero. Reranker BGE-v2-m3
  sobre 50 candidatos; hasta 10 pasajes literales, cada uno encabezado con el nombre oficial de su norma para que el
  extractor de citas del evaluador la reconozca.
- **Agente reformulador** (Qwen3-8B, texto libre). Lista las normas y artículos que regulan el caso; se buscan por
  nombre y sus artículos entran como candidatos al reranker, que decide. Motivo: en las preguntas de caso («ampliar la
  avenida 68») el texto no se parece al de la norma («acciones populares»); la auditoría de fallas
  (`src/auditar_fallas.py`) mostró que la norma correcta no llegaba a los candidatos, aunque el reranker la habría
  aceptado.
- **Herramientas deterministas.** La *calculadora* convierte montos de la pregunta a SMMLV y UVT con los decretos y
  resoluciones del corpus (leídos en su artículo 1). El *normalizador* avisa cuando la pregunta u opción cita una ley
  con el año equivocado (el banco trae «Ley 1564 de 2002» por la de 2012).
- **Redactor.** Qwen3-8B, greedy (temperatura 0), contexto de 6.144 tokens. En cerradas, la letra se elige por la
  probabilidad del siguiente token, calculada en float32 (en bf16 dos letras empataban y decidía el orden alfabético),
  y luego se redacta la justificación con esa letra fijada.
- **Verificación.** JSON reparado; toda cita que no figura en los 10 pasajes se elimina (no la respuesta); se agregan
  las normas de la evidencia que el evaluador reconoce; validación contra el esquema oficial; un reintento con otra
  penalización de repetición si el JSON salió inválido. Abstención solo sin evidencia.

## 2. Selección de encoder y decoder

| Componente | Elegido | Motivo |
|---|---|---|
| Encoder | BAAI/bge-m3 (568 M, MIT) | Mejor recuperación en E06 frente a multilingual-e5-large y Qwen3-Embedding (respaldo literal top-10 0,854, MRR 0,546) |
| Reranker | BAAI/bge-reranker-v2-m3 | Reordena 50 candidatos; Qwen3-Reranker 0.6B/4B comparado en `src/comparar.py --rerankers` |
| Decoder | Qwen/Qwen3-8B (8.191 M, Apache-2.0) | Sugerido por nombre en la §3.1. Sobre la misma recuperación: 40,00 frente a 38,43 de Qwen2.5-7B-Instruct |

Revisiones fijadas en `configs/sistema.json`; solo modelos abiertos; ningún modelo cerrado en el sistema (OpenRouter
lo usa únicamente el evaluador oficial para el componente de texto libre).

## 3. Resultados sobre las 50 preguntas de muestra

Evaluador oficial, componente determinista (cerradas 20, citas 20, abstención 10). Cada fila cambia una cosa.

| Corrida (RTX 4090) | Total /50 | Cambio |
|---|---:|---|
| Primera corrida completa | 29,07 | Corpus de ~19.400 documentos, Qwen2.5-7B |
| Ajustes de generación | 36,24 | JSON robusto, letra por probabilidad, respaldo completo de citas |
| Letra razonada | 37,80 | Primero la justificación, luego la letra |
| Ajustes de recuperación en texto libre | 38,43 | Reserva para la norma nombrada, tope por documento, mínimo normativo |
| Qwen3-8B | 40,00 | Cambio de decoder |
| Tres agentes, corpus corregido | pendiente | Configuración de entrega |

Medido y **descartado** (no mejoró o empeoró): permutar el orden de las opciones (Zheng et al., 2024) y descarte en
dos pasos (Ma y Du, 2023), 38,43–40,00; verificación en cadena CoVe, +0 y +3,3 s por pregunta; búsqueda por opción
en cerradas, −1 cerrada; un segundo modelo (Salamandra-7B) como juez de pasajes y segunda opinión: aprobó todos los
pasajes y Qwen3-8B falla con probabilidad 1,0 en sus errores, así que votar no los corrige; expansión de consulta solo
con evidencia «débil»: perdía la pregunta que la expansión sí resolvía.

Texto libre (30 puntos, RAGAS *answer correctness* contra la respuesta esperada): pendiente de la corrida con juez;
aproximación local gratuita en `legalrag.evaluation.ragas_local`. Tiempo por pregunta sin caché frente a los ~22 s
disponibles: pendiente.

## 4. Limitaciones

- La muestra es de 50 preguntas: una cerrada vale 1,33 puntos. Se prefirieron cambios que corrigen errores
  sistemáticos (segmentación, nombres de norma, precisión numérica) sobre ajustes que mueven una sola pregunta.
- Errores del banco: respuestas con la norma mal digitada (pregunta 58) o discutibles (128); fundamentos de doctrina
  sin norma (671). El normalizador mitiga los primeros; los demás no se pueden resolver con evidencia.
- Qwen3-8B es categórico en sus errores en cerradas (probabilidad 1,0 en la letra incorrecta): ni permutar opciones
  ni una segunda opinión los corrigieron.
- El control de citas compara normas, no certifica la corrección jurídica ni la vigencia (marcada `por_verificar`).
- La reformulación corre en todas las preguntas de texto libre (+~3 s): el disparador «solo si la evidencia es débil»
  que sugiere el enunciado, calibrado con la muestra, perdía casos; el tiempo total sigue dentro del presupuesto.
- Determinismo: una pregunta a la vez y temperatura 0; la generación en lote de varias preguntas cambia resultados
  (kernels no invariantes al tamaño del lote), por lo que no se usa.
