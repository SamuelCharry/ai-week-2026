# Cerberus: iteraciones (marks)

Cada mark se corre con `python3 src/main.py --mark N` (comprueba datos e índice, responde la muestra de 50 y pasa
el evaluador oficial). Para comparar dos salidas pregunta por pregunta: `scripts/comparar_salidas.py`. Cada mark se corre sobre la muestra de 50 y se compara con el anterior con `scripts/comparar_salidas.py`
(evaluador oficial, cambios pregunta por pregunta y RAGAS≈). Las opciones nuevas de cada mark se agregan sin
cambiar el comportamiento de los anteriores: sin ellas, el sistema reproduce el mark previo.

| Mark | Opciones de `run` | Muestra (determinista /50) |
|---|---|---|
| 42 | `--hybrid --no-hyde --mc-thinking --multi-query` | 42,90 (`reports/eval_phase6.json`) |
| 43 | Mark 42 + `--herramientas-v2 --normalizador` | por medir |

```bash
python -m legalrag.cli run --questions data/oficial/data/sample_50.jsonl --output salidas/mark42.jsonl \
  --hybrid --no-hyde --mc-thinking --multi-query
python -m legalrag.cli run --questions data/oficial/data/sample_50.jsonl --output salidas/mark43.jsonl \
  --hybrid --no-hyde --mc-thinking --multi-query --herramientas-v2 --normalizador
python scripts/comparar_salidas.py salidas/mark42.jsonl salidas/mark43.jsonl
```

## Mark 43 (de Cerberus sobre Mark 42)

- **Herramientas v2** (`tools/legal_tools.py`, `tool_block_v2`): SMLMV 2026 = $1.750.905 (Decreto 0159 de 2026; la
  tabla v1 tenía $1.500.000); UVT 2015–2026 en ambos sentidos (pesos ↔ UVT); sin año en la pregunta, los dos años
  más recientes (v1 suponía 2024); los plazos de liquidación del art. 11 de la Ley 1150 solo si se habla de liquidar
  un contrato (v1 los agregaba ante cualquier «plazo»).
- **Normalizador de citas** (`tools/normalizador.py`): si la pregunta u opción cita una ley o decreto con un año
  que no existe en el corpus pero sí con otro año, lo anota en el prompt (pregunta 58: «Ley 1564 de 2002» → 2012).
