#!/usr/bin/env bash
# Ejecuta variantes del pipeline sobre sample_50 y agrega una fila por variante
# al archivo reports/ablaciones.csv. Permite comparar cambios de prompt o retrieval
# sin perder las corridas anteriores.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venv/bin/python3}
mkdir -p salidas reports

run_variant() {
  local name="$1"; shift
  local out="salidas/exp_${name}.jsonl"
  rm -f "$out" "${out%.jsonl}.trace.jsonl" "${out%.jsonl}.run.json"
  echo "===== VARIANT $name ====="
  "$PY" -m legalrag.cli run \
    --questions data/oficial/data/sample_50.jsonl \
    --output "$out" --hybrid --no-hyde "$@"
  echo
  "$PY" data/oficial/scripts/evaluate.py \
    --submission "$out" --split sample \
    --out "reports/eval_${name}.json"
  "$PY" scripts/append_row.py "$name" "reports/eval_${name}.json" "$out"
}

for v in "$@"; do
  case "$v" in
    base)      run_variant base ;;
    thinking)  run_variant thinking --mc-thinking ;;
    tools)     run_variant tools --mc-thinking ;;
    fewshot)   run_variant fewshot --mc-thinking ;;
    multiquery) run_variant multiquery --mc-thinking --multi-query ;;
    phase6)    run_variant phase6 --mc-thinking --multi-query ;;
    no-augment) run_variant no_augment --augment-max 0 ;;
    *) echo "unknown variant: $v" >&2; exit 2 ;;
  esac
done
