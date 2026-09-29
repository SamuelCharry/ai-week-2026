#!/usr/bin/env bash
# Comando único de reproducción. Construye el contenedor, descarga corpus, índice y
# modelos, responde las 50 preguntas de muestra y ejecuta el evaluador oficial.
# data/ se monta para conservar descargas y resultados entre corridas.
# Argumentos extra pasan a `python -m legalrag run` (p. ej. --ragas).
set -euo pipefail
cd "$(dirname "$0")"
docker build -t p34k-derecho .
docker run --rm --gpus all -e OPENROUTER_API_KEY -v "$PWD/data:/app/data" p34k-derecho "$@"
