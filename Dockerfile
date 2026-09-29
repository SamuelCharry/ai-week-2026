# Reproducción en contenedor limpio: responde las 50 preguntas de muestra y las evalúa.
# Requiere GPU NVIDIA y NVIDIA Container Toolkit. Uso: ./reproducir.sh
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt data/oficial/scripts/requirements-evaluador.txt /tmp/requisitos/
RUN pip install torch --index-url https://download.pytorch.org/whl/cu128 \
    && pip install -r /tmp/requisitos/requirements.txt -r /tmp/requisitos/requirements-evaluador.txt

COPY . .
ENTRYPOINT ["python", "-m", "legalrag", "run"]
