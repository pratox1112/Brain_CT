# BrainCT inference API: ONNX Runtime INT8 on CPU. Research use only.
#
# Build from the repo root (the model files come from artifacts/, never from git):
#   uv export --frozen --no-dev --extra serve --no-emit-project --no-hashes -o serving/requirements.txt
#   docker build -f serving/Dockerfile -t brainct-api:0.1.0 .
# Run:
#   docker run --rm -p 8000:8000 brainct-api:0.1.0

# ---- build stage: install dependencies + the package into a virtualenv -------------------
FROM python:3.11.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Dependencies first (cached unless requirements change). Base + [serve] only: no PyTorch.
COPY serving/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY pyproject.toml README.md /src/
COPY src /src/src
RUN pip install --no-deps /src

# ---- runtime stage: just the virtualenv, the model and an unprivileged user --------------
FROM python:3.11.11-slim-bookworm

# libgomp1: OpenMP runtime that some onnxruntime builds load
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv
COPY artifacts/resnet50_baseline/onnx/model_int8.onnx /app/model/model_int8.onnx
COPY artifacts/resnet50_baseline/metrics_val.json /app/model/metrics_val.json

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    BRAINCT_MODEL_PATH=/app/model/model_int8.onnx \
    BRAINCT_THRESHOLDS_PATH=/app/model/metrics_val.json \
    BRAINCT_THREADS=8 \
    BRAINCT_BATCHING=false

USER app
WORKDIR /app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "brainct.serving", "--host", "0.0.0.0", "--port", "8000"]
