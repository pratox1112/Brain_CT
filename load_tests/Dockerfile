# Locust load generator for the BrainCT API, run as a Kubernetes Job (job.yaml).
# No model, no RSNA data: it sends synthetic DICOMs built by tests/dicom_factory.py.
#
# Build from the repo root (load_tests/Dockerfile.dockerignore limits what is sent):
#   uv export --frozen --only-group loadtest --no-emit-project --no-hashes -o load_tests/requirements.txt
#   docker build -f load_tests/Dockerfile -t brainct-loadtest:0.1.0 .

FROM python:3.11.11-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY load_tests/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt \
    && useradd --system --create-home --uid 10001 app

# Locust puts the locustfile's folder on sys.path, so dicom_factory is importable from here.
COPY load_tests/locustfile.py tests/dicom_factory.py /app/

USER app
WORKDIR /app
ENTRYPOINT ["locust", "-f", "/app/locustfile.py"]
