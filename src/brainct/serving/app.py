"""FastAPI service: POST a CT DICOM, get per-subtype hemorrhage probabilities. Research use only.

Run locally (laptop):  uv run python -m brainct.serving        -> http://127.0.0.1:8000/docs
Settings come from BRAINCT_* environment variables (see settings.py).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from brainct.constants import DISCLAIMER, LABELS
from brainct.preprocessing.dicom import DicomError
from brainct.serving.model import InferenceModel
from brainct.serving.schemas import (
    ErrorResponse,
    ModelInfo,
    PredictResponse,
    PreprocessingInfo,
    Timing,
)
from brainct.serving.settings import Settings

ENDPOINTS = {"/predict", "/health", "/ready", "/metrics", "/docs", "/openapi.json"}
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse, "description": text}
    for code, text in {
        400: "empty upload",
        413: "upload too large",
        422: "not a usable CT DICOM",
        503: "model not loaded",
    }.items()
}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    # A registry per app keeps metrics separate when tests create several apps.
    registry = CollectorRegistry()
    requests_total = Counter(
        "brainct_requests", "HTTP requests", ["endpoint", "status"], registry=registry
    )
    request_latency = Histogram(
        "brainct_request_latency_seconds",
        "End-to-end request latency",
        ["endpoint"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    stage_latency = Histogram(
        "brainct_stage_seconds",
        "Time per prediction stage",
        ["stage"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    model_loaded = Gauge("brainct_model_loaded", "1 if the model is loaded", registry=registry)

    state: dict[str, Any] = {"model": None, "error": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # If loading fails the service still starts: /health stays up, /ready reports 503,
        # so Kubernetes can tell "process alive" from "able to serve".
        try:
            state["model"] = InferenceModel(
                settings.model_path, settings.thresholds_path, settings.model_name, settings.threads
            )
            model_loaded.set(1)
        except Exception as exc:  # any load failure is reported via /ready
            state["error"] = f"{type(exc).__name__}: {exc}"
            model_loaded.set(0)
        yield

    app = FastAPI(
        title="BrainCT hemorrhage detection (research use only)",
        description=DISCLAIMER,
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        path = request.url.path
        endpoint = path if path in ENDPOINTS else "other"  # bounded label values
        requests_total.labels(endpoint, str(response.status_code)).inc()
        request_latency.labels(endpoint).observe(time.perf_counter() - start)
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        body = ErrorResponse(detail=str(exc.detail)).model_dump()
        return JSONResponse(body, status_code=exc.status_code)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", responses={503: {"model": ErrorResponse}})
    async def ready() -> dict[str, str]:
        model: InferenceModel | None = state["model"]
        if model is None:
            raise HTTPException(503, f"model not loaded: {state['error']}")
        return {"status": "ready", "model": model.name, "version": model.version}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/predict", response_model=PredictResponse, responses=ERROR_RESPONSES)
    async def predict(
        file: Annotated[UploadFile, File(description="one CT slice as a DICOM file")],
    ) -> PredictResponse:
        start = time.perf_counter()
        model: InferenceModel | None = state["model"]
        if model is None:
            raise HTTPException(503, f"model not loaded: {state['error']}")
        data = await file.read(settings.max_upload_bytes + 1)
        if not data:
            raise HTTPException(400, "empty upload")
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(413, f"upload larger than {settings.max_upload_mb} MB")

        try:
            prepared = await run_in_threadpool(model.prepare, data)
        except DicomError as exc:
            raise HTTPException(422, str(exc)) from exc
        if prepared.info.modality != "CT":
            raise HTTPException(422, f"expected a CT image, got Modality={prepared.info.modality}")

        inference_start = time.perf_counter()
        probs = (await run_in_threadpool(model.predict_batch, prepared.x[None]))[0]
        inference_ms = (time.perf_counter() - inference_start) * 1000
        stage_latency.labels("preprocess").observe(prepared.preprocess_ms / 1000)
        stage_latency.labels("inference").observe(inference_ms / 1000)

        probabilities = {label: float(p) for label, p in zip(LABELS, probs, strict=True)}
        return PredictResponse(
            probabilities=probabilities,
            predictions={k: v >= model.thresholds[k] for k, v in probabilities.items()},
            thresholds=model.thresholds,
            model=ModelInfo(
                name=model.name,
                version=model.version,
                format=model.format,
                input_size=model.image_size,
            ),
            preprocessing=PreprocessingInfo(
                original_shape=prepared.info.original_shape,
                sign_fixed=prepared.info.sign_fixed,
                padded=prepared.info.padded,
            ),
            timing=Timing(
                preprocess_ms=prepared.preprocess_ms,
                inference_ms=inference_ms,
                total_ms=(time.perf_counter() - start) * 1000,
            ),
        )

    return app
