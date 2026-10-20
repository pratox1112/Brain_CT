"""FastAPI service: POST a CT DICOM, get per-subtype hemorrhage probabilities. Research use only.

Run locally (laptop):  uv run python -m brainct.serving        -> http://127.0.0.1:8000/docs
Settings come from BRAINCT_* environment variables (see settings.py).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    generate_latest,
)

from brainct.constants import DISCLAIMER, LABELS
from brainct.preprocessing.dicom import DicomError
from brainct.serving.batching import MicroBatcher
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
# Dense from 10 to 300 ms, where served requests fall, so p95/p99 estimated from the
# histogram are not off by a whole coarse bucket.
LATENCY_BUCKETS = (
    0.005, 0.01, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075,
    0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0, 2.5, 5.0,
)  # fmt: skip
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
    # process_cpu_seconds_total, process_resident_memory_bytes, ... per pod, read from /proc
    # (Linux only; on Windows/macOS it reports nothing). Lets Grafana show CPU per pod
    # without cAdvisor or kube-state-metrics.
    ProcessCollector(registry=registry)
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
    batch_sizes = Histogram(
        "brainct_batch_size",
        "Images per model call",
        buckets=(1, 2, 4, 8, 16, 32),
        registry=registry,
    )

    state: dict[str, Any] = {"model": None, "error": None, "batcher": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # If loading fails the service still starts: /health stays up, /ready reports 503,
        # so Kubernetes can tell "process alive" from "able to serve".
        try:
            state["model"] = InferenceModel(
                settings.model_path, settings.thresholds_path, settings.model_name, settings.threads
            )
            if settings.warmup:
                # The first ONNX Runtime call allocates memory and is slow (27 ms vs ~11 ms
                # measured on the laptop); do it now, not on a user's request.
                size = state["model"].image_size
                state["model"].predict_batch(np.zeros((1, 3, size, size), dtype=np.float32))
            model_loaded.set(1)
        except Exception as exc:  # any load failure is reported via /ready
            state["model"] = None
            state["error"] = f"{type(exc).__name__}: {exc}"
            model_loaded.set(0)
        if state["model"] is not None and settings.batching:
            state["batcher"] = MicroBatcher(
                state["model"].predict_batch,
                settings.max_batch,
                settings.max_wait_ms,
                on_batch=batch_sizes.observe,
            )
            await state["batcher"].start()
        yield
        if state["batcher"] is not None:
            await state["batcher"].stop()

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
        batching = "on" if state["batcher"] is not None else "off"
        return {
            "status": "ready",
            "model": model.name,
            "version": model.version,
            "batching": batching,
        }

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
        batcher: MicroBatcher | None = state["batcher"]
        if batcher is not None:
            probs, batch_size = await batcher.predict(prepared.x)
        else:
            probs = (await run_in_threadpool(model.predict_batch, prepared.x[None]))[0]
            batch_size = 1
            batch_sizes.observe(1)
        # With batching this includes the wait for the batch to fill (at most max_wait_ms).
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
                batch_size=batch_size,
            ),
        )

    return app
