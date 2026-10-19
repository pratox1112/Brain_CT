"""Pydantic response models (also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from brainct.constants import DISCLAIMER

# A real response from the laptop (2026-10-19, one RSNA test slice), shown in /docs.
EXAMPLE_RESPONSE = {
    "probabilities": {
        "any": 0.9748,
        "epidural": 0.0008,
        "intraparenchymal": 0.0905,
        "intraventricular": 0.0046,
        "subarachnoid": 0.9560,
        "subdural": 0.0529,
    },
    "predictions": {
        "any": True,
        "epidural": False,
        "intraparenchymal": False,
        "intraventricular": False,
        "subarachnoid": True,
        "subdural": False,
    },
    "thresholds": {
        "any": 0.3804,
        "epidural": 0.1840,
        "intraparenchymal": 0.4479,
        "intraventricular": 0.3760,
        "subarachnoid": 0.2630,
        "subdural": 0.3554,
    },
    "model": {
        "name": "resnet50_baseline-int8",
        "version": "ef6cea9f5cb4",
        "format": "onnx-int8",
        "input_size": 256,
    },
    "preprocessing": {"original_shape": [512, 512], "sign_fixed": False, "padded": False},
    # The first request after startup, before warmup existed (so inference was slow).
    "timing": {"preprocess_ms": 12.6, "inference_ms": 27.2, "total_ms": 40.8, "batch_size": 1},
    "disclaimer": DISCLAIMER,
}


class ModelInfo(BaseModel):
    name: str
    version: str = Field(description="first 12 hex digits of the model file's SHA-256")
    format: str
    input_size: int


class PreprocessingInfo(BaseModel):
    original_shape: tuple[int, int]
    sign_fixed: bool = Field(description="12-bit sign fix applied (mislabelled unsigned pixels)")
    padded: bool = Field(description="non-square image padded with air before resizing")


class Timing(BaseModel):
    preprocess_ms: float
    inference_ms: float = Field(description="includes any wait for a micro-batch to fill")
    total_ms: float
    batch_size: int = Field(1, description="images in the model call this request ran in")


class PredictResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_RESPONSE]})

    probabilities: dict[str, float] = Field(description="per label, in [0, 1]")
    predictions: dict[str, bool] = Field(description="probability >= the val-chosen threshold")
    thresholds: dict[str, float]
    model: ModelInfo
    preprocessing: PreprocessingInfo
    timing: Timing
    disclaimer: str = DISCLAIMER


class ErrorResponse(BaseModel):
    detail: str
    disclaimer: str = DISCLAIMER
