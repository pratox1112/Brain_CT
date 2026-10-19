"""Pydantic response models (also drive the OpenAPI docs at /docs)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from brainct.constants import DISCLAIMER


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
    inference_ms: float
    total_ms: float


class PredictResponse(BaseModel):
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
