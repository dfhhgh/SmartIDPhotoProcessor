"""Pydantic schemas for the SmartIDPhotoProcessor API.

These schemas define the JSON contract for request/response serialization.
Internal domain objects (numpy arrays, InsightFace Face, etc.) are NOT exposed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Validation metric
# ---------------------------------------------------------------------------

class ValidationMetricSchema(BaseModel):
    """Serialized validation metric."""

    type: str = Field(..., description="Validation rule type (e.g. blur, brightness)")
    passed: bool = Field(..., description="Whether the metric passed")
    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0.0 and 1.0")
    message: str = Field(default="", description="Human-readable message")


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

class ValidationResultSchema(BaseModel):
    """Serialized validation result."""

    is_valid: bool = Field(..., description="True when all metrics passed")
    total_metrics: int = Field(..., ge=1, description="Total number of metrics evaluated")
    passed_count: int = Field(..., ge=0, description="Number of metrics that passed")
    failed_count: int = Field(..., ge=0, description="Number of metrics that failed")
    failed_metrics: list[ValidationMetricSchema] = Field(
        default_factory=list,
        description="Only the metrics that failed",
    )


# ---------------------------------------------------------------------------
# Face info (safe subset of InsightFace Face)
# ---------------------------------------------------------------------------

class FaceInfoSchema(BaseModel):
    """Safe subset of face detection information.

    Does NOT expose embeddings, landmarks, or model internals.
    """

    bbox: list[float] = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2]",
        min_length=4,
        max_length=4,
    )
    detection_score: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    age: int | None = Field(default=None, description="Estimated age if available")
    gender: str | None = Field(default=None, description="Estimated gender if available")


# ---------------------------------------------------------------------------
# Export result
# ---------------------------------------------------------------------------

class ExportResultSchema(BaseModel):
    """Serialized export metadata.

    Does NOT include the actual exported image bytes.
    """

    original_size: list[int] = Field(
        ...,
        description="[width, height] of the original image",
        min_length=2,
        max_length=2,
    )
    exported_size: list[int] = Field(
        ...,
        description="[width, height] of the exported image",
        min_length=2,
        max_length=2,
    )
    content_size: list[int] = Field(
        ...,
        description="[width, height] of the content area",
        min_length=2,
        max_length=2,
    )
    offset_x: int = Field(..., description="Horizontal offset of content")
    offset_y: int = Field(..., description="Vertical offset of content")
    padding_color: list[int] = Field(
        ...,
        description="[B, G, R] padding color",
        min_length=3,
        max_length=3,
    )
    was_upscaled: bool = Field(..., description="Whether the image was upscaled")
    upscale_factor: float = Field(..., gt=0.0, description="Upscale factor applied")
    quality: str = Field(..., description="Export quality level (safe/warning)")
    warnings: list[str] = Field(default_factory=list, description="Export warnings")


# ---------------------------------------------------------------------------
# Reverse search candidate
# ---------------------------------------------------------------------------

class CandidateMatchSchema(BaseModel):
    """Serialized reverse search candidate match."""

    vector_id: int = Field(..., description="FAISS vector ID")
    person_id: str = Field(..., description="Reference person identifier")
    label: str = Field(..., description="Human-readable label")
    image: str = Field(..., description="Relative path of reference image")
    similarity: float = Field(..., description="Cosine similarity score")


# ---------------------------------------------------------------------------
# Reverse search result
# ---------------------------------------------------------------------------

class ReverseSearchResultSchema(BaseModel):
    """Serialized reverse search result.

    Evidence only — cannot trigger PASS/REVIEW/REJECT.
    """

    status: str = Field(..., description="Search status (completed/unavailable/disabled/error)")
    candidates: list[CandidateMatchSchema] = Field(
        default_factory=list,
        description="Ranked candidate matches",
    )
    top_k: int = Field(..., ge=1, description="Requested top-k limit")
    query_dimension: int = Field(..., ge=1, description="Embedding dimensionality")
    processing_time_ms: float = Field(..., ge=0.0, description="Search wall-clock time")
    error_message: str | None = Field(default=None, description="Error details if status is error")


# ---------------------------------------------------------------------------
# Validation response
# ---------------------------------------------------------------------------

class ValidateResponseSchema(BaseModel):
    """Response for POST /api/v1/validate."""

    is_valid: bool = Field(..., description="Overall validation result")
    validation: ValidationResultSchema = Field(..., description="Detailed validation results")
    face: FaceInfoSchema | None = Field(default=None, description="Selected face info")
    export: ExportResultSchema | None = Field(default=None, description="Export metadata")
    reverse_search: ReverseSearchResultSchema | None = Field(
        default=None,
        description="Reverse search evidence (disabled by default)",
    )


# ---------------------------------------------------------------------------
# Health / readiness responses
# ---------------------------------------------------------------------------

class HealthResponseSchema(BaseModel):
    """Response for GET /api/v1/health."""

    status: str = Field(default="ok", description="Process health status")


class ReadyResponseSchema(BaseModel):
    """Response for GET /api/v1/ready."""

    status: str = Field(..., description="Readiness status (ready/not_ready)")
    models_loaded: bool = Field(..., description="Whether models are loaded and ready")
    detail: str = Field(default="", description="Additional readiness information")


# ---------------------------------------------------------------------------
# Error response
# ---------------------------------------------------------------------------

class ErrorResponseSchema(BaseModel):
    """Standard error response."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Human-readable error description")
