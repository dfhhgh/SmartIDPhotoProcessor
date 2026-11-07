"""API routes for SmartIDPhotoProcessor.

Implements:
- POST /api/v1/validate — photo validation
- GET /api/v1/health — process liveness
- GET /api/v1/ready — model readiness
"""

from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from smartidphotoprocessor.api.schemas import (
    CandidateMatchSchema,
    ErrorResponseSchema,
    ExportResultSchema,
    FaceInfoSchema,
    HealthResponseSchema,
    ReadyResponseSchema,
    ReverseSearchResultSchema,
    ValidateResponseSchema,
    ValidationMetricSchema,
    ValidationResultSchema,
)
from smartidphotoprocessor.api.executor import run_in_executor
from smartidphotoprocessor.exceptions.face_exceptions import FacePipelineError
from smartidphotoprocessor.models.photo_processing_result import PhotoProcessingResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_IMAGE_DIMENSION = 4096
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
_UPLOAD_CHUNK_SIZE = 64 * 1024  # 64 KB per read

# ---------------------------------------------------------------------------
# Gender mapping (InsightFace integer → semantic string)
# ---------------------------------------------------------------------------
_GENDER_MAP = {0: "female", 1: "male"}

# ---------------------------------------------------------------------------
# Pipeline singleton with thread-safe initialization
# ---------------------------------------------------------------------------
_pipeline = None
_pipeline_ready = False
_pipeline_init_lock = threading.Lock()


def _get_pipeline():
    """Return the singleton pipeline, initializing on first call.

    Thread-safe: initialization is serialized via _pipeline_init_lock.
    After initialization, concurrent requests share the same instance
    without lock overhead.
    """
    global _pipeline, _pipeline_ready
    if _pipeline is not None:
        return _pipeline
    with _pipeline_init_lock:
        if _pipeline is not None:
            return _pipeline
        from smartidphotoprocessor.pipeline.photo_validation_pipeline import PhotoValidationPipeline

        logger.info("Initializing PhotoValidationPipeline (first request).")
        _pipeline = PhotoValidationPipeline()
        _pipeline_ready = True
        logger.info("PhotoValidationPipeline initialized successfully.")
    return _pipeline


def get_pipeline_ready() -> bool:
    """Return whether the pipeline has been initialized."""
    return _pipeline_ready


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_face_info(face) -> FaceInfoSchema | None:
    """Extract safe fields from an InsightFace Face object.

    Gender is normalized from InsightFace integer (0/1) to semantic string.
    """
    if face is None:
        return None
    bbox = face.bbox.tolist() if hasattr(face, "bbox") and face.bbox is not None else None
    if bbox is None:
        return None

    raw_gender = getattr(face, "gender", None)
    if raw_gender is not None and raw_gender in _GENDER_MAP:
        gender = _GENDER_MAP[raw_gender]
    else:
        gender = None

    return FaceInfoSchema(
        bbox=[float(v) for v in bbox],
        detection_score=float(getattr(face, "det_score", 0.0)),
        age=int(face.age) if hasattr(face, "age") and face.age is not None else None,
        gender=gender,
    )


def _serialize_export_result(export) -> ExportResultSchema | None:
    """Serialize export metadata (no image bytes)."""
    if export is None:
        return None
    return ExportResultSchema(
        original_size=list(export.original_size),
        exported_size=list(export.exported_size),
        content_size=list(export.content_size),
        offset_x=export.offset_x,
        offset_y=export.offset_y,
        padding_color=list(export.padding_color),
        was_upscaled=export.was_upscaled,
        upscale_factor=export.upscale_factor,
        quality=export.export_quality,
        warnings=export.export_warnings,
    )


def _serialize_reverse_search(rs) -> ReverseSearchResultSchema | None:
    """Serialize reverse search result."""
    if rs is None:
        return None
    candidates = [
        CandidateMatchSchema(
            vector_id=c.vector_id,
            person_id=c.person_id,
            label=c.label,
            image=c.image,
            similarity=c.similarity,
        )
        for c in rs.candidates
    ]
    return ReverseSearchResultSchema(
        status=str(rs.status),
        candidates=candidates,
        top_k=rs.top_k,
        query_dimension=rs.query_dimension,
        processing_time_ms=rs.processing_time_ms,
        error_message=rs.error_message,
    )


def _serialize_validation_result(vr) -> ValidationResultSchema:
    """Serialize validation result."""
    return ValidationResultSchema(
        is_valid=vr.is_valid,
        total_metrics=len(vr.metrics),
        passed_count=len(vr.passed_metrics),
        failed_count=len(vr.failed_metrics),
        failed_metrics=[
            ValidationMetricSchema(
                type=str(m.type),
                passed=m.passed,
                score=m.score,
                message=m.message,
            )
            for m in vr.failed_metrics
        ],
    )


def _serialize_processing_result(result: PhotoProcessingResult) -> ValidateResponseSchema:
    """Serialize the full processing result into the API response contract."""
    return ValidateResponseSchema(
        is_valid=result.validation_result.is_valid,
        validation=_serialize_validation_result(result.validation_result),
        face=_serialize_face_info(result.selected_face),
        export=_serialize_export_result(result.export_result),
        reverse_search=_serialize_reverse_search(result.reverse_search_result),
    )


# ---------------------------------------------------------------------------
# Image decoding
# ---------------------------------------------------------------------------

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes into a BGR uint8 numpy array.

    Raises:
        HTTPException: 400 if the image cannot be decoded.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Unsupported or corrupted format.")
    if image.size == 0:
        raise HTTPException(status_code=400, detail="Decoded image is empty.")
    h, w = image.shape[:2]
    if h > MAX_IMAGE_DIMENSION or w > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions {w}x{h} exceed maximum {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}.",
        )
    return image


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponseSchema,
    tags=["ops"],
)
async def health():
    """Process liveness check.

    Returns 200 if the process is alive. Does NOT check model readiness.
    """
    return HealthResponseSchema(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponseSchema,
    tags=["ops"],
)
async def ready():
    """Model readiness check.

    Returns whether the pipeline has been initialized and models are loaded.
    The pipeline loads all required models (InsightFace, BiSeNet, Aux Head)
    during construction. If the pipeline object exists, models are loaded.
    """
    models_loaded = get_pipeline_ready()
    return ReadyResponseSchema(
        status="ready" if models_loaded else "not_ready",
        models_loaded=models_loaded,
        detail="Pipeline initialized" if models_loaded else "Pipeline not yet initialized",
    )


@router.post(
    "/validate",
    response_model=ValidateResponseSchema,
    responses={
        400: {"model": ErrorResponseSchema, "description": "Invalid request"},
        422: {"model": ErrorResponseSchema, "description": "Validation error"},
        500: {"model": ErrorResponseSchema, "description": "Internal server error"},
    },
    tags=["validation"],
)
async def validate(file: UploadFile = File(...)):
    """Validate a photo against ID document standards.

    Accepts a JPEG, PNG, WebP, or TIFF image upload.
    Returns detailed validation results including face detection,
    quality metrics, and export metadata.
    """
    start_time = time.perf_counter()

    # 1. Validate upload exists
    if file is None or file.filename is None or file.filename == "":
        raise HTTPException(status_code=400, detail="No file uploaded.")

    # 2. Validate content type (advisory; some clients send incorrect types)
    if file.content_type and file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported content type: {file.content_type}. "
            f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}.",
        )

    # 3. Read bytes incrementally with early size rejection
    chunks: list[bytes] = []
    total_size = 0
    try:
        while True:
            chunk = await file.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=f"File too large: exceeds {MAX_UPLOAD_BYTES} bytes.",
                )
            chunks.append(chunk)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    contents = b"".join(chunks)

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 4. Decode image
    image = _decode_image(contents)

    # 5. Run pipeline (offloaded to bounded executor to avoid blocking event loop)
    try:
        pipeline = _get_pipeline()
        result = await run_in_executor(pipeline.validate, image)
    except FacePipelineError as exc:
        logger.warning("Pipeline validation error: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except (TypeError, KeyError) as exc:
        # Internal programming errors — not client input errors
        logger.exception("Internal pipeline type error.")
        raise HTTPException(status_code=500, detail="Internal processing error.")
    except ValueError as exc:
        # Distinguish client input errors from internal errors.
        # Pipeline raises ValueError for expected validation outcomes
        # (e.g. "no face detected", "face too small").
        msg = str(exc).lower()
        _expected_validation = any(
            kw in msg
            for kw in ("face", "detect", "select", "crop", "align", "parse", "size", "valid", "metric")
        )
        if _expected_validation:
            logger.warning("Validation error: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc))
        logger.exception("Unexpected ValueError.")
        raise HTTPException(status_code=500, detail="Internal processing error.")
    except Exception as exc:
        logger.exception("Unexpected pipeline error.")
        raise HTTPException(status_code=500, detail="Internal processing error.")

    # 6. Serialize response
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "Validation completed: is_valid=%s, elapsed=%.0fms",
        result.validation_result.is_valid,
        elapsed_ms,
    )

    return _serialize_processing_result(result)
