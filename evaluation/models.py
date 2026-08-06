"""
Evaluation data models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from models.validation_type import ValidationType


@dataclass(frozen=True, slots=True)
class FacePartEvaluation:
    """Detailed evaluation metrics for a specific FacePart."""
    part_name: str
    pixels: int
    area_ratio: float
    parser_confidence: float
    landmark_confidence: float
    pose_confidence: float
    occlusion_confidence: float
    eye_support_confidence: float
    final_confidence: float
    passed: bool


@dataclass(frozen=True, slots=True)
class ValidatorEvaluation:
    """Evaluation metrics for an individual validator."""
    validator_type: ValidationType
    validator_name: str
    passed: bool
    score: float
    message: str
    execution_time_ms: float | None = None


@dataclass(frozen=True, slots=True)
class FaceMetrics:
    """Metrics associated with the detected face."""
    bbox: tuple[float, float, float, float]
    area_ratio: float
    center: tuple[float, float]
    crop_size: tuple[int, int]
    alignment_size: tuple[int, int]
    yaw: float
    pitch: float
    roll: float
    detection_confidence: float


@dataclass(frozen=True, slots=True)
class RootCauseResult:
    """Root cause classification for a validation failure."""
    cause: str
    confidence: float
    explanation: str
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ImageEvaluationResult:
    """Complete evaluation report for a single evaluated image."""
    image_name: str
    resolution: tuple[int, int]
    execution_time_ms: float
    detected_faces: int
    primary_face_confidence: float
    selection_confidence: float
    face_metrics: FaceMetrics | None
    validators: list[ValidatorEvaluation]
    semantic_parts: dict[str, FacePartEvaluation]
    parser_statistics: dict[str, dict[str, Any]]
    root_cause: RootCauseResult | None
    overall_passed: bool
    output_paths: dict[str, str] = field(default_factory=dict)
