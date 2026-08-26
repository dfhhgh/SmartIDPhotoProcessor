"""
Root cause analyzer for identifying the exact primary failure reason of failed validations.
"""

from __future__ import annotations

from evaluation.models import FacePartEvaluation, RootCauseResult, ValidatorEvaluation
from models.parsing.face_part import FacePart
from models.validation_type import ValidationType


# Canonical root cause mapping from ValidationType.
_ROOT_CAUSE_MAP: dict[ValidationType, RootCauseResult] = {
    ValidationType.BLUR: RootCauseResult(
        cause="LOW_SHARPNESS",
        confidence=0.95,
        explanation="Image sharpness score fell below the required threshold.",
    ),
    ValidationType.BRIGHTNESS: RootCauseResult(
        cause="LOW_BRIGHTNESS",
        confidence=0.90,
        explanation="Image brightness is outside acceptable exposure bounds.",
    ),
    ValidationType.CONTRAST: RootCauseResult(
        cause="LOW_IMAGE_CONTRAST",
        confidence=0.90,
        explanation="Image contrast standard deviation is below threshold.",
    ),
    ValidationType.FACE_SIZE: RootCauseResult(
        cause="FACE_TOO_SMALL",
        confidence=0.95,
        explanation="Face bounding box area ratio relative to image is out of bounds.",
    ),
    ValidationType.HEAD_POSE: RootCauseResult(
        cause="HEAD_POSE",
        confidence=0.95,
        explanation="Head pitch, yaw, or roll exceeds maximum allowed angular limits.",
    ),
    ValidationType.OCCLUSION: RootCauseResult(
        cause="REAL_OCCLUSION",
        confidence=0.92,
        explanation="Prohibited head covering (non-religious hat/cap) detected.",
    ),
    ValidationType.FACE_AMBIGUITY: RootCauseResult(
        cause="FACE_AMBIGUITY",
        confidence=0.85,
        explanation="Multiple faces detected with similar confidence scores.",
    ),
}

# Ordered pipeline stages — first failure is the primary root cause.
_PIPELINE_ORDER: list[ValidationType] = [
    ValidationType.BLUR,
    ValidationType.BRIGHTNESS,
    ValidationType.CONTRAST,
    ValidationType.FACE_SIZE,
    ValidationType.HEAD_POSE,
    ValidationType.FACE_VISIBILITY,
    ValidationType.OCCLUSION,
    ValidationType.FACE_AMBIGUITY,
]


class RootCauseAnalyzer:
    """Analyzes validation results and semantic evidence to determine the

    precise primary root cause of any validation failure.  Uses only
    structured production data — never re-runs semantic logic.
    """

    @staticmethod
    def analyze(
        validators: list[ValidatorEvaluation],
        semantic_parts: dict[str, FacePartEvaluation] | None = None,
    ) -> RootCauseResult | None:
        """Determine the single primary root cause for a failed evaluation.

        Args:
            validators: Production validator evaluations (from ValidationResult.metrics).
            semantic_parts: Per-part semantic evidence (from SemanticEvidenceEngine
                public APIs).  Used to distinguish parser false negatives from
                threshold-strict failures.

        Returns:
            A RootCauseResult identifying the primary failure, or None when
            all validators passed.
        """
        failed_by_type = {
            v.validator_type: v
            for v in validators
            if not v.passed
        }
        if not failed_by_type:
            return None

        for vtype in _PIPELINE_ORDER:
            if vtype not in failed_by_type:
                continue

            if vtype == ValidationType.FACE_VISIBILITY:
                return RootCauseAnalyzer._analyze_face_visibility(
                    failed_by_type[vtype],
                    semantic_parts,
                )

            template = _ROOT_CAUSE_MAP.get(vtype)
            if template is not None:
                return RootCauseResult(
                    cause=template.cause,
                    confidence=template.confidence,
                    explanation=template.explanation,
                    evidence=[f"{vtype.value} validator failed."],
                )

        first_failed = min(failed_by_type.values(), key=lambda v: v.score)
        return RootCauseResult(
            cause="UNKNOWN",
            confidence=0.50,
            explanation="Validation failed due to unspecified criteria.",
            evidence=[f"Failed validators: {[v.validator_type.value for v in failed_by_type.values()]}"],
        )

    @staticmethod
    def _analyze_face_visibility(
        metric: ValidatorEvaluation,
        semantic_parts: dict[str, FacePartEvaluation] | None,
    ) -> RootCauseResult:
        """Determine whether a face-visibility failure is a parser false negative.

        Compares parser confidence against eye-support confidence for each
        eye part.  When the parser misses a region but the independent
        eye-support channel confirms visibility, the parser produced a
        false negative.  When both channels agree the region is absent,
        the threshold is genuinely strict.
        """
        if semantic_parts:
            eye_parts = [
                semantic_parts.get(FacePart.LEFT_EYE.name),
                semantic_parts.get(FacePart.RIGHT_EYE.name),
            ]
            for eye_pe in eye_parts:
                if eye_pe is None:
                    continue
                if eye_pe.parser_confidence == 0.0 and eye_pe.eye_support_confidence > 0.0:
                    return RootCauseResult(
                        cause="MODEL_PARSER_FALSE_NEGATIVE",
                        confidence=0.88,
                        explanation="BiSeNet parser failed to segment eye region despite independent evidence.",
                        evidence=[
                            f"Parser confidence for {eye_pe.part_name}: {eye_pe.parser_confidence:.2f}.",
                            f"Eye support confidence for {eye_pe.part_name}: {eye_pe.eye_support_confidence:.2f}.",
                        ],
                    )

        return RootCauseResult(
            cause="THRESHOLD_TOO_STRICT",
            confidence=0.80,
            explanation="Mandatory facial part ratio fell below minimum threshold.",
            evidence=[f"FaceVisibilityValidator score: {metric.score:.2f}."],
        )
