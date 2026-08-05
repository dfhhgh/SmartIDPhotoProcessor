"""
Face visibility validator.
"""

from __future__ import annotations

import numpy as np
from insightface.app.common import Face

from config.constants import (
    FACE_VISIBILITY_COMPOSITE_REGION_COUNT,
    FACE_VISIBILITY_MIN_PART_RATIOS,
    FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR,
    FACE_VISIBILITY_REQUIRED_PARTS,
)
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from models.validation_stage import ValidationStage
from reasoning.semantic_engine import SemanticEvidenceEngine
from validators.base_validator import BaseValidator

# Human-readable labels for mandatory regions, used only in messages.
# Kept private to this module: it is presentation logic, not a threshold,
# so it does not belong in constants.py.
_PART_DISPLAY_NAMES: dict[FacePart, str] = {
    FacePart.LEFT_EYE: "Left eye",
    FacePart.RIGHT_EYE: "Right eye",
    FacePart.LEFT_BROW: "Left eyebrow",
    FacePart.RIGHT_BROW: "Right eyebrow",
    FacePart.NOSE: "Nose",
    FacePart.MOUTH: "Mouth",
    FacePart.UPPER_LIP: "Upper lip",
    FacePart.LOWER_LIP: "Lower lip",
}

# InsightFace 5-point landmark indices (viewer perspective).
# Index 0 = right eye, Index 1 = left eye.
_EYE_LANDMARK_INDICES: dict[FacePart, int] = {
    FacePart.RIGHT_EYE: 0,
    FacePart.LEFT_EYE: 1,
}


class FaceVisibilityValidator(BaseValidator):
    """Validates that all mandatory anatomical facial regions are sufficiently visible.

    The primary source of evidence is the BiSeNet parsing mask. When the
    parser fails to detect eyes because of transparent eyeglasses, but the
    EYE_GLASS class is present in the mask, InsightFace facial landmarks
    from the Face object are used as secondary confirmation. If the
    corresponding eye landmark exists and is valid, the eye is treated as
    visible instead of missing.

    This validator does NOT detect occluding accessories such as sunglasses,
    hats, or masks; that responsibility belongs to a dedicated
    OcclusionValidator.
    """

    @property
    def stage(self) -> ValidationStage:
        """Return the validation stage for this validator."""
        return ValidationStage.PARSING

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationMetric:
        """Validate facial region visibility using a BiSeNet parsing result.

        When the parser misses an eye but EYE_GLASS is present in the mask,
        InsightFace landmarks from *face* are consulted as secondary
        evidence. A valid eye landmark overrides the parser's missing-eye
        decision, preventing false rejections for transparent prescription
        glasses.

        Args:
            image: Image data to validate.
            face: Optional detected face providing InsightFace landmarks
                for secondary eye-visibility confirmation.
            parsing_result: Semantic face-parsing result describing which
                anatomical regions are present and how large they are.

        Returns:
            A ValidationMetric containing a quality score clamped to the
            range [0.0, 1.0], where 1.0 indicates every mandatory region
            is present and sufficiently visible.

        Raises:
            TypeError: If image is not a NumPy array, or parsing_result is
                not a FaceParsingResult.
            ValueError: If image is None/empty, or parsing_result is None.
        """
        if image is None:
            raise ValueError(
                "Image must not be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "Image must be a numpy array."
            )

        if image.size == 0:
            raise ValueError(
                "Image must not be empty."
            )

        if parsing_result is None:
            raise ValueError(
                "Parsing result must not be None."
            )

        if not isinstance(
            parsing_result,
            FaceParsingResult,
        ):
            raise TypeError(
                "Parsing result must be a FaceParsingResult."
            )

        engine = SemanticEvidenceEngine(
            parsing_result=parsing_result,
            face=face,
        )

        missing_parts, landmark_overridden = (
            self._find_missing_parts_with_engine(
                parsing_result=parsing_result,
                engine=engine,
            )
        )
        insufficient_parts = self._find_insufficient_parts(
            parsing_result=parsing_result,
            missing_parts=missing_parts,
            landmark_overridden_parts=landmark_overridden,
        )

        score = self._compute_score(
            missing_parts=missing_parts,
            insufficient_parts=insufficient_parts,
        )
        passed = not missing_parts and not insufficient_parts
        message = self._build_message(
            missing_parts=missing_parts,
            insufficient_parts=insufficient_parts,
        )

        return ValidationMetric(
            type=ValidationType.FACE_VISIBILITY,
            passed=passed,
            score=score,
            message=message,
        )

    def _find_missing_parts_with_engine(
        self,
        parsing_result: FaceParsingResult,
        engine: SemanticEvidenceEngine,
    ) -> tuple[tuple[FacePart, ...], frozenset[FacePart]]:
        """Identify mandatory regions absent using the SemanticEvidenceEngine."""
        missing: list[FacePart] = []
        overridden: set[FacePart] = set()

        # 1. Required individual parts (eyes, nose)
        for part in FACE_VISIBILITY_REQUIRED_PARTS:
            is_visible = (
                engine.is_eye_visible(part, min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[part])
                if part in (FacePart.LEFT_EYE, FacePart.RIGHT_EYE)
                else parsing_result.has_part(part)
            )
            if not is_visible:
                if part in (FacePart.LEFT_EYE, FacePart.RIGHT_EYE) and engine._compute_landmark_confidence(part) > 0.0 and parsing_result.has_part(FacePart.EYE_GLASS):
                    overridden.add(part)
                else:
                    missing.append(part)

        # 2. Composite mouth region
        if not engine.is_mouth_visible(
            mouth_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.MOUTH],
            upper_lip_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.UPPER_LIP],
            lower_lip_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.LOWER_LIP],
        ):
            missing.append(FacePart.MOUTH)

        # 3. Composite eyebrow regions
        if not engine.is_eyebrow_visible(
            brow=FacePart.LEFT_BROW,
            eye=FacePart.LEFT_EYE,
            brow_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.LEFT_BROW],
            eye_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.LEFT_EYE],
        ):
            missing.append(FacePart.LEFT_BROW)

        if not engine.is_eyebrow_visible(
            brow=FacePart.RIGHT_BROW,
            eye=FacePart.RIGHT_EYE,
            brow_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.RIGHT_BROW],
            eye_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.RIGHT_EYE],
        ):
            missing.append(FacePart.RIGHT_BROW)

        return tuple(missing), frozenset(overridden)

    def _find_missing_parts_with_landmark_override(
        self,
        parsing_result: FaceParsingResult,
        face: Face | None = None,
    ) -> tuple[tuple[FacePart, ...], frozenset[FacePart]]:
        """Backward compatibility wrapper for internal/test usage."""
        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        return self._find_missing_parts_with_engine(parsing_result=parsing_result, engine=engine)

    def _find_missing_parts(
        self,
        parsing_result: FaceParsingResult,
        face: Face | None = None,
    ) -> tuple[FacePart, ...]:
        """Backward compatibility wrapper for internal/test usage."""
        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        missing, _ = self._find_missing_parts_with_engine(parsing_result=parsing_result, engine=engine)
        return missing

    def _find_insufficient_parts(
        self,
        parsing_result: FaceParsingResult,
        missing_parts: tuple[FacePart, ...],
        landmark_overridden_parts: frozenset[FacePart] = frozenset(),
    ) -> tuple[FacePart, ...]:
        """Identify present mandatory regions below their minimum area ratio.

        Parts that were overridden by valid InsightFace landmarks are
        excluded alongside missing parts, since the landmark confirmation
        supersedes the parser's pixel-level evidence.

        Note: LEFT_BROW, RIGHT_BROW, and MOUTH are composite checks
        handled by FaceParsingResult methods and are not iterated here.

        Args:
            parsing_result: Semantic face-parsing result to inspect.
            missing_parts: Mandatory parts already known to be absent.
                Excluded here so they are penalized only once, as missing
                rather than as insufficient.
            landmark_overridden_parts: Eye parts whose missing status was
                overridden by valid InsightFace landmarks. Also excluded
                from the insufficient-parts check.

        Returns:
            Mandatory FacePart values present but below their configured
            minimum visibility ratio.
        """
        excluded = set(missing_parts) | landmark_overridden_parts

        return tuple(
            part
            for part in FACE_VISIBILITY_REQUIRED_PARTS
            if part not in excluded
            and parsing_result.part_ratio(part) < FACE_VISIBILITY_MIN_PART_RATIOS[part]
        )

    def _has_valid_eye_landmark(
        self,
        face: Face | None,
        part: FacePart,
    ) -> bool:
        """Check whether the InsightFace landmark for a specific eye is
        present and contains valid coordinates.

        A landmark is considered valid when:

        - The Face object is not None.
        - face.kps is a 2-D NumPy ndarray with at least 2 columns.
        - The array has enough rows for the requested eye index.
        - The coordinate pair for the requested eye contains no NaN
          or infinite values.

        This validation is intentionally flexible: it does not require a
        specific landmark count (5, 68, 106, etc.), so it will continue
        working if a future InsightFace model exposes more landmarks.

        Args:
            face: Optional detected face providing InsightFace landmarks.
            part: An eye FacePart (LEFT_EYE or RIGHT_EYE).

        Returns:
            True if the corresponding eye landmark exists and is valid.
        """
        if face is None:
            return False

        kps = getattr(face, "kps", None)

        if kps is None or not isinstance(kps, np.ndarray):
            return False

        if kps.ndim != 2 or kps.shape[1] != 2:
            return False

        idx = _EYE_LANDMARK_INDICES[part]

        if idx >= kps.shape[0]:
            return False

        return bool(
            np.isfinite(kps[idx]).all()
        )

    def _compute_score(
        self,
        missing_parts: tuple[FacePart, ...],
        insufficient_parts: tuple[FacePart, ...],
    ) -> float:
        """Compute an overall visibility score from region-level failures.

        Each mandatory region contributes an equal share of the total
        score. A missing region loses its full share; a region that is
        present but below its minimum ratio loses only a fraction of its
        share, controlled by FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR.

        The total number of checkable regions is the individual required
        parts plus the composite semantic regions (mouth, eyebrows).

        Args:
            missing_parts: Mandatory regions entirely absent from the mask.
            insufficient_parts: Mandatory regions present but too small.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        # Individual required parts + composite semantic regions.
        total_parts = (
            len(FACE_VISIBILITY_REQUIRED_PARTS)
            + FACE_VISIBILITY_COMPOSITE_REGION_COUNT
        )
        part_weight = 1.0 / total_parts

        missing_penalty = len(missing_parts) * part_weight
        insufficient_penalty = (
            len(insufficient_parts)
            * part_weight
            * FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR
        )

        score = 1.0 - missing_penalty - insufficient_penalty

        return float(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            )
        )

    def _build_message(
        self,
        missing_parts: tuple[FacePart, ...],
        insufficient_parts: tuple[FacePart, ...],
    ) -> str:
        """Build a human-readable message describing visibility failures.

        Args:
            missing_parts: Mandatory parts entirely absent from the mask.
            insufficient_parts: Mandatory parts present but too small.

        Returns:
            A descriptive message string.
        """
        if not missing_parts and not insufficient_parts:
            return "All required facial features are sufficiently visible."

        issues: list[str] = [
            f"{self._describe_part(part)} is not visible."
            for part in missing_parts
        ]
        issues.extend(
            f"{self._describe_part(part)} visibility is below the required threshold."
            for part in insufficient_parts
        )

        return " ".join(issues)

    @staticmethod
    def _describe_part(part: FacePart) -> str:
        """Return a human-readable label for a mandatory FacePart.

        Args:
            part: Facial region to describe.

        Returns:
            A display label such as "Left eye" or "Right eyebrow".
        """
        return _PART_DISPLAY_NAMES[part]