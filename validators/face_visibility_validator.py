"""
Face visibility validator.
"""

from __future__ import annotations

import numpy as np
from insightface.app.common import Face

from config.constants import (
    FACE_VISIBILITY_MIN_PART_RATIOS,
    FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR,
    FACE_VISIBILITY_REQUIRED_PARTS,
)
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from models.validation_stage import ValidationStage
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

        missing_parts, landmark_overridden = (
            self._find_missing_parts_with_landmark_override(
                parsing_result=parsing_result,
                face=face,
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

    def _find_missing_parts_with_landmark_override(
        self,
        parsing_result: FaceParsingResult,
        face: Face | None,
    ) -> tuple[tuple[FacePart, ...], frozenset[FacePart]]:
        """Identify mandatory regions absent from the parsing mask, with
        landmark-based override for eye parts when eyeglasses are present.

        Decision logic for each mandatory part:

        1. Parser detects the part -> present (never missing).
        2. Parser misses the part AND no EYE_GLASS in mask -> missing.
        3. Parser misses the part AND EYE_GLASS present AND corresponding
           InsightFace landmark exists and is valid -> treat as present.
        4. Parser misses the part AND EYE_GLASS present BUT landmark is
           missing or invalid -> missing.

        Args:
            parsing_result: Semantic face-parsing result to inspect.
            face: Optional detected face providing InsightFace landmarks.

        Returns:
            A tuple of (missing_parts, landmark_overridden_parts) where
            missing_parts are the parts determined to be genuinely absent,
            and landmark_overridden_parts is the set of eye parts that
            were originally missing but overridden by valid landmarks.
        """
        raw_missing = self._find_missing_parts(
            parsing_result=parsing_result,
        )

        if not raw_missing:
            return raw_missing, frozenset()

        has_glasses = parsing_result.has_part(FacePart.EYE_GLASS)

        if not has_glasses:
            return raw_missing, frozenset()

        overridden: set[FacePart] = set()
        result: list[FacePart] = []

        for part in raw_missing:
            if part in _EYE_LANDMARK_INDICES and self._has_valid_eye_landmark(
                face=face,
                part=part,
            ):
                overridden.add(part)
            else:
                result.append(part)

        return tuple(result), frozenset(overridden)

    def _find_missing_parts(
        self,
        parsing_result: FaceParsingResult,
    ) -> tuple[FacePart, ...]:
        """Identify mandatory facial regions that are entirely absent from
        the parsing mask.

        This is the baseline parser-only check, without any landmark
        override logic. It is called internally by
        _find_missing_parts_with_landmark_override.

        The mouth region is checked as a single composite semantic unit
        via FaceParsingResult.has_visible_mouth_region(), keeping all
        parsing knowledge inside the parsing result.

        Args:
            parsing_result: Semantic face-parsing result to inspect.

        Returns:
            Mandatory FacePart values with zero pixels in the mask, in
            the order defined by FACE_VISIBILITY_REQUIRED_PARTS, plus
            FacePart.MOUTH when the composite mouth region is not visible.
        """
        result = [
            part
            for part in FACE_VISIBILITY_REQUIRED_PARTS
            if not parsing_result.has_part(part)
        ]

        if not parsing_result.has_visible_mouth_region(
            mouth_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.MOUTH],
            upper_lip_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.UPPER_LIP],
            lower_lip_min_ratio=FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.LOWER_LIP],
        ):
            result.append(FacePart.MOUTH)

        return tuple(result)

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
        - face.kps is a NumPy ndarray with shape (5, 2).
        - The coordinate pair for the requested eye index contains no NaN
          or infinite values.

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

        if kps.shape != (5, 2):
            return False

        idx = _EYE_LANDMARK_INDICES[part]

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
        parts plus the composite mouth region (which is checked as a
        single semantic unit via FaceParsingResult).

        Args:
            missing_parts: Mandatory parts entirely absent from the mask.
            insufficient_parts: Mandatory parts present but too small.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        # 5 individual parts + 1 composite mouth region = 6 semantic checks.
        total_parts = len(FACE_VISIBILITY_REQUIRED_PARTS) + 1
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