"""
Face visibility validator.
"""

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


class FaceVisibilityValidator(BaseValidator):
    """Validates that all mandatory anatomical facial regions are sufficiently visible.

    This validator only checks whether required facial regions are present
    in the BiSeNet parsing mask and large enough to be considered visible.
    It does NOT detect occluding accessories such as sunglasses, hats, or
    masks; that responsibility belongs to a dedicated OcclusionValidator.
    """

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationMetric:
        """Validate facial region visibility using a BiSeNet parsing result.

        Args:
            image: Image data to validate.
            face: Optional detected face. Unused by this validator; kept
                for signature compatibility with BaseValidator.
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
        _ = face

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

        missing_parts = self._find_missing_parts(
            parsing_result=parsing_result,
        )
        insufficient_parts = self._find_insufficient_parts(
            parsing_result=parsing_result,
            missing_parts=missing_parts,
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

    def _find_missing_parts(
        self,
        parsing_result: FaceParsingResult,
    ) -> tuple[FacePart, ...]:
        """Identify mandatory facial regions that are entirely absent.

        Args:
            parsing_result: Semantic face-parsing result to inspect.

        Returns:
            Mandatory FacePart values with zero pixels in the mask, in
            the order defined by FACE_VISIBILITY_REQUIRED_PARTS.
        """
        return tuple(
            part
            for part in FACE_VISIBILITY_REQUIRED_PARTS
            if not parsing_result.has_part(part)
        )

    def _find_insufficient_parts(
        self,
        parsing_result: FaceParsingResult,
        missing_parts: tuple[FacePart, ...],
    ) -> tuple[FacePart, ...]:
        """Identify present mandatory regions below their minimum area ratio.

        Args:
            parsing_result: Semantic face-parsing result to inspect.
            missing_parts: Mandatory parts already known to be absent.
                Excluded here so they are penalized only once, as missing
                rather than as insufficient.

        Returns:
            Mandatory FacePart values present but below their configured
            minimum visibility ratio.
        """
        return tuple(
            part
            for part in FACE_VISIBILITY_REQUIRED_PARTS
            if part not in missing_parts
            and parsing_result.part_ratio(part) < FACE_VISIBILITY_MIN_PART_RATIOS[part]
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

        Args:
            missing_parts: Mandatory parts entirely absent from the mask.
            insufficient_parts: Mandatory parts present but too small.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        total_parts = len(FACE_VISIBILITY_REQUIRED_PARTS)
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