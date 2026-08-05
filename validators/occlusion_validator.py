"""
Occlusion validator.
"""

import numpy as np
from insightface.app.common import Face

from config.constants import OCCLUSION_PROHIBITED_PARTS
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from models.validation_stage import ValidationStage
from reasoning.semantic_engine import SemanticEvidenceEngine
from validators.base_validator import BaseValidator


class OcclusionValidator(BaseValidator):
    """Validates that no prohibited semantic objects occlude the ID photo.

    This validator only detects prohibited semantic classes returned by
    the face parser (e.g. hats). It does NOT check whether mandatory
    anatomical regions are visible; that responsibility belongs to
    FaceVisibilityValidator. It also does NOT distinguish normal
    eyeglasses from sunglasses; a dedicated GlassesValidator handles
    that distinction, so FacePart.EYE_GLASS is never treated as an
    occlusion here. Hair is likewise allowed: hair covering the eyes
    only affects FaceVisibilityValidator's assessment, not this one.
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
        """Validate the absence of prohibited occluding objects.

        Args:
            image: Image data to validate.
            face: Optional detected face. Unused by this validator; kept
                for signature compatibility with BaseValidator.
            parsing_result: Semantic face-parsing result describing which
                semantic classes are present in the image.

        Returns:
            A ValidationMetric containing a quality score clamped to the
            range [0.0, 1.0], where 1.0 indicates no prohibited objects
            were detected.

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

        prohibited_occlusions = self._find_prohibited_occlusions(
            parsing_result=parsing_result,
            engine=engine,
        )

        score = self._compute_score(
            prohibited_occlusions=prohibited_occlusions,
        )
        passed = not prohibited_occlusions
        message = self._build_message(
            prohibited_occlusions=prohibited_occlusions,
        )

        return ValidationMetric(
            type=ValidationType.OCCLUSION,
            passed=passed,
            score=score,
            message=message,
        )

    def _find_prohibited_occlusions(
        self,
        parsing_result: FaceParsingResult,
        engine: SemanticEvidenceEngine | None = None,
    ) -> tuple[FacePart, ...]:
        """Identify prohibited semantic parts present in the parsing result,

        delegating head covering semantic reasoning to the SemanticEvidenceEngine
        to distinguish prohibited hard headwear (caps, helmets) from allowed
        religious head coverings (hijabs, headscarves) when the face is fully visible
        and pose is frontal.
        """
        if engine is None:
            engine = SemanticEvidenceEngine(parsing_result=parsing_result)

        prohibited: list[FacePart] = []
        for part in OCCLUSION_PROHIBITED_PARTS:
            if part == FacePart.HAT:
                if engine.is_head_covering_prohibited():
                    prohibited.append(part)
            else:
                if parsing_result.has_part(part):
                    prohibited.append(part)
        return tuple(prohibited)

    def _compute_score(
        self,
        prohibited_occlusions: tuple[FacePart, ...],
    ) -> float:
        """Compute an overall occlusion score from detected prohibited parts.

        Each prohibited part contributes an equal share of the total
        score. A detected prohibited part loses its full share, mirroring
        the weighting approach used by FaceVisibilityValidator for
        missing mandatory regions.

        Args:
            prohibited_occlusions: Prohibited parts detected in the mask.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        total_parts = len(OCCLUSION_PROHIBITED_PARTS)

        if total_parts == 0:
            return 1.0

        part_weight = 1.0 / total_parts
        penalty = len(prohibited_occlusions) * part_weight

        score = 1.0 - penalty

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
        prohibited_occlusions: tuple[FacePart, ...],
    ) -> str:
        """Build a human-readable message describing detected occlusions.

        Args:
            prohibited_occlusions: Prohibited parts detected in the mask.

        Returns:
            A descriptive message string.
        """
        if not prohibited_occlusions:
            return "No prohibited occlusions detected."

        issues: list[str] = [
            f"{self._describe_part(part)} detected."
            for part in prohibited_occlusions
        ]

        return " ".join(issues)

    @staticmethod
    def _describe_part(part: FacePart) -> str:
        """Return a human-readable label for a prohibited FacePart.

        The label is derived directly from the enum member name (e.g.
        HAT -> "Hat"), so newly added prohibited parts are automatically
        supported without touching this method.

        Args:
            part: Prohibited semantic part to describe.

        Returns:
            A display label such as "Hat".
        """
        return part.name.replace("_", " ").title()