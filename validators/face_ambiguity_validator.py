"""
Face ambiguity validator.
"""

from config.constants import (
    FACE_SELECTION_AMBIGUITY_MAX_RATIO,
    FACE_SELECTION_MIN_PRIMARY_SCORE,
    FACE_SELECTION_MIN_COMPETITIVE_SCORE,
    FACE_SELECTION_MIN_SCORE_MARGIN,
)
from models.selection_result import SelectionResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.base_selection_validator import BaseSelectionValidator


class FaceAmbiguityValidator(BaseSelectionValidator):
    """Validates that the selector's primary-face decision is reliable.

    This validator does NOT count faces and does NOT decide whether a
    photo contains "too many people". It only interprets the confidence
    metadata already computed by FaceSelector (SelectionResult): a single
    small billboard or a distant face in the background should score far
    below the primary subject and therefore leave the selection
    unambiguous, while two people of similar size and centrality compete
    strongly for "primary face" and should be flagged as ambiguous.

    The decision rule utilizes multiple signals available in SelectionResult:
    - selected_score
    - second_best_score
    - ambiguity_ratio
    - score_margin

    A distant tiny face, advertisement, poster, TV, banner, reflection, etc.
    should almost never trigger rejection unless it is genuinely competitive
    with a strong primary subject.
    """

    def validate(self, selection_result: SelectionResult) -> ValidationMetric:
        """Validate selection reliability using multiple confidence signals.

        Args:
            selection_result: The selector's chosen face plus confidence
                metadata.

        Returns:
            A ValidationMetric using ValidationType.FACE_AMBIGUITY, where a
            higher score indicates a more confidently unambiguous selection.

        Raises:
            TypeError: If selection_result is not a SelectionResult.
            ValueError: If selection_result is None.
        """
        if selection_result is None:
            raise ValueError("Selection result must not be None.")

        if not isinstance(selection_result, SelectionResult):
            raise TypeError("Selection result must be a SelectionResult.")

        selected_score = selection_result.selected_score
        second_best_score = selection_result.second_best_score
        ambiguity_ratio = selection_result.ambiguity_ratio
        score_margin = selection_result.score_margin

        # 1. Check if primary score meets minimum threshold
        if selected_score < FACE_SELECTION_MIN_PRIMARY_SCORE:
            passed = False
            score = selected_score
            message = (
                f"Selected face score ({selected_score:.2f}) is below the minimum "
                f"primary score threshold ({FACE_SELECTION_MIN_PRIMARY_SCORE:.2f})."
            )
        elif second_best_score is None or selection_result.detected_faces_count <= 1:
            passed = True
            score = 1.0
            message = "Primary face selection is unambiguous."
        else:
            # 2. Check if runner-up is a strong competitive face (not billboard/poster/noise)
            is_competitive = second_best_score >= FACE_SELECTION_MIN_COMPETITIVE_SCORE

            if not is_competitive:
                passed = True
                score = float(min(max(1.0 - ambiguity_ratio * 0.2, 0.0), 1.0))
                message = "Primary face selection is unambiguous (runner-up is non-competitive)."
            else:
                # 3. Both primary and runner-up are strong; evaluate ambiguity via ratio and margin
                is_ambiguous_ratio = ambiguity_ratio >= FACE_SELECTION_AMBIGUITY_MAX_RATIO
                is_ambiguous_margin = score_margin <= FACE_SELECTION_MIN_SCORE_MARGIN

                passed = not (is_ambiguous_ratio or is_ambiguous_margin)
                score = self._compute_score(ambiguity_ratio=ambiguity_ratio, score_margin=score_margin)
                message = self._build_message(
                    passed=passed,
                    selected_score=selected_score,
                    second_best_score=second_best_score,
                    ambiguity_ratio=ambiguity_ratio,
                    score_margin=score_margin,
                )

        return ValidationMetric(
            type=ValidationType.FACE_AMBIGUITY,
            passed=passed,
            score=score,
            message=message,
        )

    def _compute_score(self, ambiguity_ratio: float, score_margin: float) -> float:
        """Convert ambiguity ratio and score margin into a bounded confidence score."""
        margin_factor = min(max(score_margin, 0.0), 1.0)
        ratio_factor = 1.0 - ambiguity_ratio
        score = 0.5 * ratio_factor + 0.5 * margin_factor
        return float(min(max(score, 0.0), 1.0))

    def _build_message(
        self,
        passed: bool,
        selected_score: float,
        second_best_score: float,
        ambiguity_ratio: float,
        score_margin: float,
    ) -> str:
        """Build a human-readable message describing selection reliability."""
        if passed:
            return "Primary face selection is unambiguous."

        return (
            "Genuine ambiguity detected between strong candidate faces "
            f"(selected score: {selected_score:.2f}, runner-up score: {second_best_score:.2f}, "
            f"ambiguity ratio: {ambiguity_ratio:.2f}, score margin: {score_margin:.2f})."
        )
