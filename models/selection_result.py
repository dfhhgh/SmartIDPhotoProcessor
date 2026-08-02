"""
Face selection result model.

Captures not just which face FaceSelector chose, but how confident that
choice was, so downstream components can reason about selection
reliability without re-scoring faces or duplicating selector logic.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass, field

from insightface.app.common import Face

from models.ranked_face import RankedFace


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Represents the outcome of primary-face selection, with confidence.

    Attributes:
        selected_face: The face chosen as the primary subject.
        selected_score: The winning face's final weighted score, in [0.0, 1.0].
        second_best_score: The runner-up face's final weighted score, in
            [0.0, 1.0], or None when only one face was detected.
        score_margin: selected_score - second_best_score. When no runner-up
            exists, this equals selected_score (maximal, uncontested margin).
        ambiguity_ratio: second_best_score / selected_score, in [0.0, 1.0].
            0.0 means no competition (or only one face detected); values
            close to 1.0 mean the runner-up scored almost as well as the
            winner, i.e. the selection is ambiguous.
        detected_faces_count: Total number of faces detected in the image,
            including the selected one.
        ranked_faces: Tuple of RankedFace models representing every detected
            face paired with its score, sorted descending by score.
    """

    selected_face: Face
    selected_score: float
    second_best_score: float | None
    score_margin: float
    ambiguity_ratio: float
    detected_faces_count: int
    ranked_faces: tuple[RankedFace, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate the result values after initialization."""
        if self.selected_face is None:
            raise ValueError("Selected face must not be None.")

        if not isinstance(self.selected_score, numbers.Real) or isinstance(
            self.selected_score, bool
        ):
            raise TypeError("Selected score must be numeric.")

        if not 0.0 <= float(self.selected_score) <= 1.0:
            raise ValueError("Selected score must be between 0.0 and 1.0.")

        if self.second_best_score is not None:
            if not isinstance(self.second_best_score, numbers.Real) or isinstance(
                self.second_best_score, bool
            ):
                raise TypeError("Second best score must be numeric.")

            if not 0.0 <= float(self.second_best_score) <= 1.0:
                raise ValueError("Second best score must be between 0.0 and 1.0.")

        if not isinstance(self.score_margin, numbers.Real) or isinstance(
            self.score_margin, bool
        ):
            raise TypeError("Score margin must be numeric.")

        if not isinstance(self.ambiguity_ratio, numbers.Real) or isinstance(
            self.ambiguity_ratio, bool
        ):
            raise TypeError("Ambiguity ratio must be numeric.")

        if not 0.0 <= float(self.ambiguity_ratio) <= 1.0:
            raise ValueError("Ambiguity ratio must be between 0.0 and 1.0.")

        if not isinstance(self.detected_faces_count, int) or isinstance(
            self.detected_faces_count, bool
        ):
            raise TypeError("Detected faces count must be an int.")

        if self.detected_faces_count < 1:
            raise ValueError("Detected faces count must be at least 1.")

        if not isinstance(self.ranked_faces, tuple):
            raise TypeError("ranked_faces must be a tuple.")

        if len(self.ranked_faces) != self.detected_faces_count:
            raise ValueError("Length of ranked_faces must match detected_faces_count.")

        for rf in self.ranked_faces:
            if not isinstance(rf, RankedFace):
                raise TypeError("All items in ranked_faces must be RankedFace instances.")

        scores = [rf.score for rf in self.ranked_faces]
        if scores != sorted(scores, reverse=True):
            raise ValueError("ranked_faces must be sorted in descending order by score.")

        if self.ranked_faces:
            if self.selected_face is not self.ranked_faces[0].face:
                raise ValueError("selected_face must match the first RankedFace.")
            if abs(float(self.selected_score) - float(self.ranked_faces[0].score)) > 1e-6:
                raise ValueError("selected_score must match the first RankedFace score.")

            if len(self.ranked_faces) > 1:
                if self.second_best_score is None or abs(float(self.second_best_score) - float(self.ranked_faces[1].score)) > 1e-6:
                    raise ValueError("second_best_score must match the second RankedFace score.")
            else:
                if self.second_best_score is not None:
                    raise ValueError("second_best_score must be None when only one face is detected.")

    @property
    def has_competing_face(self) -> bool:
        """Return True when more than one face was detected."""
        return self.second_best_score is not None

    @property
    def all_scores(self) -> tuple[float, ...]:
        """Return every candidate's final weighted score, sorted descending."""
        return tuple(rf.score for rf in self.ranked_faces)
