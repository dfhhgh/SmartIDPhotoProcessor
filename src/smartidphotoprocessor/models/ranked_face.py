"""
Ranked face model.
"""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from insightface.app.common import Face


@dataclass(frozen=True, slots=True)
class RankedFace:
    """Represents a detected face paired with its computed final score.

    Attributes:
        face: The detected Face object.
        score: The computed final weighted score, in [0.0, 1.0].
    """

    face: Face
    score: float

    def __post_init__(self) -> None:
        """Validate ranked face attributes after initialization."""
        if self.face is None:
            raise ValueError("Face must not be None.")

        if not isinstance(self.score, numbers.Real) or isinstance(self.score, bool):
            raise TypeError("Face score must be numeric.")

        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("Face score must be between 0.0 and 1.0.")
