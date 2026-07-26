"""
Base validator contract.
"""

from abc import ABC, abstractmethod

import numpy as np
from insightface.app.common import Face

from models.validation_metric import ValidationMetric


class BaseValidator(ABC):
    """Defines the contract for all image validation rules."""

    @abstractmethod
    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
    ) -> ValidationMetric:
        """Validate an image and optional detected face.

        Args:
            image: Image data to validate.
            face: Optional detected face used by face-aware validators.

        Returns:
            A validation metric describing whether the rule passed or failed.
        """
        pass
