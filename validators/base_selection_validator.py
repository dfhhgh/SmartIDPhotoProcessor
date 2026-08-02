"""
Base selection-validator contract.

Mirrors BaseValidator's pattern, but for the selection stage: validators
implementing this contract interpret a SelectionResult only. They must
never re-score faces, access FaceDetector, or duplicate FaceSelector logic.
"""

from abc import ABC, abstractmethod

from models.selection_result import SelectionResult
from models.validation_metric import ValidationMetric


class BaseSelectionValidator(ABC):
    """Defines the contract for validators that assess selection reliability."""

    @abstractmethod
    def validate(self, selection_result: SelectionResult) -> ValidationMetric:
        """Assess whether a face selection is reliable enough to proceed.

        Args:
            selection_result: The selector's chosen face plus confidence
                metadata (scores, margin, ambiguity ratio, candidate count).

        Returns:
            A validation metric describing whether the selection is
            reliable enough to continue the pipeline.
        """
        pass
