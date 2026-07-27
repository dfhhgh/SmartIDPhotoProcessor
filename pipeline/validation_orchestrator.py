"""
Validation pipeline orchestrator.

Executes a sequence of validators against an image and optional face /
parsing data, collecting the results into a single ValidationResult.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from insightface.app.common import Face

from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_result import ValidationResult
from pipeline.validator_factory import create_default_validators
from validators.base_validator import BaseValidator

if TYPE_CHECKING:
    import numpy as np


class ValidationOrchestrator:
    """Orchestrate the execution of a validator pipeline.

    The orchestrator iterates over a fixed sequence of validators,
    passes the same inputs to each one, and aggregates the resulting
    :class:`ValidationMetric` objects into a :class:`ValidationResult`.

    It does **not** perform any validation logic itself; that
    responsibility belongs entirely to the individual validators.
    """

    def __init__(
        self,
        validators: tuple[BaseValidator, ...] | None = None,
    ) -> None:
        """Initialise the orchestrator with a validator pipeline.

        Args:
            validators: An ordered tuple of validators to execute.
                When ``None`` (the default), the pipeline is created
                by :func:`create_default_validators`.
        """
        if validators is None:
            validators = create_default_validators()

        self._validators: tuple[BaseValidator, ...] = tuple(validators)

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationResult:
        """Run every registered validator and return the aggregated result.

        Args:
            image: BGR image data to validate.
            face: Optional detected face. Required by face-aware validators
                (e.g. :class:`FaceSizeValidator`, :class:`HeadPoseValidator`).
            parsing_result: Optional semantic parsing result. Required by
                parsing-dependent validators (e.g.
                :class:`FaceVisibilityValidator`, :class:`OcclusionValidator`).

        Returns:
            A :class:`ValidationResult` containing one
            :class:`ValidationMetric` per registered validator.
        """
        metrics = [
            validator.validate(
                image=image,
                face=face,
                parsing_result=parsing_result,
            )
            for validator in self._validators
        ]

        return ValidationResult(metrics=metrics)
