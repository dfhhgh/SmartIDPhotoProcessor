"""
Validation pipeline orchestrator.

Executes a sequence of validators against an image and optional face /
parsing data, collecting the results into a single ValidationResult,
with optimized multi-stage execution and short-circuiting to avoid
unnecessary model inference (FaceParserService and GlassesDetectorClassifier).
Validators are dynamically grouped by their explicit `stage` property.

Two execution modes are supported (see ValidationExecutionMode):
- PRODUCTION: stage short-circuiting, lazy inference (existing behavior).
- DEVELOPMENT: every stage and every validator always runs, for debugging,
  calibration, dataset analysis, benchmarking, and threshold tuning.

Execution mode affects orchestration only -- no validator is aware of it.
"""

from __future__ import annotations

import logging
from typing import Callable, TYPE_CHECKING

from insightface.app.common import Face

from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_execution_mode import ValidationExecutionMode
from models.validation_metric import ValidationMetric
from models.validation_result import ValidationResult
from models.validation_stage import ValidationStage
from pipeline.validator_factory import create_default_validators
from services.face_parser_service import FaceParserService
from validators.base_validator import BaseValidator
from validators.face_size_validator import FaceSizeValidator

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class ValidationOrchestrator:
    """Orchestrate the execution of a validator pipeline in optimized stages.

    Validators are dynamically grouped by their ``stage`` property into:
    - ValidationStage.CHEAP: Blur, Brightness, Contrast, FaceSize, HeadPose.
    - ValidationStage.PARSING: FaceVisibility, Occlusion.
    - ValidationStage.GLASSES: GlassesValidator.

    In ValidationExecutionMode.PRODUCTION (default), the orchestrator stops
    immediately on the first failing stage without running FaceParserService
    or GlassesDetectorClassifier if not reached.

    In ValidationExecutionMode.DEVELOPMENT, every stage always runs and every
    metric is collected, regardless of earlier failures.
    """

    def __init__(
        self,
        validators: tuple[BaseValidator, ...] | None = None,
        parser_service: FaceParserService | None = None,
        execution_mode: ValidationExecutionMode = ValidationExecutionMode.PRODUCTION,
    ) -> None:
        """Initialise the orchestrator with a validator pipeline and parser service."""
        self._is_custom_validators = validators is not None
        if validators is None:
            validators = create_default_validators()

        self._validators: tuple[BaseValidator, ...] = tuple(validators)
        self._parser_service = (
            parser_service if parser_service is not None else FaceParserService()
        )
        self._execution_mode = execution_mode

        # Strategy dispatch table: one executor per execution mode. Adding a
        # future mode means adding one entry here and one private method --
        # no branching logic elsewhere in the class.
        self._executors: dict[
            ValidationExecutionMode,
            Callable[
                [np.ndarray, Face | None, FaceParsingResult | None, np.ndarray | None, Face | None, dict[ValidationStage, list[BaseValidator]]],
                ValidationResult,
            ],
        ] = {
            ValidationExecutionMode.PRODUCTION: self._execute_production,
            ValidationExecutionMode.DEVELOPMENT: self._execute_development,
        }

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
        original_image: np.ndarray | None = None,
        original_face: Face | None = None,
    ) -> ValidationResult:
        """Run validators according to the configured execution mode.

        Args:
            image: BGR image data to validate (typically aligned image for quality checks).
            face: Optional detected face (typically aligned face).
            parsing_result: Optional semantic parsing result.
            original_image: Optional original uploaded image, used by FaceSizeValidator.
            original_face: Optional original selected face in original image coordinate space.

        Returns:
            A :class:`ValidationResult` containing collected metrics.
        """
        # If custom validators are provided (e.g. in unit tests), run all of them in order
        if self._is_custom_validators:
            metrics = []
            for validator in self._validators:
                v_img = (original_image if original_image is not None else image) if isinstance(validator, FaceSizeValidator) else image
                v_face = (original_face if original_face is not None else face) if isinstance(validator, FaceSizeValidator) else face
                metrics.append(
                    validator.validate(
                        image=v_img,
                        face=v_face,
                        parsing_result=parsing_result,
                    )
                )
            return ValidationResult(metrics=metrics)

        stages = self._group_by_stage()
        executor = self._executors[self._execution_mode]
        return executor(image, face, parsing_result, original_image, original_face, stages)

    def _group_by_stage(self) -> dict[ValidationStage, list[BaseValidator]]:
        """Group configured validators by their declared ``stage`` property.

        Every validator is required to expose a valid ``stage``. There is no
        silent fallback: a validator with a missing or invalid stage is a
        configuration error and must fail fast rather than being grouped
        into an arbitrary default stage.
        """
        stages: dict[ValidationStage, list[BaseValidator]] = {
            stage: [] for stage in ValidationStage
        }

        for validator in self._validators:
            v_stage = validator.stage
            if v_stage not in stages:
                raise ValueError(
                    f"Validator {validator!r} has an invalid stage: {v_stage!r}. "
                    f"Expected one of: {', '.join(str(stage) for stage in ValidationStage)}."
                )
            stages[v_stage].append(validator)

        return stages

    @staticmethod
    def _run_stage(
        validators: list[BaseValidator],
        image: np.ndarray,
        face: Face | None,
        parsing_result: FaceParsingResult | None,
        original_image: np.ndarray | None = None,
        original_face: Face | None = None,
    ) -> list[ValidationMetric]:
        """Run every validator in a stage and return their metrics, in order."""
        metrics = []
        for validator in validators:
            v_img = (original_image if original_image is not None else image) if isinstance(validator, FaceSizeValidator) else image
            v_face = (original_face if original_face is not None else face) if isinstance(validator, FaceSizeValidator) else face
            metrics.append(
                validator.validate(image=v_img, face=v_face, parsing_result=parsing_result)
            )
        return metrics

    def _execute_production(
        self,
        image: np.ndarray,
        face: Face | None,
        parsing_result: FaceParsingResult | None,
        original_image: np.ndarray | None,
        original_face: Face | None,
        stages: dict[ValidationStage, list[BaseValidator]],
    ) -> ValidationResult:
        """Run stages with short-circuiting and lazy inference (existing behavior)."""
        metrics: list[ValidationMetric] = []

        # Stage 1: CHEAP validators (Blur, Brightness, Contrast, FaceSize, HeadPose)
        cheap_metrics = self._run_stage(
            stages[ValidationStage.CHEAP], image, face, None, original_image, original_face
        )
        metrics.extend(cheap_metrics)
        if not all(metric.passed for metric in cheap_metrics):
            logger.info("Stage CHEAP validation failed. Short-circuiting pipeline.")
            return ValidationResult(metrics=metrics)

        # Stage 2: PARSING validators (FaceVisibility, Occlusion)
        if stages[ValidationStage.PARSING]:
            if parsing_result is None:
                logger.info("Stage CHEAP passed. Initializing FaceParserService for Stage PARSING...")
                parsing_result = self._parser_service.parse(image)

            parsing_metrics = self._run_stage(
                stages[ValidationStage.PARSING], image, face, parsing_result, original_image, original_face
            )
            metrics.extend(parsing_metrics)
            if not all(metric.passed for metric in parsing_metrics):
                logger.info("Stage PARSING validation failed. Short-circuiting pipeline.")
                return ValidationResult(metrics=metrics)

        # Stage 3: GLASSES validators (GlassesValidator)
        if stages[ValidationStage.GLASSES]:
            glasses_metrics = self._run_stage(
                stages[ValidationStage.GLASSES], image, face, parsing_result, original_image, original_face
            )
            metrics.extend(glasses_metrics)

        return ValidationResult(metrics=metrics)

    def _execute_development(
        self,
        image: np.ndarray,
        face: Face | None,
        parsing_result: FaceParsingResult | None,
        original_image: np.ndarray | None,
        original_face: Face | None,
        stages: dict[ValidationStage, list[BaseValidator]],
    ) -> ValidationResult:
        """Run every stage and every validator unconditionally, collecting all metrics."""
        metrics: list[ValidationMetric] = []

        # Stage 1: CHEAP validators -- always run, failures do not stop the pipeline.
        metrics.extend(
            self._run_stage(stages[ValidationStage.CHEAP], image, face, None, original_image, original_face)
        )

        # Stage 2: PARSING validators -- always run if any exist, regardless of
        # Stage CHEAP outcome. Parsing is still computed lazily: only invoked
        # if there is a PARSING-stage validator to consume it.
        if stages[ValidationStage.PARSING] and parsing_result is None:
            logger.info("Development mode: initializing FaceParserService for Stage PARSING...")
            parsing_result = self._parser_service.parse(image)

        metrics.extend(
            self._run_stage(stages[ValidationStage.PARSING], image, face, parsing_result, original_image, original_face)
        )

        # Stage 3: GLASSES validators -- always run, regardless of earlier failures.
        metrics.extend(
            self._run_stage(stages[ValidationStage.GLASSES], image, face, parsing_result, original_image, original_face)
        )

        return ValidationResult(metrics=metrics)
