"""Tests for ValidationOrchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_result import ValidationResult
from models.validation_type import ValidationType
from models.validation_stage import ValidationStage
from pipeline.validation_orchestrator import ValidationOrchestrator
from validators.base_validator import BaseValidator


# ------------------------------------------------------------------ #
# Fake validators
# ------------------------------------------------------------------ #


class _RecordingValidator(BaseValidator):
    """Validator that records every call it receives and returns a preset metric."""

    def __init__(
        self,
        name: str,
        metric: ValidationMetric,
        execution_log: list[str],
    ) -> None:
        self._name = name
        self._metric = metric
        self._execution_log = execution_log

        self.received_image = None
        self.received_face = None
        self.received_parsing_result = None

    @property
    def stage(self) -> ValidationStage:
        return ValidationStage.CHEAP

    def validate(
        self,
        image: np.ndarray,
        face=None,
        parsing_result=None,
    ) -> ValidationMetric:
        self.received_image = image
        self.received_face = face
        self.received_parsing_result = parsing_result
        self._execution_log.append(self._name)
        return self._metric


class _RaisingValidator(BaseValidator):
    """Validator that always raises RuntimeError."""

    def __init__(self, message: str = "boom") -> None:
        self._message = message

    @property
    def stage(self) -> ValidationStage:
        return ValidationStage.CHEAP

    def validate(
        self,
        image: np.ndarray,
        face=None,
        parsing_result=None,
    ) -> ValidationMetric:
        raise RuntimeError(self._message)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _metric(
    vtype: ValidationType = ValidationType.BLUR,
    passed: bool = True,
    score: float = 1.0,
    message: str = "ok",
) -> ValidationMetric:
    return ValidationMetric(
        type=vtype,
        passed=passed,
        score=score,
        message=message,
    )


# ------------------------------------------------------------------ #
# 1. Execution order
# ------------------------------------------------------------------ #


def test_runs_all_validators_in_order():
    """Three fake validators must execute in the exact order they were provided."""
    log: list[str] = []

    v1 = _RecordingValidator("v1", _metric(), log)
    v2 = _RecordingValidator("v2", _metric(), log)
    v3 = _RecordingValidator("v3", _metric(), log)

    orchestrator = ValidationOrchestrator(
        validators=(v1, v2, v3),
    )

    orchestrator.validate(image=np.zeros((1, 1, 3), dtype=np.uint8))

    assert log == ["v1", "v2", "v3"]


# ------------------------------------------------------------------ #
# 2. Return type
# ------------------------------------------------------------------ #


def test_returns_validation_result():
    """validate() must return a ValidationResult instance."""
    orchestrator = ValidationOrchestrator(
        validators=(_RecordingValidator("a", _metric(), []),),
    )

    result = orchestrator.validate(
        image=np.zeros((1, 1, 3), dtype=np.uint8),
    )

    assert isinstance(result, ValidationResult)


# ------------------------------------------------------------------ #
# 3. Metric collection
# ------------------------------------------------------------------ #


def test_collects_all_metrics():
    """Every metric returned by a validator must appear in the result."""
    m1 = _metric(vtype=ValidationType.BLUR, passed=True, score=0.9, message="sharp")
    m2 = _metric(vtype=ValidationType.BRIGHTNESS, passed=False, score=0.2, message="dark")
    m3 = _metric(vtype=ValidationType.CONTRAST, passed=True, score=0.7, message="ok")

    orchestrator = ValidationOrchestrator(
        validators=(
            _RecordingValidator("a", m1, []),
            _RecordingValidator("b", m2, []),
            _RecordingValidator("c", m3, []),
        ),
    )

    result = orchestrator.validate(
        image=np.zeros((1, 1, 3), dtype=np.uint8),
    )

    assert len(result.metrics) == 3
    assert result.metrics[0] is m1
    assert result.metrics[1] is m2
    assert result.metrics[2] is m3


# ------------------------------------------------------------------ #
# 4. Input passthrough (identity)
# ------------------------------------------------------------------ #


def test_passes_same_inputs_to_every_validator():
    """Every validator must receive the exact same objects, not copies."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    face = object()
    parsing_result = object()

    v1 = _RecordingValidator("a", _metric(), [])
    v2 = _RecordingValidator("b", _metric(), [])

    orchestrator = ValidationOrchestrator(
        validators=(v1, v2),
    )

    orchestrator.validate(
        image=image,
        face=face,
        parsing_result=parsing_result,
    )

    for v in (v1, v2):
        assert v.received_image is image
        assert v.received_face is face
        assert v.received_parsing_result is parsing_result


# ------------------------------------------------------------------ #
# 5. No fail-fast
# ------------------------------------------------------------------ #


def test_does_not_stop_after_failed_metric():
    """A failed metric from an earlier validator must not prevent
    later validators from executing."""
    log: list[str] = []

    failing = _RecordingValidator(
        "failing",
        _metric(passed=False, score=0.0, message="fail"),
        log,
    )
    passing = _RecordingValidator(
        "passing",
        _metric(passed=True, score=1.0, message="ok"),
        log,
    )

    orchestrator = ValidationOrchestrator(
        validators=(failing, passing),
    )

    result = orchestrator.validate(
        image=np.zeros((1, 1, 3), dtype=np.uint8),
    )

    assert log == ["failing", "passing"]
    assert len(result.metrics) == 2
    assert result.metrics[0].passed is False
    assert result.metrics[1].passed is True


# ------------------------------------------------------------------ #
# 6. Exception propagation
# ------------------------------------------------------------------ #


def test_propagates_validator_exceptions():
    """Exceptions raised by a validator must propagate unhandled."""
    orchestrator = ValidationOrchestrator(
        validators=(_RaisingValidator("kaboom"),),
    )

    with pytest.raises(RuntimeError, match="kaboom"):
        orchestrator.validate(
            image=np.zeros((1, 1, 3), dtype=np.uint8),
        )


# ------------------------------------------------------------------ #
# 7. Default Validator Collection Verification
# ------------------------------------------------------------------ #


def test_validation_orchestrator_executes_default_validators():
    """Verify that ValidationOrchestrator successfully executes the default validator collection."""
    from unittest.mock import MagicMock, patch
    from insightface.app.common import Face

    from models.parsing.face_parsing_result import FaceParsingResult
    from services.eyewear_classifier import EyewearClassifier

    # Create dummy inputs that satisfy all validators
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    face = Face()
    face.bbox = np.array([10, 10, 50, 50], dtype=np.float32)
    face.pose = [0.0, 0.0, 0.0]

    parsing_result = MagicMock(spec=FaceParsingResult)
    parsing_result.has_part.return_value = True
    parsing_result.part_ratio.return_value = 0.5

    # Mock the eyewear classifier used by GlassesValidator
    mock_classifier = MagicMock(spec=EyewearClassifier)
    from models.eyewear_prediction import EyewearPrediction
    from models.eyewear_type import EyewearType
    mock_classifier.classify.return_value = EyewearPrediction(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )

    # Use patched constructor or factories to build actual default validators
    from pipeline.validator_factory import create_default_validators
    with patch("pipeline.validator_factory.GlassesDetectorClassifier", return_value=mock_classifier):
        orchestrator = ValidationOrchestrator()  # Loads defaults automatically

    # This should run through all validators (Blur, Brightness, Contrast, FaceSize,
    # HeadPose, Glasses, FaceVisibility, Occlusion) without any signature TypeError exceptions.
    result = orchestrator.validate(
        image=image,
        face=face,
        parsing_result=parsing_result,
    )

    assert isinstance(result, ValidationResult)
    assert len(result.metrics) > 0


def test_validation_orchestrator_short_circuits_on_stage1_failure():
    """If a Stage 1 default validator fails, execution stops immediately and FaceParserService is not called."""
    mock_parser = MagicMock()

    v1 = MagicMock()
    v1.validate.return_value = _metric(ValidationType.BLUR, passed=False, score=0.1)
    v2 = MagicMock()
    v3 = MagicMock()
    v4 = MagicMock()
    v5 = MagicMock()
    v6 = MagicMock()
    v7 = MagicMock()
    v8 = MagicMock()

    with patch("pipeline.validation_orchestrator.create_default_validators", return_value=(v1, v2, v3, v4, v5, v6, v7, v8)):
        orchestrator = ValidationOrchestrator(parser_service=mock_parser)
        result = orchestrator.validate(image=np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(result.metrics) == 1
    assert result.metrics[0].passed is False
    mock_parser.parse.assert_not_called()
    v2.validate.assert_not_called()

