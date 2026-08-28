"""Tests for ValidationOrchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.validation_metric import ValidationMetric
from models.validation_result import ValidationResult
from models.validation_stage import ValidationStage
from models.validation_type import ValidationType
from pipeline.validation_orchestrator import ValidationOrchestrator


def _metric(
    v_type: ValidationType,
    passed: bool = True,
    score: float = 1.0,
) -> ValidationMetric:
    """Helper to create a validation metric."""
    return ValidationMetric(
        type=v_type,
        passed=passed,
        score=score,
        message="Test message.",
    )


def test_validation_orchestrator_initialization_defaults():
    """Verify default initialization creates built-in validators and parser service."""
    orchestrator = ValidationOrchestrator()
    assert len(orchestrator._validators) == 7


def test_validation_orchestrator_custom_validators():
    """Verify custom validators are executed in order."""
    v1 = MagicMock()
    v1.validate.return_value = _metric(ValidationType.BLUR)
    v2 = MagicMock()
    v2.validate.return_value = _metric(ValidationType.BRIGHTNESS)

    orchestrator = ValidationOrchestrator(validators=(v1, v2))
    result = orchestrator.validate(image=np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(result.metrics) == 2
    v1.validate.assert_called_once()
    v2.validate.assert_called_once()


def test_validation_orchestrator_short_circuits_on_stage1_failure():
    """If a Stage 1 default validator fails, execution stops immediately after Stage CHEAP and FaceParserService is not called."""
    mock_parser = MagicMock()

    v1 = MagicMock()
    v1.stage = ValidationStage.CHEAP
    v1.validate.return_value = _metric(ValidationType.BLUR, passed=False, score=0.1)
    v2 = MagicMock()
    v2.stage = ValidationStage.CHEAP
    v2.validate.return_value = _metric(ValidationType.BRIGHTNESS)
    v3 = MagicMock()
    v3.stage = ValidationStage.CHEAP
    v3.validate.return_value = _metric(ValidationType.CONTRAST)
    v4 = MagicMock()
    v4.stage = ValidationStage.CHEAP
    v4.validate.return_value = _metric(ValidationType.FACE_SIZE)
    v5 = MagicMock()
    v5.stage = ValidationStage.CHEAP
    v5.validate.return_value = _metric(ValidationType.HEAD_POSE)
    v6 = MagicMock()
    v6.stage = ValidationStage.PARSING
    v6.validate.return_value = _metric(ValidationType.FACE_VISIBILITY)
    v7 = MagicMock()
    v7.stage = ValidationStage.PARSING
    v7.validate.return_value = _metric(ValidationType.OCCLUSION)

    with patch("pipeline.validation_orchestrator.create_default_validators", return_value=(v1, v2, v3, v4, v5, v6, v7)):
        orchestrator = ValidationOrchestrator(parser_service=mock_parser)
        result = orchestrator.validate(image=np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(result.metrics) == 5
    assert result.metrics[0].passed is False
    mock_parser.parse.assert_not_called()
    v6.validate.assert_not_called()
    v7.validate.assert_not_called()


def test_orchestrator_routes_crop_data_to_face_size_validator():
    """FaceSizeValidator receives crop_image and crop_face when provided."""
    mock_parser = MagicMock()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    crop_img = np.zeros((200, 150, 3), dtype=np.uint8)

    v1 = MagicMock()
    v1.stage = ValidationStage.CHEAP
    v1.validate.return_value = _metric(ValidationType.BLUR)
    v2 = MagicMock(spec=["stage", "validate"])
    v2.stage = ValidationStage.CHEAP
    # Make it pass isinstance check for FaceSizeValidator
    from validators.face_size_validator import FaceSizeValidator
    v2.__class__ = FaceSizeValidator
    v2.validate.return_value = _metric(ValidationType.FACE_SIZE)

    with patch("pipeline.validation_orchestrator.create_default_validators", return_value=(v1, v2)):
        orchestrator = ValidationOrchestrator(parser_service=mock_parser)
        result = orchestrator.validate(
            image=img,
            crop_image=crop_img,
            crop_face=MagicMock(),
        )

    assert len(result.metrics) == 2
    v2.validate.assert_called_once()
    call_kwargs = v2.validate.call_args
    assert call_kwargs.kwargs.get("crop_image") is crop_img
    assert call_kwargs.kwargs.get("crop_face") is not None


def test_orchestrator_does_not_pass_crop_to_non_face_size_validators():
    """Non-FaceSizeValidator validators must NOT receive crop_image/crop_face."""
    mock_parser = MagicMock()
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    crop_img = np.zeros((200, 150, 3), dtype=np.uint8)

    v1 = MagicMock()
    v1.stage = ValidationStage.CHEAP
    v1.validate.return_value = _metric(ValidationType.BLUR)

    with patch("pipeline.validation_orchestrator.create_default_validators", return_value=(v1,)):
        orchestrator = ValidationOrchestrator(parser_service=mock_parser)
        result = orchestrator.validate(
            image=img,
            crop_image=crop_img,
            crop_face=MagicMock(),
        )

    assert len(result.metrics) == 1
    v1.validate.assert_called_once()
    call_kwargs = v1.validate.call_args
    assert "crop_image" not in call_kwargs.kwargs
    assert "crop_face" not in call_kwargs.kwargs
