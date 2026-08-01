"""Tests for scripts/check_models.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from scripts.check_models import (
    check_insightface,
    check_face_parser,
    check_glasses_detector,
)


def test_check_insightface_success():
    """Verify check_insightface succeeds when FaceService loads successfully."""
    with patch("scripts.check_models.FaceService") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.use_gpu = True
        mock_instance.gpu_id = 0
        mock_instance.get_model.return_value = MagicMock()

        success, device, elapsed = check_insightface()
        assert success is True
        assert device == "GPU"
        assert isinstance(elapsed, float)


def test_check_insightface_failure():
    """Verify check_insightface handles exceptions correctly."""
    with patch("scripts.check_models.FaceService") as mock_cls:
        mock_cls.side_effect = RuntimeError("Failed to load")

        success, error, elapsed = check_insightface()
        assert success is False
        assert "Failed to load" in error
        assert isinstance(elapsed, float)


def test_check_face_parser_success():
    """Verify check_face_parser succeeds when FaceParserService loads session successfully."""
    with patch("scripts.check_models.FaceParserService") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_session = MagicMock()
        mock_session.get_providers.return_value = ["CUDAExecutionProvider"]
        mock_instance._ensure_loaded.return_value = mock_session

        success, device, elapsed = check_face_parser()
        assert success is True
        assert device == "GPU"
        assert isinstance(elapsed, float)


def test_check_face_parser_failure():
    """Verify check_face_parser handles exceptions correctly."""
    with patch("scripts.check_models.FaceParserService") as mock_cls:
        mock_cls.side_effect = Exception("Model not found")

        success, error, elapsed = check_face_parser()
        assert success is False
        assert "Model not found" in error
        assert isinstance(elapsed, float)


def test_check_glasses_detector_success():
    """Verify check_glasses_detector succeeds when GlassesDetectorClassifier loads successfully."""
    with patch("scripts.check_models.GlassesDetectorClassifier") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance._device = "cpu"
        mock_instance._ensure_loaded.return_value = (MagicMock(), MagicMock())

        success, device, elapsed = check_glasses_detector()
        assert success is True
        assert device == "CPU"
        assert isinstance(elapsed, float)


def test_check_glasses_detector_failure():
    """Verify check_glasses_detector handles exceptions correctly."""
    with patch("scripts.check_models.GlassesDetectorClassifier") as mock_cls:
        mock_cls.side_effect = Exception("Init failed")

        success, error, elapsed = check_glasses_detector()
        assert success is False
        assert "Init failed" in error
        assert isinstance(elapsed, float)
