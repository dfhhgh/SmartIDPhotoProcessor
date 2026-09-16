"""Tests for the SmartIDPhotoProcessor API endpoints."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from insightface.app.common import Face

from smartidphotoprocessor.api.app import app
from smartidphotoprocessor.api import routes
from smartidphotoprocessor.models.photo_processing_result import PhotoProcessingResult
from smartidphotoprocessor.models.validation_result import ValidationResult
from smartidphotoprocessor.models.validation_metric import ValidationMetric
from smartidphotoprocessor.models.validation_type import ValidationType

TEST_IMAGE = Path(__file__).resolve().parents[2] / "test_images" / "experiments" / "09e1122b113e4b2ce108f45d622915ae.jpg"


@pytest.fixture(autouse=True)
def _reset_pipeline():
    """Reset the singleton pipeline before each test."""
    routes._pipeline = None
    routes._pipeline_ready = False
    yield
    routes._pipeline = None
    routes._pipeline_ready = False


@pytest.fixture()
def client():
    """Create a fresh TestClient."""
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200

    def test_status_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.json()["status"] == "ok"

    def test_no_model_check(self, client):
        r = client.get("/api/v1/health")
        data = r.json()
        assert "models_loaded" not in data


# ---------------------------------------------------------------------------
# Ready
# ---------------------------------------------------------------------------

class TestReady:
    def test_not_ready_by_default(self, client):
        r = client.get("/api/v1/ready")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "not_ready"
        assert data["models_loaded"] is False

    def test_ready_after_pipeline_init(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline_ready = True
        r = client.get("/api/v1/ready")
        data = r.json()
        assert data["status"] == "ready"
        assert data["models_loaded"] is True


# ---------------------------------------------------------------------------
# Validate — request validation
# ---------------------------------------------------------------------------

class TestValidateRequest:
    def test_no_file_returns_422(self, client):
        r = client.post("/api/v1/validate")
        assert r.status_code == 422

    def test_empty_filename_returns_422(self, client):
        r = client.post(
            "/api/v1/validate",
            files={"file": ("", b"some bytes", "image/jpeg")},
        )
        assert r.status_code == 422

    def test_empty_upload_returns_400(self, client):
        r = client.post(
            "/api/v1/validate",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()

    def test_invalid_bytes_returns_400(self, client):
        r = client.post(
            "/api/v1/validate",
            files={"file": ("bad.jpg", b"not image data", "image/jpeg")},
        )
        assert r.status_code == 400
        assert "decode" in r.json()["detail"].lower() or "corrupt" in r.json()["detail"].lower()

    def test_unsupported_content_type_returns_400(self, client):
        r = client.post(
            "/api/v1/validate",
            files={"file": ("file.pdf", b"some data", "application/pdf")},
        )
        assert r.status_code == 400
        assert "content type" in r.json()["detail"].lower()

    def test_oversized_upload_returns_400(self, client):
        large = b"\x00" * (routes.MAX_UPLOAD_BYTES + 1)
        r = client.post(
            "/api/v1/validate",
            files={"file": ("huge.jpg", large, "image/jpeg")},
        )
        assert r.status_code == 400
        assert "large" in r.json()["detail"].lower() or "too" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Validate — mock pipeline
# ---------------------------------------------------------------------------

class TestValidateMockPipeline:
    def _mock_result(self, is_valid=True):
        face = Face(d={
            "bbox": np.array([10.0, 20.0, 100.0, 120.0], dtype=np.float32),
            "det_score": 0.99,
            "age": 25,
            "gender": 1,
        })

        metric = ValidationMetric(
            type=ValidationType.BLUR,
            passed=is_valid,
            score=0.1 if is_valid else 0.9,
            message="ok" if is_valid else "too blurry",
        )
        vr = ValidationResult(metrics=[metric])

        return PhotoProcessingResult(
            validation_result=vr,
            selected_face=face,
            aligned_image=np.zeros((112, 112, 3), dtype=np.uint8) if is_valid else None,
            cropped_image=np.zeros((112, 112, 3), dtype=np.uint8) if is_valid else None,
            export_result=None,
            reverse_search_result=None,
        )

    def test_valid_image_returns_200(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = self._mock_result(is_valid=True)
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post(
                "/api/v1/validate",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["is_valid"] is True
        assert data["validation"]["is_valid"] is True
        assert data["face"] is not None
        assert "bbox" in data["face"]

    def test_invalid_image_returns_200_with_is_valid_false(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = self._mock_result(is_valid=False)
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post(
                "/api/v1/validate",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["is_valid"] is False
        assert len(data["validation"]["failed_metrics"]) > 0

    def test_no_raw_images_exposed(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = self._mock_result(is_valid=True)
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post(
                "/api/v1/validate",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )
        data = r.json()
        # Should not contain raw image arrays or embeddings
        assert "aligned_image" not in data
        assert "cropped_image" not in data
        assert "embedding" not in data.get("face", {})

    def test_pipeline_error_returns_422(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.side_effect = ValueError("no face found")
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post(
                "/api/v1/validate",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )
        assert r.status_code == 422
        assert "face found" in r.json()["detail"]

    def test_unexpected_error_returns_500(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.side_effect = RuntimeError("model crashed")
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post(
                "/api/v1/validate",
                files={"file": ("photo.jpg", f, "image/jpeg")},
            )
        assert r.status_code == 500
        assert "internal" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Validate — real end-to-end
# ---------------------------------------------------------------------------

class TestValidateEndToEnd:
    @pytest.mark.skipif(
        not TEST_IMAGE.exists(),
        reason="Test image not available",
    )
    def test_real_image_pipeline(self, client):
        """Full end-to-end: HTTP -> FastAPI -> decode -> pipeline -> JSON."""
        r = client.post(
            "/api/v1/validate",
            files={"file": ("photo.jpg", TEST_IMAGE.read_bytes(), "image/jpeg")},
        )
        assert r.status_code == 200
        data = r.json()
        assert "is_valid" in data
        assert "validation" in data
        assert "face" in data
        assert isinstance(data["validation"]["total_metrics"], int)
        assert data["validation"]["total_metrics"] >= 1


# ---------------------------------------------------------------------------
# OpenAPI
# ---------------------------------------------------------------------------

class TestOpenAPI:
    def test_docs_endpoint(self, client):
        r = client.get("/docs")
        assert r.status_code == 200

    def test_openapi_json(self, client):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        data = r.json()
        assert data["openapi"] == "3.1.0"
        assert data["info"]["title"] == "SmartIDPhotoProcessor"
        assert "/api/v1/validate" in data["paths"]
        assert "/api/v1/health" in data["paths"]
        assert "/api/v1/ready" in data["paths"]
