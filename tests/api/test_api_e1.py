"""Phase E.1 — API correctness & event-loop non-blocking tests."""

from __future__ import annotations

import asyncio
import io
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from smartidphotoprocessor.api.app import app
from smartidphotoprocessor.api import routes


@pytest.fixture(autouse=True)
def reset_pipeline():
    """Reset pipeline singleton between tests."""
    routes._pipeline = None
    routes._pipeline_ready = False
    yield
    routes._pipeline = None
    routes._pipeline_ready = False


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_image_bytes():
    """Return minimal valid JPEG bytes for testing."""
    # Minimal JPEG: SOI + APP0 + DQT + SOF0 + DHT + SOS + EOI
    # Use a real small image from test_images
    path = "test_images/experiments/glasses.jpg"
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        pytest.skip("Test image not available")


# ---------------------------------------------------------------------------
# Part 7: API Correctness Tests
# ---------------------------------------------------------------------------

class TestAPIContractUnchanged:
    """Verify the API contract is unchanged after executor change."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_ready_returns_200_before_initialization(self, client):
        resp = client.get("/api/v1/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["models_loaded"] is False

    def test_validate_empty_upload_returns_400(self, client):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("empty.jpg", io.BytesIO(b""), "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_validate_no_file_returns_422(self, client):
        resp = client.post("/api/v1/validate")
        assert resp.status_code == 422

    def test_validate_unsupported_content_type_returns_400(self, client):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_validate_garbage_bytes_returns_400(self, client):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("bad.jpg", io.BytesIO(b"not valid jpeg bytes"), "image/jpeg")},
        )
        assert resp.status_code == 400

    def test_validate_real_image_returns_200(self, client, sample_image_bytes):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("photo.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "is_valid" in data
        assert "validation" in data
        assert "face" in data
        assert "export" in data
        assert "reverse_search" in data

    def test_validate_response_schema(self, client, sample_image_bytes):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("photo.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        data = resp.json()
        # Top-level fields
        assert isinstance(data["is_valid"], bool)
        v = data["validation"]
        assert isinstance(v["is_valid"], bool)
        assert isinstance(v["total_metrics"], int)
        assert isinstance(v["passed_count"], int)
        assert isinstance(v["failed_count"], int)
        assert isinstance(v["failed_metrics"], list)
        # Face info
        f = data["face"]
        assert isinstance(f["bbox"], list)
        assert len(f["bbox"]) == 4
        assert isinstance(f["detection_score"], float)
        assert f["gender"] in ("male", "female", None)

    def test_gender_is_normalized(self, client, sample_image_bytes):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("photo.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        data = resp.json()
        if data["face"] is not None:
            assert data["face"]["gender"] in ("male", "female")

    def test_no_embeddings_exposed(self, client, sample_image_bytes):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("photo.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        raw = resp.text
        assert "embedding" not in raw.lower()
        assert "normed_embedding" not in raw.lower()

    def test_reverse_search_disabled_by_default(self, client, sample_image_bytes):
        resp = client.post(
            "/api/v1/validate",
            files={"file": ("photo.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
        )
        data = resp.json()
        assert data["reverse_search"] is None


class TestConcurrentRequestIsolation:
    """Verify concurrent requests do not contaminate each other."""

    def test_concurrent_requests_same_result(self, client, sample_image_bytes):
        """Multiple concurrent requests should all get independent results."""
        results = []

        def do_request():
            resp = client.post(
                "/api/v1/validate",
                files={"file": ("photo.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")},
            )
            return resp.json()

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(do_request) for _ in range(4)]
            results = [f.result() for f in futures]

        for r in results:
            assert r["is_valid"] == results[0]["is_valid"]
            assert r["validation"]["total_metrics"] == results[0]["validation"]["total_metrics"]
            if r["face"] and results[0]["face"]:
                assert r["face"]["gender"] == results[0]["face"]["gender"]


# ---------------------------------------------------------------------------
# Part 8: Event Loop Non-Blocking Test
# ---------------------------------------------------------------------------

class TestEventLoopNonBlocking:
    """Verify that synchronous ML work is offloaded from the event loop."""

    def test_pipeline_runs_in_executor_not_on_event_loop(self):
        """The pipeline.validate call should execute in the executor thread pool,
        not on the asyncio event loop thread."""
        loop_thread = threading.current_thread()
        executor_thread = None

        original_get_pipeline = routes._get_pipeline

        def mock_get_pipeline():
            pipeline = original_get_pipeline()
            original_validate = pipeline.validate

            def tracking_validate(image):
                nonlocal executor_thread
                executor_thread = threading.current_thread()
                return original_validate(image)

            pipeline.validate = tracking_validate
            return pipeline

        routes._pipeline = None
        routes._pipeline_ready = False

        with patch.object(routes, "_get_pipeline", side_effect=mock_get_pipeline):
            client = TestClient(app)
            image_bytes = None
            try:
                with open("test_images/experiments/glasses.jpg", "rb") as f:
                    image_bytes = f.read()
            except FileNotFoundError:
                pytest.skip("Test image not available")

            resp = client.post(
                "/api/v1/validate",
                files={"file": ("photo.jpg", io.BytesIO(image_bytes), "image/jpeg")},
            )
            assert resp.status_code == 200
            assert executor_thread is not None
            assert executor_thread is not loop_thread, (
                "Pipeline should run in executor thread, not event loop thread"
            )

    def test_concurrent_requests_use_executor(self):
        """Multiple concurrent requests should be dispatched to executor threads."""
        executor_threads = set()
        lock = threading.Lock()

        original_get_pipeline = routes._get_pipeline

        def mock_get_pipeline():
            pipeline = original_get_pipeline()
            original_validate = pipeline.validate

            def tracking_validate(image):
                tid = threading.current_thread().name
                with lock:
                    executor_threads.add(tid)
                return original_validate(image)

            pipeline.validate = tracking_validate
            return pipeline

        routes._pipeline = None
        routes._pipeline_ready = False

        with patch.object(routes, "_get_pipeline", side_effect=mock_get_pipeline):
            client = TestClient(app)
            image_bytes = None
            try:
                with open("test_images/experiments/glasses.jpg", "rb") as f:
                    image_bytes = f.read()
            except FileNotFoundError:
                pytest.skip("Test image not available")

            def do_request():
                return client.post(
                    "/api/v1/validate",
                    files={"file": ("photo.jpg", io.BytesIO(image_bytes), "image/jpeg")},
                )

            with ThreadPoolExecutor(max_workers=4) as ex:
                futures = [ex.submit(do_request) for _ in range(4)]
                for f in futures:
                    assert f.result().status_code == 200

            assert len(executor_threads) >= 2, (
                f"Expected multiple executor threads, got: {executor_threads}"
            )
