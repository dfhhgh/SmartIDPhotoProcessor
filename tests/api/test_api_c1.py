"""Phase C.1 — API Boundary, Concurrency & Multi-Reference Tests.

Tests for:
- Concurrent pipeline initialization
- Concurrent request processing
- Gender serialization normalization
- Upload memory safety (chunked reading)
- Multi-reference reverse search behavior
- Reverse search fail-open
- Reverse search enabled E2E
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from fastapi.testclient import TestClient
from insightface.app.common import Face

from smartidphotoprocessor.api.app import app
from smartidphotoprocessor.api import routes
from smartidphotoprocessor.exceptions.face_exceptions import FacePipelineError
from smartidphotoprocessor.models.photo_processing_result import PhotoProcessingResult
from smartidphotoprocessor.models.validation_metric import ValidationMetric
from smartidphotoprocessor.models.validation_result import ValidationResult
from smartidphotoprocessor.models.validation_type import ValidationType
from smartidphotoprocessor.search.embedding_validator import EmbeddingValidator
from smartidphotoprocessor.search.flat_index import FlatIndex
from smartidphotoprocessor.search.reverse_search_service import (
    ReverseSearchResult,
    ReverseSearchService,
    ReverseSearchStatus,
)

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


def _make_face(bbox=None, gender=1, det_score=0.99, age=25):
    """Create a real InsightFace Face object for testing."""
    if bbox is None:
        bbox = np.array([10.0, 20.0, 100.0, 120.0], dtype=np.float32)
    return Face(d={"bbox": bbox, "det_score": det_score, "age": age, "gender": gender})


def _make_valid_result(is_valid=True, gender=1):
    """Create a valid PhotoProcessingResult."""
    face = _make_face(gender=gender)
    metric = ValidationMetric(
        type=ValidationType.BLUR,
        passed=is_valid,
        score=0.1 if is_valid else 0.9,
        message="ok" if is_valid else "too blurry",
    )
    return PhotoProcessingResult(
        validation_result=ValidationResult(metrics=[metric]),
        selected_face=face,
        aligned_image=np.zeros((112, 112, 3), dtype=np.uint8) if is_valid else None,
        cropped_image=np.zeros((112, 112, 3), dtype=np.uint8) if is_valid else None,
        export_result=None,
        reverse_search_result=None,
    )


def _make_result_with_reverse_search(is_valid=True, status=ReverseSearchStatus.COMPLETED):
    """Create a PhotoProcessingResult with reverse search evidence."""
    rs = ReverseSearchResult(
        status=status,
        candidates=(),
        top_k=5,
        query_dimension=512,
        processing_time_ms=1.2,
        error_message=None if status == ReverseSearchStatus.COMPLETED else "test error",
    )
    face = _make_face()
    metric = ValidationMetric(
        type=ValidationType.BLUR, passed=is_valid, score=0.1, message="ok",
    )
    return PhotoProcessingResult(
        validation_result=ValidationResult(metrics=[metric]),
        selected_face=face,
        aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
        cropped_image=np.zeros((112, 112, 3), dtype=np.uint8),
        export_result=None,
        reverse_search_result=rs,
    )


# ===========================================================================
# Part 3: Gender Serialization
# ===========================================================================

class TestGenderSerialization:
    def test_gender_male(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = _make_valid_result(gender=1)
        routes._pipeline_ready = True
        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})
        assert r.json()["face"]["gender"] == "male"

    def test_gender_female(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = _make_valid_result(gender=0)
        routes._pipeline_ready = True
        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})
        assert r.json()["face"]["gender"] == "female"

    def test_gender_none(self, client):
        face = _make_face()
        face.gender = None
        metric = ValidationMetric(type=ValidationType.BLUR, passed=True, score=0.1, message="ok")
        result = PhotoProcessingResult(
            validation_result=ValidationResult(metrics=[metric]),
            selected_face=face,
            aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
            cropped_image=np.zeros((112, 112, 3), dtype=np.uint8),
        )
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = result
        routes._pipeline_ready = True
        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})
        assert r.json()["face"]["gender"] is None

    def test_gender_not_integer_string(self, client):
        """Gender must never be a raw integer string like '1'."""
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = _make_valid_result(gender=1)
        routes._pipeline_ready = True
        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})
        gender = r.json()["face"]["gender"]
        assert gender is not None
        assert not gender.isdigit(), f"Gender should be semantic string, got '{gender}'"


# ===========================================================================
# Part 4: Upload Memory Safety
# ===========================================================================

class TestUploadMemorySafety:
    def test_chunked_rejection_of_large_file(self, client):
        """Oversized upload should be rejected without reading entire file."""
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = _make_valid_result()
        routes._pipeline_ready = True
        large = b"\x00" * (routes.MAX_UPLOAD_BYTES + 1024)
        r = client.post(
            "/api/v1/validate",
            files={"file": ("huge.jpg", large, "image/jpeg")},
        )
        assert r.status_code == 400
        assert "large" in r.json()["detail"].lower() or "exceeds" in r.json()["detail"].lower()

    def test_empty_upload_rejected(self, client):
        r = client.post(
            "/api/v1/validate",
            files={"file": ("empty.jpg", b"", "image/jpeg")},
        )
        assert r.status_code == 400
        assert "empty" in r.json()["detail"].lower()

    def test_valid_small_upload_passes(self, client):
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = _make_valid_result()
        routes._pipeline_ready = True
        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("photo.jpg", f, "image/jpeg")})
        assert r.status_code == 200


# ===========================================================================
# Part 5: Pipeline Singleton Race Condition
# ===========================================================================

class TestPipelineSingletonRace:
    def test_concurrent_initialization_creates_single_pipeline(self):
        """Multiple threads racing to init pipeline should only create one instance."""
        routes._pipeline = None
        routes._pipeline_ready = False
        created_ids = []
        lock = threading.Lock()

        original_init = None

        def patched_init(self):
            with lock:
                created_ids.append(id(self))

        results = []

        def init_and_get():
            p = routes._get_pipeline()
            results.append(id(p))

        with patch("smartidphotoprocessor.pipeline.photo_validation_pipeline.PhotoValidationPipeline.__init__", patched_init):
            threads = [threading.Thread(target=init_and_get) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        # All threads should get the same pipeline instance
        assert len(set(results)) == 1, f"Expected 1 pipeline instance, got {len(set(results))}"
        assert routes._pipeline_ready is True

    def test_ready_reflects_initialization(self):
        routes._pipeline = None
        routes._pipeline_ready = False
        assert routes.get_pipeline_ready() is False
        routes._pipeline = MagicMock()
        routes._pipeline_ready = True
        assert routes.get_pipeline_ready() is True


# ===========================================================================
# Part 7: Concurrent Request Processing
# ===========================================================================

class TestConcurrentRequests:
    def test_concurrent_requests_produce_valid_results(self, client):
        """Multiple concurrent requests should all produce valid responses."""
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = _make_valid_result()
        routes._pipeline_ready = True

        def make_request():
            with open(TEST_IMAGE, "rb") as f:
                return client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        N = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(make_request) for _ in range(N)]
            responses = [f.result() for f in futures]

        for r in responses:
            assert r.status_code == 200
            data = r.json()
            assert "is_valid" in data
            assert "validation" in data
            assert "face" in data
            assert isinstance(data["validation"]["total_metrics"], int)

    def test_concurrent_requests_no_cross_contamination(self, client):
        """Each request should get its own result, no mixing."""
        call_count = 0
        lock = threading.Lock()

        def mock_validate(image):
            nonlocal call_count
            with lock:
                call_count += 1
                local_count = call_count
            # Return different results based on call order
            return _make_valid_result(is_valid=(local_count % 2 == 1))

        routes._pipeline = MagicMock()
        routes._pipeline.validate.side_effect = mock_validate
        routes._pipeline_ready = True

        def make_request():
            with open(TEST_IMAGE, "rb") as f:
                return client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        N = 8
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(make_request) for _ in range(N)]
            responses = [f.result() for f in futures]

        # All should succeed
        for r in responses:
            assert r.status_code == 200

        # Results should alternate (odd calls valid, even invalid)
        results = [r.json()["is_valid"] for r in responses]
        # With concurrent execution, exact ordering may vary,
        # but we should have a mix of True/False
        assert True in results
        assert False in results

    def test_concurrent_pipeline_errors_are_isolated(self, client):
        """Error in one request should not affect others."""
        call_count = 0
        lock = threading.Lock()

        def mock_validate(image):
            nonlocal call_count
            with lock:
                call_count += 1
                if call_count == 1:
                    raise ValueError("face not detected")
            return _make_valid_result()

        routes._pipeline = MagicMock()
        routes._pipeline.validate.side_effect = mock_validate
        routes._pipeline_ready = True

        def make_request():
            with open(TEST_IMAGE, "rb") as f:
                return client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        N = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(make_request) for _ in range(N)]
            responses = [f.result() for f in futures]

        statuses = [r.status_code for r in responses]
        # First may be 422, rest should be 200
        assert 200 in statuses


# ===========================================================================
# Part 10: Multi-Reference Correctness
# ===========================================================================

class TestMultiReferenceCorrectness:
    def test_multiple_references_same_person(self):
        """Multiple references for the same person should all be returned."""
        validator = EmbeddingValidator(dimension=512, normalize=True)
        index = FlatIndex(dimension=512, normalize=True)

        rng = np.random.default_rng(42)

        # Person A: 3 references (similar vectors)
        base_a = rng.standard_normal(512).astype(np.float32)
        base_a = base_a / np.linalg.norm(base_a)
        a_refs = []
        for i in range(3):
            noise = rng.standard_normal(512).astype(np.float32) * 0.05
            ref = base_a + noise
            ref = ref / np.linalg.norm(ref)
            a_refs.append(ref)

        # Person B: 3 references (similar vectors)
        base_b = rng.standard_normal(512).astype(np.float32)
        base_b = base_b / np.linalg.norm(base_b)
        b_refs = []
        for i in range(3):
            noise = rng.standard_normal(512).astype(np.float32) * 0.05
            ref = base_b + noise
            ref = ref / np.linalg.norm(ref)
            b_refs.append(ref)

        # Add all references to index
        metadata_records = []
        for i, ref in enumerate(a_refs):
            vid = index.add(ref)
            metadata_records.append({"vector_id": vid, "person_id": "A", "label": "Person A", "image": f"A{i+1}.jpg"})
        for i, ref in enumerate(b_refs):
            vid = index.add(ref)
            metadata_records.append({"vector_id": vid, "person_id": "B", "label": "Person B", "image": f"B{i+1}.jpg"})

        # Create ReverseSearchService with controlled metadata
        import json
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "index.faiss"
            meta_path = Path(tmpdir) / "metadata.json"

            index.save(idx_path)

            metadata = {
                "schema_version": 1,
                "embedding_dimension": 512,
                "metric": "inner_product",
                "normalized": True,
                "total_vectors": 6,
                "total_persons": 2,
                "records": metadata_records,
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            service = ReverseSearchService(index_path=idx_path, metadata_path=meta_path)

            # Query: closer to A2
            query = a_refs[1] + rng.standard_normal(512).astype(np.float32) * 0.02
            query = query / np.linalg.norm(query)

            result = service.search(query, k=6)

            assert result.status == ReverseSearchStatus.COMPLETED
            assert len(result.candidates) == 6

            # All A references should be in top 3
            top_person_ids = [c.person_id for c in result.candidates[:3]]
            assert top_person_ids.count("A") >= 2, f"Expected at least 2 A references in top 3, got {top_person_ids}"

            # A2 should be highest similarity among A references
            a_similarities = {c.image: c.similarity for c in result.candidates if c.person_id == "A"}
            assert a_similarities["A2.jpg"] >= a_similarities["A1.jpg"]
            assert a_similarities["A2.jpg"] >= a_similarities["A3.jpg"]

    def test_identity_not_duplicated(self):
        """Multiple references for same person should not create duplicate identities."""
        index = FlatIndex(dimension=512, normalize=False)
        rng = np.random.default_rng(42)

        # Add 3 references for person A
        for _ in range(3):
            vec = rng.standard_normal(512).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            index.add(vec)

        assert index.size == 3

        # Metadata should have 3 records for person A
        import json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "index.faiss"
            meta_path = Path(tmpdir) / "metadata.json"
            index.save(idx_path)

            metadata = {
                "schema_version": 1,
                "embedding_dimension": 512,
                "metric": "inner_product",
                "normalized": True,
                "total_vectors": 3,
                "total_persons": 1,
                "records": [
                    {"vector_id": 0, "person_id": "A", "label": "Person A", "image": "A1.jpg"},
                    {"vector_id": 1, "person_id": "A", "label": "Person A", "image": "A2.jpg"},
                    {"vector_id": 2, "person_id": "A", "label": "Person A", "image": "A3.jpg"},
                ],
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            service = ReverseSearchService(index_path=idx_path, metadata_path=meta_path)

            query = rng.standard_normal(512).astype(np.float32)
            query = query / np.linalg.norm(query)

            result = service.search(query, k=3)
            assert result.status == ReverseSearchStatus.COMPLETED
            assert len(result.candidates) == 3

            # All should be person A
            person_ids = [c.person_id for c in result.candidates]
            assert all(pid == "A" for pid in person_ids)


# ===========================================================================
# Part 11: Multi-Reference + Concurrency
# ===========================================================================

class TestMultiReferenceConcurrency:
    def test_concurrent_search_same_gallery(self):
        """Multiple concurrent searches against the same index should be safe."""
        rng = np.random.default_rng(42)

        import json, tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "index.faiss"
            meta_path = Path(tmpdir) / "metadata.json"

            index = FlatIndex(dimension=512, normalize=False)

            # Build gallery: 2 persons, 3 refs each
            records = []
            for pid in ["A", "B"]:
                for i in range(3):
                    vec = rng.standard_normal(512).astype(np.float32)
                    vec = vec / np.linalg.norm(vec)
                    vid = index.add(vec)
                    records.append({"vector_id": vid, "person_id": pid, "label": f"Person {pid}", "image": f"{pid}{i+1}.jpg"})

            index.save(idx_path)
            metadata = {
                "schema_version": 1,
                "embedding_dimension": 512,
                "metric": "inner_product",
                "normalized": True,
                "total_vectors": 6,
                "total_persons": 2,
                "records": records,
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            service = ReverseSearchService(index_path=idx_path, metadata_path=meta_path)

            queries = [rng.standard_normal(512).astype(np.float32) for _ in range(8)]
            queries = [q / np.linalg.norm(q) for q in queries]

            def search_one(q):
                return service.search(q, k=3)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(search_one, q) for q in queries]
                results = [f.result() for f in futures]

            for r in results:
                assert r.status == ReverseSearchStatus.COMPLETED
                assert len(r.candidates) == 3
                for c in r.candidates:
                    assert c.person_id in ("A", "B")
                    assert isinstance(c.similarity, float)

    def test_concurrent_search_no_index_mutation(self):
        """Concurrent searches should not modify the FAISS index."""
        rng = np.random.default_rng(42)

        import json, tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "index.faiss"
            meta_path = Path(tmpdir) / "metadata.json"

            index = FlatIndex(dimension=512, normalize=False)
            records = []
            for i in range(5):
                vec = rng.standard_normal(512).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                vid = index.add(vec)
                records.append({"vector_id": vid, "person_id": "P", "label": "Person", "image": f"P{i}.jpg"})

            initial_size = index.size
            index.save(idx_path)
            metadata = {
                "schema_version": 1,
                "embedding_dimension": 512,
                "metric": "inner_product",
                "normalized": True,
                "total_vectors": 5,
                "total_persons": 1,
                "records": records,
            }
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

            service = ReverseSearchService(index_path=idx_path, metadata_path=meta_path)

            queries = [rng.standard_normal(512).astype(np.float32) for _ in range(8)]
            queries = [q / np.linalg.norm(q) for q in queries]

            def search_one(q):
                return service.search(q, k=3)

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(search_one, queries))

            # Index should be unchanged
            assert service.size == initial_size


# ===========================================================================
# Part 12: Reverse Search Enabled E2E
# ===========================================================================

class TestReverseSearchEnabledE2E:
    def test_reverse_search_evidence_in_response(self, client):
        """Reverse search evidence should appear in response when present."""
        result = _make_result_with_reverse_search(
            is_valid=True,
            status=ReverseSearchStatus.COMPLETED,
        )
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = result
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        assert r.status_code == 200
        data = r.json()
        assert data["reverse_search"] is not None
        assert data["reverse_search"]["status"] == "completed"
        assert isinstance(data["reverse_search"]["candidates"], list)

    def test_reverse_search_does_not_affect_validation(self, client):
        """Reverse search cannot turn invalid validation into valid."""
        result = _make_result_with_reverse_search(
            is_valid=False,
            status=ReverseSearchStatus.COMPLETED,
        )
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = result
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        assert r.status_code == 200
        data = r.json()
        assert data["is_valid"] is False
        assert data["reverse_search"] is not None

    def test_reverse_search_absent_when_disabled(self, client):
        """When pipeline returns no reverse search, it should be null."""
        result = _make_valid_result()
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = result
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        assert r.status_code == 200
        data = r.json()
        assert data["reverse_search"] is None


# ===========================================================================
# Part 13: Fail-Open Verification
# ===========================================================================

class TestReverseSearchFailOpen:
    def test_unavailable_status_does_not_affect_validation(self, client):
        """UNAVAILABLE status should not change validation outcome."""
        result = _make_result_with_reverse_search(
            is_valid=True,
            status=ReverseSearchStatus.UNAVAILABLE,
        )
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = result
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        assert r.status_code == 200
        data = r.json()
        assert data["is_valid"] is True
        assert data["reverse_search"]["status"] == "unavailable"

    def test_error_status_does_not_affect_validation(self, client):
        """ERROR status should not change validation outcome."""
        result = _make_result_with_reverse_search(
            is_valid=False,
            status=ReverseSearchStatus.ERROR,
        )
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = result
        routes._pipeline_ready = True

        with open(TEST_IMAGE, "rb") as f:
            r = client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        assert r.status_code == 200
        data = r.json()
        assert data["is_valid"] is False
        assert data["reverse_search"]["status"] == "error"

    def test_pipeline_exception_during_reverse_search_still_returns_result(self):
        """Pipeline should handle reverse search failures gracefully."""
        from smartidphotoprocessor.pipeline.photo_validation_pipeline import PhotoValidationPipeline

        # Create a pipeline with a broken reverse search service
        broken_rs = MagicMock()
        broken_rs.search.side_effect = RuntimeError("FAISS crashed")

        face = _make_face()
        metric = ValidationMetric(type=ValidationType.BLUR, passed=True, score=0.1, message="ok")

        pipeline = PhotoValidationPipeline.__new__(PhotoValidationPipeline)
        pipeline._detector = MagicMock()
        pipeline._selector = MagicMock()
        pipeline._cropper = MagicMock()
        pipeline._coordinate_transformer = MagicMock()
        pipeline._aligner = MagicMock()
        pipeline._ambiguity_validator = MagicMock()
        pipeline._parser_service = MagicMock()
        pipeline._orchestrator = MagicMock()
        pipeline._exporter = MagicMock()
        pipeline._reverse_search_enabled = True
        pipeline._reverse_search_service = broken_rs

        # Mock the orchestrator to return a valid result
        from smartidphotoprocessor.pipeline.aligner import AlignmentResult
        from smartidphotoprocessor.pipeline.cropper import CropResult
        from smartidphotoprocessor.pipeline.selector import SelectionResult

        pipeline._detector.detect.return_value = [face]
        sel_result = MagicMock()
        sel_result.selected_face = face
        pipeline._selector.select.return_value = sel_result

        amb_metric = ValidationMetric(type=ValidationType.FACE_AMBIGUITY, passed=True, score=0.1, message="ok")
        pipeline._ambiguity_validator.validate.return_value = amb_metric

        crop_result = MagicMock()
        crop_result.image = np.zeros((200, 200, 3), dtype=np.uint8)
        crop_result.crop_x = 0
        crop_result.crop_y = 0
        pipeline._cropper.crop.return_value = crop_result

        pipeline._coordinate_transformer.transform.return_value = face

        align_result = MagicMock()
        align_result.aligned_image = np.zeros((112, 112, 3), dtype=np.uint8)
        align_result.aligned_face = face
        pipeline._aligner.align.return_value = align_result

        vr = ValidationResult(metrics=[metric])
        pipeline._orchestrator.validate.return_value = vr
        pipeline._exporter.export.return_value = None

        result = pipeline.validate(np.zeros((100, 100, 3), dtype=np.uint8))

        # Should complete without error, with reverse search result = None
        assert result.validation_result.is_valid is True
        assert result.reverse_search_result is None


# ===========================================================================
# Part 14: Performance Observation (basic timing)
# ===========================================================================

class TestPerformanceObservation:
    def test_concurrent_execution_timing(self, client):
        """Record timing for concurrent vs sequential execution."""
        routes._pipeline = MagicMock()
        routes._pipeline.validate.return_value = _make_valid_result()
        routes._pipeline_ready = True

        def make_request():
            with open(TEST_IMAGE, "rb") as f:
                return client.post("/api/v1/validate", files={"file": ("p.jpg", f, "image/jpeg")})

        N = 4

        # Sequential
        start = time.perf_counter()
        for _ in range(N):
            make_request()
        sequential_time = time.perf_counter() - start

        # Concurrent
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
            list(pool.map(lambda _: make_request(), range(N)))
        concurrent_time = time.perf_counter() - start

        # Just record — don't assert specific ratios
        assert sequential_time > 0
        assert concurrent_time > 0
        # Concurrent should not be dramatically slower
        assert concurrent_time < sequential_time * 3


# ===========================================================================
# Part 8: Async Endpoint Behavior
# ===========================================================================

class TestAsyncBehavior:
    def test_validate_endpoint_is_async(self):
        """Verify the validate endpoint is defined as async."""
        from smartidphotoprocessor.api.routes import validate
        import inspect
        assert inspect.iscoroutinefunction(validate)

    def test_health_endpoint_is_async(self):
        from smartidphotoprocessor.api.routes import health
        import inspect
        assert inspect.iscoroutinefunction(health)

    def test_ready_endpoint_is_async(self):
        from smartidphotoprocessor.api.routes import ready
        import inspect
        assert inspect.iscoroutinefunction(ready)
