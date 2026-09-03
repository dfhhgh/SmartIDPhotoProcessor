"""Tests for search.reverse_search_service — Runtime Reverse Search service.

Covers:
- Successful initialization & artifact compatibility checks
- Missing / corrupted artifact handling (IndexError, ArtifactError)
- Search correctness, ranking, and NumPy agreement
- Top-K clamping and validation
- Embedding validation contract enforcement
- Empty index behavior (ReverseSearchUnavailableError)
- Multiple reference images per person (no person-level aggregation)
- State isolation & error robustness (invalid query does not corrupt state)
- Determinism across repeated searches
- Read-only thread safety under concurrent queries
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

from search.embedding_validator import EmbeddingError
from search.flat_index import FlatIndex
from search.index_builder import IndexBuilder
from search.reverse_search_service import (
    ArtifactError,
    CandidateMatch,
    ReverseSearchResult,
    ReverseSearchService,
    ReverseSearchStatus,
    ReverseSearchUnavailableError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_vector(dim: int = 512) -> np.ndarray:
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _create_mock_artifacts(
    tmp_path: Path,
    vectors: list[np.ndarray],
    records_meta: list[dict[str, Any]],
    dimension: int = 512,
    schema_version: int = 1,
    metric: str = "inner_product",
    normalized: bool = True,
) -> tuple[Path, Path]:
    """Helper to write custom FAISS index and metadata for error testing."""
    index_path = tmp_path / "reference_index.faiss"
    metadata_path = tmp_path / "metadata.json"

    idx = FlatIndex(dimension=dimension, normalize=False)
    if vectors:
        idx.add(np.vstack(vectors))
    idx.save(index_path)

    metadata = {
        "schema_version": schema_version,
        "embedding_dimension": dimension,
        "metric": metric,
        "normalized": normalized,
        "total_vectors": idx.size,
        "total_persons": len(set(r["person_id"] for r in records_meta)) if records_meta else 0,
        "records": records_meta,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return index_path, metadata_path


# ---------------------------------------------------------------------------
# Initialization & Artifact Validation Tests
# ---------------------------------------------------------------------------

class TestReverseSearchServiceInit:
    def test_successful_init(self, tmp_path: Path) -> None:
        v1 = _make_valid_vector()
        v2 = _make_valid_vector()
        records = [
            {"vector_id": 0, "person_id": "alice", "label": "alice", "image": "alice/1.jpg"},
            {"vector_id": 1, "person_id": "bob", "label": "bob", "image": "bob/1.jpg"},
        ]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v1, v2], records)

        svc = ReverseSearchService(idx_path, meta_path)
        assert svc.size == 2
        assert not svc.is_empty
        assert svc.dimension == 512

    def test_missing_index_raises_artifact_error(self, tmp_path: Path) -> None:
        idx_path = tmp_path / "missing.faiss"
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text("{}", encoding="utf-8")

        with pytest.raises(ArtifactError, match="FAISS index file not found"):
            ReverseSearchService(idx_path, meta_path)

    def test_missing_metadata_raises_artifact_error(self, tmp_path: Path) -> None:
        idx_path = tmp_path / "index.faiss"
        idx = FlatIndex(512)
        idx.save(idx_path)
        meta_path = tmp_path / "missing.json"

        with pytest.raises(ArtifactError, match="Metadata file not found"):
            ReverseSearchService(idx_path, meta_path)

    def test_invalid_json_raises_artifact_error(self, tmp_path: Path) -> None:
        idx_path = tmp_path / "index.faiss"
        idx = FlatIndex(512)
        idx.save(idx_path)
        meta_path = tmp_path / "metadata.json"
        meta_path.write_text("NOT JSON {{", encoding="utf-8")

        with pytest.raises(ArtifactError, match="invalid JSON"):
            ReverseSearchService(idx_path, meta_path)

    def test_missing_top_level_field_raises(self, tmp_path: Path) -> None:
        idx_path = tmp_path / "index.faiss"
        idx = FlatIndex(512)
        idx.save(idx_path)
        meta_path = tmp_path / "metadata.json"
        metadata = {"schema_version": 1}  # Missing embedding_dimension, etc.
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(ArtifactError, match="missing required top-level fields"):
            ReverseSearchService(idx_path, meta_path)

    def test_unsupported_schema_version_raises(self, tmp_path: Path) -> None:
        v = _make_valid_vector()
        records = [{"vector_id": 0, "person_id": "p", "label": "p", "image": "p/1.jpg"}]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v], records, schema_version=99)

        with pytest.raises(ArtifactError, match="Unsupported metadata schema version"):
            ReverseSearchService(idx_path, meta_path)

    def test_dimension_mismatch_raises(self, tmp_path: Path) -> None:
        v = _make_valid_vector(dim=512)
        records = [{"vector_id": 0, "person_id": "p", "label": "p", "image": "p/1.jpg"}]
        # Metadata declares dim=256, but index has 512
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v], records, dimension=512)
        # Manually tamper metadata dimension
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["embedding_dimension"] = 256
        meta_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ArtifactError, match="Embedding dimension mismatch"):
            ReverseSearchService(idx_path, meta_path)

    def test_metric_mismatch_raises(self, tmp_path: Path) -> None:
        v = _make_valid_vector()
        records = [{"vector_id": 0, "person_id": "p", "label": "p", "image": "p/1.jpg"}]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v], records, metric="L2")

        with pytest.raises(ArtifactError, match="Unsupported metric"):
            ReverseSearchService(idx_path, meta_path)

    def test_normalized_false_raises(self, tmp_path: Path) -> None:
        v = _make_valid_vector()
        records = [{"vector_id": 0, "person_id": "p", "label": "p", "image": "p/1.jpg"}]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v], records, normalized=False)

        with pytest.raises(ArtifactError, match="normalized.*must be True"):
            ReverseSearchService(idx_path, meta_path)

    def test_total_vectors_mismatch_raises(self, tmp_path: Path) -> None:
        v = _make_valid_vector()
        records = [{"vector_id": 0, "person_id": "p", "label": "p", "image": "p/1.jpg"}]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v], records)
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["total_vectors"] = 999
        meta_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ArtifactError, match="Total vectors mismatch"):
            ReverseSearchService(idx_path, meta_path)

    def test_records_count_mismatch_raises(self, tmp_path: Path) -> None:
        v = _make_valid_vector()
        records = [
            {"vector_id": 0, "person_id": "p1", "label": "p1", "image": "p1/1.jpg"},
            {"vector_id": 1, "person_id": "p2", "label": "p2", "image": "p2/1.jpg"},
        ]
        # Index has 1 vector, metadata has 2 records
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v], records)
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["total_vectors"] = 1  # wait, index.size is 1, but len(records) is 2
        meta_path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ArtifactError, match="Records count mismatch"):
            ReverseSearchService(idx_path, meta_path)

    def test_duplicate_vector_id_raises(self, tmp_path: Path) -> None:
        v1 = _make_valid_vector()
        v2 = _make_valid_vector()
        records = [
            {"vector_id": 0, "person_id": "p1", "label": "p1", "image": "p1/1.jpg"},
            {"vector_id": 0, "person_id": "p2", "label": "p2", "image": "p2/1.jpg"},  # duplicate ID 0
        ]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v1, v2], records)

        with pytest.raises(ArtifactError, match="Duplicate vector_id"):
            ReverseSearchService(idx_path, meta_path)

    def test_missing_vector_id_raises(self, tmp_path: Path) -> None:
        v1 = _make_valid_vector()
        v2 = _make_valid_vector()
        records = [
            {"vector_id": 0, "person_id": "p1", "label": "p1", "image": "p1/1.jpg"},
            {"vector_id": 5, "person_id": "p2", "label": "p2", "image": "p2/1.jpg"},  # ID 5 instead of 1
        ]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v1, v2], records)

        with pytest.raises(ArtifactError, match="Vector ID coverage mismatch"):
            ReverseSearchService(idx_path, meta_path)


# ---------------------------------------------------------------------------
# Search Functionality Tests
# ---------------------------------------------------------------------------

class TestReverseSearchServiceSearch:
    @pytest.fixture
    def populated_service(self, tmp_path: Path) -> tuple[ReverseSearchService, list[np.ndarray]]:
        rng = np.random.RandomState(42)
        vectors = []
        records = []
        for i in range(5):
            v = rng.randn(512).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors.append(v)
            records.append({
                "vector_id": i,
                "person_id": f"person_{i:03d}",
                "label": f"label_{i}",
                "image": f"person_{i:03d}/img.jpg",
            })

        idx_path, meta_path = _create_mock_artifacts(tmp_path, vectors, records)
        svc = ReverseSearchService(idx_path, meta_path)
        return svc, vectors

    def test_exact_match_query_returns_self_first(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        query = vectors[2]  # Query with vector 2
        result = svc.search(query, k=1)

        assert isinstance(result, ReverseSearchResult)
        assert len(result.candidates) == 1
        top = result.candidates[0]
        assert top.vector_id == 2
        assert top.person_id == "person_002"
        assert top.similarity > 0.999

    def test_ranking_preserves_numpy_cosine_similarity(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        db_matrix = np.vstack(vectors)
        query = _make_valid_vector()

        result = svc.search(query, k=5)

        # Compute NumPy reference similarity
        numpy_sims = query @ db_matrix.T
        numpy_ranking = np.argsort(-numpy_sims)

        faiss_ids = [c.vector_id for c in result.candidates]
        assert faiss_ids == list(numpy_ranking)

        # Check similarity values agree
        for c in result.candidates:
            expected_sim = numpy_sims[c.vector_id]
            assert abs(c.similarity - expected_sim) < 1e-5

    def test_top_k_default_and_custom(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, _ = populated_service
        query = _make_valid_vector()

        res_default = svc.search(query)
        assert res_default.top_k == 5
        assert len(res_default.candidates) == 5

        res_custom = svc.search(query, k=2)
        assert res_custom.top_k == 2
        assert len(res_custom.candidates) == 2

    def test_k_larger_than_index_clamped(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, _ = populated_service
        query = _make_valid_vector()
        result = svc.search(query, k=100)
        assert len(result.candidates) == svc.size

    def test_invalid_k_raises(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, _ = populated_service
        query = _make_valid_vector()
        with pytest.raises(ValueError, match="positive"):
            svc.search(query, k=0)
        with pytest.raises(ValueError, match="positive"):
            svc.search(query, k=-3)

    def test_invalid_query_embedding_raises(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, _ = populated_service
        # Wrong dimension
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            svc.search(np.zeros(256, dtype=np.float32))

        # NaN
        v = _make_valid_vector()
        v[0] = np.nan
        with pytest.raises(EmbeddingError, match="NaN or Inf"):
            svc.search(v)

        # Zero vector
        with pytest.raises(EmbeddingError, match="Zero-norm"):
            svc.search(np.zeros(512, dtype=np.float32))

    def test_empty_index_returns_unavailable_status(self, tmp_path: Path) -> None:
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [], [])
        svc = ReverseSearchService(idx_path, meta_path)
        assert svc.is_empty

        result = svc.search(_make_valid_vector())
        assert isinstance(result, ReverseSearchResult)
        assert result.status == ReverseSearchStatus.UNAVAILABLE
        assert result.candidates == ()
        assert result.error_message is not None

    def test_multiple_reference_images_same_person(self, tmp_path: Path) -> None:
        """Verify multiple reference images for the same person are returned separately (no aggregation)."""
        v1 = _make_valid_vector()
        v2 = _make_valid_vector() # make v2 very close to v1
        v2 = v1 + np.random.randn(512).astype(np.float32) * 0.001
        v2 /= np.linalg.norm(v2)

        records = [
            {"vector_id": 0, "person_id": "alice", "label": "alice", "image": "alice/img1.jpg"},
            {"vector_id": 1, "person_id": "alice", "label": "alice", "image": "alice/img2.jpg"},
        ]
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [v1, v2], records)
        svc = ReverseSearchService(idx_path, meta_path)

        result = svc.search(v1, k=2)
        assert len(result.candidates) == 2
        assert result.candidates[0].person_id == "alice"
        assert result.candidates[1].person_id == "alice"
        assert result.candidates[0].image != result.candidates[1].image

    def test_determinism(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, _ = populated_service
        query = _make_valid_vector()

        res1 = svc.search(query, k=3)
        res2 = svc.search(query, k=3)

        assert len(res1.candidates) == len(res2.candidates)
        for c1, c2 in zip(res1.candidates, res2.candidates):
            assert c1.vector_id == c2.vector_id
            assert c1.person_id == c2.person_id
            assert c1.image == c2.image
            assert c1.similarity == c2.similarity

    def test_error_does_not_corrupt_state(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        # Trigger an error with invalid query
        with pytest.raises(EmbeddingError):
            svc.search(np.zeros(512, dtype=np.float32))

        # Subsequent valid search must succeed normally
        result = svc.search(vectors[0], k=1)
        assert len(result.candidates) == 1
        assert result.candidates[0].vector_id == 0

    def test_concurrent_read_safety(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        """Verify multiple threads can perform read-only searches simultaneously."""
        svc, vectors = populated_service
        errors = []
        results = []

        def worker(q: np.ndarray) -> None:
            try:
                res = svc.search(q, k=3)
                results.append(res)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(vectors[i % len(vectors)],))
            for i in range(20)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 20
        for res in results:
            assert len(res.candidates) == 3


# ---------------------------------------------------------------------------
# Phase 13.12 — Contract Enhancement Tests
# ---------------------------------------------------------------------------

class TestReverseSearchResultContract:
    """Tests for the enhanced ReverseSearchResult contract (Phase 13.12)."""

    @pytest.fixture
    def populated_service(self, tmp_path: Path) -> tuple[ReverseSearchService, list[np.ndarray]]:
        rng = np.random.RandomState(42)
        vectors = []
        records = []
        for i in range(5):
            v = rng.randn(512).astype(np.float32)
            v /= np.linalg.norm(v)
            vectors.append(v)
            records.append({
                "vector_id": i,
                "person_id": f"person_{i:03d}",
                "label": f"label_{i}",
                "image": f"person_{i:03d}/img.jpg",
            })
        idx_path, meta_path = _create_mock_artifacts(tmp_path, vectors, records)
        svc = ReverseSearchService(idx_path, meta_path)
        return svc, vectors

    def test_successful_search_has_completed_status(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        result = svc.search(vectors[0], k=3)
        assert result.status == ReverseSearchStatus.COMPLETED
        assert result.error_message is None
        assert result.processing_time_ms >= 0.0

    def test_successful_search_has_valid_timing(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        result = svc.search(vectors[0], k=3)
        assert isinstance(result.processing_time_ms, float)
        assert result.processing_time_ms >= 0.0
        assert result.processing_time_ms < 10000.0  # Should complete in < 10s

    def test_successful_search_has_query_dimension(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        result = svc.search(vectors[0], k=3)
        assert result.query_dimension == 512

    def test_empty_index_returns_unavailable_with_error_message(self, tmp_path: Path) -> None:
        idx_path, meta_path = _create_mock_artifacts(tmp_path, [], [])
        svc = ReverseSearchService(idx_path, meta_path)
        result = svc.search(_make_valid_vector())
        assert result.status == ReverseSearchStatus.UNAVAILABLE
        assert result.error_message is not None
        assert "empty" in result.error_message.lower()
        assert result.candidates == ()
        assert result.processing_time_ms >= 0.0

    def test_result_is_frozen(self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]) -> None:
        svc, vectors = populated_service
        result = svc.search(vectors[0], k=3)
        assert isinstance(result, ReverseSearchResult)
        # frozen=True prevents attribute assignment
        with pytest.raises(AttributeError):
            result.status = ReverseSearchStatus.ERROR  # type: ignore[misc]

    def test_result_has_all_required_fields(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        result = svc.search(vectors[0], k=3)
        assert hasattr(result, "status")
        assert hasattr(result, "candidates")
        assert hasattr(result, "top_k")
        assert hasattr(result, "query_dimension")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "error_message")

    def test_error_status_via_invalid_embedding(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        """Verify that EmbeddingError still propagates (caller handles via try/except)."""
        svc, _ = populated_service
        with pytest.raises(EmbeddingError):
            svc.search(np.zeros(512, dtype=np.float32))

    def test_status_enum_values(self) -> None:
        assert ReverseSearchStatus.COMPLETED == "completed"
        assert ReverseSearchStatus.UNAVAILABLE == "unavailable"
        assert ReverseSearchStatus.DISABLED == "disabled"
        assert ReverseSearchStatus.ERROR == "error"

    def test_result_candidate_fields_populated(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        result = svc.search(vectors[0], k=2)
        assert len(result.candidates) == 2
        for c in result.candidates:
            assert isinstance(c, CandidateMatch)
            assert isinstance(c.vector_id, int)
            assert isinstance(c.person_id, str)
            assert isinstance(c.label, str)
            assert isinstance(c.image, str)
            assert isinstance(c.similarity, float)

    def test_search_timing_is_non_negative(
        self, populated_service: tuple[ReverseSearchService, list[np.ndarray]]
    ) -> None:
        svc, vectors = populated_service
        for _ in range(10):
            result = svc.search(vectors[0], k=1)
            assert result.processing_time_ms >= 0.0
