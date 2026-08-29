"""Tests for search.flat_index — the FAISS core abstraction.

These tests verify:
- Add / search correctness
- NumPy brute-force cosine similarity agreement
- Save / load round-trip
- Deterministic repeated searches
- Error handling for invalid inputs
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from search.embedding_validator import EmbeddingError
from search.flat_index import FlatIndex, SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 512


def _make_valid(dim: int = DIM) -> np.ndarray:
    """Random L2-normalized float32 vector."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_batch(n: int, dim: int = DIM) -> np.ndarray:
    """N random L2-normalized float32 vectors."""
    batch = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    return batch / norms


def _numpy_cosine_similarity(
    query: np.ndarray,
    database: np.ndarray,
) -> np.ndarray:
    """Brute-force cosine similarity using NumPy.

    Parameters
    ----------
    query:
        (dim,) or (B, dim) float32, assumed L2-normalized.
    database:
        (N, dim) float32, assumed L2-normalized.

    Returns
    -------
    numpy.ndarray
        (B, N) similarity matrix.  Each row contains cosine similarities
        between one query and every database vector.
    """
    if query.ndim == 1:
        query = query.reshape(1, -1)
    return query @ database.T


def _make_deterministic_vectors() -> tuple[np.ndarray, list[str]]:
    """Create a small deterministic dataset with known relationships.

    Returns
    -------
    database : (6, 512) float32
        L2-normalized vectors.
    labels : list[str]
        Human-readable descriptions for each vector.
    """
    rng = np.random.RandomState(42)
    labels = []
    vectors = []

    # v0: base vector
    v0 = rng.randn(DIM).astype(np.float32)
    v0 /= np.linalg.norm(v0)
    vectors.append(v0)
    labels.append("v0: base")

    # v1: very similar to v0 (small perturbation)
    noise = rng.randn(DIM).astype(np.float32) * 0.01
    v1 = v0 + noise
    v1 /= np.linalg.norm(v1)
    vectors.append(v1)
    labels.append("v1: similar to v0")

    # v2: moderately similar to v0
    noise2 = rng.randn(DIM).astype(np.float32) * 0.1
    v2 = v0 + noise2
    v2 /= np.linalg.norm(v2)
    vectors.append(v2)
    labels.append("v2: moderately similar to v0")

    # v3: clearly different from v0
    v3 = rng.randn(DIM).astype(np.float32)
    v3 /= np.linalg.norm(v3)
    vectors.append(v3)
    labels.append("v3: different from v0")

    # v4: another different vector
    v4 = rng.randn(DIM).astype(np.float32)
    v4 /= np.linalg.norm(v4)
    vectors.append(v4)
    labels.append("v4: different from v0")

    # v5: very similar to v3
    noise3 = rng.randn(DIM).astype(np.float32) * 0.01
    v5 = v3 + noise3
    v5 /= np.linalg.norm(v5)
    vectors.append(v5)
    labels.append("v5: similar to v3")

    database = np.vstack(vectors).astype(np.float32)
    return database, labels


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def index() -> FlatIndex:
    return FlatIndex(dimension=DIM)


@pytest.fixture
def populated_index() -> tuple[FlatIndex, np.ndarray, list[str]]:
    """Index with 6 deterministic vectors, plus the database and labels."""
    db, labels = _make_deterministic_vectors()
    idx = FlatIndex(dimension=DIM)
    idx.add(db)
    return idx, db, labels


# ---------------------------------------------------------------------------
# Empty index
# ---------------------------------------------------------------------------

class TestEmptyIndex:
    def test_empty_size(self, index: FlatIndex) -> None:
        assert index.size == 0
        assert index.is_empty

    def test_search_empty_raises(self, index: FlatIndex) -> None:
        v = _make_valid()
        with pytest.raises(ValueError, match="empty"):
            index.search(v, k=5)


# ---------------------------------------------------------------------------
# Add
# ---------------------------------------------------------------------------

class TestAdd:
    def test_add_single(self, index: FlatIndex) -> None:
        v = _make_valid()
        start_id = index.add(v)
        assert start_id == 0
        assert index.size == 1

    def test_add_batch(self, index: FlatIndex) -> None:
        batch = _make_batch(10)
        start_id = index.add(batch)
        assert start_id == 0
        assert index.size == 10

    def test_sequential_add_ids(self, index: FlatIndex) -> None:
        v1 = _make_valid()
        v2 = _make_valid()
        id1 = index.add(v1)
        id2 = index.add(v2)
        assert id1 == 0
        assert id2 == 1
        assert index.size == 2

    def test_add_after_batch(self, index: FlatIndex) -> None:
        batch = _make_batch(5)
        start = index.add(batch)
        assert start == 0
        v = _make_valid()
        start2 = index.add(v)
        assert start2 == 5
        assert index.size == 6


# ---------------------------------------------------------------------------
# Search — correctness vs NumPy
# ---------------------------------------------------------------------------

class TestSearchCorrectness:
    def test_search_returns_correct_shape(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        idx, db, _ = populated_index
        query = db[0]
        result = idx.search(query, k=3)
        assert isinstance(result, SearchResult)
        assert result.ids.shape == (3,)
        assert result.distances.shape == (3,)

    def test_search_own_vector_returns_self(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        """Searching for a vector already in the index returns it as top-1."""
        idx, db, labels = populated_index
        query = db[0]  # v0
        result = idx.search(query, k=1)
        assert result.ids[0] == 0, f"Expected id 0, got {result.ids[0]}"
        assert result.distances[0] > 0.99, (
            f"Expected similarity ~1.0, got {result.distances[0]}"
        )

    def test_numpy_ranking_agrees_with_faiss(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        """FAISS ranking must match NumPy brute-force ranking."""
        idx, db, labels = populated_index

        for query_idx in range(len(labels)):
            query = db[query_idx]
            faiss_result = idx.search(query, k=6)

            numpy_sim = _numpy_cosine_similarity(query, db).flatten()
            numpy_ranking = np.argsort(-numpy_sim)  # descending

            # Top-k must match
            faiss_ids = list(faiss_result.ids)
            numpy_ids = list(numpy_ranking[: len(faiss_ids)])
            assert faiss_ids == numpy_ids, (
                f"Query {query_idx}: FAISS IDs {faiss_ids} != NumPy IDs {numpy_ids}"
            )

    def test_similarity_values_agree(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        """FAISS distances must match NumPy cosine similarity within tolerance."""
        idx, db, labels = populated_index

        for query_idx in range(len(labels)):
            query = db[query_idx]
            faiss_result = idx.search(query, k=6)

            numpy_sim = _numpy_cosine_similarity(query, db).flatten()
            numpy_sorted = np.sort(numpy_sim)[::-1]

            np.testing.assert_allclose(
                faiss_result.distances,
                numpy_sorted[: len(faiss_result.distances)],
                atol=1e-5,
                err_msg=f"Query {query_idx}: FAISS distances != NumPy similarities",
            )

    def test_similar_vector_ranked_higher_than_different(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        """v1 (similar to v0) should rank higher than v3 (different) for query v0."""
        idx, db, _ = populated_index
        result = idx.search(db[0], k=6)
        # v1 (id=1) should appear before v3 (id=3)
        pos_v1 = list(result.ids).index(1)
        pos_v3 = list(result.ids).index(3)
        assert pos_v1 < pos_v3


# ---------------------------------------------------------------------------
# Top-K edge cases
# ---------------------------------------------------------------------------

class TestTopK:
    def test_k_larger_than_index(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        """k > ntotal should return ntotal results."""
        idx, db, _ = populated_index
        result = idx.search(db[0], k=100)
        assert len(result.ids) == idx.size

    def test_k_equals_one(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        idx, db, _ = populated_index
        result = idx.search(db[0], k=1)
        assert len(result.ids) == 1

    def test_k_zero_raises(self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]) -> None:
        idx, db, _ = populated_index
        with pytest.raises(ValueError, match="positive"):
            idx.search(db[0], k=0)

    def test_k_negative_raises(self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]) -> None:
        idx, db, _ = populated_index
        with pytest.raises(ValueError, match="positive"):
            idx.search(db[0], k=-1)


# ---------------------------------------------------------------------------
# Deterministic repeated search
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_repeated_search_identical(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        idx, db, _ = populated_index
        query = db[0]
        r1 = idx.search(query, k=5)
        r2 = idx.search(query, k=5)
        np.testing.assert_array_equal(r1.ids, r2.ids)
        np.testing.assert_allclose(r1.distances, r2.distances, atol=1e-7)

    def test_search_not_mutated(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        idx, db, _ = populated_index
        query = db[0]
        db_before = db.copy()
        idx.search(query, k=5)
        np.testing.assert_array_equal(db, db_before)


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_load_search_identical(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        idx, db, _ = populated_index
        query = db[0]
        original = idx.search(query, k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.faiss"
            idx.save(path)

            loaded = FlatIndex.load(path)
            assert loaded.size == idx.size
            assert loaded.dimension == idx.dimension

            after_load = loaded.search(query, k=5)
            np.testing.assert_array_equal(original.ids, after_load.ids)
            np.testing.assert_allclose(
                original.distances, after_load.distances, atol=1e-6
            )

    def test_save_load_preserves_all_vectors(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        idx, db, _ = populated_index
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.faiss"
            idx.save(path)
            loaded = FlatIndex.load(path)
            assert loaded.size == 6

    def test_save_load_ranking_agrees_with_numpy(
        self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]
    ) -> None:
        """Loaded index must produce the same ranking as NumPy brute-force."""
        idx, db, _ = populated_index
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_index.faiss"
            idx.save(path)
            loaded = FlatIndex.load(path)

            for query_idx in range(6):
                query = db[query_idx]
                faiss_result = loaded.search(query, k=6)
                numpy_sim = _numpy_cosine_similarity(query, db).flatten()
                numpy_ranking = np.argsort(-numpy_sim)
                assert list(faiss_result.ids) == list(numpy_ranking)

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            FlatIndex.load("/nonexistent/path/index.faiss")

    def test_save_creates_parent_dirs(self) -> None:
        idx = FlatIndex(dimension=DIM)
        idx.add(_make_valid())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "dir" / "index.faiss"
            idx.save(path)
            assert path.exists()


# ---------------------------------------------------------------------------
# Error handling — invalid embeddings
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_add_wrong_dim(self, index: FlatIndex) -> None:
        v = np.random.randn(256).astype(np.float32)
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            index.add(v)

    def test_add_nan(self, index: FlatIndex) -> None:
        v = _make_valid()
        v[0] = np.nan
        with pytest.raises(EmbeddingError, match="NaN or Inf"):
            index.add(v)

    def test_add_zero_vector(self, index: FlatIndex) -> None:
        v = np.zeros(DIM, dtype=np.float32)
        with pytest.raises(EmbeddingError, match="Zero-norm"):
            index.add(v)

    def test_search_wrong_dim(self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]) -> None:
        idx, _, _ = populated_index
        v = np.random.randn(256).astype(np.float32)
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            idx.search(v, k=3)

    def test_search_nan(self, populated_index: tuple[FlatIndex, np.ndarray, list[str]]) -> None:
        idx, _, _ = populated_index
        v = _make_valid()
        v[0] = np.nan
        with pytest.raises(EmbeddingError, match="NaN or Inf"):
            idx.search(v, k=3)


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_empties_index(self, index: FlatIndex) -> None:
        index.add(_make_batch(10))
        assert index.size == 10
        index.reset()
        assert index.size == 0
        assert index.is_empty


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestRepr:
    def test_repr(self, index: FlatIndex) -> None:
        r = repr(index)
        assert "FlatIndex" in r
        assert "512" in r
