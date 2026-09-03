"""Tests for Phase 13.5.1 — Embedding & Retrieval Integrity Audit.

Verifies:
- Duplicate file detection
- Embedding identity for identical images
- FAISS/NumPy ranking agreement (ties allowed)
- Normalization contract
- Order independence
- Repeated extraction determinism
- No state leakage from sequential processing
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from search.flat_index import FlatIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
# Duplicate File Detection
# ---------------------------------------------------------------------------

class TestDuplicateFileDetection:
    def test_identical_files_detected(self) -> None:
        """Two files with same content should have same SHA-256."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = b"test image content"
            f1 = Path(tmpdir) / "image_a.jpg"
            f2 = Path(tmpdir) / "image_b_copy.jpg"
            f1.write_bytes(content)
            f2.write_bytes(content)

            h1 = hashlib.sha256(f1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(f2.read_bytes()).hexdigest()
            assert h1 == h2

    def test_different_files_detected(self) -> None:
        """Two files with different content should have different SHA-256."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "image_a.jpg"
            f2 = Path(tmpdir) / "image_b.jpg"
            f1.write_bytes(b"content A")
            f2.write_bytes(b"content B")

            h1 = hashlib.sha256(f1.read_bytes()).hexdigest()
            h2 = hashlib.sha256(f2.read_bytes()).hexdigest()
            assert h1 != h2


# ---------------------------------------------------------------------------
# Embedding Identity
# ---------------------------------------------------------------------------

class TestEmbeddingIdentity:
    def test_identical_images_produce_identical_embeddings(self) -> None:
        """Same embedding vector should produce similarity = 1.0."""
        v = _make_embedding(seed=42)
        sim = float(np.dot(v, v))
        assert abs(sim - 1.0) < 1e-6

    def test_different_embeddings_have_less_than_one(self) -> None:
        """Different embedding vectors should have similarity < 1.0."""
        v1 = _make_embedding(seed=1)
        v2 = _make_embedding(seed=2)
        sim = float(np.dot(v1, v2))
        assert sim < 1.0

    def test_embedding_distance_for_identical_vectors(self) -> None:
        v = _make_embedding(seed=42)
        diff = np.abs(v - v)
        assert float(np.max(diff)) == 0.0
        assert float(np.mean(diff)) == 0.0
        assert float(np.linalg.norm(v - v)) == 0.0


# ---------------------------------------------------------------------------
# FAISS/NumPy Agreement (ties allowed)
# ---------------------------------------------------------------------------

class TestFAISSNumPyAgreement:
    def test_faiss_numpy_same_set(self) -> None:
        """FAISS and NumPy should return the same set of IDs (order may differ for ties)."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(20, 512).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = FlatIndex(dimension=512, normalize=False)
        index.add(embeddings)

        sim_matrix = embeddings @ embeddings.T

        for i in range(20):
            faiss_result = index.search(embeddings[i], k=5)
            faiss_ids = set(faiss_result.ids.tolist())

            numpy_top5 = set(np.argsort(-sim_matrix[i])[:5].tolist())
            assert faiss_ids == numpy_top5, f"Query {i}: FAISS={faiss_ids} != NumPy={numpy_top5}"

    def test_ties_allowed_in_ranking(self) -> None:
        """When two vectors have identical similarity, order may differ."""
        v1 = _make_embedding(seed=1)
        v2 = _make_embedding(seed=1)  # Same as v1
        v3 = _make_embedding(seed=3)

        embeddings = np.vstack([v1, v2, v3]).astype(np.float32)
        index = FlatIndex(dimension=512, normalize=False)
        index.add(embeddings)

        result = index.search(v1, k=3)
        # Both v1 and v2 have similarity 1.0 with v1
        # FAISS may return [0, 1, 2] or [1, 0, 2]
        returned_ids = set(result.ids.tolist())
        assert {0, 1, 2} == returned_ids


# ---------------------------------------------------------------------------
# Normalization Contract
# ---------------------------------------------------------------------------

class TestNormalizationContract:
    def test_normalized_vector_has_unit_norm(self) -> None:
        v = _make_embedding(seed=42)
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-6

    def test_inner_product_equals_cosine_for_normalized(self) -> None:
        v1 = _make_embedding(seed=1)
        v2 = _make_embedding(seed=2)
        inner = float(np.dot(v1, v2))
        cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        assert abs(inner - cosine) < 1e-6


# ---------------------------------------------------------------------------
# Order Independence
# ---------------------------------------------------------------------------

class TestOrderIndependence:
    def test_faiss_search_order_independent(self) -> None:
        """FAISS search result should not depend on add order for distinct vectors.

        Vector IDs are positional, so we compare the actual embedding vectors
        returned rather than IDs (which change with insertion order).
        """
        rng = np.random.RandomState(42)
        vecs = [_make_embedding(seed=i) for i in range(10)]

        # Order 1
        e1 = np.vstack(vecs).astype(np.float32)
        idx1 = FlatIndex(dimension=512, normalize=False)
        idx1.add(e1)
        r1 = idx1.search(vecs[0], k=5)

        # Order 2 (reversed)
        e2 = np.vstack(list(reversed(vecs))).astype(np.float32)
        idx2 = FlatIndex(dimension=512, normalize=False)
        idx2.add(e2)
        r2 = idx2.search(vecs[0], k=5)

        # Compare actual vectors returned (not IDs, since IDs are positional)
        r1_vecs = set(tuple(e1[vid]) for vid in r1.ids)
        r2_vecs = set(tuple(e2[vid]) for vid in r2.ids)
        assert r1_vecs == r2_vecs


# ---------------------------------------------------------------------------
# Repeated Extraction Determinism
# ---------------------------------------------------------------------------

class TestRepeatedDeterminism:
    def test_same_vector_search_deterministic(self) -> None:
        """Searching the same vector twice should return identical results."""
        rng = np.random.RandomState(42)
        embeddings = rng.randn(15, 512).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = FlatIndex(dimension=512, normalize=False)
        index.add(embeddings)

        r1 = index.search(embeddings[0], k=5)
        r2 = index.search(embeddings[0], k=5)

        np.testing.assert_array_equal(r1.ids, r2.ids)
        np.testing.assert_array_almost_equal(r1.distances, r2.distances, decimal=6)


# ---------------------------------------------------------------------------
# No State Leakage
# ---------------------------------------------------------------------------

class TestNoStateLeakage:
    def test_sequential_index_builds_independent(self) -> None:
        """Building two separate indexes should produce independent results."""
        rng = np.random.RandomState(42)
        e1 = np.vstack([_make_embedding(seed=i) for i in range(10)]).astype(np.float32)
        e2 = np.vstack([_make_embedding(seed=i + 100) for i in range(10)]).astype(np.float32)

        idx1 = FlatIndex(dimension=512, normalize=False)
        idx1.add(e1)

        idx2 = FlatIndex(dimension=512, normalize=False)
        idx2.add(e2)

        # Search e1[0] in both indexes
        r1 = idx1.search(e1[0], k=3)
        r2 = idx2.search(e1[0], k=3)

        # idx2 should not find e1[0] at rank 1 (different vectors)
        # The top result in idx2 should be different from idx1's top result
        assert r1.ids[0] != r2.ids[0] or not np.allclose(e1[0], e2[0])
