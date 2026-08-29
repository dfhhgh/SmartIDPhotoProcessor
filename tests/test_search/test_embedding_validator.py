"""Tests for search.embedding_validator."""

from __future__ import annotations

import numpy as np
import pytest

from search.embedding_validator import EmbeddingError, EmbeddingValidator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def validator() -> EmbeddingValidator:
    return EmbeddingValidator(dimension=512)


@pytest.fixture
def validator_norm() -> EmbeddingValidator:
    return EmbeddingValidator(dimension=512, normalize=True)


def _make_valid(dim: int = 512) -> np.ndarray:
    """Return a random L2-normalized float32 vector."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_batch(n: int, dim: int = 512) -> np.ndarray:
    """Return *n* random L2-normalized float32 vectors."""
    batch = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    return batch / norms


# ---------------------------------------------------------------------------
# Valid inputs
# ---------------------------------------------------------------------------

class TestValidInputs:
    def test_single_vector(self, validator: EmbeddingValidator) -> None:
        v = _make_valid()
        result = validator.validate(v)
        assert result.shape == (1, 512)
        assert result.dtype == np.float32

    def test_batch(self, validator: EmbeddingValidator) -> None:
        batch = _make_batch(10)
        result = validator.validate(batch)
        assert result.shape == (10, 512)
        assert result.dtype == np.float32

    def test_single_element_batch(self, validator: EmbeddingValidator) -> None:
        v = _make_valid().reshape(1, -1)
        result = validator.validate(v)
        assert result.shape == (1, 512)

    def test_contiguous_output(self, validator: EmbeddingValidator) -> None:
        v = _make_valid()
        result = validator.validate(v)
        assert result.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# Invalid inputs — type / shape
# ---------------------------------------------------------------------------

class TestInvalidTypeShape:
    def test_rejects_list(self, validator: EmbeddingValidator) -> None:
        with pytest.raises(EmbeddingError, match="numpy.ndarray"):
            validator.validate([1.0, 2.0])

    def test_rejects_3d(self, validator: EmbeddingValidator) -> None:
        arr = np.random.randn(2, 3, 512).astype(np.float32)
        with pytest.raises(EmbeddingError, match="3-D"):
            validator.validate(arr)

    def test_rejects_wrong_dim_1d(self, validator: EmbeddingValidator) -> None:
        v = np.random.randn(256).astype(np.float32)
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            validator.validate(v)

    def test_rejects_wrong_dim_2d(self, validator: EmbeddingValidator) -> None:
        batch = np.random.randn(5, 256).astype(np.float32)
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            validator.validate(batch)

    def test_rejects_empty(self, validator: EmbeddingValidator) -> None:
        arr = np.array([], dtype=np.float32)
        with pytest.raises(EmbeddingError, match="empty"):
            validator.validate(arr)


# ---------------------------------------------------------------------------
# Invalid inputs — dtype
# ---------------------------------------------------------------------------

class TestInvalidDtype:
    def test_rejects_float64(self, validator: EmbeddingValidator) -> None:
        v = _make_valid().astype(np.float64)
        with pytest.raises(EmbeddingError, match="float32"):
            validator.validate(v)

    def test_rejects_int32(self, validator: EmbeddingValidator) -> None:
        v = np.ones(512, dtype=np.int32)
        with pytest.raises(EmbeddingError, match="float32"):
            validator.validate(v)


# ---------------------------------------------------------------------------
# Invalid inputs — NaN / Inf / zero
# ---------------------------------------------------------------------------

class TestInvalidValues:
    def test_rejects_nan(self, validator: EmbeddingValidator) -> None:
        v = _make_valid()
        v[0] = np.nan
        with pytest.raises(EmbeddingError, match="NaN or Inf"):
            validator.validate(v)

    def test_rejects_inf(self, validator: EmbeddingValidator) -> None:
        v = _make_valid()
        v[0] = np.inf
        with pytest.raises(EmbeddingError, match="NaN or Inf"):
            validator.validate(v)

    def test_rejects_neg_inf(self, validator: EmbeddingValidator) -> None:
        v = _make_valid()
        v[0] = -np.inf
        with pytest.raises(EmbeddingError, match="NaN or Inf"):
            validator.validate(v)

    def test_rejects_zero_vector(self, validator: EmbeddingValidator) -> None:
        v = np.zeros(512, dtype=np.float32)
        with pytest.raises(EmbeddingError, match="Zero-norm"):
            validator.validate(v)

    def test_rejects_zero_in_batch(self, validator: EmbeddingValidator) -> None:
        batch = _make_batch(5)
        batch[2] = 0.0
        with pytest.raises(EmbeddingError, match="Zero-norm"):
            validator.validate(batch)


# ---------------------------------------------------------------------------
# Normalization mode
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalize_unnormalized(self, validator_norm: EmbeddingValidator) -> None:
        v = np.random.randn(512).astype(np.float32)
        result = validator_norm.validate(v)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_normalize_already_normalized(self, validator_norm: EmbeddingValidator) -> None:
        v = _make_valid()
        result = validator_norm.validate(v)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_no_normalize_preserves(self, validator: EmbeddingValidator) -> None:
        v = np.random.randn(512).astype(np.float32)
        original_norm = np.linalg.norm(v)
        result = validator.validate(v)
        result_norm = np.linalg.norm(result)
        assert abs(result_norm - original_norm) < 1e-5


# ---------------------------------------------------------------------------
# Dimension property
# ---------------------------------------------------------------------------

class TestDimension:
    def test_custom_dimension(self) -> None:
        v = EmbeddingValidator(dimension=256)
        assert v.dimension == 256

    def test_default_dimension(self) -> None:
        v = EmbeddingValidator()
        assert v.dimension == 512
