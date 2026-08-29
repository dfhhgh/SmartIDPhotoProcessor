"""Embedding validation contract for vector search.

Enforces that input embeddings are valid float32 L2-normalized vectors
before they enter the FAISS index.  This module is domain-agnostic — it
knows nothing about faces, students, or celebrities.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class EmbeddingError(Exception):
    """Raised when an embedding violates the input contract."""


class EmbeddingValidator:
    """Validate and normalize embeddings for FAISS indexing.

    Contract
    --------
    * dtype must be float32.
    * 1-D vectors must have shape ``(dim,)``.
    * Batch arrays must have shape ``(N, dim)``.
    * No element may be NaN or Inf.
    * No vector may be all-zeros (zero-norm).
    * All vectors must be finite.

    Normalization
    -------------
    The validator does **not** re-normalize by default.  Input embeddings
    are expected to arrive pre-normalized (e.g. ``face.normed_embedding``
    from InsightFace).  If ``normalize=True`` is passed, each vector is
    L2-normalized in-place before validation completes — this is a
    convenience for callers that cannot guarantee pre-normalization.
    """

    def __init__(self, dimension: int = 512, normalize: bool = False) -> None:
        self._dimension = dimension
        self._normalize = normalize

    @property
    def dimension(self) -> int:
        return self._dimension

    def validate(self, embeddings: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Validate *embeddings* and return a clean array ready for FAISS.

        Parameters
        ----------
        embeddings:
            A 1-D ``(dim,)`` or 2-D ``(N, dim)`` float32 array.

        Returns
        -------
        numpy.ndarray
            A 2-D ``(N, dim)`` float32 array (row-major, contiguous).

        Raises
        ------
        EmbeddingError
            If any contract rule is violated.
        """
        if not isinstance(embeddings, np.ndarray):
            raise EmbeddingError(
                f"Embedding must be a numpy.ndarray, got {type(embeddings).__name__}."
            )

        if embeddings.size == 0:
            raise EmbeddingError("Embedding array must not be empty.")

        if embeddings.dtype != np.float32:
            raise EmbeddingError(
                f"Embedding dtype must be float32, got {embeddings.dtype}."
            )

        # Ensure 2-D
        if embeddings.ndim == 1:
            if embeddings.shape[0] != self._dimension:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: expected {self._dimension}, "
                    f"got {embeddings.shape[0]}."
                )
            embeddings = embeddings.reshape(1, -1)
        elif embeddings.ndim == 2:
            if embeddings.shape[1] != self._dimension:
                raise EmbeddingError(
                    f"Embedding dimension mismatch: expected {self._dimension}, "
                    f"got {embeddings.shape[1]}."
                )
        else:
            raise EmbeddingError(
                f"Embedding must be 1-D or 2-D, got {embeddings.ndim}-D."
            )

        # NaN / Inf check
        if not np.all(np.isfinite(embeddings)):
            raise EmbeddingError("Embedding contains NaN or Inf values.")

        # Optional L2 normalization
        if self._normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            # Avoid division by zero — zero vectors are caught below.
            safe_norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / safe_norms

        # Zero-vector check (after optional normalization)
        norms = np.linalg.norm(embeddings, axis=1)
        zero_mask = norms == 0.0
        if np.any(zero_mask):
            indices = np.where(zero_mask)[0].tolist()
            raise EmbeddingError(
                f"Zero-norm (all-zero) embeddings at indices {indices}."
            )

        # Ensure contiguous float32 row-major
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        return embeddings
