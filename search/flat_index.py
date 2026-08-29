"""FAISS IndexFlatIP wrapper for exact inner-product search.

This is the core search abstraction for Phase 13.1.  It wraps a
FAISS ``IndexFlatIP`` index with:

* L2-normalized float32 embedding contract enforcement.
* Top-k search returning vector IDs and similarity scores.
* Save / load to disk.
* Index size reporting.

The component is **domain-agnostic** — it knows nothing about faces,
students, celebrities, or validation rules.  It works exclusively
with NumPy float32 vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import numpy.typing as npt

from search.embedding_validator import EmbeddingError, EmbeddingValidator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SearchResult:
    """Result of a single top-k search query.

    Attributes
    ----------
    ids:
        Vector IDs (row indices in the index), shape ``(k,)``.
        IDs correspond to the order in which vectors were added
        (0-indexed).
    distances:
        Inner-product similarity scores, shape ``(k,)``.
        For L2-normalized vectors this equals cosine similarity.
    """

    ids: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float32]


# ---------------------------------------------------------------------------
# FlatIndex
# ---------------------------------------------------------------------------

class FlatIndex:
    """Thin wrapper around ``faiss.IndexFlatIP``.

    Parameters
    ----------
    dimension:
        Vector dimensionality (default 512, matching ArcFace output).
    normalize:
        If ``True``, input embeddings are L2-normalized before indexing
        and search.  Default ``False`` — callers are expected to supply
        pre-normalized vectors (e.g. ``face.normed_embedding``).
    """

    def __init__(self, dimension: int = 512, normalize: bool = False) -> None:
        self._dimension = dimension
        self._normalize = normalize
        self._validator = EmbeddingValidator(dimension=dimension, normalize=normalize)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        return self._dimension

    @property
    def size(self) -> int:
        """Number of vectors currently in the index."""
        return self._index.ntotal

    @property
    def is_empty(self) -> bool:
        """``True`` if the index contains no vectors."""
        return self._index.ntotal == 0

    # ------------------------------------------------------------------
    # Add
    # ------------------------------------------------------------------

    def add(self, embeddings: npt.NDArray[np.float32]) -> int:
        """Add *embeddings* to the index.

        Parameters
        ----------
        embeddings:
            1-D ``(dim,)`` or 2-D ``(N, dim)`` float32 array.

        Returns
        -------
        int
            The starting vector ID assigned to the first added vector.
            IDs are sequential: ``[start, start + N)``.

        Raises
        ------
        EmbeddingError
            If the embedding violates the contract.
        """
        validated = self._validator.validate(embeddings)
        n = validated.shape[0]
        start_id = self._next_id
        self._index.add(validated)
        self._next_id += n
        return start_id

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: npt.NDArray[np.float32],
        k: int = 5,
    ) -> SearchResult:
        """Search the index for the *k* most similar vectors.

        Parameters
        ----------
        query:
            1-D ``(dim,)`` or 2-D ``(B, dim)`` float32 query vector(s).
        k:
            Number of nearest neighbours to return.  Clamped to
            ``min(k, self.size)``.

        Returns
        -------
        SearchResult
            Contains ``ids`` and ``distances`` arrays.

        Raises
        ------
        EmbeddingError
            If the query violates the contract.
        ValueError
            If the index is empty or ``k <= 0``.
        """
        if self._index.ntotal == 0:
            raise ValueError("Cannot search an empty index.")

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")

        validated = self._validator.validate(query)

        # Clamp k to index size
        k_eff = min(k, self._index.ntotal)

        # faiss.search expects 2-D even for a single query
        if validated.ndim == 1:
            validated = validated.reshape(1, -1)

        distances, ids = self._index.search(validated, k_eff)

        # For batch queries, return last query result (single-query API)
        # When batch_size == 1, distances shape is (1, k_eff)
        return SearchResult(
            ids=ids[0].astype(np.int64),
            distances=distances[0].astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Write the index to disk.

        Parameters
        ----------
        path:
            File path.  Parent directories are created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        logger.info("Index saved to %s (%d vectors, dim=%d).", path, self.size, self._dimension)

    @classmethod
    def load(cls, path: Path | str, normalize: bool = False) -> FlatIndex:
        """Load an index from disk.

        Parameters
        ----------
        path:
            File path previously written by :meth:`save`.
        normalize:
            Whether the loaded index should re-normalize on future
            additions / searches.

        Returns
        -------
        FlatIndex
            A new wrapper around the loaded FAISS index.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")

        faiss_index = faiss.read_index(str(path))
        if not isinstance(faiss_index, faiss.IndexFlatIP):
            raise ValueError(
                f"Loaded index is {type(faiss_index).__name__}, expected IndexFlatIP."
            )

        dim = faiss_index.d
        instance = cls.__new__(cls)
        instance._dimension = dim
        instance._normalize = normalize
        instance._validator = EmbeddingValidator(dimension=dim, normalize=normalize)
        instance._index = faiss_index
        instance._next_id = faiss_index.ntotal
        logger.info("Index loaded from %s (%d vectors, dim=%d).", path, faiss_index.ntotal, dim)
        return instance

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Remove all vectors from the index."""
        self._index.reset()
        self._next_id = 0

    def __repr__(self) -> str:
        return (
            f"FlatIndex(dimension={self._dimension}, size={self.size}, "
            f"normalize={self._normalize})"
        )
