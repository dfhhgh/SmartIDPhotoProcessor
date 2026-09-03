"""Runtime Reverse Search service.

Loads pre-built FAISS artifacts (reference_index.faiss + metadata.json)
and provides a clean, isolated runtime abstraction to query for similar
reference faces given a query embedding.

The service is domain-agnostic regarding validation rules (no PASS/REVIEW/REJECT
decisions) and operates strictly on read-only immutable artifacts.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from search.embedding_validator import EmbeddingError, EmbeddingValidator
from search.flat_index import FlatIndex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ReverseSearchError(Exception):
    """Base exception for all reverse search runtime errors."""


class ReverseSearchUnavailableError(ReverseSearchError):
    """Raised when the reverse search service cannot operate (e.g. empty index)."""


class ArtifactError(ReverseSearchError):
    """Raised when artifacts (FAISS index or metadata.json) are missing or invalid."""


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class ReverseSearchStatus(StrEnum):
    """Status of a reverse search operation.

    Attributes
    ----------
    COMPLETED:
        Search completed successfully; candidates may be empty or populated.
    UNAVAILABLE:
        Service cannot operate (empty index, missing artifacts).
    DISABLED:
        Reverse search is not enabled in configuration.
    ERROR:
        An unexpected error occurred during search.
    """

    COMPLETED = "completed"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CandidateMatch:
    """Represents a single matching reference face candidate.

    Attributes
    ----------
    vector_id:
        FAISS vector ID.
    person_id:
        Reference person identifier.
    label:
        Human-readable label or name.
    image:
        Relative path of the reference image.
    similarity:
        Cosine similarity score (inner product of L2-normalized vectors),
        in the range [-1.0, 1.0].
    """

    vector_id: int
    person_id: str
    label: str
    image: str
    similarity: float


@dataclass(frozen=True, slots=True)
class ReverseSearchResult:
    """Result of a reverse search query.

    Attributes
    ----------
    status:
        Outcome of the search operation.
    candidates:
        Tuple of candidate matches sorted by similarity descending.
        Empty when status is not COMPLETED or no matches were found.
    top_k:
        Requested top-k limit.
    query_dimension:
        Dimensionality of the query embedding.
    processing_time_ms:
        Wall-clock time of the search operation in milliseconds.
    error_message:
        Human-readable error description when status is ERROR or UNAVAILABLE.
        None when status is COMPLETED or DISABLED.
    """

    status: ReverseSearchStatus
    candidates: tuple[CandidateMatch, ...]
    top_k: int
    query_dimension: int
    processing_time_ms: float
    error_message: str | None = None


# ---------------------------------------------------------------------------
# ReverseSearchService
# ---------------------------------------------------------------------------

class ReverseSearchService:
    """Runtime service for querying the local reference-face FAISS index.

    Parameters
    ----------
    index_path:
        Path to the FAISS index file (``reference_index.faiss``).
    metadata_path:
        Path to the metadata JSON file (``metadata.json``).
    """

    def __init__(
        self,
        index_path: Path | str,
        metadata_path: Path | str,
    ) -> None:
        self._index_path = Path(index_path)
        self._metadata_path = Path(metadata_path)

        self._index: FlatIndex | None = None
        self._validator: EmbeddingValidator | None = None
        self._metadata_records: dict[int, dict[str, Any]] = {}
        self._metadata_raw: dict[str, Any] = {}

        self._load_and_validate_artifacts()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of reference vectors in the index."""
        if self._index is None:
            return 0
        return self._index.size

    @property
    def is_empty(self) -> bool:
        """``True`` if the index contains no vectors."""
        return self.size == 0

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        if self._index is None:
            return 512
        return self._index.dimension

    # ------------------------------------------------------------------
    # Artifact loading & validation
    # ------------------------------------------------------------------

    def _load_and_validate_artifacts(self) -> None:
        """Load index and metadata, enforcing strict artifact compatibility checks."""
        # 1. Check file existence
        if not self._index_path.exists():
            raise ArtifactError(f"FAISS index file not found: {self._index_path}")

        if not self._metadata_path.exists():
            raise ArtifactError(f"Metadata file not found: {self._metadata_path}")

        # 2. Load FAISS index
        try:
            self._index = FlatIndex.load(self._index_path, normalize=False)
        except Exception as exc:
            raise ArtifactError(f"Failed to load FAISS index from {self._index_path}: {exc}") from exc

        # 3. Load metadata JSON
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"Metadata file contains invalid JSON: {self._metadata_path}: {exc}") from exc
        except Exception as exc:
            raise ArtifactError(f"Failed to read metadata file {self._metadata_path}: {exc}") from exc

        if not isinstance(data, dict):
            raise ArtifactError("Metadata root must be a JSON object (dictionary).")

        self._metadata_raw = data

        # 4. Validate top-level schema fields
        required_fields = {
            "schema_version",
            "embedding_dimension",
            "metric",
            "normalized",
            "total_vectors",
            "total_persons",
            "records",
        }
        missing = required_fields - set(data.keys())
        if missing:
            raise ArtifactError(f"Metadata is missing required top-level fields: {sorted(missing)}")

        # 5. Validate compatibility rules
        schema_version = data["schema_version"]
        if schema_version != 1:
            raise ArtifactError(f"Unsupported metadata schema version: {schema_version} (expected 1).")

        meta_dim = data["embedding_dimension"]
        if meta_dim != self._index.dimension:
            raise ArtifactError(
                f"Embedding dimension mismatch: metadata says {meta_dim}, "
                f"FAISS index has {self._index.dimension}."
            )

        metric = data["metric"]
        if metric != "inner_product":
            raise ArtifactError(f"Unsupported metric: {metric!r} (expected 'inner_product').")

        normalized = data["normalized"]
        if normalized is not True:
            raise ArtifactError(f"Metadata 'normalized' must be True, got {normalized!r}.")

        total_vectors = data["total_vectors"]
        if total_vectors != self._index.size:
            raise ArtifactError(
                f"Total vectors mismatch: metadata reports {total_vectors}, "
                f"FAISS index has {self._index.size}."
            )

        records = data["records"]
        if not isinstance(records, list):
            raise ArtifactError("Metadata 'records' must be a list.")

        if len(records) != self._index.size:
            raise ArtifactError(
                f"Records count mismatch: metadata has {len(records)} records, "
                f"FAISS index has {self._index.size} vectors."
            )

        # 6. Validate individual records & build O(1) lookup table
        seen_vector_ids: set[int] = set()
        expected_ids = set(range(self._index.size))

        for idx, rec in enumerate(records):
            if not isinstance(rec, dict):
                raise ArtifactError(f"Metadata record at index {idx} must be a dictionary.")

            rec_fields = {"vector_id", "person_id", "label", "image"}
            rec_missing = rec_fields - set(rec.keys())
            if rec_missing:
                raise ArtifactError(f"Record at index {idx} is missing fields: {sorted(rec_missing)}")

            vid = rec["vector_id"]
            if not isinstance(vid, int) or isinstance(vid, bool):
                raise ArtifactError(f"Record at index {idx} has non-integer vector_id: {vid!r}.")

            if vid in seen_vector_ids:
                raise ArtifactError(f"Duplicate vector_id found in metadata: {vid}.")
            seen_vector_ids.add(vid)

            person_id = rec["person_id"]
            if not isinstance(person_id, str) or not person_id.strip():
                raise ArtifactError(f"Record vector_id {vid} has invalid person_id: {person_id!r}.")

            label = rec["label"]
            if not isinstance(label, str):
                raise ArtifactError(f"Record vector_id {vid} has non-string label: {label!r}.")

            image = rec["image"]
            if not isinstance(image, str) or not image.strip():
                raise ArtifactError(f"Record vector_id {vid} has invalid image path: {image!r}.")

            self._metadata_records[vid] = rec

        if seen_vector_ids != expected_ids:
            missing_ids = expected_ids - seen_vector_ids
            extra_ids = seen_vector_ids - expected_ids
            raise ArtifactError(
                f"Vector ID coverage mismatch. Missing: {sorted(missing_ids)}, Extra: {sorted(extra_ids)}."
            )

        # 7. Initialize embedding validator for queries
        self._validator = EmbeddingValidator(dimension=self._index.dimension, normalize=False)

        logger.info(
            "ReverseSearchService initialized successfully: index=%s, vectors=%d, persons=%d.",
            self._index_path,
            self.size,
            data["total_persons"],
        )

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    def search(
        self,
        embedding: npt.NDArray[np.float32],
        k: int = 5,
    ) -> ReverseSearchResult:
        """Search the reference index for the *k* most similar faces.

        Parameters
        ----------
        embedding:
            A 1-D ``(512,)`` float32 L2-normalized query vector.
        k:
            Number of nearest neighbours to return (default 5).
            Clamped to ``min(k, self.size)``.

        Returns
        -------
        ReverseSearchResult
            Contains status, ranked candidate matches, timing, and error info.

        Raises
        ------
        ReverseSearchUnavailableError
            If the service is operating on an empty index.
        EmbeddingError
            If the query embedding violates the 512-D float32 normalized contract.
        ValueError
            If *k* is not positive.
        """
        start_time = time.perf_counter()

        if self.is_empty:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ReverseSearchResult(
                status=ReverseSearchStatus.UNAVAILABLE,
                candidates=(),
                top_k=k,
                query_dimension=self.dimension,
                processing_time_ms=elapsed_ms,
                error_message="Cannot search an empty reference index.",
            )

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}.")

        # Validate query embedding contract via Phase 13.1 validator
        if self._validator is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ReverseSearchResult(
                status=ReverseSearchStatus.UNAVAILABLE,
                candidates=(),
                top_k=k,
                query_dimension=self.dimension,
                processing_time_ms=elapsed_ms,
                error_message="Service validator is not initialized.",
            )

        try:
            validated_query = self._validator.validate(embedding)
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Invalid query embedding: {exc}") from exc

        # Execute FAISS search via FlatIndex
        if self._index is None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ReverseSearchResult(
                status=ReverseSearchStatus.UNAVAILABLE,
                candidates=(),
                top_k=k,
                query_dimension=self.dimension,
                processing_time_ms=elapsed_ms,
                error_message="Service FAISS index is not initialized.",
            )

        faiss_result = self._index.search(validated_query, k=k)

        # Resolve FAISS vector IDs through O(1) metadata lookup
        candidates: list[CandidateMatch] = []
        for vid, dist in zip(faiss_result.ids, faiss_result.distances):
            vid_int = int(vid)
            rec = self._metadata_records.get(vid_int)
            if rec is None:
                raise ArtifactError(f"FAISS returned vector_id {vid_int} which is missing from metadata.")

            candidates.append(
                CandidateMatch(
                    vector_id=vid_int,
                    person_id=rec["person_id"],
                    label=rec["label"],
                    image=rec["image"],
                    similarity=float(dist),
                )
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return ReverseSearchResult(
            status=ReverseSearchStatus.COMPLETED,
            candidates=tuple(candidates),
            top_k=k,
            query_dimension=self._index.dimension,
            processing_time_ms=elapsed_ms,
        )
