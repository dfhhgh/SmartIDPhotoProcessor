"""Data models for dataset acquisition."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Person:
    person_id: str
    display_name: str
    category: str = "unknown"
    aliases: tuple[str, ...] = ()
    search_queries: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "category": self.category,
            "aliases": list(self.aliases),
            "search_queries": list(self.search_queries),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Person:
        return cls(
            person_id=d["person_id"],
            display_name=d["display_name"],
            category=d.get("category", "unknown"),
            aliases=tuple(d.get("aliases", [])),
            search_queries=tuple(d.get("search_queries", [])),
        )


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    person_id: str
    source: str
    source_url: str
    local_path: str
    license: str = "unknown"
    attribution: str = ""
    query: str = ""
    download_timestamp: str = ""
    sha256: str = ""
    file_size: int = 0
    width: int = 0
    height: int = 0
    faces_detected: int = 0
    face_selected: bool = False
    face_confidence: float = 0.0
    duplicate_of: str = ""
    duplicate_type: str = ""
    image_category: str = "photograph"
    identity_status: str = "confirmed"
    status: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "person_id": self.person_id,
            "source": self.source,
            "source_url": self.source_url,
            "local_path": self.local_path,
            "license": self.license,
            "attribution": self.attribution,
            "query": self.query,
            "download_timestamp": self.download_timestamp,
            "sha256": self.sha256,
            "file_size": self.file_size,
            "width": self.width,
            "height": self.height,
            "faces_detected": self.faces_detected,
            "face_selected": self.face_selected,
            "face_confidence": self.face_confidence,
            "duplicate_of": self.duplicate_of,
            "duplicate_type": self.duplicate_type,
            "image_category": self.image_category,
            "identity_status": self.identity_status,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ImageRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class SearchResult:
    source: str
    source_url: str
    image_url: str
    title: str = ""
    description: str = ""
    license: str = "unknown"
    attribution: str = ""
    width: int = 0
    height: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DownloadResult:
    success: bool
    image_record: ImageRecord | None = None
    error: str = ""


@dataclass(frozen=True, slots=True)
class ReviewItem:
    image_id: str
    person_id: str
    source: str
    source_url: str
    local_path: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "person_id": self.person_id,
            "source": self.source,
            "source_url": self.source_url,
            "local_path": self.local_path,
            "reason": self.reason,
            "metadata": self.metadata,
        }


def compute_stats_from_records(records: list[ImageRecord]) -> "CollectionStats":
    """Derive CollectionStats purely from ImageRecord data.

    This is the single source of truth for validation statistics.
    """
    return CollectionStats(
        total_searched=len(records),
        total_downloaded=len(records),
        total_valid=sum(1 for r in records if r.status == "valid"),
        total_duplicates=0,
        total_no_face=sum(1 for r in records if r.status == "no_face"),
        total_multi_face=sum(1 for r in records if r.faces_detected > 1),
        total_invalid_image=sum(1 for r in records if r.status == "invalid"),
        total_representation=sum(1 for r in records if r.image_category == "representation"),
        total_identity_uncertain=sum(1 for r in records if r.identity_status == "uncertain"),
        persons_completed=0,
        persons_incomplete=0,
    )


REJECTION_REASONS = frozenset({
    "representation",
    "no_face",
    "multi_face",
    "download_error",
    "decode_error",
    "invalid_image",
    "duplicate",
    "other",
})


@dataclass(frozen=True, slots=True)
class RejectionDetail:
    person_id: str
    source: str
    source_url: str
    rejection_reason: str
    title: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "source": self.source,
            "source_url": self.source_url,
            "rejection_reason": self.rejection_reason,
            "title": self.title,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RejectionDetail":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class RejectionStats:
    total_candidates: int = 0
    accepted: int = 0
    rejected_total: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    per_person: dict[str, dict[str, int]] = field(default_factory=dict)
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "accepted": self.accepted,
            "rejected_total": self.rejected_total,
            "rejections_by_reason": dict(self.rejections_by_reason),
            "per_person": dict(self.per_person),
            "per_source": dict(self.per_source),
        }


REVIEW_DECISIONS = frozenset({"PENDING", "ACCEPT", "REJECT", "UNCERTAIN"})

REVIEW_REASONS = frozenset({
    "CORRECT_IDENTITY",
    "WRONG_IDENTITY",
    "REPRESENTATION",
    "NO_FACE",
    "MULTIPLE_FACES",
    "FACE_NOT_TARGET",
    "LOW_QUALITY",
    "DUPLICATE",
    "NEAR_DUPLICATE",
    "INSUFFICIENT_CONTEXT",
    "OTHER",
})


@dataclass(frozen=True, slots=True)
class ManualReviewRecord:
    person_id: str
    display_name: str
    image_id: str
    image_path: str
    source: str
    source_url: str
    automated_status: str
    automated_image_category: str
    automated_identity_status: str
    faces_detected: int
    face_confidence: float = 0.0
    manual_decision: str = "PENDING"
    manual_reason: str = ""
    reviewer_notes: str = ""

    def __post_init__(self) -> None:
        if self.manual_decision not in REVIEW_DECISIONS:
            raise ValueError(
                f"manual_decision must be one of {sorted(REVIEW_DECISIONS)}, "
                f"got {self.manual_decision!r}"
            )
        if self.manual_reason and self.manual_reason not in REVIEW_REASONS:
            raise ValueError(
                f"manual_reason must be one of {sorted(REVIEW_REASONS)} or empty, "
                f"got {self.manual_reason!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "display_name": self.display_name,
            "image_id": self.image_id,
            "image_path": self.image_path,
            "source": self.source,
            "source_url": self.source_url,
            "automated_status": self.automated_status,
            "automated_image_category": self.automated_image_category,
            "automated_identity_status": self.automated_identity_status,
            "faces_detected": self.faces_detected,
            "face_confidence": self.face_confidence,
            "manual_decision": self.manual_decision,
            "manual_reason": self.manual_reason,
            "reviewer_notes": self.reviewer_notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ManualReviewRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass(slots=True)
class ReviewStats:
    total_images: int = 0
    pending: int = 0
    accepted: int = 0
    rejected: int = 0
    uncertain: int = 0
    acceptance_rate: float = 0.0
    per_person: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_images": self.total_images,
            "pending": self.pending,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "uncertain": self.uncertain,
            "acceptance_rate": self.acceptance_rate,
            "per_person": self.per_person,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionRunResult:
    """Structured result from a single download_candidates() call.

    Metric definitions:
      candidates_discovered: Number yielded by the source iterator.
      candidates_examined: Number presented to the validation gate after
        skipping already-downloaded and already-rejected URLs.
      candidates_skipped_existing: Number skipped because source_url was
        already downloaded in prior state.
      candidates_skipped_rejected: Number skipped because source_url was
        already rejected in prior state.
      accepted: Number that passed the full validation gate and were saved.
      rejected: Number processed by the gate and rejected.

    Invariants:
      discovered >= examined + skipped_existing + skipped_rejected
      examined == accepted + rejected
      acceptance_rate == accepted / examined (0.0 if examined == 0)
    """
    records: list[ImageRecord]
    rejection_details: list[RejectionDetail]
    candidates_discovered: int = 0
    candidates_examined: int = 0
    candidates_skipped_existing: int = 0
    candidates_skipped_rejected: int = 0
    accepted: int = 0
    rejected: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.candidates_examined if self.candidates_examined > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_discovered": self.candidates_discovered,
            "candidates_examined": self.candidates_examined,
            "candidates_skipped_existing": self.candidates_skipped_existing,
            "candidates_skipped_rejected": self.candidates_skipped_rejected,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "acceptance_rate": self.acceptance_rate,
        }


@dataclass(frozen=True, slots=True)
class CollectionStats:
    total_searched: int = 0
    total_downloaded: int = 0
    total_valid: int = 0
    total_duplicates: int = 0
    total_no_face: int = 0
    total_multi_face: int = 0
    total_invalid_image: int = 0
    total_representation: int = 0
    total_identity_uncertain: int = 0
    persons_completed: int = 0
    persons_incomplete: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_searched": self.total_searched,
            "total_downloaded": self.total_downloaded,
            "total_valid": self.total_valid,
            "total_duplicates": self.total_duplicates,
            "total_no_face": self.total_no_face,
            "total_multi_face": self.total_multi_face,
            "total_invalid_image": self.total_invalid_image,
            "total_representation": self.total_representation,
            "total_identity_uncertain": self.total_identity_uncertain,
            "persons_completed": self.persons_completed,
            "persons_incomplete": self.persons_incomplete,
        }
