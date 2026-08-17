"""
Persistent collection state for incremental dataset building.

Tracks seen source results, download history, and collection progress
across runs. State is persisted as JSON and loaded on startup.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class SeenResult:
    """A source search result that has been encountered before."""

    source: str
    """Source provider name (e.g. 'pexels')."""

    source_id: str
    """Provider-specific unique identifier."""

    download_url: str
    """URL of the downloaded image."""

    local_path: str
    """Relative path to the local file (relative to raw dir)."""

    category: str
    """Category this result was collected for."""

    query: str
    """Search query that produced this result."""

    status: str
    """Status: 'accepted', 'rejected_duplicate', 'rejected_face', 'rejected_other'."""

    rejection_reason: str | None = None
    """Specific rejection reason if status is 'rejected_*'."""

    collected_at: str = ""
    """ISO timestamp when this result was processed."""


@dataclass
class CategoryProgress:
    """Collection progress for a single category."""

    target: int = 0
    """Target number of images."""

    accepted: int = 0
    """Number of accepted images."""

    rejected_duplicate: int = 0
    """Number rejected as duplicates."""

    rejected_face: int = 0
    """Number rejected by face filter."""

    rejected_other: int = 0
    """Number rejected for other reasons."""

    last_page: dict[str, int] = field(default_factory=dict)
    """Last processed page per query+source key."""

    queries_processed: list[str] = field(default_factory=list)
    """List of query+source keys that have been fully processed."""


class CollectionState:
    """Persistent state for incremental collection.

    Tracks which source results have been seen, their status,
    and per-category collection progress. State is persisted
    as JSON and loaded on startup.

    Parameters
    ----------
    state_path:
        Path to the JSON state file.
    """

    def __init__(self, state_path: Path) -> None:
        self._state_path: Path = state_path
        self._seen: dict[str, SeenResult] = {}
        self._categories: dict[str, CategoryProgress] = {}

        if state_path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_seen(self, source: str, source_id: str) -> bool:
        """Check if a source result has been seen before.

        Parameters
        ----------
        source:
            Source provider name.
        source_id:
            Provider-specific unique identifier.

        Returns
        -------
        bool
            True if this result was already processed.
        """
        key = self._make_key(source, source_id)
        return key in self._seen

    def is_url_seen(self, download_url: str) -> bool:
        """Check if a download URL has been seen before.

        Parameters
        ----------
        download_url:
            The image download URL.

        Returns
        -------
        bool
            True if this URL was already processed.
        """
        for result in self._seen.values():
            if result.download_url == download_url:
                return True
        return False

    def get_status(self, source: str, source_id: str) -> str | None:
        """Get the status of a previously seen result.

        Parameters
        ----------
        source:
            Source provider name.
        source_id:
            Provider-specific unique identifier.

        Returns
        -------
        str or None
            The status string, or None if not seen.
        """
        key = self._make_key(source, source_id)
        result = self._seen.get(key)
        return result.status if result else None

    def record_seen(
        self,
        source: str,
        source_id: str,
        download_url: str,
        local_path: str,
        category: str,
        query: str,
        status: str,
        rejection_reason: str | None = None,
    ) -> None:
        """Record that a source result has been processed.

        Parameters
        ----------
        source:
            Source provider name.
        source_id:
            Provider-specific unique identifier.
        download_url:
            URL of the downloaded image.
        local_path:
            Relative path to the local file.
        category:
            Category this result was collected for.
        query:
            Search query that produced this result.
        status:
            Processing status ('accepted', 'rejected_*').
        rejection_reason:
            Specific rejection reason if applicable.
        """
        key = self._make_key(source, source_id)

        self._seen[key] = SeenResult(
            source=source,
            source_id=source_id,
            download_url=download_url,
            local_path=local_path,
            category=category,
            query=query,
            status=status,
            rejection_reason=rejection_reason,
            collected_at=datetime.now(timezone.utc).isoformat(),
        )

    def get_category_progress(self, category: str) -> CategoryProgress:
        """Get collection progress for a category.

        Parameters
        ----------
        category:
            Category name.

        Returns
        -------
        CategoryProgress
            Current progress for this category.
        """
        if category not in self._categories:
            self._categories[category] = CategoryProgress()
        return self._categories[category]

    def set_category_target(self, category: str, target: int) -> None:
        """Set the target for a category.

        Parameters
        ----------
        category:
            Category name.
        target:
            Target number of images.
        """
        progress = self.get_category_progress(category)
        progress.target = target

    def increment_accepted(self, category: str) -> None:
        """Increment the accepted count for a category.

        Parameters
        ----------
        category:
            Category name.
        """
        progress = self.get_category_progress(category)
        progress.accepted += 1

    def increment_rejected(self, category: str, reason: str) -> None:
        """Increment a rejection counter for a category.

        Parameters
        ----------
        category:
            Category name.
        reason:
            Rejection reason ('duplicate', 'face', 'other').
        """
        progress = self.get_category_progress(category)
        if reason == "duplicate":
            progress.rejected_duplicate += 1
        elif reason == "face":
            progress.rejected_face += 1
        else:
            progress.rejected_other += 1

    def get_last_page(self, category: str, query: str, source: str) -> int:
        """Get the last processed page for a query+source combination.

        Parameters
        ----------
        category:
            Category name.
        query:
            Search query.
        source:
            Source provider name.

        Returns
        -------
        int
            Last processed page number (0 if never processed).
        """
        progress = self.get_category_progress(category)
        key = f"{source}::{query}"
        return progress.last_page.get(key, 0)

    def set_last_page(
        self, category: str, query: str, source: str, page: int
    ) -> None:
        """Record the last processed page for a query+source combination.

        Parameters
        ----------
        category:
            Category name.
        query:
            Search query.
        source:
            Source provider name.
        page:
            Last processed page number.
        """
        progress = self.get_category_progress(category)
        key = f"{source}::{query}"
        progress.last_page[key] = page

    def mark_query_processed(
        self, category: str, query: str, source: str
    ) -> None:
        """Mark a query+source as fully processed.

        Parameters
        ----------
        category:
            Category name.
        query:
            Search query.
        source:
            Source provider name.
        """
        progress = self.get_category_progress(category)
        key = f"{source}::{query}"
        if key not in progress.queries_processed:
            progress.queries_processed.append(key)

    def is_query_processed(
        self, category: str, query: str, source: str
    ) -> bool:
        """Check if a query+source has been fully processed.

        Parameters
        ----------
        category:
            Category name.
        query:
            Search query.
        source:
            Source provider name.

        Returns
        -------
        bool
            True if this query+source was already fully processed.
        """
        progress = self.get_category_progress(category)
        key = f"{source}::{query}"
        return key in progress.queries_processed

    def save(self) -> None:
        """Persist state to disk."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "seen": {k: asdict(v) for k, v in self._seen.items()},
            "categories": {
                k: asdict(v) for k, v in self._categories.items()
            },
        }

        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_seen_count(self) -> int:
        """Return the number of seen results."""
        return len(self._seen)

    def get_accepted_count(self) -> int:
        """Return the total number of accepted results."""
        return sum(
            1 for r in self._seen.values() if r.status == "accepted"
        )

    def get_all_accepted_paths(self) -> set[str]:
        """Return the set of all accepted local paths.

        Returns
        -------
        set[str]
            Set of relative paths to accepted images.
        """
        return {
            r.local_path
            for r in self._seen.values()
            if r.status == "accepted"
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(source: str, source_id: str) -> str:
        """Create a unique key for a source result."""
        return f"{source}::{source_id}"

    def _load(self) -> None:
        """Load state from disk."""
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for k, v in data.get("seen", {}).items():
                self._seen[k] = SeenResult(**v)

            for k, v in data.get("categories", {}).items():
                self._categories[k] = CategoryProgress(**v)

        except (json.JSONDecodeError, KeyError, TypeError):
            # If state file is corrupted, start fresh
            self._seen.clear()
            self._categories.clear()
