"""Abstract base class for image sources."""

from __future__ import annotations

import abc
from typing import Iterator

from dataset_acquisition.models import SearchResult


class ImageSource(abc.ABC):
    """Abstract interface for image search and download providers."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable source name."""

    @abc.abstractmethod
    def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> Iterator[SearchResult]:
        """Search for images matching *query*.

        Yields SearchResult objects. Implementations should handle
        pagination internally.
        """

    @abc.abstractmethod
    def download_url(self, url: str) -> bytes | None:
        """Download raw image bytes from *url*.

        Returns None on failure. Implementations should handle
        retries and timeouts.
        """

    def close(self) -> None:
        """Release any held resources."""
