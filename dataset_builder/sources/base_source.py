"""
Abstract contract for image source providers.

Every image source (Pexels, Pixabay, Flickr, etc.) must implement
``BaseSource``.  The Dataset Builder interacts only with this
interface, remaining completely unaware of provider-specific logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from config.settings import Settings


# ------------------------------------------------------------------
# Lightweight data transfer objects
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SearchResult:
    """A single image result returned by a source search query."""

    id: str
    """Provider-specific unique identifier."""

    download_url: str
    """URL pointing to the full-resolution image."""

    preview_url: str
    """URL pointing to a smaller preview/thumbnail."""

    page_url: str
    """URL of the image's public detail page."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    photographer: str
    """Name of the photographer or author."""

    license_name: str
    """Human-readable license identifier."""

    query: str
    """Search query that produced this result."""

    source: str
    """Identifier of the source provider (e.g. 'pexels', 'pixabay')."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a single image download attempt."""

    success: bool
    """True when the image was saved without error."""

    local_path: Path | None
    """Filesystem path to the saved image, or None on failure."""

    error_message: str
    """Empty string on success; human-readable error description otherwise."""

    download_time_seconds: float
    """Elapsed wall-clock time for the download in seconds."""


@dataclass(frozen=True)
class ImageMetadata:
    """Structured metadata captured for every downloaded image."""

    id: str
    """Provider-specific unique identifier."""

    source: str
    """Identifier of the source provider."""

    local_path: Path
    """Absolute path to the saved image on disk."""

    download_url: str
    """Original URL from which the image was fetched."""

    page_url: str
    """Public detail page URL."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    photographer: str
    """Name of the photographer or author."""

    license_name: str
    """Human-readable license identifier."""

    query: str
    """Search query that produced this result."""


# ------------------------------------------------------------------
# Abstract source contract
# ------------------------------------------------------------------


class BaseSource(ABC):
    """Abstract base class that every image source must implement.

    The Dataset Builder orchestrates downloads, filtering, and
    metadata management through this interface alone, keeping
    provider-specific details fully encapsulated.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings

    # ------------------------------------------------------------------
    # Read-only property
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the source identifier (e.g. ``'pexels'``, ``'pixabay'``)."""

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def validate_configuration(self) -> bool:
        """Validate API keys and required configuration.

        Returns ``True`` when the source is ready to operate.

        Raises
        ------
        ValueError
            When a required API key is missing or invalid.
        RuntimeError
            When configuration validation fails for any other reason.
        """

    @abstractmethod
    def search(self, query: str, page: int, per_page: int) -> list[SearchResult]:
        """Execute a search query and return matching image results.

        Parameters
        ----------
        query:
            Free-text search string.
        page:
            1-indexed page number.
        per_page:
            Number of results per page.

        Returns
        -------
        list[SearchResult]
            Matching images.  No downloading occurs.
        """

    @abstractmethod
    def download(
        self, result: SearchResult, destination_directory: Path
    ) -> DownloadResult:
        """Download a single image to the given directory.

        Parameters
        ----------
        result:
            The search result to download.
        destination_directory:
            Target folder for the saved file.

        Returns
        -------
        DownloadResult
            Outcome of the download attempt.
        """

    @abstractmethod
    def build_metadata(self, result: SearchResult, local_file: Path) -> ImageMetadata:
        """Construct structured metadata for a downloaded image.

        Parameters
        ----------
        result:
            The original search result.
        local_file:
            Path to the saved image on disk.

        Returns
        -------
        ImageMetadata
            Complete metadata record.
        """

    @abstractmethod
    def close(self) -> None:
        """Release any open sessions, connections, or resources."""
