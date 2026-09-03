"""Service manager for loading and managing the singleton ReverseSearchService.

Follows the same singleton + lazy-loading architecture as FaceService
and FaceParserService.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from config.settings import Settings
from search.reverse_search_service import ReverseSearchService

logger = logging.getLogger(__name__)


class ReverseSearchServiceManager:
    """Manages the process-wide singleton instance of ReverseSearchService."""

    _instance: ReverseSearchServiceManager | None = None
    _initialized: bool = False
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> ReverseSearchServiceManager:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        index_path: Path | str | None = None,
        metadata_path: Path | str | None = None,
        enabled: bool | None = None,
    ) -> None:
        if self._initialized:
            return

        settings = Settings()
        self._enabled: bool = (
            enabled if enabled is not None else getattr(settings, "REVERSE_SEARCH_ENABLED", False)
        )
        self._index_path: Path | None = (
            Path(index_path) if index_path is not None else getattr(settings, "REVERSE_SEARCH_INDEX_PATH", None)
        )
        self._metadata_path: Path | None = (
            Path(metadata_path) if metadata_path is not None else getattr(settings, "REVERSE_SEARCH_METADATA_PATH", None)
        )

        self._service: ReverseSearchService | None = None
        self._load_lock = threading.Lock()
        self._initialized = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_service(self) -> ReverseSearchService | None:
        """Return the loaded ReverseSearchService singleton, or None if disabled."""
        if not self._enabled:
            return None

        if self._service is None:
            with self._load_lock:
                if self._service is None:
                    if self._index_path is None or self._metadata_path is None:
                        raise RuntimeError(
                            "Reverse search is enabled, but REVERSE_SEARCH_INDEX_PATH "
                            "or REVERSE_SEARCH_METADATA_PATH is not configured."
                        )
                    logger.info(
                        "Initializing ReverseSearchService from index=%s, metadata=%s...",
                        self._index_path,
                        self._metadata_path,
                    )
                    self._service = ReverseSearchService(
                        index_path=self._index_path,
                        metadata_path=self._metadata_path,
                    )
        return self._service
