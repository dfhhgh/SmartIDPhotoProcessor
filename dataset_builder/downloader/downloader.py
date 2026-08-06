"""
Orchestration layer for coordinated dataset downloads.

The Downloader iterates categories, queries, and sources, delegating
all network and persistence details to :class:`BaseSource` implementations.
It never performs HTTP requests or file I/O directly.
"""

from __future__ import annotations

from pathlib import Path

from dataset_builder.config.settings import Settings
from dataset_builder.queries.query_loader import QueryLoader
from dataset_builder.sources.base_source import BaseSource, ImageMetadata


class Downloader:
    """Coordinate dataset downloads across categories and sources.

    The Downloader acts purely as an orchestrator: it loads queries,
    iterates sources, and collects :class:`ImageMetadata` records.
    All actual network and file operations are delegated to the
    supplied :class:`BaseSource` instances.

    Images are saved into category-specific subdirectories under
    ``Settings.RAW_IMAGES_DIR`` (e.g. ``raw/hijab/``, ``raw/mask/``).

    Usage
    -----
    ::

        with Downloader(settings, query_loader, sources) as dl:
            dl.download_all()
            metadata = dl.metadata
    """

    def __init__(
        self,
        settings: Settings,
        query_loader: QueryLoader,
        sources: list[BaseSource],
    ) -> None:
        self._settings: Settings = settings
        self._query_loader: QueryLoader = query_loader
        self._sources: list[BaseSource] = sources
        self._metadata: list[ImageMetadata] = []

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Downloader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> list[ImageMetadata]:
        """Return all :class:`ImageMetadata` collected during this session."""
        return list(self._metadata)

    def download_category(self, category_name: str) -> None:
        """Download images for a single category from all enabled sources.

        Images are saved into ``RAW_IMAGES_DIR/<category_name>/``.

        Parameters
        ----------
        category_name:
            Category identifier matching a ``.txt`` file in the queries
            directory.

        Raises
        ------
        FileNotFoundError
            When the category file does not exist.
        ValueError
            When the category file contains no valid queries.
        """
        queries = self._query_loader.load_category(category_name)

        for source in self._sources:
            for query in queries:
                self._download_query(source, query, category_name)

    def download_all(self) -> None:
        """Download images for every available category from all sources.

        Iterates categories returned by :meth:`QueryLoader.categories`,
        then iterates every enabled source and every query within each
        category.
        """
        categories = self._query_loader.categories()

        for category_name in categories:
            self.download_category(category_name)

    def close(self) -> None:
        """Close all source connections and release resources."""
        for source in self._sources:
            source.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_query(
        self, source: BaseSource, query: str, category_name: str
    ) -> None:
        """Search and download images for a single query from one source."""
        search_results = source.search(
            query=query,
            page=1,
            per_page=self._settings.MAX_IMAGES_PER_QUERY,
        )

        destination = self._settings.RAW_IMAGES_DIR / category_name
        destination.mkdir(parents=True, exist_ok=True)

        for result in search_results:
            download_result = source.download(result, destination)

            if not download_result.success:
                continue

            if download_result.local_path is None:
                continue

            image_metadata = source.build_metadata(result, download_result.local_path)
            self._metadata.append(image_metadata)
