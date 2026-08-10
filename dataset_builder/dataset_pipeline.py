"""
Top-level orchestrator for the Dataset Builder pipeline.

Coordinates every stage of the dataset-building workflow without
duplicating any business logic.  All component instances are
injected via the constructor; nothing is created internally.

Pipeline Order
--------------
1. :class:`Downloader` -- fetch images from external sources.
2. :class:`MetadataManager` -- persist download metadata.
3. :class:`DuplicateRemover` -- detect and move perceptual duplicates.
4. :class:`FaceFilter` -- accept/reject images based on face geometry.
5. :class:`QualityFilter` -- accept/reject images based on technical quality.
6. :class:`StatisticsAggregator` -- assemble a single summary object.
7. :class:`ReportGenerator` -- write Markdown, JSON, and CSV reports.

Design Principles
-----------------
- **Clean Architecture**: the orchestrator depends only on abstractions
  injected at construction time.
- **SOLID**: each responsibility belongs to the injected component.
- **Zero duplicated logic**: no HTTP, no detection, no filtering, no
  statistics computation, no report rendering.
- **No global state**: all dependencies are held as instance attributes.
"""

from __future__ import annotations

from pathlib import Path

from dataset_builder.config.settings import Settings
from dataset_builder.downloader.downloader import Downloader
from dataset_builder.duplicate_remover.duplicate_remover import DuplicateRemover
from dataset_builder.face_filter.face_filter import FaceFilter
from dataset_builder.metadata.metadata_manager import MetadataManager
from dataset_builder.quality_filter.quality_filter import QualityFilter
from dataset_builder.reports.report_generator import ReportGenerator
from dataset_builder.statistics.statistics import (
    DatasetStatistics,
    StatisticsAggregator,
)


class DatasetBuilder:
    """High-level coordinator for the dataset-building pipeline.

    Receives all collaborators via dependency injection and exposes
    a single :meth:`build` method that executes the full pipeline.

    The builder never performs HTTP requests, duplicate detection,
    face detection, quality validation, statistics computation, or
    report generation itself.  It delegates every operation to the
    appropriate injected component.

    Parameters
    ----------
    settings:
        Application configuration.
    query_loader:
        Loads search queries from disk.
    downloader:
        Fetches images from external sources.
    metadata_manager:
        Persists download metadata.
    duplicate_remover:
        Detects and moves perceptual duplicates.
    face_filter:
        Accepts/rejects images based on face geometry.
    quality_filter:
        Accepts/rejects images based on technical quality.
    statistics_aggregator:
        Assembles a single summary from stage outputs.
    report_generator:
        Renders Markdown, JSON, and CSV reports.

    Examples
    --------
    ::

        builder = DatasetBuilder(
            settings=settings,
            query_loader=query_loader,
            downloader=downloader,
            metadata_manager=metadata_manager,
            duplicate_remover=duplicate_remover,
            face_filter=face_filter,
            quality_filter=quality_filter,
            statistics_aggregator=statistics_aggregator,
            report_generator=report_generator,
        )
        stats = builder.build()
    """

    def __init__(
        self,
        *,
        settings: Settings,
        query_loader: object,
        downloader: Downloader,
        metadata_manager: MetadataManager,
        duplicate_remover: DuplicateRemover,
        face_filter: FaceFilter,
        quality_filter: QualityFilter,
        statistics_aggregator: StatisticsAggregator,
        report_generator: ReportGenerator,
    ) -> None:
        self._settings: Settings = settings
        self._query_loader: object = query_loader
        self._downloader: Downloader = downloader
        self._metadata_manager: MetadataManager = metadata_manager
        self._duplicate_remover: DuplicateRemover = duplicate_remover
        self._face_filter: FaceFilter = face_filter
        self._quality_filter: QualityFilter = quality_filter
        self._statistics_aggregator: StatisticsAggregator = statistics_aggregator
        self._report_generator: ReportGenerator = report_generator

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def settings(self) -> Settings:
        """Return the application settings."""
        return self._settings

    @property
    def downloader(self) -> Downloader:
        """Return the downloader instance."""
        return self._downloader

    @property
    def metadata_manager(self) -> MetadataManager:
        """Return the metadata manager instance."""
        return self._metadata_manager

    @property
    def duplicate_remover(self) -> DuplicateRemover:
        """Return the duplicate remover instance."""
        return self._duplicate_remover

    @property
    def face_filter(self) -> FaceFilter:
        """Return the face filter instance."""
        return self._face_filter

    @property
    def quality_filter(self) -> QualityFilter:
        """Return the quality filter instance."""
        return self._quality_filter

    @property
    def statistics_aggregator(self) -> StatisticsAggregator:
        """Return the statistics aggregator instance."""
        return self._statistics_aggregator

    @property
    def report_generator(self) -> ReportGenerator:
        """Return the report generator instance."""
        return self._report_generator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        categories: list[str] | None = None,
    ) -> DatasetStatistics:
        """Execute the full dataset-building pipeline.

        Parameters
        ----------
        categories:
            Optional list of category names to process.  When ``None``,
            all available categories are processed (default behaviour).

        Returns
        -------
        DatasetStatistics
            Immutable summary of the entire pipeline run.
        """
        self._step_download(categories=categories)
        self._step_metadata()
        self._step_duplicate_removal()
        self._step_face_filter()
        self._step_quality_filter()
        return self._step_statistics_and_reports()

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    def _step_download(
        self,
        *,
        categories: list[str] | None = None,
    ) -> None:
        """STEP 1: Download images from all enabled sources.

        Parameters
        ----------
        categories:
            Optional list of category names to download.  When ``None``,
            all available categories are downloaded.
        """
        if categories is not None:
            self._downloader.download_categories(categories)
        else:
            self._downloader.download_all()

    def _step_metadata(self) -> None:
        """STEP 2: Persist downloader metadata to disk."""
        metadata = self._downloader.metadata
        self._metadata_manager.add_many(metadata)

        metadata_dir = self._settings.METADATA_DIR
        self._metadata_manager.save_json(metadata_dir / "metadata.json")
        self._metadata_manager.save_csv(metadata_dir / "metadata.csv")

    def _step_duplicate_removal(self) -> None:
        """STEP 3: Detect and move perceptual duplicates."""
        raw_dir = self._settings.RAW_IMAGES_DIR
        dup_dir = self._settings.DUPLICATES_DIR

        self._duplicate_remover.scan(raw_dir)
        self._duplicate_remover.move_duplicates(dup_dir)

    def _step_face_filter(self) -> None:
        """STEP 4: Run face detection and filtering."""
        raw_dir = self._settings.RAW_IMAGES_DIR
        self._face_filter.scan(raw_dir)

    def _step_quality_filter(self) -> None:
        """STEP 5: Run quality validation filtering."""
        raw_dir = self._settings.RAW_IMAGES_DIR
        self._quality_filter.scan(raw_dir)

    def _step_statistics_and_reports(self) -> DatasetStatistics:
        """STEP 6 & 7: Aggregate statistics and generate reports."""
        stats = self._statistics_aggregator.build(
            download_statistics=self._downloader.statistics,
            duplicate_statistics=self._duplicate_remover.statistics(),
            face_statistics=self._face_filter.statistics(),
            quality_statistics=self._quality_filter.statistics(),
            final_image_count=len(self._quality_filter.accepted),
        )

        reports_dir = self._settings.REPORTS_DIR
        self._report_generator.generate_markdown(
            stats, reports_dir / "report.md"
        )
        self._report_generator.generate_json(
            stats, reports_dir / "report.json"
        )
        self._report_generator.generate_csv(
            stats, reports_dir / "report.csv"
        )

        return stats
