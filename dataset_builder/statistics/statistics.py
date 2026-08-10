"""
Pure aggregation layer for dataset-building results.

Collects outputs from every pipeline stage into a single structured
summary.  This module performs no downloading, filtering, duplicate
detection, logging, or report generation.  It is a pure aggregation
layer intended to be consumed by :class:`ReportGenerator`,
:class:`DatasetBuilder`, console summaries, or dashboards.

Design Principles
-----------------
- **SOLID**: each responsibility lives in the originating module;
  this module only *reads* their outputs.
- **Frozen dataclasses**: every result is immutable and self-documenting.
- **No side effects**: no files are read or written; no folders scanned.
- **No duplicated logic**: values are taken verbatim from upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_builder.downloader.downloader import DownloadStatistics
from dataset_builder.duplicate_remover.duplicate_remover import DuplicateStatistics
from dataset_builder.face_filter.face_filter import FaceFilterStatistics
from dataset_builder.quality_filter.quality_filter import QualityFilterStatistics


# ------------------------------------------------------------------
# Summary dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class CollectionStatistics:
    """Thin wrapper around :class:`DownloadStatistics`.

    Maps download-stage output into the naming convention used by
    the statistics aggregation layer.
    """

    total_queries: int
    """Number of unique queries executed across all categories and sources."""

    total_sources: int
    """Number of image sources that contributed results."""

    downloaded_images: int
    """Number of images successfully saved to disk."""

    failed_downloads: int
    """Number of download attempts that did not produce a file."""

    download_success_rate: float
    """Fraction of attempts that succeeded (0.0 -- 1.0)."""


@dataclass(frozen=True)
class DuplicateStatisticsSummary:
    """Thin wrapper around :class:`DuplicateStatistics`.

    Reads values directly from the output of
    :meth:`DuplicateRemover.statistics` without recomputation.
    """

    total_images: int
    """Total number of images scanned for duplicates."""

    unique_images: int
    """Number of unique (non-duplicate) images."""

    duplicate_images: int
    """Number of duplicate images (candidates for removal)."""

    duplicate_groups: int
    """Number of distinct duplicate groups."""

    duplicate_ratio: float
    """Fraction of images that are duplicates (0.0 -- 1.0)."""


@dataclass(frozen=True)
class FaceFilterStatisticsSummary:
    """Thin wrapper around :class:`FaceFilterStatistics`.

    Reads all fields directly from the upstream output including
    the ``rejection_reason_distribution`` which is computed inside
    :meth:`FaceFilter.statistics`.
    """

    processed_images: int
    """Total number of images processed by the face filter."""

    accepted_images: int
    """Number of images that passed face filtering."""

    rejected_images: int
    """Number of images that failed face filtering."""

    acceptance_rate: float
    """Fraction of images accepted (0.0 -- 1.0)."""

    rejection_reason_distribution: dict[str, int]
    """Mapping of rejection reason to its occurrence count."""


@dataclass(frozen=True)
class QualityFilterStatisticsSummary:
    """Thin wrapper around :class:`QualityFilterStatistics`.

    Reads all fields directly from the upstream output including
    the ``rejection_reason_distribution`` which is computed inside
    :meth:`QualityFilter.statistics`.
    """

    processed_images: int
    """Total number of images processed by the quality filter."""

    accepted_images: int
    """Number of images that passed quality filtering."""

    rejected_images: int
    """Number of images that failed quality filtering."""

    acceptance_rate: float
    """Fraction of images accepted (0.0 -- 1.0)."""

    rejection_reason_distribution: dict[str, int]
    """Mapping of rejection reason to its occurrence count."""


# ------------------------------------------------------------------
# Main summary
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetStatistics:
    """Complete summary of the dataset-building pipeline.

    Composed of one summary per stage plus two cross-cutting
    metrics derived from the combined pipeline output.
    """

    collection: CollectionStatistics
    """Download / collection stage results."""

    duplicates: DuplicateStatisticsSummary
    """Duplicate detection stage results."""

    face_filter: FaceFilterStatisticsSummary
    """Face filtering stage results."""

    quality_filter: QualityFilterStatisticsSummary
    """Quality filtering stage results."""

    total_final_images: int
    """Number of images remaining after all pipeline stages."""

    overall_retention_rate: float
    """Fraction of originally downloaded images that survive the
    full pipeline (0.0 -- 1.0)."""


# ------------------------------------------------------------------
# Aggregator
# ------------------------------------------------------------------


class StatisticsAggregator:
    """Assemble a :class:`DatasetStatistics` from pipeline outputs.

    The aggregator receives pre-computed outputs from each stage and
    maps them into a single :class:`DatasetStatistics` object.  No
    computation is performed beyond:

    - Mapping upstream fields to summary fields.
    - Computing ``overall_retention_rate``.

    All values come verbatim from the originating modules.
    No rejection distributions are computed here.
    No download success rates are assumed.

    Usage
    -----
    ::

        aggregator = StatisticsAggregator()
        stats = aggregator.build(
            download_statistics=downloader.statistics,
            duplicate_statistics=remover.statistics(),
            face_statistics=face_filter.statistics(),
            quality_statistics=quality_filter.statistics(),
            final_image_count=len(final_images),
        )
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        download_statistics: DownloadStatistics,
        duplicate_statistics: DuplicateStatistics,
        face_statistics: FaceFilterStatistics,
        quality_statistics: QualityFilterStatistics,
        final_image_count: int,
    ) -> DatasetStatistics:
        """Construct a :class:`DatasetStatistics` from stage outputs.

        Parameters
        ----------
        download_statistics:
            Output of :attr:`Downloader.statistics`.
        duplicate_statistics:
            Output of :meth:`DuplicateRemover.statistics`.
        face_statistics:
            Output of :meth:`FaceFilter.statistics`.
        quality_statistics:
            Output of :meth:`QualityFilter.statistics`.
        final_image_count:
            Number of images remaining after the complete pipeline.

        Returns
        -------
        DatasetStatistics
            Immutable summary of the entire pipeline.
        """
        collection = self._map_collection(download_statistics)
        dup_summary = self._map_duplicate_summary(duplicate_statistics)
        face_summary = self._map_face_summary(face_statistics)
        quality_summary = self._map_quality_summary(quality_statistics)

        downloaded = collection.downloaded_images
        retention = (
            final_image_count / downloaded
            if downloaded > 0
            else 0.0
        )

        return DatasetStatistics(
            collection=collection,
            duplicates=dup_summary,
            face_filter=face_summary,
            quality_filter=quality_summary,
            total_final_images=final_image_count,
            overall_retention_rate=retention,
        )

    # ------------------------------------------------------------------
    # Internal mappers (read-only, no computation)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_collection(stats: DownloadStatistics) -> CollectionStatistics:
        """Map :class:`DownloadStatistics` to :class:`CollectionStatistics`."""
        return CollectionStatistics(
            total_queries=stats.total_queries,
            total_sources=stats.total_sources,
            downloaded_images=stats.successful_downloads,
            failed_downloads=stats.failed_downloads,
            download_success_rate=stats.success_rate,
        )

    @staticmethod
    def _map_duplicate_summary(
        stats: DuplicateStatistics,
    ) -> DuplicateStatisticsSummary:
        """Map :class:`DuplicateStatistics` to summary."""
        return DuplicateStatisticsSummary(
            total_images=stats.total_images,
            unique_images=stats.unique_images,
            duplicate_images=stats.duplicate_images,
            duplicate_groups=stats.duplicate_groups,
            duplicate_ratio=stats.duplicate_ratio,
        )

    @staticmethod
    def _map_face_summary(
        stats: FaceFilterStatistics,
    ) -> FaceFilterStatisticsSummary:
        """Map :class:`FaceFilterStatistics` to summary."""
        return FaceFilterStatisticsSummary(
            processed_images=stats.total_images,
            accepted_images=stats.accepted_images,
            rejected_images=stats.rejected_images,
            acceptance_rate=stats.acceptance_ratio,
            rejection_reason_distribution=dict(stats.rejection_reason_distribution),
        )

    @staticmethod
    def _map_quality_summary(
        stats: QualityFilterStatistics,
    ) -> QualityFilterStatisticsSummary:
        """Map :class:`QualityFilterStatistics` to summary."""
        return QualityFilterStatisticsSummary(
            processed_images=stats.total_images,
            accepted_images=stats.accepted_images,
            rejected_images=stats.rejected_images,
            acceptance_rate=stats.acceptance_ratio,
            rejection_reason_distribution=dict(stats.rejection_reason_distribution),
        )
