"""
Composition root for the Dataset Builder pipeline.

This module is the **only** place where concrete implementations are
instantiated and wired together.  Every other module receives its
dependencies via constructor injection.

Execute via::

    python -m dataset_builder
    python -m dataset_builder --categories hijab
    python -m dataset_builder --categories hijab --max-per-query 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file before any Settings() instantiation
load_dotenv()

from dataset_builder.config.settings import Settings
from dataset_builder.dataset_pipeline import DatasetBuilder
from dataset_builder.downloader.downloader import Downloader
from dataset_builder.duplicate_remover.duplicate_remover import DuplicateRemover
from dataset_builder.face_filter.face_filter import FaceFilter
from dataset_builder.metadata.metadata_manager import MetadataManager
from dataset_builder.quality_filter.quality_filter import QualityFilter
from dataset_builder.queries.query_loader import QueryLoader
from dataset_builder.reports.report_generator import ReportGenerator
from dataset_builder.sources.base_source import BaseSource
from dataset_builder.sources.pexels import PexelsSource
from dataset_builder.sources.pixabay import PixabaySource
from dataset_builder.statistics.statistics import StatisticsAggregator


# ------------------------------------------------------------------
# Source factory
# ------------------------------------------------------------------


def _create_sources(settings: Settings) -> list[BaseSource]:
    """Instantiate and validate every enabled source.

    Invalid sources are skipped with a warning.  Returns an empty
    list when zero sources are valid.

    Parameters
    ----------
    settings:
        Application configuration.

    Returns
    -------
    list[BaseSource]
        Validated source instances ready for download.
    """
    source_registry: dict[str, type[BaseSource]] = {
        "pexels": PexelsSource,
        "pixabay": PixabaySource,
    }

    sources: list[BaseSource] = []

    for source_name in settings.ENABLED_SOURCES:
        source_cls = source_registry.get(source_name)
        if source_cls is None:
            _print_warning(f"Unknown source '{source_name}' — skipped.")
            continue

        try:
            source = source_cls(settings)
            source.validate_configuration()
            sources.append(source)
            _print_info(f"  {source_name}: enabled")
        except ValueError as exc:
            _print_warning(f"  {source_name}: disabled — {exc}")
        except Exception as exc:
            _print_warning(f"  {source_name}: disabled — {exc}")

    return sources


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------


_PREFIX = "Dataset Builder"

_BANNER = (
    "\n"
    "====================================\n"
    "  Dataset Builder\n"
    "====================================\n"
)


def _print_info(message: str) -> None:
    """Print an informational line to stdout."""
    print(message)


def _print_warning(message: str) -> None:
    """Print a warning line to stderr."""
    print(f"  Warning: {message}", file=sys.stderr)


def _print_error(message: str) -> None:
    """Print an error line to stderr."""
    print(f"\nError: {message}", file=sys.stderr)


def _print_summary(stats: object) -> None:
    """Print the execution summary to stdout."""
    collection = getattr(stats, "collection", None)
    duplicates = getattr(stats, "duplicates", None)
    face_filter = getattr(stats, "face_filter", None)
    quality_filter = getattr(stats, "quality_filter", None)

    downloaded = getattr(collection, "downloaded_images", 0) if collection else 0
    dup_removed = getattr(duplicates, "duplicate_images", 0) if duplicates else 0
    face_accepted = getattr(face_filter, "accepted_images", 0) if face_filter else 0
    quality_accepted = getattr(quality_filter, "accepted_images", 0) if quality_filter else 0
    final = getattr(stats, "total_final_images", 0)

    print("\n------------------------------------")
    print("Dataset Builder Finished")
    print("------------------------------------")
    print(f"Downloaded:         {downloaded}")
    print(f"Duplicates removed: {dup_removed}")
    print(f"Face accepted:      {face_accepted}")
    print(f"Quality accepted:   {quality_accepted}")
    print(f"Final images:       {final}")
    print("")
    print("Reports:            reports/")
    print("Metadata:           dataset/metadata/")
    print("------------------------------------")
    print("Execution successful.")
    print("")


# ------------------------------------------------------------------
# Composition root
# ------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``categories`` and ``max_per_query``.
    """
    parser = argparse.ArgumentParser(
        prog="dataset_builder",
        description="Build face-image datasets from public photo APIs.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CATEGORY",
        help=(
            "One or more category names to process "
            "(default: all available categories)."
        ),
    )
    parser.add_argument(
        "--max-per-query",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum number of images to fetch per search query "
            "(default: use Settings value)."
        ),
    )
    return parser.parse_args(argv)


def _validate_categories(
    requested: list[str] | None,
    available: list[str],
) -> list[str] | None:
    """Validate requested categories against available ones.

    Parameters
    ----------
    requested:
        Categories requested via CLI, or ``None`` for all.
    available:
        Categories discovered by :class:`QueryLoader`.

    Returns
    -------
    list[str] | None
        Validated category list, or ``None`` if all should be processed.

    Raises
    ------
    SystemExit
        When a requested category is not found.
    """
    if requested is None:
        return None

    invalid = [c for c in requested if c not in available]
    if invalid:
        for name in invalid:
            _print_error(f"Category not found: {name}")
        _print_info(f"Available categories: {', '.join(available)}")
        sys.exit(1)

    return requested


def _validate_max_per_query(value: int | None) -> int | None:
    """Validate the max-per-query argument.

    Parameters
    ----------
    value:
        Parsed value from argparse, or ``None``.

    Returns
    -------
    int | None
        Validated value, or ``None``.

    Raises
    ------
    SystemExit
        When the value is not a positive integer.
    """
    if value is None:
        return None
    if value <= 0:
        _print_error(
            f"--max-per-query must be a positive integer, got {value}."
        )
        sys.exit(1)
    return value


def main() -> None:
    """Wire all dependencies and execute the full pipeline.

    Catches ``KeyboardInterrupt`` and unexpected exceptions to
    provide clean exit messages without stack traces.
    """
    print(_BANNER)

    try:
        _run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        _print_error(f"Unexpected error: {exc}")
        sys.exit(1)


def _run() -> None:
    """Internal runner separated for clean exception handling."""
    args = parse_args()
    settings = Settings()

    # ---- Validate queries directory ----
    queries_dir = settings.QUERIES_DIR
    if not queries_dir.exists():
        _print_error(f"Queries directory not found: {queries_dir}")
        sys.exit(1)

    query_loader = QueryLoader(queries_dir)
    available_categories = query_loader.categories()
    if not available_categories:
        _print_error(f"No query files found in: {queries_dir}")
        sys.exit(1)

    _print_info(f"Categories found: {len(available_categories)}")

    # ---- Validate CLI arguments ----
    selected_categories = _validate_categories(
        args.categories, available_categories
    )
    max_per_query = _validate_max_per_query(args.max_per_query)

    if selected_categories is not None:
        _print_info(f"Selected categories: {', '.join(selected_categories)}")
    if max_per_query is not None:
        _print_info(f"Max images per query: {max_per_query}")

    # ---- Create and validate sources ----
    _print_info("Validating sources...")
    sources = _create_sources(settings)
    if not sources:
        _print_error("No valid sources. Set PEXELS_API_KEY or PIXABAY_API_KEY.")
        sys.exit(1)

    # ---- Wire all dependencies ----
    metadata_manager = MetadataManager()
    duplicate_remover = DuplicateRemover(settings)
    face_filter = FaceFilter(settings)
    quality_filter = QualityFilter(settings)
    statistics_aggregator = StatisticsAggregator()
    report_generator = ReportGenerator(settings)

    downloader = Downloader(
        settings, query_loader, sources, max_per_query=max_per_query
    )

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

    # ---- Execute pipeline ----
    _print_info("Starting pipeline...\n")
    stats = builder.build(categories=selected_categories)

    # ---- Summary ----
    _print_summary(stats)
