"""
Incremental dataset collection for BiSeNet fine-tuning.

Supports:
- incremental collection (never re-downloads existing images)
- source-level and perceptual deduplication
- face filtering (single face only)
- pagination support
- persistent collection state
- dry-run mode (preview without downloading)
- per-category limits
- resumable collection
- logging and reporting

Usage:
    python scripts/collect_pilot.py                    # dry-run by default
    python scripts/collect_pilot.py --execute          # actually download
    python scripts/collect_pilot.py --execute --pilot  # pilot collection
    python scripts/collect_pilot.py --categories normal eyeglasses
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from dataset_builder.collection.incremental_collector import (
    CollectionStats,
    IncrementalCollector,
)
from dataset_builder.config.collection_config import (
    CategoryConfig,
    CategoryType,
    CollectionConfig,
    Priority,
)
from dataset_builder.config.settings import Settings
from dataset_builder.queries.query_loader import QueryLoader
from dataset_builder.sources.base_source import BaseSource
from dataset_builder.sources.pexels import PexelsSource
from dataset_builder.sources.pixabay import PixabaySource
from dataset_builder.sources.openverse import OpenverseSource
from dataset_builder.sources.wikimedia_commons import WikimediaCommonsSource


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCE_REGISTRY: dict[str, type[BaseSource]] = {
    "pexels": PexelsSource,
    "pixabay": PixabaySource,
    "openverse": OpenverseSource,
    "wikimedia_commons": WikimediaCommonsSource,
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="collect_pilot",
        description="Incremental dataset collection for BiSeNet fine-tuning.",
    )

    parser.add_argument(
        "--inventory",
        action="store_true",
        default=False,
        help="Show raw dataset inventory and exit (no collection).",
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        default=False,
        help="Actually download images (default is dry-run).",
    )

    parser.add_argument(
        "--pilot",
        action="store_true",
        default=False,
        help="Use pilot collection limits (30 images per category).",
    )

    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        metavar="CATEGORY",
        help="Categories to collect (default: all).",
    )

    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        metavar="SOURCE",
        help="Sources to use (default: all enabled).",
    )

    parser.add_argument(
        "--max-per-category",
        type=int,
        default=None,
        metavar="N",
        help="Max images per category.",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help="Max pages per query (default: from settings).",
    )

    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        metavar="N",
        help="Max total images across all categories.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for downloaded images.",
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for collection reports.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Source creation
# ---------------------------------------------------------------------------

def create_sources(
    settings: Settings,
    source_names: list[str] | None = None,
) -> list[BaseSource]:
    """Create and validate source instances."""
    names = source_names or list(SOURCE_REGISTRY.keys())
    sources: list[BaseSource] = []

    for name in names:
        cls = SOURCE_REGISTRY.get(name)
        if cls is None:
            print(f"  Warning: Unknown source '{name}' — skipped.")
            continue

        try:
            source = cls(settings)
            source.validate_configuration()
            sources.append(source)
            print(f"  {name}: enabled")
        except ValueError as exc:
            print(f"  {name}: disabled — {exc}")
        except Exception as exc:
            print(f"  {name}: disabled — {exc}")

    return sources


# ---------------------------------------------------------------------------
# Inventory display
# ---------------------------------------------------------------------------

def print_inventory(settings: Settings) -> None:
    """Print the raw dataset inventory and exit."""
    from dataset_builder.collection.inventory import Inventory

    inv = Inventory(settings)
    result = inv.inventory_all()

    print("=" * 60)
    print("RAW DATASET INVENTORY")
    print("=" * 60)

    for cat_name in sorted(result.categories.keys()):
        cat_inv = result.categories[cat_name]
        print(f"\n  {cat_name}:")
        print(f"    valid:           {cat_inv.valid_images}")
        print(f"    invalid:         {cat_inv.invalid_images}")
        print(f"    duplicates:      {cat_inv.duplicate_images}")
        print(f"    unique_valid:    {cat_inv.unique_valid}")

    if result.uncategorized_files:
        print(f"\n  uncategorized (files directly under raw/):")
        for f in result.uncategorized_files:
            print(f"    {f.name}")

    if result.unknown_dirs:
        print(f"\n  unknown directories:")
        for d in result.unknown_dirs:
            print(f"    {d}/")

    print(f"\n  {'TOTAL':18s}")
    print(f"    files:           {result.total_files}")
    print(f"    valid:           {result.total_valid}")
    print(f"    invalid:         {result.total_invalid}")
    print(f"    duplicates:      {result.total_duplicates}")
    print(f"    unique_valid:    {result.total_unique_valid}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_report(stats: CollectionStats) -> None:
    """Print a human-readable collection report."""
    print("\n" + "=" * 60)
    print("COLLECTION REPORT")
    print("=" * 60)
    print(f"Mode: {stats.mode}")
    print(f"Generated: {stats.generated_at}")
    print(f"Total target: {stats.total_target}")
    print(f"Total existing valid: {stats.total_existing_valid}")
    print(f"Total remaining: {stats.total_remaining}")
    print(f"Total accepted this run: {stats.total_accepted}")
    print()

    for cat in stats.categories:
        if cat.remaining == 0 and cat.accepted_this_run == 0:
            status = "COMPLETE"
        elif cat.accepted_this_run > 0:
            status = "COLLECTED"
        else:
            status = "INCOMPLETE"

        print(f"{cat.category}:")
        print(f"  Target: {cat.target}")
        print(f"  Existing valid: {cat.existing_valid}")
        print(f"  Existing duplicates: {cat.existing_duplicates}")
        print(f"  Existing invalid: {cat.existing_invalid}")
        print(f"  Remaining: {cat.remaining}")
        print(f"  Accepted this run: {cat.accepted_this_run} [{status}]")
        print(f"  Final count: {cat.final_valid_unique}")
        print(f"  Search attempts: {cat.search_attempts}")
        print(f"  Download attempts: {cat.download_attempts}")
        print(f"  Rejected (seen): {cat.rejected_seen}")
        print(f"  Rejected (duplicate): {cat.rejected_duplicate}")
        print(f"  Rejected (multiple faces): {cat.rejected_multiple_faces}")
        print(f"  Rejected (no face): {cat.rejected_no_face}")
        print(f"  Rejected (face too small): {cat.rejected_face_too_small}")
        print(f"  Rejected (profile): {cat.rejected_profile}")
        print(f"  Rejected (decode): {cat.rejected_decode}")
        print(f"  Rejected (download): {cat.rejected_download}")
        print(f"  Queries executed: {cat.queries_executed}")
        print(f"  Pages searched: {cat.pages_searched}")

    print("=" * 60)


def save_report(stats: CollectionStats, output_path: Path) -> None:
    """Save collection report as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generated_at": stats.generated_at,
        "mode": stats.mode,
        "total_target": stats.total_target,
        "total_existing_valid": stats.total_existing_valid,
        "total_remaining": stats.total_remaining,
        "total_accepted": stats.total_accepted,
        "categories": [
            {
                "category": cat.category,
                "target": cat.target,
                "existing_valid": cat.existing_valid,
                "existing_duplicates": cat.existing_duplicates,
                "existing_invalid": cat.existing_invalid,
                "remaining": cat.remaining,
                "accepted_this_run": cat.accepted_this_run,
                "final_valid_unique": cat.final_valid_unique,
                "search_attempts": cat.search_attempts,
                "download_attempts": cat.download_attempts,
                "rejected_seen": cat.rejected_seen,
                "rejected_duplicate": cat.rejected_duplicate,
                "rejected_multiple_faces": cat.rejected_multiple_faces,
                "rejected_no_face": cat.rejected_no_face,
                "rejected_face_too_small": cat.rejected_face_too_small,
                "rejected_profile": cat.rejected_profile,
                "rejected_decode": cat.rejected_decode,
                "rejected_download": cat.rejected_download,
                "queries_executed": cat.queries_executed,
                "pages_searched": cat.pages_searched,
            }
            for cat in stats.categories
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nReport saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run incremental collection."""
    args = parse_args()

    # Configuration
    config = CollectionConfig()
    settings = Settings()

    # Handle --inventory mode
    if args.inventory:
        print_inventory(settings)
        return 0

    dry_run = not args.execute
    mode = "dry_run" if dry_run else ("pilot" if args.pilot else "full")

    print("=" * 60)
    print("INCREMENTAL DATASET COLLECTION")
    print("=" * 60)
    print(f"Mode: {mode.upper()}")
    print(f"Dry run: {dry_run}")
    print()

    # Query loader
    query_loader = QueryLoader(settings.QUERIES_DIR)
    available = query_loader.categories()
    print(f"Available categories: {', '.join(available)}")

    # Filter categories
    if args.categories:
        selected = [c for c in args.categories if c in available]
        invalid = [c for c in args.categories if c not in available]
        if invalid:
            print(f"Warning: Unknown categories: {', '.join(invalid)}")
    else:
        selected = available

    print(f"Selected categories: {', '.join(selected)}")

    # Create sources
    print("\nCreating sources...")
    sources = create_sources(settings, args.sources)
    if not sources:
        print("Error: No valid sources.")
        return 1

    # Create collector
    print("\nInitializing incremental collector...")
    collector = IncrementalCollector(
        settings=settings,
        config=config,
        query_loader=query_loader,
        sources=sources,
    )

    # Collection limits
    cat_limit = args.max_per_category or (
        config.pilot_images_per_category if args.pilot else config.max_images_per_category
    )
    total_limit = args.max_total

    print(f"Category limit: {cat_limit}")
    if total_limit:
        print(f"Total limit: {total_limit}")

    # Execute collection
    print("\n" + "-" * 40)
    print("Starting collection...")
    print("-" * 40)

    start_time = time.time()

    stats = collector.collect_all(
        dry_run=dry_run,
        categories=selected,
        category_limit=cat_limit,
        max_pages=args.max_pages,
    )

    elapsed = time.time() - start_time

    # Print and save report
    print_report(stats)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = args.report_dir or settings.REPORTS_DIR
    report_path = report_dir / f"collection_{mode}_{timestamp}.json"
    save_report(stats, report_path)

    # Only save state in real mode
    if not dry_run:
        collector.save_state()

    print(f"\n{'DRY RUN' if dry_run else 'COLLECTION'} COMPLETE")
    print(f"Total accepted: {stats.total_accepted}")
    print(f"Elapsed time: {elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
