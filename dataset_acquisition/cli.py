"""CLI for dataset acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from dataset_acquisition.models import Person, CollectionStats
from dataset_acquisition.downloader import Downloader
from dataset_acquisition.splitter import split_reference_query, copy_split_to_disk
from dataset_acquisition.manifest import generate_manifest, generate_quality_report
from dataset_acquisition.sources.wikimedia import WikimediaSource


def load_people(path: Path) -> list[Person]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Person.from_dict(p) for p in data["people"]]


def check_leakage(reference_dir: Path, query_dir: Path) -> list[str]:
    """Verify no identical files exist between reference and query splits."""
    ref_hashes: dict[str, str] = {}
    for p in reference_dir.rglob("*"):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            ref_hashes[h] = str(p)

    issues: list[str] = []
    for p in query_dir.rglob("*"):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            if h in ref_hashes:
                issues.append(f"LEAKAGE: {p} matches {ref_hashes[h]}")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dataset_acquisition",
        description="Identity-labeled celebrity dataset acquisition tool.",
    )
    parser.add_argument(
        "--people",
        type=Path,
        default=Path("dataset_acquisition/people.json"),
        help="Path to people.json manifest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/celebrity-v2-pilot"),
        help="Output directory.",
    )
    parser.add_argument(
        "--max-images-per-person",
        type=int,
        default=10,
        help="Maximum images to collect per person.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic operations.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous download state.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.5,
        help="Delay between downloads in seconds (default: 2.5).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without downloading.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["wikimedia_commons"],
        help="Sources to use.",
    )
    parser.add_argument(
        "--min-reference",
        type=int,
        default=2,
        help="Minimum reference images per person.",
    )
    parser.add_argument(
        "--min-query",
        type=int,
        default=1,
        help="Minimum query images per person.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.people.exists():
        print(f"Error: People manifest not found: {args.people}", file=sys.stderr)
        return 1

    people = load_people(args.people)
    print(f"Loaded {len(people)} target identities.")

    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Output: {args.output}")
        print(f"Max images/person: {args.max_images_per_person}")
        print(f"Sources: {args.sources}")
        print(f"Seed: {args.seed}")
        print("\nTarget identities:")
        for p in people:
            queries = p.search_queries or [p.display_name]
            print(f"  {p.person_id}: {p.display_name} ({p.category}) — queries: {queries}")
        return 0

    face_service = None
    try:
        from services.face_service import FaceService
        face_service = FaceService()
        print("InsightFace/FaceService loaded for face validation.")
    except Exception as exc:
        print(f"Warning: Could not load FaceService ({exc}). Using fallback.")

    sources = []
    if "wikimedia_commons" in args.sources:
        sources.append(WikimediaSource(delay=args.delay))

    if not sources:
        print("Error: No valid sources specified.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    downloader = Downloader(
        output_dir=args.output,
        sources=sources,
        max_images_per_person=args.max_images_per_person,
        delay=args.delay,
        face_service=face_service,
    )

    all_records = []
    stats = CollectionStats()

    for i, person in enumerate(people):
        print(f"\n[{i+1}/{len(people)}] Processing: {person.display_name}")
        records = downloader.download_person(person)
        all_records.extend(records)
        print(f"  Downloaded: {len(records)} images")

    seen_hashes = set()
    unique_records = []
    for r in all_records:
        if r.sha256 not in seen_hashes:
            seen_hashes.add(r.sha256)
            unique_records.append(r)
        else:
            stats.total_duplicates += 1

    stats.total_no_face = sum(1 for r in unique_records if r.status == "no_face")
    stats.total_representation = sum(1 for r in unique_records if r.status == "representation")
    stats.total_identity_uncertain = sum(1 for r in unique_records if r.identity_status == "uncertain")
    stats.total_valid = sum(1 for r in unique_records if r.status == "valid")

    print(f"\nAfter deduplication: {len(unique_records)} unique images ({stats.total_duplicates} duplicates)")
    print(f"  Valid: {stats.total_valid}")
    print(f"  No-face: {stats.total_no_face}")
    print(f"  Representation: {stats.total_representation}")
    print(f"  Identity-uncertain: {stats.total_identity_uncertain}")

    split = split_reference_query(
        unique_records,
        reference_ratio=0.6,
        min_reference=args.min_reference,
        min_query=args.min_query,
        seed=args.seed,
    )

    copy_stats = copy_split_to_disk(split, args.output)
    print(f"\nReference: {copy_stats['reference_images']} images")
    print(f"Query: {copy_stats['query_images']} images")
    print(f"Excluded persons: {copy_stats['excluded_persons']}")

    ref_dir = args.output / "reference"
    query_dir = args.output / "query"
    if ref_dir.exists() and query_dir.exists():
        leakage = check_leakage(ref_dir, query_dir)
        if leakage:
            print(f"\nWARNING: {len(leakage)} cross-split leakage issues detected!")
            for issue in leakage:
                print(f"  {issue}")
        else:
            print("\nCross-split leakage check: PASSED (no shared hashes)")

    # Derive stats from ImageRecord data (single source of truth)
    stats = CollectionStats(
        total_searched=len(all_records),
        total_downloaded=len(unique_records),
        total_valid=sum(1 for r in unique_records if r.status == "valid"),
        total_duplicates=stats.total_duplicates,
        total_no_face=sum(1 for r in unique_records if r.faces_detected == 0),
        total_multi_face=sum(1 for r in unique_records if r.faces_detected > 1),
        total_representation=sum(1 for r in unique_records if r.image_category == "representation"),
        total_identity_uncertain=sum(1 for r in unique_records if r.identity_status == "uncertain"),
        persons_completed=len(split["reference"]),
        persons_incomplete=len(split["excluded"]),
    )

    metadata_dir = args.output / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest = generate_manifest(
        output_dir=metadata_dir,
        version="celebrity-v2-pilot",
        persons=people,
        records=unique_records,
        split=split,
        stats=stats,
        seed=args.seed,
    )

    report_path = generate_quality_report(
        output_dir=metadata_dir,
        version="celebrity-v2-pilot",
        records=unique_records,
        split=split,
        stats=stats,
        persons=people,
    )

    records_path = metadata_dir / "images.json"
    records_path.write_text(
        json.dumps([r.to_dict() for r in unique_records], indent=2),
        encoding="utf-8",
    )

    review_queue = downloader.get_review_queue()
    if review_queue:
        review_path = metadata_dir / "review_queue.json"
        review_path.write_text(
            json.dumps([r.to_dict() for r in review_queue], indent=2),
            encoding="utf-8",
        )
        print(f"\nReview queue: {len(review_queue)} items -> {review_path}")

    print(f"\nManifest: {metadata_dir / 'dataset_manifest.json'}")
    print(f"Quality report: {report_path}")
    print(f"Image records: {records_path}")

    for source in sources:
        source.close()

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
