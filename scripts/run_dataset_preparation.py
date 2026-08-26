"""Run the dataset preparation pipeline.

Usage:
    python scripts/run_dataset_preparation.py
    python scripts/run_dataset_preparation.py --skip-quality-scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset_builder.collection.dataset_preparation import DatasetPreparation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset preparation pipeline for parser fine-tuning",
    )
    parser.add_argument(
        "--skip-face-detection",
        action="store_true",
        default=False,
        help="Skip face detection phase",
    )
    parser.add_argument(
        "--skip-quality-scores",
        action="store_true",
        default=False,
        help="Skip quality scoring phase",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "dataset_builder" / "dataset" / "raw"
    output_dir = project_root / "dataset_builder" / "dataset" / "parser_finetune"

    pipeline = DatasetPreparation(
        raw_dir=raw_dir,
        output_dir=output_dir,
        skip_face_detection=args.skip_face_detection,
        skip_quality_scores=args.skip_quality_scores,
    )

    manifest = pipeline.run()

    # Exit code based on results
    if manifest.pending_count > 0:
        print(f"\nWARNING: {manifest.pending_count} images still PENDING")
        sys.exit(1)
    else:
        print(f"\nAll {manifest.total_images} images have decisions.")
        sys.exit(0)


if __name__ == "__main__":
    main()
