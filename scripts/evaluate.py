"""
CLI evaluation entry point script.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.evaluator import Evaluator
from evaluation.charts import ChartsGenerator
from evaluation.statistics import StatisticsAggregator


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.evaluate <image_path_or_directory>")
        sys.exit(1)

    target_path = sys.argv[1]
    output_dir = "evaluation_results"

    print(f"Starting evaluation on: {target_path}")
    evaluator = Evaluator(output_dir=output_dir)
    results = evaluator.evaluate_path(target_path)

    if results:
        stats = StatisticsAggregator.aggregate(results)
        charts_gen = ChartsGenerator(output_dir)
        charts_gen.generate_charts(results, stats)
        print(f"Evaluation complete! Processed {len(results)} images. Results saved to '{output_dir}/'.")
    else:
        print("No valid images found or evaluated.")


if __name__ == "__main__":
    main()
