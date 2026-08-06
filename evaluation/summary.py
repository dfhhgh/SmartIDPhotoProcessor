"""
Dataset summary and statistics text/markdown report generator.
"""

from __future__ import annotations

import os

from evaluation.models import ImageEvaluationResult


class SummaryGenerator:
    """Generates dataset-wide summary reports and statistics."""

    def __init__(self, output_dir: str) -> None:
        self._reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(self._reports_dir, exist_ok=True)

    def generate_dataset_summary(self, results: list[ImageEvaluationResult], stats: dict) -> str:
        """Generate a comprehensive dataset summary report in markdown format."""
        summary_path = os.path.join(self._reports_dir, "dataset_summary.md")

        lines: list[str] = []
        lines.append("# Dataset Evaluation Summary")
        lines.append("")
        lines.append("## OVERVIEW")
        lines.append(f"- **Images Processed**: {stats['images_processed']}")
        lines.append(f"- **Passed**: {stats['passed']}")
        lines.append(f"- **Failed**: {stats['failed']}")
        lines.append(f"- **Pass Rate**: {stats['pass_rate']:.2f}%")
        lines.append(f"- **Average Quality Score**: {stats['average_score']:.2f}")
        lines.append(f"- **Average Parser Confidence**: {stats['average_parser_confidence']:.2f}")
        lines.append(f"- **Average Landmark Confidence**: {stats['average_landmark_confidence']:.2f}")
        lines.append(f"- **Average Semantic Confidence**: {stats['average_semantic_confidence']:.2f}")
        lines.append(f"- **Average Processing Time**: {stats['average_processing_time_ms']:.2f} ms")
        lines.append("")

        lines.append("## VALIDATOR FAILURE FREQUENCY")
        lines.append("| Validator | Failure Count |")
        lines.append("|---|---|")
        for v_name, count in stats["failure_frequencies"].items():
            lines.append(f"| {v_name} | {count} |")
        lines.append("")

        lines.append("## SEMANTIC FAILURE SUMMARY")
        lines.append("| FacePart | Failure Count |")
        lines.append("|---|---|")
        for part_name, count in stats["semantic_failure_summary"].items():
            lines.append(f"| {part_name} | {count} |")
        lines.append("")

        lines.append("## ROOT CAUSE DISTRIBUTION")
        lines.append("| Root Cause | Count |")
        lines.append("|---|---|")
        # Sort by count descending
        sorted_causes = sorted(stats["root_cause_distribution"].items(), key=lambda x: x[1], reverse=True)
        for cause, count in sorted_causes:
            lines.append(f"| `{cause}` | {count} |")
        lines.append("")

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return summary_path
