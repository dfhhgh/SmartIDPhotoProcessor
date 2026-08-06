"""
Charts and visualization plots generator.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.models import ImageEvaluationResult


class ChartsGenerator:
    """Generates and saves visual analytic charts (histograms, pass rates, distributions)."""

    def __init__(self, output_dir: str) -> None:
        self._charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(self._charts_dir, exist_ok=True)

    def generate_charts(self, results: list[ImageEvaluationResult], dataset_stats: dict) -> dict[str, str]:
        """Generate all required analytical charts and return their file paths."""
        paths: dict[str, str] = {}

        # 1. Validator Pass Rate Bar Chart
        paths["validator_pass_rate"] = self._plot_validator_pass_rate(results)

        # 2. Failure Histogram / Frequency
        paths["failure_histogram"] = self._plot_failure_histogram(dataset_stats["failure_frequencies"])

        # 3. FacePart Failure Histogram
        paths["facepart_failure"] = self._plot_facepart_failure(dataset_stats["semantic_failure_summary"])

        # 4. Root Cause Distribution
        paths["root_cause_distribution"] = self._plot_root_cause_distribution(dataset_stats["root_cause_distribution"])

        # 5. Parser Confidence Histogram
        paths["parser_confidence_hist"] = self._plot_parser_confidence_histogram(results)

        # 6. Semantic Confidence Histogram
        paths["semantic_confidence_hist"] = self._plot_semantic_confidence_histogram(results)

        # 7. Execution Time Histogram
        paths["execution_time_hist"] = self._plot_execution_time_histogram(results)

        return paths

    def _plot_validator_pass_rate(self, results: list[ImageEvaluationResult]) -> str:
        validator_counts: dict[str, list[bool]] = {}
        for r in results:
            for v in r.validators:
                validator_counts.setdefault(v.validator_name, []).append(v.passed)

        names = list(validator_counts.keys())
        pass_rates = [sum(v) / len(v) * 100.0 for v in validator_counts.values()]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names, pass_rates, color="skyblue", edgecolor="navy")
        ax.set_ylim(0, 105)
        ax.set_ylabel("Pass Rate (%)")
        ax.set_title("Validator Pass Rates")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "validator_pass_rate.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_failure_histogram(self, failure_freqs: dict[str, int]) -> str:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(failure_freqs.keys())
        counts = list(failure_freqs.values())
        ax.bar(names, counts, color="salmon", edgecolor="darkred")
        ax.set_ylabel("Failure Count")
        ax.set_title("Validator Failure Frequency")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "failure_histogram.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_facepart_failure(self, semantic_failures: dict[str, int]) -> str:
        fig, ax = plt.subplots(figsize=(9, 5))
        names = list(semantic_failures.keys())
        counts = list(semantic_failures.values())
        ax.bar(names, counts, color="orange", edgecolor="darkorange")
        ax.set_ylabel("Failure Count")
        ax.set_title("FacePart Failure Distribution")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "facepart_failure_histogram.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_root_cause_distribution(self, root_causes: dict[str, int]) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(root_causes.keys())
        counts = list(root_causes.values())
        ax.bar(names, counts, color="mediumpurple", edgecolor="indigo")
        ax.set_ylabel("Count")
        ax.set_title("Root Cause Distribution")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "root_cause_distribution.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_parser_confidence_histogram(self, results: list[ImageEvaluationResult]) -> str:
        confs = [pe.parser_confidence for r in results for pe in r.semantic_parts.values()]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(confs, bins=15, color="lightgreen", edgecolor="darkgreen", range=(0, 1))
        ax.set_xlabel("Parser Confidence")
        ax.set_ylabel("Frequency")
        ax.set_title("Parser Confidence Histogram")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "parser_confidence_histogram.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_semantic_confidence_histogram(self, results: list[ImageEvaluationResult]) -> str:
        confs = [pe.final_confidence for r in results for pe in r.semantic_parts.values()]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(confs, bins=15, color="gold", edgecolor="darkgoldenrod", range=(0, 1))
        ax.set_xlabel("Semantic Confidence")
        ax.set_ylabel("Frequency")
        ax.set_title("Semantic Confidence Histogram")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "semantic_confidence_histogram.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def _plot_execution_time_histogram(self, results: list[ImageEvaluationResult]) -> str:
        times = [r.execution_time_ms for r in results]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(times, bins=15, color="teal", edgecolor="darkcyan")
        ax.set_xlabel("Execution Time (ms)")
        ax.set_ylabel("Frequency")
        ax.set_title("Execution Time Distribution")
        plt.tight_layout()
        path = os.path.join(self._charts_dir, "execution_time_histogram.png")
        plt.savefig(path, dpi=150)
        plt.close(fig)
        return path
