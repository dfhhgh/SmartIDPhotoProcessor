"""
Statistics and dataset metrics aggregator.
"""

from __future__ import annotations

from typing import Any

from evaluation.models import ImageEvaluationResult


class StatisticsAggregator:
    """Aggregates statistics and dataset summary metrics across evaluated images."""

    @staticmethod
    def aggregate(results: list[ImageEvaluationResult]) -> dict[str, Any]:
        """Compute dataset summary metrics from a list of image evaluations."""
        total = len(results)
        if total == 0:
            return {
                "images_processed": 0,
                "passed": 0,
                "failed": 0,
                "pass_rate": 0.0,
                "average_score": 0.0,
                "average_parser_confidence": 0.0,
                "average_landmark_confidence": 0.0,
                "average_semantic_confidence": 0.0,
                "average_processing_time_ms": 0.0,
                "failure_frequencies": {},
                "semantic_failure_summary": {},
                "root_cause_distribution": {},
            }

        passed_count = sum(1 for r in results if r.overall_passed)
        failed_count = total - passed_count
        pass_rate = (passed_count / total) * 100.0

        # Validator scores and execution times
        all_scores: list[float] = []
        all_times: list[float] = []
        parser_confs: list[float] = []
        landmark_confs: list[float] = []
        semantic_confs: list[float] = []

        failure_frequencies: dict[str, int] = {}
        semantic_failure_summary: dict[str, int] = {}
        root_cause_distribution: dict[str, int] = {}

        for r in results:
            all_times.append(r.execution_time_ms)
            for v in r.validators:
                if not v.passed:
                    failure_frequencies[v.validator_name] = failure_frequencies.get(v.validator_name, 0) + 1

            for part_name, part_eval in r.semantic_parts.items():
                parser_confs.append(part_eval.parser_confidence)
                landmark_confs.append(part_eval.landmark_confidence)
                semantic_confs.append(part_eval.final_confidence)
                if not part_eval.passed:
                    semantic_failure_summary[part_name] = semantic_failure_summary.get(part_name, 0) + 1

            if r.face_metrics:
                all_scores.append(r.face_metrics.detection_confidence)

            if r.root_cause:
                cause = r.root_cause.cause
                root_cause_distribution[cause] = root_cause_distribution.get(cause, 0) + 1

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        avg_time = sum(all_times) / total if total > 0 else 0.0
        avg_parser = sum(parser_confs) / len(parser_confs) if parser_confs else 0.0
        avg_landmark = sum(landmark_confs) / len(landmark_confs) if landmark_confs else 0.0
        avg_semantic = sum(semantic_confs) / len(semantic_confs) if semantic_confs else 0.0

        return {
            "images_processed": total,
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": pass_rate,
            "average_score": avg_score,
            "average_parser_confidence": avg_parser,
            "average_landmark_confidence": avg_landmark,
            "average_semantic_confidence": avg_semantic,
            "average_processing_time_ms": avg_time,
            "failure_frequencies": failure_frequencies,
            "semantic_failure_summary": semantic_failure_summary,
            "root_cause_distribution": root_cause_distribution,
        }
