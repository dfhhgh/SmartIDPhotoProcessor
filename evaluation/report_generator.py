"""
Report generator producing human-readable text reports for individual images and datasets.
"""

from __future__ import annotations

import os

from evaluation.models import ImageEvaluationResult


class ReportGenerator:
    """Generates structured human-readable markdown and text evaluation reports."""

    def __init__(self, output_dir: str) -> None:
        self._reports_dir = os.path.join(output_dir, "reports")
        os.makedirs(self._reports_dir, exist_ok=True)

    def generate_image_report(self, result: ImageEvaluationResult) -> str:
        """Generate a detailed markdown report for a single image evaluation."""
        base_name = os.path.splitext(result.image_name)[0]
        report_path = os.path.join(self._reports_dir, f"{base_name}_report.md")

        lines: list[str] = []
        lines.append(f"# Evaluation Report: {result.image_name}")
        lines.append("")
        lines.append("## GENERAL")
        lines.append(f"- **Image Name**: {result.image_name}")
        lines.append(f"- **Resolution**: {result.resolution[1]}x{result.resolution[0]} (WxH)")
        lines.append(f"- **Execution Time**: {result.execution_time_ms:.2f} ms")
        lines.append(f"- **Detected Faces**: {result.detected_faces}")
        lines.append(f"- **Primary Face Confidence**: {result.primary_face_confidence:.2f}")
        lines.append(f"- **Selection Confidence**: {result.selection_confidence:.2f}")
        lines.append(f"- **Overall Status**: {'PASS' if result.overall_passed else 'FAIL'}")
        lines.append("")

        lines.append("## PIPELINE")
        lines.append("| Validator | Status | Score | Message | Time (ms) |")
        lines.append("|---|---|---|---|---|")
        for v in result.validators:
            status = "PASS" if v.passed else "FAIL"
            time_str = f"{v.execution_time_ms:.2f}" if v.execution_time_ms is not None else "N/A"
            lines.append(f"| {v.validator_name} | {status} | {v.score:.2f} | {v.message} | {time_str} |")
        lines.append("")

        if result.face_metrics:
            fm = result.face_metrics
            lines.append("## FACE")
            lines.append(f"- **Bounding Box**: {fm.bbox}")
            lines.append(f"- **Area Ratio**: {fm.area_ratio:.4f}")
            lines.append(f"- **Center**: {fm.center}")
            lines.append(f"- **Crop Size**: {fm.crop_size}")
            lines.append(f"- **Alignment Size**: {fm.alignment_size}")
            lines.append(f"- **Pose (Yaw, Pitch, Roll)**: ({fm.yaw:.1f}°, {fm.pitch:.1f}°, {fm.roll:.1f}°)")
            lines.append(f"- **Detection Confidence**: {fm.detection_confidence:.2f}")
            lines.append("")

        lines.append("## SEMANTIC ANALYSIS")
        for part_name, pe in result.semantic_parts.items():
            lines.append(f"### {part_name}")
            lines.append(f"- **Pixels**: {pe.pixels}")
            lines.append(f"- **Ratio**: {pe.area_ratio:.4f}")
            lines.append(f"- **Parser Confidence**: {pe.parser_confidence:.2f}")
            lines.append(f"- **Eye Support Confidence**: {pe.eye_support_confidence:.2f}")
            lines.append(f"- **Landmark Confidence**: {pe.landmark_confidence:.2f}")
            lines.append(f"- **Pose Confidence**: {pe.pose_confidence:.2f}")
            lines.append(f"- **Occlusion Confidence**: {pe.occlusion_confidence:.2f}")
            lines.append(f"- **Final Confidence**: {pe.final_confidence:.2f}")
            lines.append(f"- **Decision**: {'PASS' if pe.passed else 'FAIL'}")
            lines.append("")

        lines.append("## PARSER STATISTICS")
        lines.append("| Class | Pixels | Percentage (%) |")
        lines.append("|---|---|---|")
        for cls_name, stats in result.parser_statistics.items():
            lines.append(f"| {cls_name} | {stats['pixels']} | {stats['percentage']:.2f}% |")
        lines.append("")

        if result.root_cause:
            rc = result.root_cause
            lines.append("## ROOT CAUSE ANALYSIS")
            lines.append(f"- **Cause**: `{rc.cause}`")
            lines.append(f"- **Confidence**: {rc.confidence:.2f}")
            lines.append(f"- **Explanation**: {rc.explanation}")
            lines.append("- **Evidence Used**:")
            for ev in rc.evidence:
                lines.append(f"  - {ev}")
            lines.append("")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path
