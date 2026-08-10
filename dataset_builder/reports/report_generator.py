"""
Reporting layer for the Dataset Builder project.

Generates Markdown, JSON, and CSV reports from a pre-built
:class:`DatasetStatistics` object.  This module performs **no**
downloading, filtering, duplicate detection, logging, statistics
computation, or file scanning.  It is a pure presentation layer.

Design Principles
-----------------
- **Single responsibility**: only present data.
- **No side effects beyond file writes**: no folders scanned, no
  counters created, no image processing.
- **Complete independence**: receives a single :class:`DatasetStatistics`
  instance and never accesses any pipeline stage directly.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from dataset_builder.config.settings import Settings
from dataset_builder.statistics.statistics import DatasetStatistics


class ReportGenerator:
    """Generate Markdown, JSON, and CSV reports from pipeline statistics.

    The generator is completely independent of every pipeline stage.
    It never downloads, scans, filters, detects duplicates, or
    computes statistics.  Its sole input is a :class:`DatasetStatistics`
    instance produced by :class:`StatisticsAggregator`.

    Parameters
    ----------
    settings:
        Application settings.  Currently unused but retained for
        future extensibility (e.g. custom report templates).

    Examples
    --------
    ::

        generator = ReportGenerator(settings)
        generator.generate_markdown(stats, Path("report.md"))
        generator.generate_json(stats, Path("report.json"))
        generator.generate_csv(stats, Path("report.csv"))
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_markdown(
        self,
        statistics: DatasetStatistics,
        output_path: Path,
    ) -> None:
        """Write a Markdown report to *output_path*.

        The report contains one section per pipeline stage plus a
        final summary.

        Parameters
        ----------
        statistics:
            Pre-built pipeline statistics.
        output_path:
            Destination file.  Parent directories are created
            automatically if they do not exist.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self._render_markdown(statistics),
            encoding="utf-8",
        )

    def generate_json(
        self,
        statistics: DatasetStatistics,
        output_path: Path,
    ) -> None:
        """Write a JSON report to *output_path*.

        Serialises the :class:`DatasetStatistics` dataclass tree
        via :func:`dataclasses.asdict`.  No values are recalculated.

        Parameters
        ----------
        statistics:
            Pre-built pipeline statistics.
        output_path:
            Destination file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(asdict(statistics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def generate_csv(
        self,
        statistics: DatasetStatistics,
        output_path: Path,
    ) -> None:
        """Write a single-row CSV summary to *output_path*.

        Suitable for importing into Excel or other spreadsheet
        applications.  One header row followed by one data row.

        Parameters
        ----------
        statistics:
            Pre-built pipeline statistics.
        output_path:
            Destination file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = self._flatten_statistics(statistics)
        headers = [k for k, _ in rows]
        values = [str(v) for _, v in rows]

        with output_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            writer.writerow(values)

    # ------------------------------------------------------------------
    # Internal renderers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_markdown(s: DatasetStatistics) -> str:
        """Render the full Markdown report as a string."""
        c = s.collection
        d = s.duplicates
        f = s.face_filter
        q = s.quality_filter

        lines: list[str] = [
            "# Dataset Builder Report",
            "",
            "## Collection",
            "",
            f"- Queries: {c.total_queries}",
            f"- Sources: {c.total_sources}",
            f"- Downloaded: {c.downloaded_images}",
            f"- Failed: {c.failed_downloads}",
            f"- Success Rate: {c.download_success_rate:.2%}",
            "",
            "## Duplicate Removal",
            "",
            f"- Total: {d.total_images}",
            f"- Unique: {d.unique_images}",
            f"- Duplicate Images: {d.duplicate_images}",
            f"- Duplicate Groups: {d.duplicate_groups}",
            f"- Duplicate Ratio: {d.duplicate_ratio:.2%}",
            "",
            "## Face Filter",
            "",
            f"- Accepted: {f.accepted_images}",
            f"- Rejected: {f.rejected_images}",
            f"- Acceptance Rate: {f.acceptance_rate:.2%}",
            "",
            "### Rejection Reasons",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ]

        for reason, count in sorted(f.rejection_reason_distribution.items()):
            lines.append(f"| {reason} | {count} |")

        lines += [
            "",
            "## Quality Filter",
            "",
            f"- Accepted: {q.accepted_images}",
            f"- Rejected: {q.rejected_images}",
            f"- Acceptance Rate: {q.acceptance_rate:.2%}",
            "",
            "### Rejection Reasons",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ]

        for reason, count in sorted(q.rejection_reason_distribution.items()):
            lines.append(f"| {reason} | {count} |")

        lines += [
            "",
            "## Final Dataset",
            "",
            f"- Final Images: {s.total_final_images}",
            f"- Overall Retention Rate: {s.overall_retention_rate:.2%}",
            "",
        ]

        return "\n".join(lines)

    @staticmethod
    def _flatten_statistics(
        s: DatasetStatistics,
    ) -> list[tuple[str, object]]:
        """Flatten the statistics tree into ``(header, value)`` pairs."""
        c = s.collection
        d = s.duplicates
        f = s.face_filter
        q = s.quality_filter

        rows: list[tuple[str, object]] = [
            ("queries", c.total_queries),
            ("sources", c.total_sources),
            ("downloaded_images", c.downloaded_images),
            ("failed_downloads", c.failed_downloads),
            ("download_success_rate", c.download_success_rate),
            ("duplicate_total_images", d.total_images),
            ("duplicate_unique_images", d.unique_images),
            ("duplicate_images", d.duplicate_images),
            ("duplicate_groups", d.duplicate_groups),
            ("duplicate_ratio", d.duplicate_ratio),
            ("face_processed_images", f.processed_images),
            ("face_accepted_images", f.accepted_images),
            ("face_rejected_images", f.rejected_images),
            ("face_acceptance_rate", f.acceptance_rate),
        ]

        for reason, count in sorted(f.rejection_reason_distribution.items()):
            rows.append((f"face_rejected_{reason}", count))

        rows += [
            ("quality_processed_images", q.processed_images),
            ("quality_accepted_images", q.accepted_images),
            ("quality_rejected_images", q.rejected_images),
            ("quality_acceptance_rate", q.acceptance_rate),
        ]

        for reason, count in sorted(q.rejection_reason_distribution.items()):
            rows.append((f"quality_rejected_{reason}", count))

        rows += [
            ("total_final_images", s.total_final_images),
            ("overall_retention_rate", s.overall_retention_rate),
        ]

        return rows
