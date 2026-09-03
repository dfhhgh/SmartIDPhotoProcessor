"""Dataset manifest and quality report generation.

Generates dataset_manifest.json and DATASET_QUALITY_REPORT.md with
per-person statistics, source distribution, and calibration-ready classification.

All statistics are derived from ImageRecord data (single source of truth).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from dataset_acquisition.models import CollectionStats, ImageRecord, Person, RejectionDetail, RejectionStats


def compute_rejection_stats(
    records: list[ImageRecord],
    rejection_details: list[RejectionDetail] | None = None,
) -> RejectionStats:
    """Compute rejection telemetry from accepted records and rejection details."""
    accepted = len(records)
    details = rejection_details or []
    rejected_total = len(details)

    rejections_by_reason: dict[str, int] = {}
    for d in details:
        rejections_by_reason[d.rejection_reason] = rejections_by_reason.get(d.rejection_reason, 0) + 1

    per_person: dict[str, dict[str, int]] = {}
    for d in details:
        pp = per_person.setdefault(d.person_id, {"accepted": 0, "rejected_total": 0})
        pp["rejected_total"] += 1
        pp[d.rejection_reason] = pp.get(d.rejection_reason, 0) + 1
    for r in records:
        pp = per_person.setdefault(r.person_id, {"accepted": 0, "rejected_total": 0})
        pp["accepted"] += 1

    per_source: dict[str, dict[str, int]] = {}
    for d in details:
        ps = per_source.setdefault(d.source, {"candidates": 0, "accepted": 0, "rejected": 0})
        ps["candidates"] += 1
        ps["rejected"] += 1
        ps[d.rejection_reason] = ps.get(d.rejection_reason, 0) + 1
    for r in records:
        ps = per_source.setdefault(r.source, {"candidates": 0, "accepted": 0, "rejected": 0})
        ps["candidates"] += 1
        ps["accepted"] += 1

    return RejectionStats(
        total_candidates=accepted + rejected_total,
        accepted=accepted,
        rejected_total=rejected_total,
        rejections_by_reason=rejections_by_reason,
        per_person=per_person,
        per_source=per_source,
    )


def generate_manifest(
    output_dir: Path,
    version: str,
    persons: list[Person],
    records: list[ImageRecord],
    split: dict[str, Any],
    stats: CollectionStats,
    seed: int = 42,
    rejection_details: list[RejectionDetail] | None = None,
) -> dict[str, Any]:
    """Generate dataset_manifest.json.

    All statistics are derived directly from ImageRecord fields.
    Includes rejection telemetry when rejection_details is provided.
    """
    source_dist: dict[str, int] = {}
    for r in records:
        source_dist[r.source] = source_dist.get(r.source, 0) + 1

    license_dist: dict[str, int] = {}
    for r in records:
        license_dist[r.license] = license_dist.get(r.license, 0) + 1

    category_dist: dict[str, int] = {}
    for r in records:
        category_dist[r.image_category] = category_dist.get(r.image_category, 0) + 1

    identity_dist: dict[str, int] = {}
    for r in records:
        identity_dist[r.identity_status] = identity_dist.get(r.identity_status, 0) + 1

    status_dist: dict[str, int] = {}
    for r in records:
        status_dist[r.status] = status_dist.get(r.status, 0) + 1

    face_count_dist: dict[str, int] = {}
    for r in records:
        if r.faces_detected == 0:
            key = "0"
        elif r.faces_detected == 1:
            key = "1"
        elif r.faces_detected <= 5:
            key = "2-5"
        else:
            key = "6+"
        face_count_dist[key] = face_count_dist.get(key, 0) + 1

    ref_total = sum(len(v) for v in split.get("reference", {}).values())
    query_total = sum(len(v) for v in split.get("query", {}).values())

    calibration_ready = sum(
        1 for r in records
        if r.status == "valid"
        and r.identity_status == "confirmed"
        and r.image_category == "photograph"
        and r.faces_detected == 1
    )

    # Per-source statistics
    source_stats: dict[str, dict[str, int]] = {}
    for r in records:
        s = source_stats.setdefault(r.source, {"total": 0, "face_selected": 0, "no_face": 0, "multi_face": 0, "representation": 0})
        s["total"] += 1
        if r.face_selected:
            s["face_selected"] += 1
        if r.faces_detected == 0:
            s["no_face"] += 1
        if r.faces_detected > 1:
            s["multi_face"] += 1
        if r.image_category == "representation":
            s["representation"] += 1

    manifest = {
        "dataset_version": version,
        "creation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_persons": len(persons),
        "total_images": len(records),
        "valid_images": sum(1 for r in records if r.status == "valid"),
        "calibration_ready_images": calibration_ready,
        "reference_images": ref_total,
        "query_images": query_total,
        "excluded_persons": len(split.get("excluded", {})),
        "source_distribution": source_dist,
        "source_statistics": source_stats,
        "license_distribution": license_dist,
        "image_category_distribution": category_dist,
        "identity_status_distribution": identity_dist,
        "status_distribution": status_dist,
        "face_count_distribution": face_count_dist,
        "random_seed": seed,
        "split_strategy": "deterministic_seeded",
        "deduplication": {
            "exact_duplicates_removed": stats.total_duplicates,
        },
        "persons_included": [
            {
                "person_id": p.person_id,
                "display_name": p.display_name,
                "category": p.category,
                "reference_count": len(split.get("reference", {}).get(p.person_id, [])),
                "query_count": len(split.get("query", {}).get(p.person_id, [])),
                "total_images": sum(1 for r in records if r.person_id == p.person_id),
                "face_selected": sum(1 for r in records if r.person_id == p.person_id and r.face_selected),
                "source_distribution": _per_person_source_dist(records, p.person_id),
            }
            for p in persons
            if p.person_id in split.get("reference", {})
        ],
        "persons_excluded": [
            {"person_id": pid, "reason": reason}
            for pid, reason in split.get("excluded", {}).items()
        ],
    }

    # Rejection telemetry
    if rejection_details is not None:
        rstats = compute_rejection_stats(records, rejection_details)
        manifest["acquisition_telemetry"] = {
            "total_candidates": rstats.total_candidates,
            "accepted": rstats.accepted,
            "rejected_total": rstats.rejected_total,
            "rejections_by_reason": rstats.rejections_by_reason,
            "per_person": rstats.per_person,
            "per_source": rstats.per_source,
        }

    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _per_person_source_dist(records: list[ImageRecord], person_id: str) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in records:
        if r.person_id == person_id:
            dist[r.source] = dist.get(r.source, 0) + 1
    return dist


def generate_quality_report(
    output_dir: Path,
    version: str,
    records: list[ImageRecord],
    split: dict[str, Any],
    stats: CollectionStats,
    persons: list[Person],
    rejection_details: list[RejectionDetail] | None = None,
) -> str:
    """Generate DATASET_QUALITY_REPORT.md.

    All statistics are derived directly from ImageRecord fields.
    Includes rejection telemetry when rejection_details is provided.
    """
    included = [p for p in persons if p.person_id in split.get("reference", {})]
    excluded = split.get("excluded", {})

    ref_counts = [len(split["reference"].get(p.person_id, [])) for p in included]
    query_counts = [len(split["query"].get(p.person_id, [])) for p in included]

    source_dist: dict[str, int] = {}
    for r in records:
        source_dist[r.source] = source_dist.get(r.source, 0) + 1

    category_dist: dict[str, int] = {}
    for r in records:
        category_dist[r.image_category] = category_dist.get(r.image_category, 0) + 1

    identity_dist: dict[str, int] = {}
    for r in records:
        identity_dist[r.identity_status] = identity_dist.get(r.identity_status, 0) + 1

    status_dist: dict[str, int] = {}
    for r in records:
        status_dist[r.status] = status_dist.get(r.status, 0) + 1

    face_count_dist: dict[str, int] = {}
    for r in records:
        if r.faces_detected == 0:
            key = "0"
        elif r.faces_detected == 1:
            key = "1"
        elif r.faces_detected <= 5:
            key = "2-5"
        else:
            key = "6+"
        face_count_dist[key] = face_count_dist.get(key, 0) + 1

    calibration_ready = sum(
        1 for r in records
        if r.status == "valid"
        and r.identity_status == "confirmed"
        and r.image_category == "photograph"
        and r.faces_detected == 1
    )

    ref_mean = f"{sum(ref_counts)/len(ref_counts):.1f}" if ref_counts else "0"
    query_mean = f"{sum(query_counts)/len(query_counts):.1f}" if query_counts else "0"
    ref_min_val = min(ref_counts) if ref_counts else 0
    ref_max_val = max(ref_counts) if ref_counts else 0
    query_min_val = min(query_counts) if query_counts else 0
    query_max_val = max(query_counts) if query_counts else 0

    lines = [
        "# Dataset Quality Report",
        "",
        f"**Version**: {version}",
        f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total requested identities | {len(persons)} |",
        f"| Identities included | {len(included)} |",
        f"| Identities excluded | {len(excluded)} |",
        f"| Total images | {len(records)} |",
        f"| Valid images | {sum(1 for r in records if r.status == 'valid')} |",
        f"| Calibration-ready images | {calibration_ready} |",
        f"| Exact duplicates | {stats.total_duplicates} |",
        f"| No-face images | {sum(1 for r in records if r.faces_detected == 0)} |",
        f"| Single-face images | {sum(1 for r in records if r.faces_detected == 1)} |",
        f"| Multiple-face images | {sum(1 for r in records if r.faces_detected > 1)} |",
        f"| Representation images | {sum(1 for r in records if r.image_category == 'representation')} |",
        f"| Identity-uncertain images | {sum(1 for r in records if r.identity_status == 'uncertain')} |",
        "",
        "## Images Per Person",
        "",
        "| Statistic | Reference | Query |",
        "|-----------|-----------|-------|",
        f"| Min | {ref_min_val} | {query_min_val} |",
        f"| Max | {ref_max_val} | {query_max_val} |",
        f"| Mean | {ref_mean} | {query_mean} |",
        f"| Total | {sum(ref_counts)} | {sum(query_counts)} |",
        "",
        "## Face Count Distribution",
        "",
    ]
    for count_key in ["0", "1", "2-5", "6+"]:
        n = face_count_dist.get(count_key, 0)
        lines.append(f"- {count_key} faces: {n}")

    lines.extend(["", "## Image Category Distribution", ""])
    for cat, count in sorted(category_dist.items()):
        lines.append(f"- {cat}: {count}")

    lines.extend(["", "## Identity Status Distribution", ""])
    for status, count in sorted(identity_dist.items()):
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Status Distribution", ""])
    for status, count in sorted(status_dist.items()):
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Source Distribution", ""])
    for source, count in sorted(source_dist.items()):
        lines.append(f"- {source}: {count}")

    if included:
        lines.extend(["", "## Per-Person Statistics", ""])
        lines.append("| Person | Category | Total | Face-Sel | No-Face | Multi-Face | Repr | Ref | Query | Sources |")
        lines.append("|--------|----------|-------|----------|---------|------------|------|-----|-------|---------|")
        for p in included:
            pid = p.person_id
            p_records = [r for r in records if r.person_id == pid]
            face_sel = sum(1 for r in p_records if r.face_selected)
            no_face = sum(1 for r in p_records if r.faces_detected == 0)
            multi_face = sum(1 for r in p_records if r.faces_detected > 1)
            repr_count = sum(1 for r in p_records if r.image_category == "representation")
            ref_count = len(split["reference"].get(pid, []))
            q_count = len(split["query"].get(pid, []))
            sources = ", ".join(sorted({r.source for r in p_records}))
            lines.append(
                f"| {p.display_name} | {p.category} | {len(p_records)} | "
                f"{face_sel} | {no_face} | {multi_face} | {repr_count} | "
                f"{ref_count} | {q_count} | {sources} |"
            )

    if excluded:
        lines.extend(["", "## Excluded Identities", ""])
        for pid, reason in sorted(excluded.items()):
            lines.append(f"- {pid}: {reason}")

    # Rejection telemetry
    if rejection_details is not None:
        rstats = compute_rejection_stats(records, rejection_details)
        lines.extend([
            "",
            "## Acquisition Telemetry",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total candidates seen | {rstats.total_candidates} |",
            f"| Accepted (dataset images) | {rstats.accepted} |",
            f"| Rejected candidates | {rstats.rejected_total} |",
            "",
            "### Rejection Reasons",
            "",
        ])
        all_reasons = ["representation", "no_face", "multi_face", "download_error",
                        "decode_error", "invalid_image", "duplicate", "other"]
        for reason in all_reasons:
            count = rstats.rejections_by_reason.get(reason, 0)
            lines.append(f"- {reason}: {count}")

        if rstats.per_person:
            lines.extend(["", "### Per-Person Rejection Statistics", ""])
            lines.append("| Person | Accepted | Rejected | Representation | No-Face | Multi-Face | Other |")
            lines.append("|--------|----------|----------|----------------|---------|------------|-------|")
            for pid in sorted(rstats.per_person.keys()):
                pp = rstats.per_person[pid]
                acc = pp.get("accepted", 0)
                rej = pp.get("rejected_total", 0)
                repr_c = pp.get("representation", 0)
                nf = pp.get("no_face", 0)
                mf = pp.get("multi_face", 0)
                other = rej - repr_c - nf - mf
                lines.append(f"| {pid} | {acc} | {rej} | {repr_c} | {nf} | {mf} | {other} |")

        if rstats.per_source:
            lines.extend(["", "### Per-Source Rejection Statistics", ""])
            lines.append("| Source | Candidates | Accepted | Rejected | Representation | No-Face | Multi-Face | Other |")
            lines.append("|--------|------------|----------|----------|----------------|---------|------------|-------|")
            for src in sorted(rstats.per_source.keys()):
                ps = rstats.per_source[src]
                cands = ps.get("candidates", 0)
                acc = ps.get("accepted", 0)
                rej = ps.get("rejected", 0)
                repr_c = ps.get("representation", 0)
                nf = ps.get("no_face", 0)
                mf = ps.get("multi_face", 0)
                other = rej - repr_c - nf - mf
                lines.append(f"| {src} | {cands} | {acc} | {rej} | {repr_c} | {nf} | {mf} | {other} |")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- This dataset measures known-identity face retrieval, not real-world celebrity recognition coverage.",
        "- It does not represent all celebrities, countries, ethnicities, ages, or image sources.",
        "- Search results depend on available tagged images from selected sources.",
        "- Identity verification relies on source metadata, not automated recognition.",
        "- Representation images (posters, paintings) are excluded from calibration-ready set.",
        "- Multiple-face images require manual review for identity confirmation.",
        "",
    ])

    report_path = output_dir / "DATASET_QUALITY_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)
