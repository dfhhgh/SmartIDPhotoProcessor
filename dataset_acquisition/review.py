"""Manual review workflow for dataset quality gate.

Generates contact sheets, review records, and quality gate analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dataset_acquisition.models import (
    ImageRecord,
    ManualReviewRecord,
    ReviewStats,
    REVIEW_DECISIONS,
)

logger = logging.getLogger(__name__)

SINGLE_FACE_THRESHOLD = 1
CONTACT_SHEET_THUMB = (320, 320)
CONTACT_SHEET_COLS = 3
CONTACT_SHEET_PADDING = 10
CONTACT_SHEET_LABEL_HEIGHT = 40
CONTACT_SHEET_BG_COLOR = (40, 40, 40)
CONTACT_SHEET_TEXT_COLOR = (255, 255, 255)
CONTACT_SHEET_BORDER_COLOR = (100, 100, 100)


def _load_pilot_images(pilot_jsonl: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(pilot_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_people(people_json: Path) -> dict[str, str]:
    with open(people_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {p["person_id"]: p["display_name"] for p in data["people"]}


def create_review_records(
    pilot_jsonl: Path,
    people_json: Path,
) -> list[ManualReviewRecord]:
    """Create PENDING review records for every image in the pilot."""
    raw_records = _load_pilot_images(pilot_jsonl)
    names = _load_people(people_json)
    records: list[ManualReviewRecord] = []
    for r in raw_records:
        record = ManualReviewRecord(
            person_id=r["person_id"],
            display_name=names.get(r["person_id"], r["person_id"]),
            image_id=r["image_id"],
            image_path=r["local_path"],
            source=r["source"],
            source_url=r["source_url"],
            automated_status=r["status"],
            automated_image_category=r["image_category"],
            automated_identity_status=r["identity_status"],
            faces_detected=r["faces_detected"],
            face_confidence=r.get("face_confidence", 0.0),
            manual_decision="PENDING",
        )
        records.append(record)
    return records


def update_review_record(
    records: list[ManualReviewRecord],
    image_id: str,
    decision: str,
    reason: str = "",
    notes: str = "",
) -> ManualReviewRecord:
    """Update a review record with a manual decision. Returns the updated record."""
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"Invalid decision: {decision!r}")
    for i, r in enumerate(records):
        if r.image_id == image_id:
            updated = ManualReviewRecord(
                person_id=r.person_id,
                display_name=r.display_name,
                image_id=r.image_id,
                image_path=r.image_path,
                source=r.source,
                source_url=r.source_url,
                automated_status=r.automated_status,
                automated_image_category=r.automated_image_category,
                automated_identity_status=r.automated_identity_status,
                faces_detected=r.faces_detected,
                face_confidence=r.face_confidence,
                manual_decision=decision,
                manual_reason=reason,
                reviewer_notes=notes,
            )
            records[i] = updated
            return updated
    raise KeyError(f"image_id {image_id!r} not found in review records")


def compute_review_stats(records: list[ManualReviewRecord]) -> ReviewStats:
    """Compute aggregate and per-person review statistics."""
    total = len(records)
    pending = sum(1 for r in records if r.manual_decision == "PENDING")
    accepted = sum(1 for r in records if r.manual_decision == "ACCEPT")
    rejected = sum(1 for r in records if r.manual_decision == "REJECT")
    uncertain = sum(1 for r in records if r.manual_decision == "UNCERTAIN")

    reviewed = total - pending
    acceptance_rate = accepted / reviewed if reviewed > 0 else 0.0

    _DECISION_KEY_MAP = {
        "PENDING": "pending",
        "ACCEPT": "accepted",
        "REJECT": "rejected",
        "UNCERTAIN": "uncertain",
    }

    per_person: dict[str, dict[str, int]] = {}
    for r in records:
        if r.person_id not in per_person:
            per_person[r.person_id] = {
                "total": 0,
                "pending": 0,
                "accepted": 0,
                "rejected": 0,
                "uncertain": 0,
            }
        pp = per_person[r.person_id]
        pp["total"] += 1
        key = _DECISION_KEY_MAP.get(r.manual_decision, r.manual_decision.lower())
        pp[key] += 1

    return ReviewStats(
        total_images=total,
        pending=pending,
        accepted=accepted,
        rejected=rejected,
        uncertain=uncertain,
        acceptance_rate=acceptance_rate,
        per_person=per_person,
    )


def classify_identity_quality(
    per_person_stats: dict[str, dict[str, int]],
    min_accepted: int = 3,
) -> dict[str, str]:
    """Determine per-identity quality status.

    Engineering rationale for min_accepted=3:
    - The reference/query split uses 60/40 ratio.
    - With 3 accepted images, reference gets ~2, query gets ~1.
    - Below 3, the identity cannot produce a meaningful split.
    """
    result: dict[str, str] = {}
    for person_id, stats in per_person_stats.items():
        accepted = stats.get("accepted", 0)
        pending = stats.get("pending", 0)
        if pending > 0:
            result[person_id] = "PENDING"
        elif accepted >= min_accepted:
            result[person_id] = "PASS"
        elif accepted > 0:
            result[person_id] = "INSUFFICIENT_DATA"
        else:
            result[person_id] = "FAILED"
    return result


def generate_contact_sheet(
    images: list[tuple[str, str, str]],
    output_path: Path,
    title: str = "Contact Sheet",
    max_images: int = 12,
) -> Path:
    """Generate a contact sheet JPEG.

    Args:
        images: list of (image_id, image_path, label) tuples.
        output_path: path to write the contact sheet.
        title: title at the top.
        max_images: maximum images to include.

    Returns:
        Path to the generated contact sheet.
    """
    from PIL import Image, ImageDraw, ImageFont

    n = min(len(images), max_images)
    if n == 0:
        img = Image.new("RGB", (400, 200), CONTACT_SHEET_BG_COLOR)
        draw = ImageDraw.Draw(img)
        draw.text((20, 90), "No images", fill=CONTACT_SHEET_TEXT_COLOR)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), "JPEG", quality=90)
        return output_path

    cols = min(n, CONTACT_SHEET_COLS)
    rows = (n + cols - 1) // cols

    thumb_w, thumb_h = CONTACT_SHEET_THUMB
    pad = CONTACT_SHEET_PADDING
    label_h = CONTACT_SHEET_LABEL_HEIGHT
    title_h = 50
    total_w = cols * (thumb_w + pad) + pad
    total_h = title_h + rows * (thumb_h + label_h + pad) + pad

    canvas = Image.new("RGB", (total_w, total_h), CONTACT_SHEET_BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
        title_font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
        title_font = font

    draw.text((pad, 10), title, fill=CONTACT_SHEET_TEXT_COLOR, font=title_font)

    for idx in range(n):
        image_id, image_path, label = images[idx]
        row = idx // cols
        col = idx % cols
        x = pad + col * (thumb_w + pad)
        y = title_h + row * (thumb_h + label_h + pad)

        try:
            thumb = Image.open(image_path)
            thumb.thumbnail((thumb_w, thumb_h), Image.LANCZOS)
            canvas.paste(thumb, (x, y))
        except Exception:
            draw.rectangle(
                [x, y, x + thumb_w, y + thumb_h],
                outline=CONTACT_SHEET_BORDER_COLOR,
                width=2,
            )
            draw.text((x + 10, y + thumb_h // 2), "ERR", fill=(255, 80, 80))

        truncated = label[:35] + "..." if len(label) > 38 else label
        draw.text(
            (x, y + thumb_h + 2),
            truncated,
            fill=CONTACT_SHEET_TEXT_COLOR,
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path), "JPEG", quality=90)
    return output_path


def generate_all_contact_sheets(
    pilot_jsonl: Path,
    people_json: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate per-identity contact sheets and a problem-case sheet."""
    raw_records = _load_pilot_images(pilot_jsonl)
    names = _load_people(people_json)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    by_person: dict[str, list[dict[str, Any]]] = {}
    for r in raw_records:
        by_person.setdefault(r["person_id"], []).append(r)

    problem_images: list[tuple[str, str, str]] = []

    for person_id, person_records in sorted(by_person.items()):
        display_name = names.get(person_id, person_id)
        images: list[tuple[str, str, str]] = []
        for r in person_records:
            status = r["status"]
            cat = r["image_category"]
            label = f"{r['image_id']} [{status}] cat={cat} faces={r['faces_detected']}"
            images.append((r["image_id"], r["local_path"], label))
            if status != "valid":
                problem_images.append((r["image_id"], r["local_path"], f"{display_name}: {label}"))

        sheet_path = output_dir / f"{person_id}.jpg"
        generate_contact_sheet(
            images,
            sheet_path,
            title=f"{display_name} ({len(images)} images)",
            max_images=12,
        )
        paths.append(sheet_path)
        logger.info("Contact sheet: %s (%d images) -> %s", display_name, len(images), sheet_path)

    if problem_images:
        problem_path = output_dir / "problem_cases.jpg"
        generate_contact_sheet(
            problem_images,
            problem_path,
            title=f"Problem Cases ({len(problem_images)} images)",
            max_images=30,
        )
        paths.append(problem_path)
        logger.info("Problem cases sheet: %d images -> %s", len(problem_images), problem_path)

    return paths


def save_review_records(records: list[ManualReviewRecord], output_dir: Path) -> tuple[Path, Path]:
    """Save review records as JSON and JSONL."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "manual_review.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, indent=2, ensure_ascii=False)

    jsonl_path = output_dir / "manual_review.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    return json_path, jsonl_path


def save_review_stats(
    stats: ReviewStats,
    identity_quality: dict[str, str],
    output_dir: Path,
) -> Path:
    """Save review statistics and quality gate results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "review_stats.json"
    data = {
        "aggregate": stats.to_dict(),
        "identity_quality": identity_quality,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
