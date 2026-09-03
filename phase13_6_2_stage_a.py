"""Phase 13.6.2 — Expanded Celebrity Dataset Collection (Stage A).

Stage A collects a small subset of identities (5 actors + 5 football players)
to evaluate source suitability, acceptance rates, and diversity before scaling.

Output: datasets/celebrity-v3-expanded/
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase13_6_2")

ROOT = Path(__file__).resolve().parent
PEOPLE_PATH = ROOT / "dataset_acquisition" / "people_v3.json"
DATASET_DIR = ROOT / "datasets" / "celebrity-v3-expanded"
OUTPUT_DIR = ROOT / "outputs" / "phase13_6_2_stage_a"
STAGE_A_COUNT = 10
MAX_IMAGES_PER_PERSON = 30
SEED = 42


def load_people() -> list:
    from dataset_acquisition.models import Person
    with open(PEOPLE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Person.from_dict(p) for p in data["people"]]


def check_leakage(reference_dir: Path, query_dir: Path) -> list[str]:
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


def generate_review_grids(
    records: list,
    people: list,
    output_dir: Path,
    max_per_person: int = 20,
) -> Path:
    """Generate contact sheet JPEGs for visual review."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("cv2 not available, skipping review grids")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    names = {p.person_id: p.display_name for p in people}
    thumb_size = (256, 256)
    cols = 4
    padding = 8
    label_h = 30
    bg_color = (40, 40, 40)
    text_color = (255, 255, 255)

    by_person: dict[str, list] = {}
    for r in records:
        by_person.setdefault(r.person_id, []).append(r)

    for pid, precs in by_person.items():
        precs = precs[:max_per_person]
        rows_count = (len(precs) + cols - 1) // cols
        cell_w = thumb_size[0] + padding
        cell_h = thumb_size[1] + label_h + padding
        grid_w = cols * cell_w + padding
        grid_h = rows_count * cell_h + padding
        grid = np.full((grid_h, grid_w, 3), bg_color, dtype=np.uint8)

        for idx, rec in enumerate(precs):
            r = idx // cols
            c = idx % cols
            x = padding + c * cell_w
            y = padding + r * cell_h

            try:
                img = cv2.imread(rec.local_path)
                if img is not None:
                    thumb = cv2.resize(img, thumb_size)
                    grid[y:y + thumb_size[1], x:x + thumb_size[0]] = thumb
            except Exception:
                pass

            label = Path(rec.local_path).stem[:20]
            cv2.putText(grid, label, (x, y + thumb_size[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1)

        display_name = names.get(pid, pid)
        cv2.putText(grid, f"{display_name} ({len(precs)} images)", (padding, grid_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out_path = output_dir / f"review_{pid}.jpg"
        cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        logger.info("Review grid: %s -> %s", pid, out_path)

    return output_dir


def compute_retrieval_diagnostics(
    index_path: Path,
    metadata_path: Path,
    records: list,
    people: list,
) -> dict:
    """Build query embeddings from query images, search against reference index, compute Top-K."""
    try:
        import cv2
        import numpy as np
        from search.flat_index import FlatIndex
        from search.reverse_search_service import ReverseSearchService
        from services.face_service import FaceService
    except ImportError as exc:
        logger.warning("Cannot compute retrieval diagnostics: %s", exc)
        return {}

    face_service = FaceService()
    face_service.get_model()

    service = ReverseSearchService(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
    )

    names = {p.person_id: p.display_name for p in people}
    query_records = [r for r in records if r.query]

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_queries = 0
    per_person_stats: dict[str, dict] = {}

    for rec in query_records:
        try:
            img = cv2.imread(rec.local_path)
            if img is None:
                continue
            model = face_service.get_model()
            faces = model.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            embedding = face.normed_embedding
            if embedding is None:
                continue

            result = service.search(embedding, k=5)
            if not result.candidates:
                continue

            total_queries += 1
            retrieved_ids = [c.person_id for c in result.candidates]
            is_top1 = retrieved_ids[0] == rec.person_id if retrieved_ids else False
            is_top3 = rec.person_id in retrieved_ids[:3]
            is_top5 = rec.person_id in retrieved_ids[:5]

            if is_top1:
                top1_correct += 1
            if is_top3:
                top3_correct += 1
            if is_top5:
                top5_correct += 1

            pp = per_person_stats.setdefault(rec.person_id, {"total": 0, "top1": 0, "top3": 0, "top5": 0})
            pp["total"] += 1
            if is_top1:
                pp["top1"] += 1
            if is_top3:
                pp["top3"] += 1
            if is_top5:
                pp["top5"] += 1

        except Exception as exc:
            logger.debug("Retrieval error for %s: %s", rec.image_id, exc)

    if total_queries == 0:
        return {"total_queries": 0}

    return {
        "total_queries": total_queries,
        "top1_accuracy": top1_correct / total_queries,
        "top3_accuracy": top3_correct / total_queries,
        "top5_accuracy": top5_correct / total_queries,
        "top1_correct": top1_correct,
        "top3_correct": top3_correct,
        "top5_correct": top5_correct,
        "per_person": per_person_stats,
    }


def verify_invariants(records: list, split: dict, index_metadata: dict | None) -> list[str]:
    """Verify all automated invariants. Returns list of violations."""
    violations: list[str] = []

    for r in records:
        if r.faces_detected != 1:
            violations.append(f"ACCEPTED record {r.image_id} has {r.faces_detected} faces (expected 1)")
        if r.image_category == "representation":
            violations.append(f"ACCEPTED record {r.image_id} is representation")
        if r.status != "valid":
            violations.append(f"ACCEPTED record {r.image_id} has status={r.status} (expected valid)")

    hashes = [r.sha256 for r in records]
    hash_counts = Counter(hashes)
    for h, count in hash_counts.items():
        if count > 1:
            dup_ids = [r.image_id for r in records if r.sha256 == h]
            violations.append(f"Cross-person duplicate: {dup_ids}")

    ref_hashes = set()
    for pid, recs in split.get("reference", {}).items():
        for r in recs:
            ref_hashes.add(r.sha256)
    query_hashes = set()
    for pid, recs in split.get("query", {}).items():
        for r in recs:
            query_hashes.add(r.sha256)
    overlap = ref_hashes & query_hashes
    if overlap:
        violations.append(f"Cross-split leakage: {len(overlap)} overlapping hashes")

    for r in records:
        if not r.person_id:
            violations.append(f"Record {r.image_id} has empty person_id")

    ref_dir = DATASET_DIR / "reference"
    query_dir = DATASET_DIR / "query"
    if ref_dir.exists():
        for pid in split.get("reference", {}):
            person_ref = ref_dir / pid
            if not person_ref.exists():
                violations.append(f"Reference dir missing for {pid}")
    if query_dir.exists():
        for pid in split.get("query", {}):
            person_query = query_dir / pid
            if not person_query.exists():
                violations.append(f"Query dir missing for {pid}")

    if index_metadata:
        expected_vectors = index_metadata.get("total_vectors", 0)
        actual_ref_count = sum(len(v) for v in split.get("reference", {}).values())
        if expected_vectors != actual_ref_count:
            violations.append(f"FAISS vector count {expected_vectors} != reference count {actual_ref_count}")
        if index_metadata.get("embedding_dimension") != 512:
            violations.append(f"FAISS dimension {index_metadata.get('embedding_dimension')} != 512")

    return violations


def main() -> int:
    t_start = time.time()

    people_all = load_people()
    actors_all = [p for p in people_all if p.category == "actor"]
    footballers_all = [p for p in people_all if p.category == "football_player"]
    people_stage_a = actors_all[:5] + footballers_all[:5]
    actors = [p for p in people_stage_a if p.category == "actor"]
    footballers = [p for p in people_stage_a if p.category == "football_player"]

    logger.info("=" * 60)
    logger.info("PHASE 13.6.2 — STAGE A COLLECTION")
    logger.info("=" * 60)
    logger.info("Identities: %d (%d actors, %d footballers)", len(people_stage_a), len(actors), len(footballers))
    for p in people_stage_a:
        logger.info("  %s: %s (%s) — %d queries", p.person_id, p.display_name, p.category, len(p.search_queries))
    logger.info("Max images/person: %d", MAX_IMAGES_PER_PERSON)
    logger.info("Dataset: %s", DATASET_DIR)

    from services.face_service import FaceService
    from dataset_acquisition.downloader import Downloader
    from dataset_acquisition.models import (
        RejectionDetail, CollectionStats, compute_stats_from_records,
    )
    from dataset_acquisition.splitter import split_reference_query, copy_split_to_disk
    from dataset_acquisition.manifest import (
        generate_manifest, generate_quality_report, compute_rejection_stats,
    )
    from dataset_acquisition.sources.wikimedia import WikimediaSource

    face_service = FaceService()
    face_service.get_model()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wikimedia = WikimediaSource(delay=5.0, max_retries=5, max_rate_limit_retries=12)
    downloader = Downloader(
        output_dir=DATASET_DIR,
        sources=[wikimedia],
        max_images_per_person=MAX_IMAGES_PER_PERSON,
        min_image_width=200,
        min_image_height=200,
        face_service=face_service,
    )

    all_rejection_details: list[RejectionDetail] = []
    per_person_timing: dict[str, float] = {}

    for i, person in enumerate(people_stage_a, 1):
        logger.info("[%d/%d] Acquiring %s ...", i, len(people_stage_a), person.display_name)
        t0 = time.time()
        records, rej_details = downloader.download_person(person)
        elapsed = time.time() - t0
        per_person_timing[person.person_id] = elapsed
        all_rejection_details.extend(rej_details)
        logger.info(
            "  -> %d images acquired, %d rejected in %.1fs",
            len(records), len(rej_details), elapsed,
        )

    wikimedia.close()

    state = downloader._load_state()
    all_records_raw = state.get("records", [])
    stage_a_ids = {p.person_id for p in people_stage_a}
    all_records = []
    from dataset_acquisition.models import ImageRecord
    for r in all_records_raw:
        if r.get("person_id") in stage_a_ids:
            all_records.append(ImageRecord.from_dict(r))

    logger.info("Total records loaded: %d", len(all_records))

    seen_hashes: set[str] = set()
    unique_records: list[ImageRecord] = []
    dup_count = 0
    for r in all_records:
        if r.sha256 not in seen_hashes:
            seen_hashes.add(r.sha256)
            unique_records.append(r)
        else:
            dup_count += 1

    logger.info("Unique: %d, cross-person duplicates: %d", len(unique_records), dup_count)

    stats = compute_stats_from_records(unique_records)
    split = split_reference_query(
        unique_records,
        reference_ratio=0.6,
        min_reference=2,
        min_query=1,
        seed=SEED,
    )

    ref_hashes = set()
    for records in split["reference"].values():
        for r in records:
            ref_hashes.add(r.sha256)
    query_hashes = set()
    for records in split["query"].values():
        for r in records:
            query_hashes.add(r.sha256)
    leakage = ref_hashes & query_hashes
    logger.info("Cross-split leakage: %d", len(leakage))

    split_dir = DATASET_DIR / "split"
    copy_split_to_disk(split, split_dir)

    rstats = compute_rejection_stats(unique_records, all_rejection_details)

    metadata_dir = DATASET_DIR / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    manifest = generate_manifest(
        metadata_dir, "celebrity-v3-expanded-stage-a", people_stage_a,
        unique_records, split, stats, seed=SEED,
        rejection_details=all_rejection_details,
    )

    report_path = generate_quality_report(
        metadata_dir, "celebrity-v3-expanded-stage-a", unique_records,
        split, stats, people_stage_a,
        rejection_details=all_rejection_details,
    )

    records_path = metadata_dir / "images.json"
    records_path.write_text(
        json.dumps([r.to_dict() for r in unique_records], indent=2),
        encoding="utf-8",
    )

    review_path = OUTPUT_DIR / "stage_a_review.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)

    per_person_summary: dict[str, dict] = {}
    for person in people_stage_a:
        p_records = [r for r in unique_records if r.person_id == person.person_id]
        pp = rstats.per_person.get(person.person_id, {})
        per_person_summary[person.person_id] = {
            "display_name": person.display_name,
            "category": person.category,
            "accepted": pp.get("accepted", 0),
            "rejected_total": pp.get("rejected_total", 0),
            "representation": pp.get("representation", 0),
            "no_face": pp.get("no_face", 0),
            "multi_face": pp.get("multi_face", 0),
            "download_error": pp.get("download_error", 0),
            "duplicate": pp.get("duplicate", 0),
            "reference_count": len(split["reference"].get(person.person_id, [])),
            "query_count": len(split["query"].get(person.person_id, [])),
            "timing_seconds": per_person_timing.get(person.person_id, 0),
            "queries_used": list(person.search_queries),
        }

    review_data = {
        "phase": "13.6.2_stage_a",
        "status": "STAGE_A_COLLECTED",
        "dataset_version": "celebrity-v3-expanded-stage-a",
        "identities": len(people_stage_a),
        "actors": len(actors),
        "footballers": len(footballers),
        "total_candidates": rstats.total_candidates,
        "accepted": rstats.accepted,
        "rejected_total": rstats.rejected_total,
        "acceptance_rate": rstats.accepted / max(1, rstats.total_candidates),
        "candidates_per_person": rstats.total_candidates / max(1, len(people_stage_a)),
        "images_per_person": rstats.accepted / max(1, len(people_stage_a)),
        "rejections_by_reason": rstats.rejections_by_reason,
        "cross_person_duplicates": dup_count,
        "cross_split_leakage": len(leakage),
        "per_person": per_person_summary,
        "per_source": rstats.per_source,
        "total_time_seconds": time.time() - t_start,
    }

    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    logger.info("\n" + "=" * 60)
    logger.info("STAGE A ACQUISITION SUMMARY")
    logger.info("=" * 60)
    logger.info("Identities: %d", len(people_stage_a))
    logger.info("Total candidates: %d", rstats.total_candidates)
    logger.info("Accepted: %d (%.1f%%)", rstats.accepted, 100 * rstats.accepted / max(1, rstats.total_candidates))
    logger.info("Rejected: %d", rstats.rejected_total)
    logger.info("Rejection reasons: %s", rstats.rejections_by_reason)
    logger.info("Cross-person duplicates: %d", dup_count)
    logger.info("Cross-split leakage: %d", len(leakage))
    for pid, ps in per_person_summary.items():
        logger.info("  %s: accepted=%d, rejected=%d (repr=%d, no_face=%d, multi=%d), ref=%d q=%d, %.1fs",
                     pid, ps["accepted"], ps["rejected_total"],
                     ps["representation"], ps["no_face"], ps["multi_face"],
                     ps["reference_count"], ps["query_count"], ps["timing_seconds"])
    logger.info("Per-source: %s", rstats.per_source)
    logger.info("Total time: %.1fs", time.time() - t_start)

    logger.info("\n--- Building FAISS reference index ---")
    index_dir = DATASET_DIR / "search_index"
    index_dir.mkdir(parents=True, exist_ok=True)
    ref_split_dir = DATASET_DIR / "split" / "reference"

    index_metadata = None
    try:
        from search.index_builder import IndexBuilder
        index, build_report = IndexBuilder(dimension=512).build(
            dataset_dir=str(ref_split_dir),
            output_dir=str(index_dir),
        )
        logger.info("FAISS index built: %d vectors", index.size)

        meta_path = index_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                index_metadata = json.load(f)
    except Exception as exc:
        logger.warning("FAISS index build failed: %s", exc)

    logger.info("\n--- Computing retrieval diagnostics ---")
    retrieval = {}
    try:
        retrieval = compute_retrieval_diagnostics(
            index_path=index_dir / "reference_index.faiss",
            metadata_path=index_dir / "metadata.json",
            records=unique_records,
            people=people_stage_a,
        )
        if retrieval:
            logger.info("Retrieval diagnostics:")
            logger.info("  Total queries: %d", retrieval["total_queries"])
            logger.info("  Top-1 accuracy: %.1f%%", 100 * retrieval["top1_accuracy"])
            logger.info("  Top-3 accuracy: %.1f%%", 100 * retrieval["top3_accuracy"])
            logger.info("  Top-5 accuracy: %.1f%%", 100 * retrieval["top5_accuracy"])
    except Exception as exc:
        logger.warning("Retrieval diagnostics failed: %s", exc)

    logger.info("\n--- Generating visual review grids ---")
    review_grid_dir = OUTPUT_DIR / "review_grids"
    try:
        generate_review_grids(unique_records, people_stage_a, review_grid_dir)
    except Exception as exc:
        logger.warning("Review grid generation failed: %s", exc)

    logger.info("\n--- Verifying automated invariants ---")
    violations = verify_invariants(unique_records, split, index_metadata)
    if violations:
        logger.warning("Invariant violations found:")
        for v in violations:
            logger.warning("  %s", v)
    else:
        logger.info("All automated invariants PASSED")

    retrieval_path = OUTPUT_DIR / "retrieval_diagnostics.json"
    with open(retrieval_path, "w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=2)

    violations_path = OUTPUT_DIR / "invariant_violations.json"
    with open(violations_path, "w", encoding="utf-8") as f:
        json.dump(violations, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("STAGE A COMPLETE — ARTIFACTS")
    logger.info("=" * 60)
    logger.info("Dataset: %s", DATASET_DIR)
    logger.info("Manifest: %s", metadata_dir / "dataset_manifest.json")
    logger.info("Quality report: %s", report_path)
    logger.info("Image records: %s", records_path)
    logger.info("FAISS index: %s", index_dir / "reference_index.faiss")
    logger.info("Retrieval diagnostics: %s", retrieval_path)
    logger.info("Review grids: %s", review_grid_dir)
    logger.info("Review data: %s", review_path)
    logger.info("Invariant violations: %s", violations_path)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
