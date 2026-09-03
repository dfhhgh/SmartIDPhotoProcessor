"""Phase 13.6.3 — Scaled Identity-Labeled Celebrity Dataset Collection & Evaluation

Collects ~20-25 identities with 10-15 accepted images each, then evaluates:
  - Reference/query split
  - FAISS index build
  - Retrieval evaluation (Top-1/3/5)
  - Genuine/impostor distributions
  - ROC-AUC, EER
  - Hard negative analysis
  - Diversity report
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Configuration ---
SEED = 42
TARGET_IMAGES_PER_PERSON = 12
MIN_IMAGES_PER_PERSON = 10
MANIFEST_PATH = Path("dataset_acquisition/people_v3_scaled.json")
OUTPUT_BASE = Path("outputs/phase13_6_3")
DATASET_BASE = Path("datasets/celebrity-v3")


def load_people(manifest_path: Path) -> list:
    """Load people from manifest file."""
    from dataset_acquisition.models import Person

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    persons = []
    for p in data["people"]:
        persons.append(Person(
            person_id=p["person_id"],
            display_name=p["display_name"],
            category=p["category"],
            aliases=tuple(p.get("aliases", [])),
            search_queries=tuple(p.get("search_queries", [])),
        ))
    return persons


def run_collection(persons: list, output_dir: Path) -> dict[str, Any]:
    """Run scaled collection with source fallback."""
    from dataset_acquisition.orchestrator import AcquisitionOrchestrator

    logger.info("=" * 60)
    logger.info("PHASE 13.6.3 — Scaled Collection")
    logger.info("=" * 60)

    # Load face service
    face_service = None
    try:
        import sys
        sys.path.insert(0, ".")
        from services.face_service import FaceService
        face_service = FaceService()
        logger.info("FaceService loaded for face validation.")
    except Exception as exc:
        logger.warning("Could not load FaceService (%s). Using fallback.", exc)

    orchestrator = AcquisitionOrchestrator(
        output_dir=output_dir,
        max_images_per_person=TARGET_IMAGES_PER_PERSON,
        max_candidates_per_source=50,
        wikimedia_delay=2.0,
        openverse_delay=1.0,
        face_service=face_service,
    )

    t0 = time.time()
    collection_result = orchestrator.collect_all(persons, TARGET_IMAGES_PER_PERSON)
    total_runtime = time.time() - t0

    collection_result["total_runtime_seconds"] = round(total_runtime, 2)
    collection_result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    collection_result["seed"] = SEED
    collection_result["target_images_per_person"] = TARGET_IMAGES_PER_PERSON

    return collection_result


def run_split_and_copy(
    records: list, persons: list, output_dir: Path
) -> dict[str, Any]:
    """Split reference/query and copy to dataset directories."""
    from dataset_acquisition.splitter import split_reference_query, copy_split_to_disk

    logger.info("=" * 60)
    logger.info("Reference/Query Split")
    logger.info("=" * 60)

    split = split_reference_query(
        records,
        reference_ratio=0.67,  # ~8 ref / 4 query for 12 images
        min_reference=2,
        min_query=1,
        seed=SEED,
    )

    # Copy to disk
    ref_dir = DATASET_BASE / "reference"
    query_dir = DATASET_BASE / "query"
    ref_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    ref_stats = copy_split_to_disk(split, DATASET_BASE)

    # Count persons in each split
    ref_persons = len(split.get("reference", {}))
    query_persons = len(split.get("query", {}))
    excluded_persons = len(split.get("excluded", {}))

    total_ref = sum(len(v) for v in split.get("reference", {}).values())
    total_query = sum(len(v) for v in split.get("query", {}).values())

    logger.info(
        "Split: %d reference images, %d query images, %d excluded persons",
        total_ref, total_query, excluded_persons,
    )

    return {
        "split": split,
        "ref_persons": ref_persons,
        "query_persons": query_persons,
        "excluded_persons": excluded_persons,
        "total_reference": total_ref,
        "total_query": total_query,
        "ref_stats": ref_stats,
    }


def run_faiss_build(ref_dir: Path, index_dir: Path) -> dict[str, Any]:
    """Build FAISS index from reference images."""
    from search.index_builder import IndexBuilder

    logger.info("=" * 60)
    logger.info("FAISS Index Build")
    logger.info("=" * 60)

    index_dir.mkdir(parents=True, exist_ok=True)

    builder = IndexBuilder(dimension=512)
    index, build_report = builder.build(
        dataset_dir=str(ref_dir),
        output_dir=str(index_dir),
        index_filename="reference_index.faiss",
        metadata_filename="metadata.json",
    )

    logger.info(
        "FAISS index built: %d vectors, %d persons",
        build_report.accepted_embeddings, build_report.total_persons,
    )
    if build_report.skipped_images > 0:
        logger.warning(
            "Skipped %d images during index build", build_report.skipped_images
        )

    return {
        "build_report": build_report.to_dict(),
        "index_size": index.size,
        "index_path": str(index_dir / "reference_index.faiss"),
        "metadata_path": str(index_dir / "metadata.json"),
    }


def run_retrieval_evaluation(
    index_path: Path, metadata_path: Path, query_dir: Path, persons: list
) -> dict[str, Any]:
    """Evaluate Top-1/3/5 retrieval on query images."""
    import sys
    sys.path.insert(0, ".")

    from search.reverse_search_service import ReverseSearchService

    logger.info("=" * 60)
    logger.info("Retrieval Evaluation")
    logger.info("=" * 60)

    service = ReverseSearchService(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
    )

    # Load query images and extract embeddings
    from services.face_service import FaceService
    face_service = FaceService()
    model = face_service.get_model()

    per_person_results: dict[str, dict[str, Any]] = {}
    all_queries = []
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_queries = 0

    for person in persons:
        person_query_dir = query_dir / person.person_id
        if not person_query_dir.exists():
            continue

        image_files = sorted(
            f for f in person_query_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        person_top1 = 0
        person_top3 = 0
        person_top5 = 0
        person_total = 0

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = model.get(img)
            if not faces:
                continue

            face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            embedding = face.normed_embedding
            if embedding is None:
                continue

            result = service.search(embedding, k=5)

            # Check if the correct person is in top-K
            top_labels = [c.person_id for c in result.candidates]
            person_total += 1
            total_queries += 1

            if person.person_id == top_labels[0] if top_labels else False:
                person_top1 += 1
                top1_correct += 1
            if person.person_id in top_labels[:3]:
                person_top3 += 1
                top3_correct += 1
            if person.person_id in top_labels[:5]:
                person_top5 += 1
                top5_correct += 1

            all_queries.append({
                "query_image": str(img_path),
                "person_id": person.person_id,
                "top1_match": top_labels[0] if top_labels else None,
                "top3_matches": top_labels[:3],
                "top5_matches": top_labels[:5],
                "similarities": [round(c.similarity, 4) for c in result.candidates],
            })

        if person_total > 0:
            per_person_results[person.person_id] = {
                "total_queries": person_total,
                "top1_correct": person_top1,
                "top3_correct": person_top3,
                "top5_correct": person_top5,
                "top1_accuracy": round(person_top1 / person_total, 4),
                "top3_accuracy": round(person_top3 / person_total, 4),
                "top5_accuracy": round(person_top5 / person_total, 4),
            }

    overall = {
        "total_queries": total_queries,
        "top1_accuracy": round(top1_correct / total_queries, 4) if total_queries > 0 else 0.0,
        "top3_accuracy": round(top3_correct / total_queries, 4) if total_queries > 0 else 0.0,
        "top5_accuracy": round(top5_correct / total_queries, 4) if total_queries > 0 else 0.0,
    }

    logger.info(
        "Retrieval: Top-1=%.2f%%, Top-3=%.2f%%, Top-5=%.2f%% (%d queries)",
        overall["top1_accuracy"] * 100,
        overall["top3_accuracy"] * 100,
        overall["top5_accuracy"] * 100,
        total_queries,
    )

    return {
        "overall": overall,
        "per_person": per_person_results,
        "query_details": all_queries,
    }


def run_calibration(
    index_path: Path, metadata_path: Path, query_dir: Path, persons: list
) -> dict[str, Any]:
    """Generate genuine/impostor pairs and compute ROC-AUC/EER."""
    import sys
    sys.path.insert(0, ".")

    from search.reverse_search_service import ReverseSearchService
    from search.calibration.pair_generator import PairGenerator
    from search.calibration.similarity_evaluator import SimilarityEvaluator
    from search.calibration.threshold_evaluator import ThresholdEvaluator

    logger.info("=" * 60)
    logger.info("Calibration: Genuine/Impostor + ROC-AUC + EER")
    logger.info("=" * 60)

    service = ReverseSearchService(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
    )

    from services.face_service import FaceService
    face_service = FaceService()
    model = face_service.get_model()

    # Extract embeddings for all query images
    identity_data: dict[str, list[tuple[str, np.ndarray]]] = {}
    for person in persons:
        person_query_dir = query_dir / person.person_id
        if not person_query_dir.exists():
            continue

        embeddings = []
        for img_path in sorted(person_query_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = model.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            emb = face.normed_embedding
            if emb is not None:
                embeddings.append((str(img_path), emb))

        if len(embeddings) >= 2:
            identity_data[person.person_id] = embeddings

    if len(identity_data) < 2:
        logger.warning("Not enough identities with embeddings for calibration.")
        return {"error": "insufficient_data"}

    # Generate pairs
    pair_gen = PairGenerator(seed=SEED)
    positive_pairs, negative_pairs = pair_gen.generate_pairs(
        identity_data, max_positive_pairs=1000, max_negative_pairs=1000
    )

    logger.info(
        "Pairs: %d genuine, %d impostor",
        len(positive_pairs), len(negative_pairs),
    )

    # Evaluate pairs
    evaluator = SimilarityEvaluator()
    pos_scores = np.array([evaluator.evaluate_pair(p).similarity for p in positive_pairs])
    neg_scores = np.array([evaluator.evaluate_pair(p).similarity for p in negative_pairs])

    # ROC-AUC
    threshold_eval = ThresholdEvaluator()
    roc_result = threshold_eval.compute_roc_auc(pos_scores, neg_scores)

    # EER
    eer_result = threshold_eval.estimate_eer_interpolated(pos_scores, neg_scores)

    # Threshold sweep
    sweep = threshold_eval.sweep_thresholds_uniform(pos_scores, neg_scores, num_steps=200)
    operating_points = threshold_eval.find_operating_points(sweep)

    logger.info("ROC-AUC: %.4f", roc_result.custom_auc)
    logger.info("EER: %.4f at threshold %.4f", eer_result.eer, eer_result.threshold)

    return {
        "genuine_pairs": len(positive_pairs),
        "impostor_pairs": len(negative_pairs),
        "genuine_score_stats": _score_stats(pos_scores),
        "impostor_score_stats": _score_stats(neg_scores),
        "roc_auc": {
            "custom_auc": round(roc_result.custom_auc, 6),
            "sklearn_auc": round(roc_result.sklearn_auc, 6),
            "absolute_difference": round(roc_result.absolute_difference, 6),
            "verified": roc_result.verified,
        },
        "eer": {
            "eer": round(eer_result.eer, 6),
            "threshold": round(eer_result.threshold, 6),
            "far_at_eer": round(eer_result.far_at_eer, 6),
            "frr_at_eer": round(eer_result.frr_at_eer, 6),
            "method": eer_result.method,
            "threshold_resolution": eer_result.threshold_resolution,
        },
        "operating_points": operating_points,
    }


def _score_stats(scores: np.ndarray) -> dict[str, Any]:
    """Compute basic statistics for a score array."""
    if len(scores) == 0:
        return {}
    return {
        "count": len(scores),
        "min": round(float(np.min(scores)), 6),
        "max": round(float(np.max(scores)), 6),
        "mean": round(float(np.mean(scores)), 6),
        "median": round(float(np.median(scores)), 6),
        "std": round(float(np.std(scores)), 6),
        "p5": round(float(np.percentile(scores, 5)), 6),
        "p25": round(float(np.percentile(scores, 25)), 6),
        "p75": round(float(np.percentile(scores, 75)), 6),
        "p95": round(float(np.percentile(scores, 95)), 6),
    }


def run_hard_negatives(
    index_path: Path, metadata_path: Path, query_dir: Path, persons: list
) -> dict[str, Any]:
    """Analyze highest impostor similarities."""
    import sys
    sys.path.insert(0, ".")

    from search.reverse_search_service import ReverseSearchService

    logger.info("=" * 60)
    logger.info("Hard Negative Analysis")
    logger.info("=" * 60)

    service = ReverseSearchService(
        index_path=str(index_path),
        metadata_path=str(metadata_path),
    )

    from services.face_service import FaceService
    face_service = FaceService()
    model = face_service.get_model()

    hard_negatives = []

    for person in persons:
        person_query_dir = query_dir / person.person_id
        if not person_query_dir.exists():
            continue

        for img_path in sorted(person_query_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = model.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            emb = face.normed_embedding
            if emb is None:
                continue

            result = service.search(emb, k=5)

            # Find the highest-scoring impostor (different person)
            for match in result.candidates:
                if match.person_id != person.person_id:
                    hard_negatives.append({
                        "query_person": person.person_id,
                        "query_image": str(img_path.name),
                        "impostor_person": match.person_id,
                        "impostor_image": match.image,
                        "similarity": round(match.similarity, 6),
                    })
                    break

    # Sort by similarity descending
    hard_negatives.sort(key=lambda x: x["similarity"], reverse=True)

    # Take top 20
    top_hard = hard_negatives[:20]

    logger.info("Analyzed %d query images for hard negatives", len(hard_negatives))
    if top_hard:
        logger.info(
            "Highest impostor similarity: %.4f (%s vs %s)",
            top_hard[0]["similarity"],
            top_hard[0]["query_person"],
            top_hard[0]["impostor_person"],
        )

    return {
        "total_query_images": len(hard_negatives),
        "top_20_hard_negatives": top_hard,
        "similarity_distribution": _score_stats(
            np.array([h["similarity"] for h in hard_negatives])
        ),
    }


def run_diversity_report(records: list) -> dict[str, Any]:
    """Generate dataset diversity report."""
    logger.info("=" * 60)
    logger.info("Diversity Report")
    logger.info("=" * 60)

    # Resolution distribution
    resolutions = [(r.width, r.height) for r in records if r.width > 0 and r.height > 0]
    areas = [w * h for w, h in resolutions]
    aspect_ratios = [w / h if h > 0 else 0 for w, h in resolutions]

    # Face confidence distribution
    confidences = [r.face_confidence for r in records if r.face_confidence > 0]

    # Source distribution
    source_dist: dict[str, int] = {}
    for r in records:
        source_dist[r.source] = source_dist.get(r.source, 0) + 1

    # Per-person counts
    person_counts: dict[str, int] = {}
    for r in records:
        person_counts[r.person_id] = person_counts.get(r.person_id, 0) + 1

    # License distribution
    license_dist: dict[str, int] = {}
    for r in records:
        license_dist[r.license] = license_dist.get(r.license, 0) + 1

    report = {
        "total_images": len(records),
        "unique_persons": len(person_counts),
        "resolution": {
            "count": len(resolutions),
            "min_area": min(areas) if areas else 0,
            "max_area": max(areas) if areas else 0,
            "mean_area": round(np.mean(areas), 0) if areas else 0,
            "median_area": round(np.median(areas), 0) if areas else 0,
        },
        "aspect_ratio": {
            "mean": round(np.mean(aspect_ratios), 4) if aspect_ratios else 0,
            "median": round(np.median(aspect_ratios), 4) if aspect_ratios else 0,
        },
        "face_confidence": _score_stats(np.array(confidences)),
        "source_distribution": source_dist,
        "license_distribution": license_dist,
        "per_person_counts": person_counts,
    }

    logger.info(
        "Diversity: %d images, %d persons, mean_area=%.0f",
        report["total_images"],
        report["unique_persons"],
        report["resolution"]["mean_area"],
    )

    return report


def run_leakage_check(split: dict) -> dict[str, Any]:
    """Check for reference/query leakage."""
    logger.info("=" * 60)
    logger.info("Leakage Check")
    logger.info("=" * 60)

    ref_hashes: set[str] = set()
    query_hashes: set[str] = set()

    for records in split.get("reference", {}).values():
        for r in records:
            ref_hashes.add(r.sha256)
    for records in split.get("query", {}).values():
        for r in records:
            query_hashes.add(r.sha256)

    overlap = ref_hashes & query_hashes

    result = {
        "reference_hashes": len(ref_hashes),
        "query_hashes": len(query_hashes),
        "overlap_count": len(overlap),
        "leakage_detected": len(overlap) > 0,
    }

    if overlap:
        logger.warning("LEAKAGE DETECTED: %d overlapping hashes!", len(overlap))
    else:
        logger.info("No leakage detected.")

    return result


def main() -> None:
    """Main execution pipeline."""
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    DATASET_BASE.mkdir(parents=True, exist_ok=True)

    # 1. Load people
    persons = load_people(MANIFEST_PATH)
    logger.info("Loaded %d persons from manifest", len(persons))

    # 2. Run collection
    collection = run_collection(persons, OUTPUT_BASE)

    # 3. Load all records
    state_path = OUTPUT_BASE / "download_state.json"
    all_records = []
    if state_path.exists():
        from dataset_acquisition.models import ImageRecord
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        all_records = [ImageRecord.from_dict(r) for r in state.get("records", [])]
    logger.info("Total records loaded: %d", len(all_records))

    # 4. Split
    split_result = run_split_and_copy(all_records, persons, OUTPUT_BASE)

    # 5. Build FAISS index
    ref_dir = DATASET_BASE / "reference"
    index_dir = DATASET_BASE / "search_index"
    faiss_result = run_faiss_build(ref_dir, index_dir)

    # 6. Retrieval evaluation
    query_dir = DATASET_BASE / "query"
    retrieval = run_retrieval_evaluation(
        Path(faiss_result["index_path"]),
        Path(faiss_result["metadata_path"]),
        query_dir,
        persons,
    )

    # 7. Calibration
    calibration = run_calibration(
        Path(faiss_result["index_path"]),
        Path(faiss_result["metadata_path"]),
        query_dir,
        persons,
    )

    # 8. Hard negatives
    hard_neg = run_hard_negatives(
        Path(faiss_result["index_path"]),
        Path(faiss_result["metadata_path"]),
        query_dir,
        persons,
    )

    # 9. Diversity report
    diversity = run_diversity_report(all_records)

    # 10. Leakage check
    leakage = run_leakage_check(split_result["split"])

    # Save all results
    results = {
        "phase": "13.6.3",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "target_images_per_person": TARGET_IMAGES_PER_PERSON,
        "collection": collection,
        "split": {
            "ref_persons": split_result["ref_persons"],
            "query_persons": split_result["query_persons"],
            "excluded_persons": split_result["excluded_persons"],
            "total_reference": split_result["total_reference"],
            "total_query": split_result["total_query"],
        },
        "faiss": faiss_result,
        "retrieval": retrieval,
        "calibration": calibration,
        "hard_negatives": hard_neg,
        "diversity": diversity,
        "leakage": leakage,
    }

    with open(OUTPUT_BASE / "dataset_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PHASE 13.6.3 COMPLETE")
    logger.info("=" * 60)
    logger.info("Results saved to %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
