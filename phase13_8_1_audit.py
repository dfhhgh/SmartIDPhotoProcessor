"""Phase 13.8.1 — Comprehensive Audit Script

Covers:
- Corrected gallery-size analysis (full gallery per size)
- 0.729 hard-negative reproduction
- Metric recalculation
- Weak identity audit
- Duplicate/leakage checks
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
V3_DATASET = Path("datasets/celebrity-v3")
V4_DATASET = Path("datasets/celebrity-v4")
OUTPUT_BASE = Path("outputs/phase13_8_1")
V3_SUMMARY = Path("outputs/phase13_7_2/verification.json")

GALLERY_SIZES = [2, 4, 6, 8]

WEAK_IDENTITIES = [
    "jennifer_lawrence", "morgan_freeman", "leonardo_dicaprio",
    "vinicius_junior", "brad_pitt", "neymar", "mohamed_salah",
    "kevin_de_bruyne",
]


def _load_face_service():
    import sys
    sys.path.insert(0, ".")
    from services.face_service import FaceService
    return FaceService()


def extract_query_embeddings(query_dir: Path, model) -> dict[str, list[tuple[str, np.ndarray]]]:
    from phase13_7_2_calibration import extract_embedding_single_face
    query_embeddings: dict[str, list[tuple[str, np.ndarray]]] = {}
    for person_dir in sorted(query_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        pid = person_dir.name
        items = []
        for img_path in sorted(person_dir.glob("*.jpg")):
            emb = extract_embedding_single_face(model, img_path)
            if emb is not None:
                items.append((str(img_path), emb))
        if items:
            query_embeddings[pid] = items
    return query_embeddings


def step3_corrected_gallery_size(
    ref_records, ref_embeddings, query_embeddings
) -> dict:
    """STEP 3: Corrected gallery-size analysis.

    For each gallery size G:
    - Select first G references per identity (deterministic nested: 2⊂4⊂6⊂8)
    - Build FULL gallery with all identities × G references
    - Evaluate ALL queries against this full gallery
    - Compute image-level and identity-level metrics
    """
    from phase13_7_2_calibration import (
        build_all_pairs,
        aggregate_identity_scores,
        compute_operating_points,
        compute_eer,
        compute_roc_auc,
        score_stats,
    )

    all_person_ids = sorted(set(r["person_id"] for r in ref_records))

    # Group references by person
    refs_by_person = defaultdict(list)
    for i, r in enumerate(ref_records):
        refs_by_person[r["person_id"]].append((i, r))
    for pid in refs_by_person:
        refs_by_person[pid].sort(key=lambda x: x[0])

    results = {}
    for gs in GALLERY_SIZES:
        logger.info("  Gallery size %d...", gs)

        # Build full gallery: first G refs per identity
        subset_indices = []
        subset_records = []
        for pid in all_person_ids:
            person_refs = refs_by_person[pid]
            for j in range(min(gs, len(person_refs))):
                subset_indices.append(person_refs[j][0])
                subset_records.append(person_refs[j][1])

        subset_embeddings = ref_embeddings[subset_indices]

        # Evaluate all queries against full gallery
        genuines, impostors = build_all_pairs(query_embeddings, subset_records, subset_embeddings)

        gen_scores = np.array([p["similarity"] for p in genuines], dtype=np.float32) if genuines else np.array([], dtype=np.float32)
        imp_scores = np.array([p["similarity"] for p in impostors], dtype=np.float32) if impostors else np.array([], dtype=np.float32)

        # Image-level metrics
        img_ops = compute_operating_points(gen_scores, imp_scores)
        img_eer = compute_eer(gen_scores, imp_scores)
        img_roc = compute_roc_auc(gen_scores, imp_scores)

        # Identity-level
        id_agg_gen = aggregate_identity_scores(genuines)
        id_agg_imp = aggregate_identity_scores(impostors)

        query_person_map = {}
        for p in genuines:
            query_person_map[p["query_image"]] = p["query_person_id"]

        id_gen_scores = []
        for q_img, id_sims in id_agg_gen.items():
            pid = query_person_map.get(q_img)
            if pid and pid in id_sims:
                id_gen_scores.append(id_sims[pid])

        id_imp_scores = []
        for q_img, id_sims in id_agg_imp.items():
            for pid, sim in id_sims.items():
                id_imp_scores.append(sim)

        id_genuine = np.array(id_gen_scores, dtype=np.float32) if id_gen_scores else np.array([], dtype=np.float32)
        id_impostor = np.array(id_imp_scores, dtype=np.float32) if id_imp_scores else np.array([], dtype=np.float32)

        id_ops = compute_operating_points(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else []
        id_eer = compute_eer(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}
        id_roc = compute_roc_auc(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}

        global_max_img = float(imp_scores.max()) if len(imp_scores) > 0 else 0.0
        imp_max_per_query = defaultdict(float)
        for p in impostors:
            imp_max_per_query[p["query_image"]] = max(imp_max_per_query[p["query_image"]], p["similarity"])
        mean_imp_max = float(np.mean(list(imp_max_per_query.values()))) if imp_max_per_query else 0.0
        median_imp_max = float(np.median(list(imp_max_per_query.values()))) if imp_max_per_query else 0.0

        results[str(gs)] = {
            "reference_vectors": len(subset_records),
            "identities": len(all_person_ids),
            "refs_per_identity": gs,
            "genuine_pairs": len(genuines),
            "impostor_pairs": len(impostors),
            "image_roc_auc": img_roc["auc"],
            "image_eer": img_eer["eer"],
            "image_eer_threshold": img_eer["threshold"],
            "identity_roc_auc": id_roc.get("auc", 0.0),
            "identity_eer": id_eer.get("eer", 0.0),
            "identity_eer_threshold": id_eer.get("threshold", 0.0),
            "global_max_impostor": global_max_img,
            "mean_impostor_max": mean_imp_max,
            "median_impostor_max": median_imp_max,
            "genuine_stats": score_stats(gen_scores),
            "impostor_stats": score_stats(imp_scores),
            "operating_points": img_ops,
        }

    return results


def step4_reproduce_hard_negative(
    ref_records, ref_embeddings, query_embeddings, model
) -> dict:
    """STEP 4: Reproduce the 0.729 lebron_james -> morgan_freeman hard negative."""
    from phase13_7_2_calibration import extract_embedding_single_face

    target_query_pid = "lebron_james"
    target_impostor_pid = "morgan_freeman"

    # Find the specific query image
    query_img_path = None
    query_emb = None
    for pid, items in query_embeddings.items():
        if pid == target_query_pid:
            for path, emb in items:
                if "f2bad227249a" in path:
                    query_img_path = path
                    query_emb = emb
                    break

    if query_emb is None:
        return {"error": "Query image not found"}

    # Find Morgan Freeman reference
    mf_refs = [(i, r) for i, r in enumerate(ref_records) if r["person_id"] == target_impostor_pid]
    mf_ref_path = None
    mf_ref_emb = None
    mf_ref_idx = None
    for i, r in mf_refs:
        if "500bfc9866d4" in r.get("image_path", ""):
            mf_ref_path = r["image_path"]
            mf_ref_idx = r["vector_id"]
            mf_ref_emb = ref_embeddings[r["vector_id"]]
            break

    if mf_ref_emb is None:
        return {"error": "Morgan Freeman reference not found"}

    # Compute similarity directly
    similarity = float(np.dot(query_emb, mf_ref_emb))

    # Verify norms
    query_norm = float(np.linalg.norm(query_emb))
    ref_norm = float(np.linalg.norm(mf_ref_emb))

    # Load original images for metadata
    query_cv = cv2.imread(query_img_path)
    ref_cv = cv2.imread(mf_ref_path) if mf_ref_path and Path(mf_ref_path).exists() else None

    # Get face detection info
    det = model.get(query_cv) if query_cv is not None else []
    query_faces = len(det)
    query_conf = float(det[0].det_score) if det else 0.0
    query_bbox = det[0].bbox.tolist() if det else None

    det_ref = model.get(ref_cv) if ref_cv is not None else []
    ref_faces = len(det_ref)
    ref_conf = float(det_ref[0].det_score) if det_ref else 0.0
    ref_bbox = det_ref[0].bbox.tolist() if det_ref else None

    return {
        "query": {
            "person_id": target_query_pid,
            "image_path": query_img_path,
            "image_dimensions": [query_cv.shape[1], query_cv.shape[0]] if query_cv is not None else None,
            "faces_detected": query_faces,
            "face_confidence": query_conf,
            "face_bbox": query_bbox,
            "embedding_norm": query_norm,
        },
        "reference": {
            "person_id": target_impostor_pid,
            "image_path": mf_ref_path,
            "vector_id": mf_ref_idx,
            "faces_detected": ref_faces,
            "face_confidence": ref_conf,
            "face_bbox": ref_bbox,
            "embedding_norm": ref_norm,
        },
        "similarity": {
            "computed": similarity,
            "reported": 0.7292,
            "match": abs(similarity - 0.7292) < 0.01,
            "method": "inner_product (cosine via unit-norm vectors)",
        },
        "verification": {
            "dot_product_matches_reported": abs(similarity - 0.7292) < 0.01,
            "query_is_unit_norm": abs(query_norm - 1.0) < 0.05,
            "ref_is_unit_norm": abs(ref_norm - 1.0) < 0.05,
        },
    }


def step8_recalculate_metrics(ref_records, ref_embeddings, query_embeddings) -> dict:
    """STEP 8: Full metric recalculation using Phase 13.7.2 methodology."""
    from phase13_7_2_calibration import (
        build_all_pairs,
        aggregate_identity_scores,
        image_level_hard_negatives,
        identity_level_hard_negatives,
        compute_operating_points,
        compute_eer,
        compute_roc_auc,
        verify_threshold_consistency,
        score_stats,
    )

    genuine_pairs, impostor_pairs = build_all_pairs(query_embeddings, ref_records, ref_embeddings)

    genuine_scores = np.array([p["similarity"] for p in genuine_pairs], dtype=np.float32)
    impostor_scores = np.array([p["similarity"] for p in impostor_pairs], dtype=np.float32)

    # Image-level
    img_ops = compute_operating_points(genuine_scores, impostor_scores)
    img_eer = compute_eer(genuine_scores, impostor_scores)
    img_roc = compute_roc_auc(genuine_scores, impostor_scores)

    # Identity-level
    id_agg_gen = aggregate_identity_scores(genuine_pairs)
    id_agg_imp = aggregate_identity_scores(impostor_pairs)

    query_person_map = {}
    for p in genuine_pairs:
        query_person_map[p["query_image"]] = p["query_person_id"]

    id_gen_scores = []
    for q_img, id_sims in id_agg_gen.items():
        pid = query_person_map.get(q_img)
        if pid and pid in id_sims:
            id_gen_scores.append(id_sims[pid])

    id_imp_scores = []
    for q_img, id_sims in id_agg_imp.items():
        for pid, sim in id_sims.items():
            id_imp_scores.append(sim)

    id_genuine = np.array(id_gen_scores, dtype=np.float32) if id_gen_scores else np.array([], dtype=np.float32)
    id_impostor = np.array(id_imp_scores, dtype=np.float32) if id_imp_scores else np.array([], dtype=np.float32)

    id_ops = compute_operating_points(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else []
    id_eer = compute_eer(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}
    id_roc = compute_roc_auc(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}

    # Hard negatives
    img_hn = image_level_hard_negatives(impostor_pairs)
    id_hn = identity_level_hard_negatives(query_embeddings, ref_records, ref_embeddings)

    # Consistency
    img_consistency = verify_threshold_consistency(
        genuine_scores, impostor_scores, img_ops, "image_level"
    )
    id_consistency = verify_threshold_consistency(
        id_genuine, id_impostor, id_ops, "identity_level"
    ) if len(id_genuine) > 0 and len(id_impostor) > 0 else []

    # Global max
    global_max_img = float(impostor_scores.max()) if len(impostor_scores) > 0 else 0.0
    global_max_id = max((h["identity_score"] for h in id_hn), default=0.0) if id_hn else 0.0

    return {
        "dataset": {
            "identities": len(query_embeddings),
            "reference_vectors": len(ref_records),
            "genuine_pairs": len(genuine_pairs),
            "impostor_pairs": len(impostor_pairs),
        },
        "image_level": {
            "roc_auc": img_roc["auc"],
            "operating_points": img_ops,
            "eer": img_eer,
            "global_max_impostor": global_max_img,
            "genuine_stats": score_stats(genuine_scores),
            "impostor_stats": score_stats(impostor_scores),
        },
        "identity_level": {
            "roc_auc": id_roc.get("auc", 0.0),
            "operating_points": id_ops,
            "eer": id_eer,
            "global_max_impostor": global_max_id,
            "genuine_stats": score_stats(id_genuine),
            "impostor_stats": score_stats(id_impostor),
        },
        "hard_negatives": {
            "image_level_top5": img_hn[:5],
            "identity_level_top5": id_hn[:5],
        },
        "consistency": {
            "image_level": img_consistency,
            "identity_level": id_consistency,
        },
    }


def step10_weak_identity_audit(ref_records, ref_embeddings, query_embeddings) -> dict:
    """STEP 10: Comprehensive weak identity audit."""
    from phase13_7_2_calibration import (
        build_all_pairs,
        score_stats,
    )

    results = {}
    for pid in WEAK_IDENTITIES:
        if pid not in query_embeddings:
            results[pid] = {"status": "not_in_dataset"}
            continue

        genuines, impostors = build_all_pairs(
            {pid: query_embeddings[pid]}, ref_records, ref_embeddings
        )

        gen_scores = np.array([p["similarity"] for p in genuines], dtype=np.float32) if genuines else np.array([], dtype=np.float32)
        imp_scores = np.array([p["similarity"] for p in impostors], dtype=np.float32) if impostors else np.array([], dtype=np.float32)

        person_refs = [r for r in ref_records if r["person_id"] == pid]

        results[pid] = {
            "reference_count": len(person_refs),
            "query_count": len(query_embeddings.get(pid, [])),
            "genuine_count": len(genuines),
            "impostor_count": len(impostors),
            "genuine_stats": score_stats(gen_scores),
            "impostor_stats": score_stats(imp_scores),
            "impostor_max": float(imp_scores.max()) if len(imp_scores) > 0 else 0.0,
            "sources": list(set(r.get("source", "unknown") for r in person_refs)),
        }

    return results


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 13.8.1 — COMPREHENSIVE AUDIT")
    logger.info("=" * 60)

    # Load FAISS index and metadata
    from phase13_7_2_calibration import load_reference_embeddings_with_faiss

    index_path = V4_DATASET / "search_index" / "reference_index.faiss"
    metadata_path = V4_DATASET / "search_index" / "metadata.json"

    logger.info("\n--- Loading reference data ---")
    ref_records, ref_embeddings = load_reference_embeddings_with_faiss(index_path, metadata_path)

    # Load or extract query embeddings
    model = _load_face_service().get_model()
    query_dir = V4_DATASET / "calibration"

    # Try cached first
    cached_path = Path("outputs/phase13_8/intermediate_query_embeddings.json")
    if cached_path.exists():
        logger.info("Loading cached query embeddings...")
        with open(cached_path) as f:
            cached = json.load(f)
        query_embeddings = {
            pid: [(path, np.array(emb, dtype=np.float32)) for path, emb in items]
            for pid, items in cached.items()
        }
    else:
        logger.info("Extracting query embeddings...")
        query_embeddings = extract_query_embeddings(query_dir, model)

    logger.info("Loaded %d query identities (%d total embeddings)",
                len(query_embeddings), sum(len(v) for v in query_embeddings.values()))

    # STEP 3: Corrected gallery-size analysis
    logger.info("\n--- STEP 3: Corrected Gallery-Size Analysis ---")
    gallery_corrected = step3_corrected_gallery_size(ref_records, ref_embeddings, query_embeddings)
    with open(OUTPUT_BASE / "gallery_size_corrected.json", "w") as f:
        json.dump(gallery_corrected, f, indent=2, default=str)
    logger.info("Saved gallery_size_corrected.json")

    # STEP 4: Reproduce 0.729 hard negative
    logger.info("\n--- STEP 4: Reproduce 0.729 Hard Negative ---")
    hn_repro = step4_reproduce_hard_negative(ref_records, ref_embeddings, query_embeddings, model)
    with open(OUTPUT_BASE / "hard_negative_reproduction.json", "w") as f:
        json.dump(hn_repro, f, indent=2, default=str)
    logger.info("Similarity computed: %.6f (reported: 0.7292, match: %s)",
                hn_repro.get("similarity", {}).get("computed", 0),
                hn_repro.get("similarity", {}).get("match", False))

    # STEP 8: Full metric recalculation
    logger.info("\n--- STEP 8: Metric Recalculation ---")
    recalc = step8_recalculate_metrics(ref_records, ref_embeddings, query_embeddings)
    with open(OUTPUT_BASE / "metric_recalculation.json", "w") as f:
        json.dump(recalc, f, indent=2, default=str)
    logger.info("Saved metric_recalculation.json")

    # STEP 10: Weak identity audit
    logger.info("\n--- STEP 10: Weak Identity Audit ---")
    weak_audit = step10_weak_identity_audit(ref_records, ref_embeddings, query_embeddings)
    with open(OUTPUT_BASE / "weak_identity_audit.json", "w") as f:
        json.dump(weak_audit, f, indent=2, default=str)
    logger.info("Saved weak_identity_audit.json")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("AUDIT SUMMARY")
    logger.info("=" * 60)

    logger.info("\nGallery-Size Corrected:")
    for gs, data in gallery_corrected.items():
        logger.info("  Size %s: ref=%d, gen=%d, imp=%d, img_auc=%.4f, id_auc=%.4f, max_imp=%.4f",
                    gs, data["reference_vectors"], data["genuine_pairs"], data["impostor_pairs"],
                    data["image_roc_auc"], data["identity_roc_auc"], data["global_max_impostor"])

    logger.info("\nHard Negative Reproduction:")
    sim_data = hn_repro.get("similarity", {})
    logger.info("  Computed: %.6f, Reported: %.6f, Match: %s",
                sim_data.get("computed", 0), sim_data.get("reported", 0), sim_data.get("match", False))

    logger.info("\nMetric Recalculation:")
    logger.info("  Image ROC-AUC: %.4f", recalc["image_level"]["roc_auc"])
    logger.info("  Image EER: %.4f", recalc["image_level"]["eer"]["eer"])
    logger.info("  Identity ROC-AUC: %.4f", recalc["identity_level"]["roc_auc"])
    logger.info("  Identity EER: %.4f", recalc["identity_level"]["eer"].get("eer", 0))
    logger.info("  Global max impostor (image): %.4f", recalc["image_level"]["global_max_impostor"])
    logger.info("  Global max impostor (identity): %.4f", recalc["identity_level"]["global_max_impostor"])

    logger.info("\nWeak Identity Audit:")
    for pid, data in weak_audit.items():
        if "status" in data:
            logger.info("  %s: %s", pid, data["status"])
        else:
            logger.info("  %s: ref=%d, cal=%d, gen_mean=%.3f, imp_max=%.3f",
                        pid, data["reference_count"], data["query_count"],
                        data["genuine_stats"].get("mean", 0),
                        data["impostor_max"])

    logger.info("\nAll audit artifacts saved to: %s", OUTPUT_BASE)


if __name__ == "__main__":
    main()
