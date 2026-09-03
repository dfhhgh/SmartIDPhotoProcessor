"""Phase 13.9 — Optimized build with progress tracking."""
from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
RAW_DIR = Path("datasets/non_celebrity-v1/raw")
OUTPUT_DIR = Path("datasets/non_celebrity-v1")
REF_DIR = OUTPUT_DIR / "reference"
HELD_DIR = OUTPUT_DIR / "held_out"
INDEX_DIR = OUTPUT_DIR / "search_index"
MANIFEST_PATH = OUTPUT_DIR / "dataset_manifest.json"
RESULTS_DIR = Path("outputs/phase13_9")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

REF_PER_ID = 6
HELD_PER_ID = 4

CELEBRITY_THRESHOLDS = {
    "eer": 0.0457, "youden_j": 0.2301, "far_5pct": 0.1039,
    "far_1pct": 0.1554, "far_0_5pct": 0.1816, "far_0_1pct": 0.2301,
}


def _load_face_service():
    sys.path.insert(0, ".")
    from services.face_service import FaceService
    return FaceService()


def discover_and_validate() -> dict[str, list[dict]]:
    """Discover raw images, validate faces, return valid images with embeddings."""
    model = _load_face_service().get_model()

    # Discover expression directories
    expression_dirs = []
    for d in RAW_DIR.iterdir():
        if d.is_dir() and d.name != "__MACOSX":
            expression_dirs.append(d)

    logger.info("Found %d expression directories", len(expression_dirs))

    # Group by identity
    all_images = defaultdict(list)
    for expr_dir in expression_dirs:
        for img_path in sorted(expr_dir.glob("*.jpg")):
            identity_id = img_path.stem.split("_")[0]
            all_images[identity_id].append(img_path)

    logger.info("Found %d identities", len(all_images))

    # Validate and extract embeddings
    valid_images = {}
    total = sum(len(v) for v in all_images.values())
    processed = 0
    rejected = defaultdict(int)

    for identity_id in sorted(all_images.keys()):
        images = all_images[identity_id]
        valid = []

        for img_path in images:
            processed += 1
            if processed % 50 == 0:
                logger.info("  Progress: %d/%d images", processed, total)

            img = cv2.imread(str(img_path))
            if img is None:
                rejected["unreadable"] += 1
                continue

            faces = model.get(img)
            if len(faces) == 0:
                rejected["no_face"] += 1
                continue
            elif len(faces) > 1:
                rejected["multiple_faces"] += 1
                continue

            face = faces[0]
            emb = face.normed_embedding if hasattr(face, "normed_embedding") and face.normed_embedding is not None else None
            if emb is None:
                rejected["no_embedding"] += 1
                continue

            emb = emb.astype(np.float32)
            norm = float(np.linalg.norm(emb))
            if norm < 0.9:
                rejected["low_norm"] += 1
                continue

            emb = emb / norm
            valid.append({
                "path": str(img_path),
                "embedding": emb,
                "confidence": float(face.det_score) if hasattr(face, "det_score") else 0.0,
                "width": img.shape[1],
                "height": img.shape[0],
            })

        if len(valid) >= REF_PER_ID + HELD_PER_ID:
            valid_images[identity_id] = valid

    logger.info("Valid: %d identities, %d images", len(valid_images), sum(len(v) for v in valid_images.values()))
    logger.info("Rejected: %s", dict(rejected))
    return valid_images


def split_and_write(valid_images: dict[str, list[dict]]) -> dict:
    """Split into ref/held and write directories."""
    for d in [REF_DIR, HELD_DIR]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": "non_celebrity-v1",
        "source": "Face Research Lab London Set (Figshare, CC BY 4.0)",
        "source_doi": "10.6084/m9.figshare.5047666.v5",
        "split": {"reference": REF_PER_ID, "held_out": HELD_PER_ID, "calibration": 0},
        "identities": len(valid_images),
        "reference_images": 0,
        "held_out_images": 0,
        "seed": SEED,
        "reference": {},
        "held_out": {},
    }

    ref_embeddings = []
    ref_metadata = []
    held_embeddings = {}
    vector_id = 0

    for identity_id in sorted(valid_images.keys()):
        images = sorted(valid_images[identity_id], key=lambda x: (-x["confidence"], x["path"]))
        ref_images = images[:REF_PER_ID]
        held_images = images[REF_PER_ID:REF_PER_ID + HELD_PER_ID]

        # Write reference
        ref_dir = REF_DIR / identity_id
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_files = []
        for i, img_data in enumerate(ref_images):
            fname = f"{identity_id}_{i:02d}.jpg"
            shutil.copy2(img_data["path"], ref_dir / fname)
            ref_files.append(fname)
            ref_embeddings.append(img_data["embedding"])
            ref_metadata.append({
                "vector_id": vector_id,
                "person_id": identity_id,
                "image_path": str(ref_dir / fname),
                "confidence": img_data["confidence"],
            })
            vector_id += 1
        manifest["reference"][identity_id] = {"count": len(ref_files), "files": ref_files}
        manifest["reference_images"] += len(ref_files)

        # Write held-out
        held_dir = HELD_DIR / identity_id
        held_dir.mkdir(parents=True, exist_ok=True)
        held_files = []
        held_embs = []
        for i, img_data in enumerate(held_images):
            fname = f"{identity_id}_{i:02d}.jpg"
            shutil.copy2(img_data["path"], held_dir / fname)
            held_files.append(fname)
            held_embs.append((str(held_dir / fname), img_data["embedding"]))
        manifest["held_out"][identity_id] = {"count": len(held_files), "files": held_files}
        manifest["held_out_images"] += len(held_files)
        held_embeddings[identity_id] = held_embs

    # Build FAISS index
    ref_array = np.array(ref_embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(512)
    faiss.normalize_L2(ref_array)
    index.add(ref_array)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "reference_index.faiss"))
    with open(INDEX_DIR / "metadata.json", "w") as f:
        json.dump(ref_metadata, f, indent=2)

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Written: %d ref, %d held, FAISS index: %d vectors",
                manifest["reference_images"], manifest["held_out_images"], index.ntotal)

    return manifest, ref_metadata, ref_array, held_embeddings


def run_integrity_checks() -> dict:
    """Check duplicates and leakage."""
    import hashlib

    ref_hashes = {}
    held_hashes = {}

    for person_dir in REF_DIR.iterdir():
        if person_dir.is_dir():
            for f in person_dir.glob("*.jpg"):
                with open(f, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                ref_hashes[h] = (person_dir.name, f.name)

    for person_dir in HELD_DIR.iterdir():
        if person_dir.is_dir():
            for f in person_dir.glob("*.jpg"):
                with open(f, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                held_hashes[h] = (person_dir.name, f.name)

    ref_set = set(ref_hashes.keys())
    held_set = set(held_hashes.keys())

    return {
        "identities": len(set(v[0] for v in ref_hashes.values())),
        "reference_images": len(ref_hashes),
        "held_out_images": len(held_hashes),
        "ref_held_leakage": len(ref_set & held_set),
        "within_ref_duplicates": len(ref_hashes) - len(ref_set),
        "within_held_duplicates": len(held_hashes) - len(held_set),
    }


def run_evaluation(ref_metadata, ref_embeddings, held_embeddings) -> dict:
    """Full calibration and gallery-size evaluation."""
    from phase13_7_2_calibration import (
        build_all_pairs, aggregate_identity_scores,
        image_level_hard_negatives, identity_level_hard_negatives,
        compute_operating_points, compute_eer, compute_roc_auc,
        verify_threshold_consistency, score_stats,
    )

    # Build ref_records format
    ref_records = []
    for r in ref_metadata:
        ref_records.append({
            "person_id": r["person_id"],
            "vector_id": r["vector_id"],
            "image_path": r["image_path"],
        })

    # Full evaluation
    genuines, impostors = build_all_pairs(held_embeddings, ref_records, ref_embeddings)
    gen_scores = np.array([p["similarity"] for p in genuines], dtype=np.float32) if genuines else np.array([], dtype=np.float32)
    imp_scores = np.array([p["similarity"] for p in impostors], dtype=np.float32) if impostors else np.array([], dtype=np.float32)

    img_ops = compute_operating_points(gen_scores, imp_scores)
    img_eer = compute_eer(gen_scores, imp_scores)
    img_roc = compute_roc_auc(gen_scores, imp_scores)

    id_agg_gen = aggregate_identity_scores(genuines)
    id_agg_imp = aggregate_identity_scores(impostors)
    query_person_map = {p["query_image"]: p["query_person_id"] for p in genuines}

    id_gen_scores = [id_sims[query_person_map.get(q_img)] for q_img, id_sims in id_agg_gen.items()
                     if query_person_map.get(q_img) and query_person_map.get(q_img) in id_sims]
    id_imp_scores = [sim for id_sims in id_agg_imp.values() for sim in id_sims.values()]

    id_genuine = np.array(id_gen_scores, dtype=np.float32) if id_gen_scores else np.array([], dtype=np.float32)
    id_impostor = np.array(id_imp_scores, dtype=np.float32) if id_imp_scores else np.array([], dtype=np.float32)

    id_ops = compute_operating_points(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else []
    id_eer = compute_eer(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}
    id_roc = compute_roc_auc(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}

    img_hn = image_level_hard_negatives(impostors)
    id_hn = identity_level_hard_negatives(held_embeddings, ref_records, ref_embeddings)

    # Fixed threshold evaluation
    fixed = {}
    for name, threshold in CELEBRITY_THRESHOLDS.items():
        tp = int(np.sum(gen_scores >= threshold))
        fn = int(np.sum(gen_scores < threshold))
        fp = int(np.sum(imp_scores >= threshold))
        tn = int(np.sum(imp_scores < threshold))
        tg = len(gen_scores)
        ti = len(imp_scores)
        fixed[name] = {
            "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "far": fp / ti if ti > 0 else 0, "frr": fn / tg if tg > 0 else 0,
            "tpr": tp / tg if tg > 0 else 0, "tnr": tn / ti if ti > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / tg if tg > 0 else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        }

    # Gallery-size analysis
    from collections import defaultdict as dd
    all_pids = sorted(set(r["person_id"] for r in ref_records))
    refs_by_pid = dd(list)
    for i, r in enumerate(ref_records):
        refs_by_pid[r["person_id"]].append((i, r))
    for pid in refs_by_pid:
        refs_by_pid[pid].sort(key=lambda x: x[0])

    gallery_sizes = {}
    for gs in [2, 4, 6]:
        idx, recs = [], []
        for pid in all_pids:
            for j in range(min(gs, len(refs_by_pid[pid]))):
                idx.append(refs_by_pid[pid][j][0])
                recs.append(refs_by_pid[pid][j][1])
        sub_emb = ref_embeddings[idx]
        g, im = build_all_pairs(held_embeddings, recs, sub_emb)
        gs_gen = np.array([p["similarity"] for p in g], dtype=np.float32) if g else np.array([], dtype=np.float32)
        gs_imp = np.array([p["similarity"] for p in im], dtype=np.float32) if im else np.array([], dtype=np.float32)
        gallery_sizes[str(gs)] = {
            "reference_vectors": len(recs),
            "genuine_pairs": len(g), "impostor_pairs": len(im),
            "image_roc_auc": compute_roc_auc(gs_gen, gs_imp)["auc"],
            "image_eer": compute_eer(gs_gen, gs_imp)["eer"],
            "global_max_impostor": float(gs_imp.max()) if len(gs_imp) > 0 else 0.0,
        }

    return {
        "dataset": {
            "identities": len(held_embeddings),
            "reference_vectors": len(ref_records),
            "genuine_pairs": len(genuines),
            "impostor_pairs": len(impostors),
        },
        "image_level": {
            "roc_auc": img_roc["auc"],
            "operating_points": img_ops,
            "eer": img_eer,
            "global_max_impostor": float(imp_scores.max()) if len(imp_scores) > 0 else 0.0,
            "genuine_stats": score_stats(gen_scores),
            "impostor_stats": score_stats(imp_scores),
        },
        "identity_level": {
            "roc_auc": id_roc.get("auc", 0.0),
            "operating_points": id_ops,
            "eer": id_eer,
            "global_max_impostor": max((h["identity_score"] for h in id_hn), default=0.0) if id_hn else 0.0,
            "genuine_stats": score_stats(id_genuine),
            "impostor_stats": score_stats(id_impostor),
        },
        "hard_negatives": {
            "image_level_top5": img_hn[:5],
            "identity_level_top5": id_hn[:5],
        },
        "fixed_thresholds": fixed,
        "gallery_sizes": gallery_sizes,
    }


def main():
    t0 = time.time()
    logger.info("=" * 60)
    logger.info("PHASE 13.9 — NON-CELEBRITY VALIDATION")
    logger.info("=" * 60)

    logger.info("\n--- Discover & Validate ---")
    valid = discover_and_validate()
    if not valid:
        logger.error("No valid images found"); sys.exit(1)

    logger.info("\n--- Split & Write ---")
    manifest, ref_meta, ref_emb, held_emb = split_and_write(valid)

    logger.info("\n--- Integrity ---")
    integrity = run_integrity_checks()
    with open(RESULTS_DIR / "integrity_checks.json", "w") as f:
        json.dump(integrity, f, indent=2)

    logger.info("\n--- Evaluation ---")
    results = run_evaluation(ref_meta, ref_emb, held_emb)
    with open(RESULTS_DIR / "calibration_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(RESULTS_DIR / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE in %.0fs", elapsed)
    logger.info("Identities: %d", len(manifest["reference"]))
    logger.info("Reference: %d, Held-out: %d", manifest["reference_images"], manifest["held_out_images"])
    logger.info("Integrity: %s", integrity)
    logger.info("Image ROC-AUC: %.4f", results["image_level"]["roc_auc"])
    logger.info("Image EER: %.4f", results["image_level"]["eer"]["eer"])
    logger.info("Identity ROC-AUC: %.4f", results["identity_level"]["roc_auc"])
    logger.info("Identity EER: %.4f", results["identity_level"]["eer"].get("eer", 0))
    logger.info("Global max impostor: %.4f", results["image_level"]["global_max_impostor"])
    for name, data in results["fixed_thresholds"].items():
        logger.info("  %s (t=%.4f): FAR=%.4f, FRR=%.4f, F1=%.4f", name, data["threshold"], data["far"], data["frr"], data["f1"])


if __name__ == "__main__":
    main()
