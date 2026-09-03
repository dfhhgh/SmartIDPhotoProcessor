"""Phase 13.8 — Analysis Pipeline (optimized, saves intermediate results)

Merges v3 (22 existing identities) + v4 new (14 complete identities).
Runs: 3-way split, FAISS index, calibration, gallery-size, weak-identity, comparison.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
V3_DATASET = Path("datasets/celebrity-v3")
V4_DATASET = Path("datasets/celebrity-v4")
OUTPUT_BASE = Path("outputs/phase13_8")
V3_SUMMARY = Path("outputs/phase13_7_2/verification.json")

REF_PER_PERSON = 8
CAL_PER_PERSON = 4
HELD_OUT_PER_PERSON = 4
CONTROLLED_PER_PERSON = REF_PER_PERSON + CAL_PER_PERSON + HELD_OUT_PER_PERSON
GALLERY_SIZES = [2, 4, 6, 8]

WEAK_IDENTITIES = [
    "jennifer_lawrence", "morgan_freeman", "leonardo_dicaprio",
    "vinicius_junior", "brad_pitt", "neymar", "mohamed_salah",
    "kevin_de_bruyne",
]


def save_intermediate(name: str, data: dict):
    path = OUTPUT_BASE / f"intermediate_{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved intermediate: %s", path)


def load_intermediate(name: str) -> dict | None:
    path = OUTPUT_BASE / f"intermediate_{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def merge_and_split() -> dict:
    logger.info("Merging v3 + new acquisitions into v4 dataset...")
    V4_DATASET.mkdir(parents=True, exist_ok=True)

    all_records: dict[str, list] = defaultdict(list)

    v3_splits = {"reference": "reference", "query": "calibration"}
    for v3_dir_name, target_split in v3_splits.items():
        split_dir = V3_DATASET / v3_dir_name
        if not split_dir.exists():
            continue
        for person_dir in sorted(split_dir.iterdir()):
            if not person_dir.is_dir():
                continue
            pid = person_dir.name
            for img_path in sorted(person_dir.glob("*.jpg")):
                all_records[pid].append({
                    "person_id": pid,
                    "image_id": img_path.stem,
                    "local_path": str(img_path),
                    "source": "v3_" + v3_dir_name,
                    "v3_split": target_split,
                })

    logger.info("v3 loaded: %d identities, %d total images",
                len(all_records), sum(len(r) for r in all_records.values()))

    state_path = OUTPUT_BASE / "new_raw" / "download_state.json"
    if state_path.exists():
        from dataset_acquisition.models import ImageRecord
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        records = [ImageRecord.from_dict(r) for r in state.get("records", [])]
        for r in records:
            all_records[r.person_id].append({
                "person_id": r.person_id,
                "image_id": r.image_id,
                "local_path": r.local_path,
                "source": r.source,
                "face_confidence": r.face_confidence,
                "width": r.width,
                "height": r.height,
            })

    logger.info("After merge: %d identities, %d total images",
                len(all_records), sum(len(r) for r in all_records.values()))

    rng = random.Random(SEED)
    split_manifest = {"reference": {}, "calibration": {}, "held_out": {}}
    counts = {"reference": {}, "calibration": {}, "held_out": {}}

    for pid in sorted(all_records.keys()):
        records = all_records[pid]
        n = len(records)
        has_v3_split = any(r.get("v3_split") for r in records)

        if has_v3_split:
            ref = [r for r in records if r.get("v3_split") == "reference"]
            cal = [r for r in records if r.get("v3_split") == "calibration"]
            held = []
        elif n < CONTROLLED_PER_PERSON:
            selected = records
            rng.shuffle(selected)
            ref = selected[:REF_PER_PERSON]
            cal = selected[REF_PER_PERSON:REF_PER_PERSON + CAL_PER_PERSON]
            held = selected[REF_PER_PERSON + CAL_PER_PERSON:]
        else:
            selected = _diversity_aware_select(records, CONTROLLED_PER_PERSON, rng)
            rng.shuffle(selected)
            ref = selected[:REF_PER_PERSON]
            cal = selected[REF_PER_PERSON:REF_PER_PERSON + CAL_PER_PERSON]
            held = selected[REF_PER_PERSON + CAL_PER_PERSON:]

        for split_name, split_records in [("reference", ref), ("calibration", cal), ("held_out", held)]:
            dest = V4_DATASET / split_name / pid
            dest.mkdir(parents=True, exist_ok=True)
            for r in split_records:
                src = Path(r["local_path"])
                if src.exists():
                    dst = dest / src.name
                    if not dst.exists():
                        shutil.copy2(str(src), str(dst))
            counts[split_name][pid] = len(split_records)
            split_manifest[split_name][pid] = [r["image_id"] for r in split_records]

    total_ref = sum(counts["reference"].values())
    total_cal = sum(counts["calibration"].values())
    total_held = sum(counts["held_out"].values())

    logger.info("v4 Split: %d ref, %d cal, %d held (%d identities)",
                total_ref, total_cal, total_held, len(counts["reference"]))

    return {"counts": counts, "totals": {
        "reference": total_ref, "calibration": total_cal, "held_out": total_held
    }}


def _diversity_aware_select(records: list, n: int, rng: random.Random) -> list:
    if len(records) <= n:
        return list(records)
    scored = []
    for r in records:
        score = 0.0
        w = r.get("width", 0) or 0
        h = r.get("height", 0) or 0
        if w > 0 and h > 0 and w * h > 500 * 500:
            score += 1.0
        fc = r.get("face_confidence", 0) or 0
        if fc > 0.8:
            score += 0.5
        scored.append((score, rng.random(), r))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [s[2] for s in scored[:n]]


def build_faiss_index() -> dict:
    index_dir = V4_DATASET / "search_index"
    index_path = index_dir / "reference_index.faiss"
    metadata_path = index_dir / "metadata.json"

    # Check if cached
    if index_path.exists() and metadata_path.exists():
        import faiss
        index = faiss.read_index(str(index_path))
        logger.info("FAISS index already exists: %d vectors, skipping build", index.ntotal)
        return {
            "accepted_embeddings": index.ntotal,
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
        }

    from search.index_builder import IndexBuilder
    ref_dir = V4_DATASET / "reference"
    index_dir.mkdir(parents=True, exist_ok=True)

    builder = IndexBuilder(dimension=512)
    index, report = builder.build(
        dataset_dir=str(ref_dir),
        output_dir=str(index_dir),
        index_filename="reference_index.faiss",
        metadata_filename="metadata.json",
    )
    logger.info("FAISS built: %d vectors", report.accepted_embeddings)
    return {
        "accepted_embeddings": report.accepted_embeddings,
        "index_path": str(index_dir / "reference_index.faiss"),
        "metadata_path": str(index_dir / "metadata.json"),
    }


def load_persons() -> list:
    from dataset_acquisition.models import Person
    with open("dataset_acquisition/people_v4.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Person(
        person_id=p["person_id"],
        display_name=p["display_name"],
        category=p["category"],
        aliases=tuple(p.get("aliases", [])),
        search_queries=tuple(p.get("search_queries", [])),
    ) for p in data["people"]]


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


def run_calibration(
    ref_records, ref_embeddings, query_embeddings
) -> dict:
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

    img_ops = compute_operating_points(genuine_scores, impostor_scores)
    img_eer = compute_eer(genuine_scores, impostor_scores)
    img_roc = compute_roc_auc(genuine_scores, impostor_scores)

    # Identity-level aggregation
    # aggregate_identity_scores returns {query_image: {person_id: max_sim}}
    id_agg_gen = aggregate_identity_scores(genuine_pairs)
    id_agg_imp = aggregate_identity_scores(impostor_pairs)

    # Build identity-level genuine scores: for each query, get max sim for its own identity
    # We need to know which person_id each query belongs to
    query_person_map = {}
    for p in genuine_pairs:
        query_person_map[p["query_image"]] = p["query_person_id"]

    id_genuine_scores = []
    for query_img, identity_sims in id_agg_gen.items():
        pid = query_person_map.get(query_img)
        if pid and pid in identity_sims:
            id_genuine_scores.append(identity_sims[pid])

    id_impostor_scores = []
    for query_img, identity_sims in id_agg_imp.items():
        for pid, sim in identity_sims.items():
            id_impostor_scores.append(sim)

    id_genuine = np.array(id_genuine_scores, dtype=np.float32) if id_genuine_scores else np.array([], dtype=np.float32)
    id_impostor = np.array(id_impostor_scores, dtype=np.float32) if id_impostor_scores else np.array([], dtype=np.float32)

    id_ops = compute_operating_points(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else []
    id_eer = compute_eer(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}
    id_roc = compute_roc_auc(id_genuine, id_impostor) if len(id_genuine) > 0 and len(id_impostor) > 0 else {}

    img_hn = image_level_hard_negatives(impostor_pairs)
    id_hn = identity_level_hard_negatives(query_embeddings, ref_records, ref_embeddings)

    img_consistency = verify_threshold_consistency(
        genuine_scores, impostor_scores, img_ops, "image_level"
    )
    id_consistency = verify_threshold_consistency(
        id_genuine, id_impostor, id_ops, "identity_level"
    ) if len(id_genuine) > 0 and len(id_impostor) > 0 else []

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
        "hard_negatives": {"image_level_top5": img_hn[:5], "identity_level_top5": id_hn[:5]},
        "consistency": {"image_level": img_consistency, "identity_level": id_consistency},
    }


def run_gallery_size_analysis(
    ref_records, ref_embeddings, query_embeddings
) -> dict:
    from phase13_7_2_calibration import (
        build_all_pairs,
    )

    results = {}
    all_person_ids = sorted(set(r["person_id"] for r in ref_records))

    for person_id in all_person_ids:
        person_refs = [(i, r) for i, r in enumerate(ref_records) if r["person_id"] == person_id]
        person_refs.sort(key=lambda x: x[0])
        person_results = {}
        for gs in GALLERY_SIZES:
            subset_indices = [person_refs[j][0] for j in range(min(gs, len(person_refs)))]
            subset_embeddings = ref_embeddings[subset_indices]
            subset_records = [ref_records[i] for i in subset_indices]

            genuines, impostors = build_all_pairs(query_embeddings, subset_records, subset_embeddings)

            gen_scores = np.array([p["similarity"] for p in genuines], dtype=np.float32) if genuines else np.array([], dtype=np.float32)
            imp_scores = np.array([p["similarity"] for p in impostors], dtype=np.float32) if impostors else np.array([], dtype=np.float32)

            person_results[gs] = {
                "genuine_count": len(genuines),
                "impostor_count": len(impostors),
                "genuine_mean": float(np.mean(gen_scores)) if len(gen_scores) > 0 else 0.0,
                "impostor_max": float(np.max(imp_scores)) if len(imp_scores) > 0 else 0.0,
            }
        results[person_id] = person_results

    overall = {}
    for gs in GALLERY_SIZES:
        all_imp_max = [results[pid][gs]["impostor_max"] for pid in results]
        overall[gs] = {"mean_impostor_max": float(np.mean(all_imp_max)) if all_imp_max else 0}

    return {"per_identity": results, "overall": overall, "gallery_sizes": GALLERY_SIZES}


def run_weak_identity_analysis(
    ref_records, ref_embeddings, query_embeddings
) -> dict:
    from phase13_7_2_calibration import (
        build_all_pairs,
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
            "query_count": len(query_embeddings.get(pid, [])),
            "reference_count": len(person_refs),
            "genuine_count": len(genuines),
            "impostor_count": len(impostors),
            "genuine_mean": float(np.mean(gen_scores)) if len(gen_scores) > 0 else 0.0,
            "genuine_min": float(np.min(gen_scores)) if len(gen_scores) > 0 else 0.0,
            "impostor_max": float(np.max(imp_scores)) if len(imp_scores) > 0 else 0.0,
            "impostor_mean": float(np.mean(imp_scores)) if len(imp_scores) > 0 else 0.0,
        }

    return results


def run_comparison(v3_path: str, v4_cal: dict) -> dict:
    if not Path(v3_path).exists():
        return {"error": "v3 verification.json not found"}
    with open(v3_path) as f:
        v3 = json.load(f)

    return {
        "v3": {
            "identities": v3["dataset"]["total_identities"],
            "reference_vectors": v3["dataset"]["total_reference"],
            "genuine_pairs": v3["dataset"]["actual_genuine_pairs"],
            "impostor_pairs": v3["dataset"]["actual_impostor_pairs"],
            "image_roc_auc": v3["image_level"]["roc_auc"],
            "identity_roc_auc": v3["identity_level"]["roc_auc"],
            "image_eer": v3["image_level"]["eer"]["eer"],
            "identity_eer": v3["identity_level"]["eer"]["eer"],
            "global_max_impostor": v3["image_level"]["global_max_impostor"],
        },
        "v4": {
            "identities": v4_cal["dataset"]["identities"],
            "reference_vectors": v4_cal["dataset"]["reference_vectors"],
            "genuine_pairs": v4_cal["dataset"]["genuine_pairs"],
            "impostor_pairs": v4_cal["dataset"]["impostor_pairs"],
            "image_roc_auc": v4_cal["image_level"]["roc_auc"],
            "identity_roc_auc": v4_cal["identity_level"]["roc_auc"],
            "image_eer": v4_cal["image_level"]["eer"]["eer"],
            "identity_eer": v4_cal["identity_level"]["eer"].get("eer", 0.0),
            "global_max_impostor": v4_cal["image_level"]["global_max_impostor"],
        },
    }


def _load_face_service():
    import sys
    sys.path.insert(0, ".")
    from services.face_service import FaceService
    return FaceService()


def main():
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("PHASE 13.8 ANALYSIS PIPELINE")
    logger.info("=" * 60)

    # Step 1: Merge + Split
    logger.info("\n--- Step 1: Merge + 3-way Split ---")
    split_result = merge_and_split()
    save_intermediate("split", split_result)

    # Step 2: Build FAISS index
    logger.info("\n--- Step 2: Build FAISS Index ---")
    faiss_result = build_faiss_index()
    save_intermediate("faiss", faiss_result)

    # Step 3: Load data (once!)
    from phase13_7_2_calibration import load_reference_embeddings_with_faiss
    logger.info("\n--- Loading reference data ---")
    ref_records, ref_embeddings = load_reference_embeddings_with_faiss(
        Path(faiss_result["index_path"]), Path(faiss_result["metadata_path"])
    )

    model = _load_face_service().get_model()
    query_dir = V4_DATASET / "calibration"

    # Check for cached query embeddings
    cached_qe = load_intermediate("query_embeddings")
    if cached_qe:
        logger.info("Loading cached query embeddings...")
        import numpy as np
        query_embeddings = {
            pid: [(path, np.array(emb, dtype=np.float32)) for path, emb in items]
            for pid, items in cached_qe.items()
        }
        logger.info("Loaded cached query embeddings for %d identities (%d total)",
                    len(query_embeddings), sum(len(v) for v in query_embeddings.values()))
    else:
        logger.info("Extracting query embeddings (once)...")
        query_embeddings = extract_query_embeddings(query_dir, model)
        logger.info("Extracted query embeddings for %d identities (%d total)",
                    len(query_embeddings), sum(len(v) for v in query_embeddings.values()))
        save_intermediate("query_embeddings", {
            pid: [(path, emb.tolist()) for path, emb in items]
            for pid, items in query_embeddings.items()
        })

    # Step 4: Calibration
    logger.info("\n--- Step 3: Calibration ---")
    cal_result = run_calibration(ref_records, ref_embeddings, query_embeddings)
    save_intermediate("calibration", cal_result)

    # Step 5: Gallery-size analysis
    logger.info("\n--- Step 4: Gallery-Size Analysis ---")
    gallery_result = run_gallery_size_analysis(ref_records, ref_embeddings, query_embeddings)
    save_intermediate("gallery_size", gallery_result)

    # Step 6: Weak-identity analysis
    logger.info("\n--- Step 5: Weak-Identity Analysis ---")
    weak_result = run_weak_identity_analysis(ref_records, ref_embeddings, query_embeddings)
    save_intermediate("weak_identity", weak_result)

    # Step 7: Comparison
    logger.info("\n--- Step 6: Comparison vs v3 ---")
    comparison = run_comparison(str(V3_SUMMARY), cal_result)

    # Save all results
    results = {
        "phase": "13.8",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "split": split_result,
        "faiss": faiss_result,
        "calibration": cal_result,
        "gallery_size_analysis": gallery_result,
        "weak_identity_analysis": weak_result,
        "comparison": comparison,
    }

    out_path = OUTPUT_BASE / "phase13_8_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PHASE 13.8 ANALYSIS COMPLETE")
    logger.info("Results: %s", out_path)
    logger.info("=" * 60)

    logger.info("\n--- SUMMARY ---")
    logger.info("Split: %s", json.dumps(split_result["totals"]))
    logger.info("FAISS: %d vectors", faiss_result["accepted_embeddings"])
    logger.info("Calibration identities: %d", cal_result["dataset"]["identities"])
    logger.info("Image ROC-AUC: %.4f", cal_result["image_level"]["roc_auc"])
    logger.info("Identity ROC-AUC: %.4f", cal_result["identity_level"]["roc_auc"])
    logger.info("Image EER: %.4f", cal_result["image_level"]["eer"]["eer"])
    logger.info("Identity EER: %.4f", cal_result["identity_level"]["eer"].get("eer", 0.0))
    logger.info("Image global max impostor: %.4f", cal_result["image_level"]["global_max_impostor"])


if __name__ == "__main__":
    main()
