"""Phase 13.8 — Expanded Diverse Validation Dataset & Calibration

Collects 20 new identities + expands 22 existing identities to 16 images each.
Builds Dataset v4 with 3-way split (reference/calibration/held-out evaluation).
Runs calibration, gallery-size analysis, weak-identity analysis, and comparison.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
MANIFEST_PATH = Path("dataset_acquisition/people_v4.json")
V3_DATASET = Path("datasets/celebrity-v3")
V4_DATASET = Path("datasets/celebrity-v4")
OUTPUT_BASE = Path("outputs/phase13_8")
V3_SUMMARY = Path("outputs/phase13_7_2/verification.json")

CONTROLLED_PER_PERSON = 16
REF_PER_PERSON = 8
CAL_PER_PERSON = 4
HELD_OUT_PER_PERSON = 4
TARGET_ACQUISITION = 20
MIN_ACQUISITION = 16
GALLERY_SIZES = [2, 4, 6, 8]

WEAK_IDENTITIES = [
    "jennifer_lawrence", "morgan_freeman", "leonardo_dicaprio",
    "vinicius_junior", "brad_pitt", "neymar", "mohamed_salah",
    "kevin_de_bruyne",
]


def load_people() -> tuple[list, dict]:
    from dataset_acquisition.models import Person
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    persons = []
    meta = {}
    for p in data["people"]:
        persons.append(Person(
            person_id=p["person_id"],
            display_name=p["display_name"],
            category=p["category"],
            aliases=tuple(p.get("aliases", [])),
            search_queries=tuple(p.get("search_queries", [])),
        ))
        meta[p["person_id"]] = p.get("dataset_status", "new")
    return persons, meta


def collect_new_identities(persons: list) -> dict[str, Any]:
    from dataset_acquisition.orchestrator import AcquisitionOrchestrator
    new_persons = [p for p in persons if p.person_id not in EXISTING_IDS]
    if not new_persons:
        return {"total_persons": 0, "total_accepted": 0}
    logger.info("Collecting %d new identities...", len(new_persons))
    face_service = _load_face_service()
    orchestrator = AcquisitionOrchestrator(
        output_dir=OUTPUT_BASE / "new_raw",
        max_images_per_person=TARGET_ACQUISITION,
        max_candidates_per_source=50,
        wikimedia_delay=2.0,
        openverse_delay=1.0,
        face_service=face_service,
    )
    t0 = time.time()
    result = orchestrator.collect_all(new_persons, TARGET_ACQUISITION)
    result["runtime_seconds"] = round(time.time() - t0, 2)
    return result


def collect_existing_expansion(persons: list) -> dict[str, Any]:
    existing_persons = [p for p in persons if p.person_id in EXISTING_IDS]
    if not existing_persons:
        return {"total_persons": 0, "total_accepted": 0}
    logger.info("Expanding %d existing identities...", len(existing_persons))
    face_service = _load_face_service()
    orchestrator = AcquisitionOrchestrator(
        output_dir=OUTPUT_BASE / "existing_raw",
        max_images_per_person=TARGET_ACQUISITION,
        max_candidates_per_source=50,
        wikimedia_delay=2.0,
        openverse_delay=1.0,
        face_service=face_service,
    )
    t0 = time.time()
    result = orchestrator.collect_all(existing_persons, TARGET_ACQUISITION)
    result["runtime_seconds"] = round(time.time() - t0, 2)
    return result


def merge_and_split() -> dict[str, Any]:
    logger.info("Merging v3 + new acquisitions into v4 dataset...")
    V4_DATASET.mkdir(parents=True, exist_ok=True)

    state_new = _load_state(OUTPUT_BASE / "new_raw" / "download_state.json")
    state_existing = _load_state(OUTPUT_BASE / "existing_raw" / "download_state.json")

    all_records: dict[str, list] = defaultdict(list)

    for pid, records in state_new.items():
        all_records[pid].extend(records)
    for pid, records in state_existing.items():
        all_records[pid].extend(records)

    logger.info("Merged records: %d identities, %d total images",
                len(all_records), sum(len(r) for r in all_records.values()))

    rng = random.Random(SEED)
    split_manifest = {"reference": {}, "calibration": {}, "held_out": {}}
    all_copied = []

    for pid in sorted(all_records.keys()):
        records = all_records[pid]
        n = len(records)
        if n < MIN_ACQUISITION:
            logger.warning("  %s: only %d images (need %d), selecting best %d",
                           pid, n, MIN_ACQUISITION, min(n, CONTROLLED_PER_PERSON))

        selected = _diversity_aware_select(records, CONTROLLED_PER_PERSON, rng)
        rng.shuffle(selected)

        ref = selected[:REF_PER_PERSON]
        cal = selected[REF_PER_PERSON:REF_PER_PERSON + CAL_PER_PERSON]
        held = selected[REF_PER_PERSON + CAL_PER_PERSON:REF_PER_PERSON + CAL_PER_PERSON + HELD_OUT_PER_PERSON]

        for split_name, split_records in [("reference", ref), ("calibration", cal), ("held_out", held)]:
            dest = V4_DATASET / split_name / pid
            dest.mkdir(parents=True, exist_ok=True)
            for r in split_records:
                src = Path(r.local_path)
                if src.exists():
                    shutil.copy2(str(src), str(dest / src.name))
                    all_copied.append({"person_id": pid, "split": split_name, "image": src.name})
            split_manifest[split_name][pid] = [r.image_id for r in split_records]

    counts = {s: {pid: len(recs) for pid, recs in split_manifest[s].items()}
              for s in ["reference", "calibration", "held_out"]}
    total_ref = sum(counts["reference"].values())
    total_cal = sum(counts["calibration"].values())
    total_held = sum(counts["held_out"].values())

    logger.info("v4 Split: %d ref, %d cal, %d held (%d identities)",
                total_ref, total_cal, total_held, len(split_manifest["reference"]))

    manifest_path = OUTPUT_BASE / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"counts": counts, "totals": {"reference": total_ref, "calibration": total_cal, "held_out": total_held}}, f, indent=2)

    return {"counts": counts, "totals": {"reference": total_ref, "calibration": total_cal, "held_out": total_held}}


def _diversity_aware_select(records: list, n: int, rng: random.Random) -> list:
    if len(records) <= n:
        return list(records)

    scored = []
    for r in records:
        score = 0.0
        if r.width > 0 and r.height > 0:
            area = r.width * r.height
            if area > 500 * 500:
                score += 1.0
        if r.face_confidence > 0.8:
            score += 0.5
        scored.append((score, rng.random(), r))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    selected = [s[2] for s in scored[:n]]

    sources = set()
    for r in selected:
        sources.add(r.source)
    for score, rand, r in scored[n:]:
        if r.source not in sources and len(selected) < n:
            selected.append(r)
            sources.add(r.source)

    return selected[:n]


def build_faiss_index() -> dict[str, Any]:
    from search.index_builder import IndexBuilder
    ref_dir = V4_DATASET / "reference"
    index_dir = V4_DATASET / "search_index"
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


def run_calibration(
    index_path: str, metadata_path: str, persons: list
) -> dict[str, Any]:
    from phase13_7_2_calibration import (
        load_reference_embeddings_with_faiss,
        extract_query_embeddings,
        build_genuine_pairs,
        build_impostor_pairs,
        image_level_hard_negatives,
        identity_level_hard_negatives,
        compute_operating_points,
        verify_threshold_consistency,
        EXPECTED_EMBEDDING_DIM,
    )

    index, metadata = load_reference_embeddings_with_faiss(Path(index_path), Path(metadata_path))
    model = _load_face_service().get_model()

    query_dir = V4_DATASET / "calibration"
    query_embeddings = extract_query_embeddings(query_dir, persons, model)

    genuine_pairs = build_genuine_pairs(query_embeddings, metadata)
    impostor_pairs = build_impostor_pairs(query_embeddings, metadata)

    from phase13_7_2_calibration import SimilarityEvaluator
    evaluator = SimilarityEvaluator()

    genuine_scores = np.array([evaluator.evaluate_pair(p).similarity for p in genuine_pairs])
    impostor_scores = np.array([evaluator.evaluate_pair(p).similarity for p in impostor_pairs])

    img_ops = compute_operating_points(genuine_scores, impostor_scores, "image_level")

    id_genuine, id_impostor = _aggregate_identity_scores(genuine_pairs, impostor_pairs)
    id_ops = compute_operating_points(id_genuine, id_impostor, "identity_level")

    img_hn = image_level_hard_negatives(impostor_pairs)
    id_hn = identity_level_hard_negatives(query_embeddings, metadata["records"], index.reconstruct_n(0, index.ntotal))

    consistent = verify_threshold_consistency(img_ops + id_ops)

    from phase13_7_2_calibration import is_accepted
    youden_img = next((op for op in img_ops if op["label"] == "youden_j"), None)
    youden_id = next((op for op in id_ops if op["label"] == "youden_j"), None)

    global_max_img = max((p["similarity"] for p in impostor_pairs), default=0.0)
    global_max_id = max((h["identity_score"] for h in id_hn), default=0.0) if id_hn else 0.0

    return {
        "dataset": {"identities": len(query_embeddings), "reference_vectors": index.ntotal,
                     "genuine_pairs": len(genuine_pairs), "impostor_pairs": len(impostor_pairs)},
        "image_level": {
            "roc_auc": float(np.trapz(
                [op["tpr"] for op in sorted(img_ops, key=lambda x: x["fpr"])],
                [op["fpr"] for op in sorted(img_ops, key=lambda x: x["fpr"])]
            )) if len(img_ops) > 1 else 0.0,
            "operating_points": img_ops,
            "global_max_impostor": global_max_img,
            "genuine_stats": _score_stats(genuine_scores),
            "impostor_stats": _score_stats(impostor_scores),
        },
        "identity_level": {
            "operating_points": id_ops,
            "global_max_impostor": global_max_id,
            "genuine_stats": _score_stats(id_genuine),
            "impostor_stats": _score_stats(id_impostor),
        },
        "hard_negatives": {"image_level_top5": img_hn[:5], "identity_level_top5": id_hn[:5]},
        "consistency": consistent,
    }


def _aggregate_identity_scores(
    genuine_pairs: list, impostor_pairs: list
) -> tuple[np.ndarray, np.ndarray]:
    gen_by_q = defaultdict(list)
    imp_by_q = defaultdict(list)
    for p in genuine_pairs:
        gen_by_q[p["query_image"]].append(p["similarity"])
    for p in impostor_pairs:
        imp_by_q[p["query_image"]].append(p["similarity"])

    id_genuine = np.array([max(sims) for sims in gen_by_q.values()]) if gen_by_q else np.array([])
    id_impostor = np.array([max(sims) for sims in imp_by_q.values()]) if imp_by_q else np.array([])
    return id_genuine, id_impostor


def run_gallery_size_analysis(
    index_path: str, metadata_path: str, persons: list
) -> dict[str, Any]:
    from phase13_7_2_calibration import (
        load_reference_embeddings_with_faiss,
        extract_query_embeddings,
        build_genuine_pairs,
        build_impostor_pairs,
        compute_operating_points,
    )

    index, metadata = load_reference_embeddings_with_faiss(Path(index_path), Path(metadata_path))
    model = _load_face_service().get_model()
    query_dir = V4_DATASET / "calibration"
    query_embeddings = extract_query_embeddings(query_dir, persons, model)

    ref_records = metadata["records"]
    ref_embeddings = index.reconstruct_n(0, index.ntotal)

    results = {}
    rng = random.Random(SEED)

    for person_id in sorted(set(r["person_id"] for r in ref_records)):
        person_refs = [(i, r) for i, r in enumerate(ref_records) if r["person_id"] == person_id]
        person_refs.sort(key=lambda x: x[0])
        person_results = {}
        for gs in GALLERY_SIZES:
            subset_indices = [person_refs[j][0] for j in range(min(gs, len(person_refs)))]
            subset_embeddings = ref_embeddings[subset_indices]
            subset_records = [ref_records[i] for i in subset_indices]

            subset_metadata = {"records": subset_records, "dimension": metadata["dimension"]}
            from search.flat_index import FlatIndex
            subset_index = FlatIndex(dimension=metadata["dimension"])
            subset_index.add(subset_embeddings)

            genuines = build_genuine_pairs(query_embeddings, {"records": subset_records}, person_filter=person_id)
            impostors = build_impostor_pairs(query_embeddings, {"records": subset_records})

            from phase13_7_2_calibration import SimilarityEvaluator
            evaluator = SimilarityEvaluator()
            gen_scores = np.array([evaluator.evaluate_pair(p).similarity for p in genuines]) if genuines else np.array([])
            imp_scores = np.array([evaluator.evaluate_pair(p).similarity for p in impostors]) if impostors else np.array([])

            ops = compute_operating_points(gen_scores, imp_scores, f"gallery_{gs}")
            person_results[gs] = {
                "genuine_count": len(genuines),
                "impostor_count": len(impostors),
                "genuine_mean": float(np.mean(gen_scores)) if len(gen_scores) > 0 else 0.0,
                "impostor_max": float(np.max(imp_scores)) if len(imp_scores) > 0 else 0.0,
                "operating_points": ops,
            }
        results[person_id] = person_results

    overall = {}
    for gs in GALLERY_SIZES:
        all_gen = []
        all_imp = []
        for pid in results:
            all_gen.extend([results[pid][gs]["genuine_count"]])
            all_imp.append(results[pid][gs]["impostor_max"])
        overall[gs] = {
            "mean_genuine_per_identity": np.mean(all_gen) if all_gen else 0,
            "mean_impostor_max": float(np.mean(all_imp)) if all_imp else 0,
        }

    return {"per_identity": results, "overall": overall, "gallery_sizes": GALLERY_SIZES}


def run_weak_identity_analysis(
    index_path: str, metadata_path: str, persons: list
) -> dict[str, Any]:
    from phase13_7_2_calibration import (
        load_reference_embeddings_with_faiss,
        extract_query_embeddings,
        build_genuine_pairs,
        build_impostor_pairs,
    )

    index, metadata = load_reference_embeddings_with_faiss(Path(index_path), Path(metadata_path))
    model = _load_face_service().get_model()
    query_dir = V4_DATASET / "calibration"
    query_embeddings = extract_query_embeddings(query_dir, persons, model)

    results = {}
    for pid in WEAK_IDENTITIES:
        if pid not in query_embeddings:
            continue

        genuines = build_genuine_pairs(query_embeddings, metadata, person_filter=pid)
        impostors = build_impostor_pairs(query_embeddings, metadata, person_filter=pid)

        from phase13_7_2_calibration import SimilarityEvaluator
        evaluator = SimilarityEvaluator()
        gen_scores = np.array([evaluator.evaluate_pair(p).similarity for p in genuines]) if genuines else np.array([])
        imp_scores = np.array([evaluator.evaluate_pair(p).similarity for p in impostors]) if impostors else np.array([])

        person_refs = [r for r in metadata["records"] if r["person_id"] == pid]

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


def run_comparison(v3_path: str, v4_cal: dict) -> dict[str, Any]:
    if not Path(v3_path).exists():
        return {"error": "v3 verification.json not found"}
    with open(v3_path) as f:
        v3 = json.load(f)

    return {
        "v3": {
            "identities": v3["dataset"]["identities"],
            "reference_vectors": v3["dataset"]["reference_vectors"],
            "genuine_pairs": v3["dataset"]["genuine_pairs"],
            "impostor_pairs": v3["dataset"]["impostor_pairs"],
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
            "image_eer": next((op["eer"] for op in v4_cal["image_level"]["operating_points"] if op["label"] == "eer"), None),
            "identity_eer": next((op["eer"] for op in v4_cal["identity_level"]["operating_points"] if op["label"] == "eer"), None),
            "global_max_impostor": v4_cal["image_level"]["global_max_impostor"],
        },
    }


def _score_stats(scores: np.ndarray) -> dict[str, float]:
    if len(scores) == 0:
        return {}
    return {
        "count": int(len(scores)),
        "min": round(float(np.min(scores)), 6),
        "max": round(float(np.max(scores)), 6),
        "mean": round(float(np.mean(scores)), 6),
        "std": round(float(np.std(scores)), 6),
    }


def _load_face_service():
    import sys
    sys.path.insert(0, ".")
    from services.face_service import FaceService
    return FaceService()


def _load_state(path: Path) -> dict[str, list]:
    from dataset_acquisition.models import ImageRecord
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)
    records = [ImageRecord.from_dict(r) for r in state.get("records", [])]
    by_person = defaultdict(list)
    for r in records:
        by_person[r.person_id].append(r)
    return dict(by_person)


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    persons, meta = load_people()
    EXISTING_IDS_SET = {p.person_id for p in persons if meta.get(p.person_id) == "existing"}
    global EXISTING_IDS
    EXISTING_IDS = EXISTING_IDS_SET

    logger.info("Loaded %d persons (%d existing, %d new)",
                len(persons), len(EXISTING_IDS), len(persons) - len(EXISTING_IDS))

    new_result = collect_new_identities(persons)
    existing_result = collect_existing_expansion(persons)

    split_result = merge_and_split()
    faiss_result = build_faiss_index()

    cal_result = run_calibration(
        faiss_result["index_path"], faiss_result["metadata_path"], persons
    )

    gallery_result = run_gallery_size_analysis(
        faiss_result["index_path"], faiss_result["metadata_path"], persons
    )

    weak_result = run_weak_identity_analysis(
        faiss_result["index_path"], faiss_result["metadata_path"], persons
    )

    comparison = run_comparison(str(V3_SUMMARY), cal_result)

    results = {
        "phase": "13.8",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "collection_new": new_result,
        "collection_existing": existing_result,
        "split": split_result,
        "faiss": faiss_result,
        "calibration": cal_result,
        "gallery_size_analysis": gallery_result,
        "weak_identity_analysis": weak_result,
        "comparison": comparison,
    }

    with open(OUTPUT_BASE / "phase13_8_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("PHASE 13.8 COMPLETE")
    logger.info("=" * 60)
    logger.info("Results: %s", OUTPUT_BASE / "phase13_8_results.json")


EXISTING_IDS: set[str] = set()

if __name__ == "__main__":
    main()
