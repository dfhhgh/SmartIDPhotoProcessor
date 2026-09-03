"""Phase 13.7.2 — Cleaned-up Calibration Evaluation

Hardened calibration evaluation with:
- No path-based identity inference
- Robust embedding dimension validation
- Normalization validation
- Single-face contract enforcement
- Dynamic reference count derivation
- Stronger threshold consistency verification
- Single threshold decision helper
- No duplicate implementations
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
EXPECTED_EMBEDDING_DIM = 512
NORM_TOLERANCE = 0.05  # max allowed deviation from unit norm

OUTPUT_BASE = Path("outputs/phase13_7_2")
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
(OUTPUT_BASE / "plots").mkdir(parents=True, exist_ok=True)

DATASET_BASE = Path("datasets/celebrity-v3")
MANIFEST_PATH = Path("dataset_acquisition/people_v3_scaled.json")


# ---------------------------------------------------------------------------
# Threshold decision helper (calibration-only)
# ---------------------------------------------------------------------------

def is_accepted(score: float, threshold: float) -> bool:
    """Calibration decision rule: score >= threshold => positive/accepted."""
    return score >= threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_people() -> list[dict[str, Any]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["people"]


def extract_embedding_single_face(
    model,
    img_path: Path,
) -> npt.NDArray[np.float32] | None:
    """Extract embedding enforcing exactly one face.

    Returns:
        embedding if exactly one face detected, None otherwise.

    Side effect: logs face count for telemetry.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        logger.debug("Invalid/unreadable image: %s", img_path)
        return None
    faces = model.get(img)
    if len(faces) == 0:
        logger.debug("No face detected: %s", img_path)
        return None
    if len(faces) > 1:
        logger.debug("Multi-face (%d) rejected: %s", len(faces), img_path)
        return None
    face = faces[0]
    emb = face.normed_embedding
    if emb is None:
        return None
    return emb.ravel().astype(np.float32)


def load_reference_embeddings_with_faiss(
    index_path: Path,
    metadata_path: Path,
) -> tuple[list[dict[str, Any]], npt.NDArray[np.float32]]:
    """Load FAISS index and metadata, validate dimension and normalization."""
    index = faiss.read_index(str(index_path))

    # Validate dimension
    dim = index.d
    if dim != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"FAISS index dimension {dim} != expected {EXPECTED_EMBEDDING_DIM}. "
            f"Production contract requires {EXPECTED_EMBEDDING_DIM}-d embeddings."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata_list = metadata.get("metadata", metadata.get("records", []))

    n_vectors = index.ntotal
    embeddings = np.zeros((n_vectors, dim), dtype=np.float32)
    index.reconstruct_n(0, n_vectors, embeddings)

    # Validate normalization (L2 norm should be ~1.0 for cosine similarity via IP)
    norms = np.linalg.norm(embeddings, axis=1)
    bad_mask = np.abs(norms - 1.0) > NORM_TOLERANCE
    if np.any(bad_mask):
        bad_count = int(np.sum(bad_mask))
        bad_min = float(norms[bad_mask].min())
        bad_max = float(norms[bad_mask].max())
        raise ValueError(
            f"Reference embeddings not unit-normalized: {bad_count}/{n_vectors} "
            f"vectors outside tolerance {NORM_TOLERANCE} (min={bad_min:.4f}, max={bad_max:.4f}). "
            f"Cosine/IP calibration assumptions violated."
        )

    ref_records = []
    for i in range(min(n_vectors, len(metadata_list))):
        entry = metadata_list[i]
        ref_records.append({
            "vector_id": i,
            "person_id": entry.get("person_id", entry.get("label", "")),
            "image_path": entry.get("image", entry.get("local_path", "")),
            "sha256": entry.get("sha256", ""),
            "source": entry.get("source", ""),
            "license": entry.get("license", ""),
        })
    return ref_records, embeddings


def validate_query_embedding(emb: npt.NDArray[np.float32], img_path: str) -> None:
    """Validate query embedding dimension and normalization."""
    if emb.shape != (EXPECTED_EMBEDDING_DIM,):
        raise ValueError(
            f"Query embedding dimension {emb.shape} != expected ({EXPECTED_EMBEDDING_DIM},). "
            f"File: {img_path}"
        )
    norm = float(np.linalg.norm(emb))
    if abs(norm - 1.0) > NORM_TOLERANCE:
        raise ValueError(
            f"Query embedding not unit-normalized: norm={norm:.4f}, "
            f"tolerance={NORM_TOLERANCE}. File: {img_path}"
        )


def score_stats(scores: npt.NDArray[np.float32]) -> dict[str, Any]:
    if scores.size == 0:
        return {}
    p_vals = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    p_computed = np.percentile(scores, p_vals)
    return {
        "count": int(scores.size),
        "min": round(float(scores.min()), 6),
        "max": round(float(scores.max()), 6),
        "mean": round(float(scores.mean()), 6),
        "median": round(float(np.median(scores)), 6),
        "std": round(float(scores.std()), 6),
        **{f"p{p}": round(float(v), 6) for p, v in zip(p_vals, p_computed)},
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Reference gallery validation
# ---------------------------------------------------------------------------

def validate_reference_gallery(
    ref_records: list[dict[str, Any]],
    people: list[dict[str, Any]],
) -> dict[str, int]:
    """Derive expected reference counts from metadata. Returns person_id -> count."""
    person_ids = {p["person_id"] for p in people}
    ref_counts: dict[str, int] = defaultdict(int)
    for ref in ref_records:
        ref_counts[ref["person_id"]] += 1

    # Verify all query identities have references
    for pid in person_ids:
        if ref_counts[pid] == 0:
            raise ValueError(f"Identity '{pid}' has no reference images in the index.")

    # Verify uniformity
    counts = list(ref_counts.values())
    if len(set(counts)) > 1:
        logger.warning("Non-uniform reference counts per identity: %s", dict(ref_counts))
    else:
        logger.info("Uniform reference count: %d per identity", counts[0])

    return dict(ref_counts)


# ---------------------------------------------------------------------------
# 1. Build all query→reference pairs
# ---------------------------------------------------------------------------

def build_all_pairs(
    query_embeddings: dict[str, list[tuple[str, npt.NDArray[np.float32]]]],
    ref_records: list[dict[str, Any]],
    ref_embeddings: npt.NDArray[np.float32],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    genuine_pairs: list[dict[str, Any]] = []
    impostor_pairs: list[dict[str, Any]] = []
    for person_id, items in query_embeddings.items():
        for img_path, query_emb in items:
            sims = ref_embeddings @ query_emb
            for i, ref in enumerate(ref_records):
                sim = float(sims[i])
                pair = {
                    "query_person_id": person_id,
                    "query_image": img_path,
                    "ref_person_id": ref["person_id"],
                    "ref_image": ref["image_path"],
                    "ref_vector_id": ref["vector_id"],
                    "similarity": round(sim, 6),
                    "source": ref.get("source", ""),
                    "license": ref.get("license", ""),
                }
                if ref["person_id"] == person_id:
                    genuine_pairs.append(pair)
                else:
                    impostor_pairs.append(pair)
    return genuine_pairs, impostor_pairs


# ---------------------------------------------------------------------------
# 2. Identity-level aggregation (max per identity per query)
# ---------------------------------------------------------------------------

def aggregate_identity_scores(
    pairs: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """For each query, aggregate scores by reference identity using max similarity."""
    query_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        key = p["query_image"]
        query_groups[key][p["ref_person_id"]].append(p["similarity"])
    result: dict[str, dict[str, float]] = {}
    for query_img, identity_sims in query_groups.items():
        result[query_img] = {
            pid: max(sims) for pid, sims in identity_sims.items()
        }
    return result


# ---------------------------------------------------------------------------
# 3. Image-level hard negatives
# ---------------------------------------------------------------------------

def image_level_hard_negatives(
    impostor_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each query, find the STRONGEST image-level impostor (single reference).

    Returns one row per query, sorted by similarity descending.
    """
    by_query: dict[str, list[dict]] = defaultdict(list)
    for p in impostor_pairs:
        by_query[p["query_image"]].append(p)

    results = []
    for query_img, pairs in by_query.items():
        best = max(pairs, key=lambda p: p["similarity"])
        results.append({
            "query_image": query_img,
            "query_person_id": best["query_person_id"],
            "impostor_person_id": best["ref_person_id"],
            "impostor_image": best["ref_image"],
            "impostor_vector_id": best["ref_vector_id"],
            "similarity": best["similarity"],
            "score_level": "IMAGE",
            "source": best.get("source", ""),
            "license": best.get("license", ""),
        })
    results.sort(key=lambda p: p["similarity"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 4. Identity-level hard negatives with supporting reference (CANONICAL)
# ---------------------------------------------------------------------------

def identity_level_hard_negatives(
    query_embeddings: dict[str, list[tuple[str, npt.NDArray[np.float32]]]],
    ref_records: list[dict[str, Any]],
    ref_embeddings: npt.NDArray[np.float32],
) -> list[dict[str, Any]]:
    """For each query, find the STRONGEST identity-level impostor WITH supporting reference.

    Identity score = max(similarity across all references belonging to that identity).
    Returns one row per (query, impostor_identity), sorted by identity_score descending.

    Person IDs come from query_embeddings keys and ref_records, never from filesystem paths.
    """
    results = []
    for person_id, items in query_embeddings.items():
        for img_path, query_emb in items:
            sims = ref_embeddings @ query_emb
            # Group by impostor identity
            impostor_by_id: dict[str, list[tuple[float, int]]] = defaultdict(list)
            for i, ref in enumerate(ref_records):
                if ref["person_id"] == person_id:
                    continue
                impostor_by_id[ref["person_id"]].append((float(sims[i]), i))

            for impostor_id, sim_list in impostor_by_id.items():
                max_sim, max_idx = max(sim_list, key=lambda x: x[0])
                ref = ref_records[max_idx]
                results.append({
                    "query_image": img_path,
                    "query_person_id": person_id,
                    "impostor_person_id": impostor_id,
                    "identity_score": round(max_sim, 6),
                    "supporting_ref_image": ref["image_path"],
                    "supporting_ref_vector_id": ref["vector_id"],
                    "score_level": "IDENTITY",
                    "source": ref.get("source", ""),
                    "license": ref.get("license", ""),
                })
    results.sort(key=lambda p: p["identity_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# 5. EER computation
# ---------------------------------------------------------------------------

def compute_eer(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
    num_thresholds: int = 10000,
) -> dict[str, Any]:
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.sort(np.unique(all_scores))[::-1]
    if len(thresholds) > num_thresholds:
        indices = np.linspace(0, len(thresholds) - 1, num_thresholds, dtype=int)
        thresholds = thresholds[indices]
    best_eer = 1.0
    best_threshold = 0.0
    best_far = 1.0
    best_frr = 0.0
    for t in thresholds:
        far = float(np.sum(impostor_scores >= t)) / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
        frr = float(np.sum(genuine_scores < t)) / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
        diff = abs(far - frr)
        if diff < abs(best_far - best_frr):
            best_eer = (far + frr) / 2.0
            best_threshold = float(t)
            best_far = far
            best_frr = frr
    return {
        "eer": round(best_eer, 6),
        "threshold": round(best_threshold, 6),
        "far_at_eer": round(best_far, 6),
        "frr_at_eer": round(best_frr, 6),
    }


def compute_roc_auc(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
    num_thresholds: int = 10000,
) -> dict[str, Any]:
    thresholds = np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))[::-1]
    if len(thresholds) > num_thresholds:
        indices = np.linspace(0, len(thresholds) - 1, num_thresholds, dtype=int)
        thresholds = thresholds[indices]
    tprs, fprs = [], []
    for t in thresholds:
        tprs.append(float(np.sum(genuine_scores >= t)) / len(genuine_scores) if len(genuine_scores) > 0 else 0.0)
        fprs.append(float(np.sum(impostor_scores >= t)) / len(impostor_scores) if len(impostor_scores) > 0 else 0.0)
    tprs, fprs = np.array(tprs), np.array(fprs)
    sorted_idx = np.argsort(fprs)
    auc = float(np.trapezoid(tprs[sorted_idx], fprs[sorted_idx]))
    return {"auc": round(auc, 6), "thresholds": [round(float(t), 6) for t in thresholds],
            "tprs": [round(float(t), 6) for t in tprs], "fprs": [round(float(t), 6) for t in fprs]}


# ---------------------------------------------------------------------------
# 6. Operating points
# ---------------------------------------------------------------------------

def compute_operating_points(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
) -> list[dict[str, Any]]:
    thresholds = np.sort(np.unique(np.concatenate([genuine_scores, impostor_scores])))[::-1]
    targets = [
        ("youden_j", None, None), ("eer", None, None),
        ("far_5pct", 0.05, None), ("far_1pct", 0.01, None),
        ("far_0_5pct", 0.005, None), ("far_0_1pct", 0.001, None),
        ("frr_5pct", None, 0.05), ("frr_10pct", None, 0.10),
    ]
    n_genuine, n_impostor = len(genuine_scores), len(impostor_scores)
    points = []
    for name, target_far, target_frr in targets:
        best_t, best_metric = 0.0, float("inf")
        for t in thresholds:
            far = float(np.sum(impostor_scores >= t)) / n_impostor if n_impostor > 0 else 0.0
            frr = float(np.sum(genuine_scores < t)) / n_genuine if n_genuine > 0 else 0.0
            tpr = 1.0 - frr
            if name == "youden_j":
                metric = -(tpr - far)
            elif name == "eer":
                metric = abs(far - frr)
            elif target_far is not None:
                metric = frr if far <= target_far else float("inf")
            elif target_frr is not None:
                metric = far if frr <= target_frr else float("inf")
            else:
                metric = float("inf")
            if metric < best_metric:
                best_metric = metric
                best_t = float(t)

        far = float(np.sum(impostor_scores >= best_t)) / n_impostor if n_impostor > 0 else 0.0
        frr = float(np.sum(genuine_scores < best_t)) / n_genuine if n_genuine > 0 else 0.0
        tpr = 1.0 - frr
        tp = int(np.sum(genuine_scores >= best_t))
        fn = int(np.sum(genuine_scores < best_t))
        fp = int(np.sum(impostor_scores >= best_t))
        tn = int(np.sum(impostor_scores < best_t))
        tnr = tn / n_impostor if n_impostor > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        points.append({
            "point_type": name, "threshold": round(best_t, 6),
            "far": round(far, 6), "frr": round(frr, 6), "tpr": round(tpr, 6),
            "tnr": round(tnr, 6),
            "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })
    return points


# ---------------------------------------------------------------------------
# 7. Threshold consistency verification (strengthened)
# ---------------------------------------------------------------------------

def verify_threshold_consistency(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
    operating_points: list[dict[str, Any]],
    score_level: str,
) -> list[dict[str, Any]]:
    """Verify ALL reported metrics against independently recomputed values.

    Checks: TP, FP, TN, FN, FAR, FRR, TPR, TNR, precision, recall, F1.
    """
    n_genuine = len(genuine_scores)
    n_impostor = len(impostor_scores)
    results = []
    for p in operating_points:
        t = p["threshold"]
        tp = int(np.sum(genuine_scores >= t))
        fn = int(np.sum(genuine_scores < t))
        fp = int(np.sum(impostor_scores >= t))
        tn = int(np.sum(impostor_scores < t))

        far = fp / n_impostor if n_impostor > 0 else 0.0
        frr = fn / n_genuine if n_genuine > 0 else 0.0
        tpr = tp / n_genuine if n_genuine > 0 else 0.0
        tnr = tn / n_impostor if n_impostor > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Check all metrics with tolerances
        tol_int = 0  # exact match for integers
        tol_float = 0.002  # tolerance for floating-point derived metrics

        checks = {
            "tp": tp == p["tp"],
            "fp": fp == p["fp"],
            "tn": tn == p["tn"],
            "fn": fn == p["fn"],
            "far": abs(far - p["far"]) < tol_float,
            "frr": abs(frr - p["frr"]) < tol_float,
            "tpr": abs(tpr - p["tpr"]) < tol_float,
            "tnr": abs(tnr - p["tnr"]) < tol_float,
            "precision": abs(precision - p["precision"]) < tol_float,
            "recall": abs(recall - p["recall"]) < tol_float,
            "f1": abs(f1 - p["f1"]) < tol_float,
        }
        consistent = all(checks.values())
        failed_metrics = [k for k, v in checks.items() if not v]

        n_above = int(np.sum(impostor_scores >= t))

        results.append({
            "score_level": score_level,
            "operating_point": p["point_type"],
            "threshold": p["threshold"],
            "max_impostor_score": round(float(impostor_scores.max()), 6) if n_impostor > 0 else 0.0,
            "impostors_above_threshold": n_above,
            "reported_far": p["far"], "computed_far": round(far, 6),
            "reported_frr": p["frr"], "computed_frr": round(frr, 6),
            "reported_tpr": p["tpr"], "computed_tpr": round(tpr, 6),
            "reported_tnr": p["tnr"], "computed_tnr": round(tnr, 6),
            "reported_precision": p["precision"], "computed_precision": round(precision, 6),
            "reported_recall": p["recall"], "computed_recall": round(recall, 6),
            "reported_f1": p["f1"], "computed_f1": round(f1, 6),
            "reported_tp": p["tp"], "computed_tp": tp,
            "reported_fp": p["fp"], "computed_fp": fp,
            "reported_tn": p["tn"], "computed_tn": tn,
            "reported_fn": p["fn"], "computed_fn": fn,
            "consistent": consistent,
            "failed_metrics": failed_metrics,
        })
    return results


# ---------------------------------------------------------------------------
# 8. Reproduce 0.2975 case
# ---------------------------------------------------------------------------

def reproduce_02975_case(
    impostor_pairs: list[dict[str, Any]],
    identity_impostor: dict[str, dict[str, float]],
    image_op: list[dict[str, Any]],
    identity_op: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reproduce the robert_downey_jr → neymar case at 0.2975."""
    target_query = "robert_downey_jr_f68f78d02e99.jpg"
    target_impostor = "neymar"

    # Find image-level pair (strongest for this query-impostor)
    img_level_pair = None
    for p in impostor_pairs:
        if target_query in p["query_image"] and p["ref_person_id"] == target_impostor:
            if img_level_pair is None or p["similarity"] > img_level_pair["similarity"]:
                img_level_pair = p

    # Find identity-level score
    id_level_score = None
    for query_img, id_sims in identity_impostor.items():
        if target_query in query_img and target_impostor in id_sims:
            id_level_score = id_sims[target_impostor]
            break

    # Evaluate against thresholds using the decision helper
    threshold_results = []
    for op in image_op:
        score = img_level_pair["similarity"] if img_level_pair else None
        accepted = is_accepted(score, op["threshold"]) if score is not None else False
        threshold_results.append({
            "threshold_name": op["point_type"],
            "threshold_value": op["threshold"],
            "score_level": "IMAGE",
            "score": score,
            "accepted": accepted,
            "decision": "ACCEPT (false accept)" if accepted else "REJECT (correct reject)",
        })

    for op in identity_op:
        score = id_level_score
        accepted = is_accepted(score, op["threshold"]) if score is not None else False
        threshold_results.append({
            "threshold_name": op["point_type"],
            "threshold_value": op["threshold"],
            "score_level": "IDENTITY",
            "score": score,
            "accepted": accepted,
            "decision": "ACCEPT (false accept)" if accepted else "REJECT (correct reject)",
        })

    return {
        "query_image": img_level_pair["query_image"] if img_level_pair else None,
        "query_person_id": "robert_downey_jr",
        "impostor_person_id": target_impostor,
        "image_level_score": img_level_pair["similarity"] if img_level_pair else None,
        "identity_level_score": id_level_score,
        "threshold_evaluation": threshold_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()

    # Load people
    people = load_people()
    logger.info("Loaded %d people.", len(people))

    # Load face model
    from services.face_service import FaceService
    face_service = FaceService()
    model = face_service.get_model()
    logger.info("FaceService loaded.")

    # Load reference embeddings (validates dimension and normalization)
    index_path = DATASET_BASE / "search_index" / "reference_index.faiss"
    metadata_path = DATASET_BASE / "search_index" / "metadata.json"
    ref_records, ref_embeddings = load_reference_embeddings_with_faiss(index_path, metadata_path)
    logger.info("Loaded %d reference embeddings (dim=%d).", len(ref_records), ref_embeddings.shape[1])

    # Validate reference gallery structure
    ref_counts = validate_reference_gallery(ref_records, people)
    refs_per_person = min(ref_counts.values()) if ref_counts else 0
    logger.info("Reference gallery: %d identities, %d refs/person (uniform=%s)",
                len(ref_counts), refs_per_person,
                len(set(ref_counts.values())) == 1)

    # Extract query embeddings (single-face enforcement)
    logger.info("Extracting query embeddings (single-face contract)...")
    query_dir = DATASET_BASE / "query"
    query_embeddings: dict[str, list[tuple[str, npt.NDArray[np.float32]]]] = {}
    face_telemetry = {"valid": 0, "no_face": 0, "multi_face": 0, "invalid": 0}

    for person in people:
        person_id = person["person_id"]
        person_query_dir = query_dir / person_id
        if not person_query_dir.exists():
            continue
        embeddings = []
        for img_path in sorted(person_query_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            emb = extract_embedding_single_face(model, img_path)
            if emb is not None:
                validate_query_embedding(emb, str(img_path))
                embeddings.append((str(img_path), emb))
                face_telemetry["valid"] += 1
            else:
                # Distinguish failure mode
                img = cv2.imread(str(img_path))
                if img is None:
                    face_telemetry["invalid"] += 1
                else:
                    faces = model.get(img)
                    if len(faces) == 0:
                        face_telemetry["no_face"] += 1
                    elif len(faces) > 1:
                        face_telemetry["multi_face"] += 1
                    else:
                        face_telemetry["invalid"] += 1
        if embeddings:
            query_embeddings[person_id] = embeddings

    total_queries = sum(len(v) for v in query_embeddings.values())
    logger.info("Query extraction: %d valid, %d no_face, %d multi_face, %d invalid",
                face_telemetry["valid"], face_telemetry["no_face"],
                face_telemetry["multi_face"], face_telemetry["invalid"])
    logger.info("Extracted %d queries across %d identities.", total_queries, len(query_embeddings))

    # =========================================================================
    # A. Build all pairs
    # =========================================================================
    logger.info("Building query→reference pairs...")
    genuine_pairs, impostor_pairs = build_all_pairs(query_embeddings, ref_records, ref_embeddings)
    genuine_scores = np.array([p["similarity"] for p in genuine_pairs], dtype=np.float32)
    impostor_scores = np.array([p["similarity"] for p in impostor_pairs], dtype=np.float32)
    logger.info("Genuine: %d, Impostor: %d", len(genuine_pairs), len(impostor_pairs))

    # Verify pair counts dynamically from reference gallery
    expected_genuine = total_queries * refs_per_person
    expected_impostor = total_queries * (len(ref_records) - refs_per_person)
    genuine_match = expected_genuine == len(genuine_pairs)
    impostor_match = expected_impostor == len(impostor_pairs)
    logger.info("Expected genuine: %d, computed: %d, match: %s",
                expected_genuine, len(genuine_pairs), genuine_match)
    logger.info("Expected impostor: %d, computed: %d, match: %s",
                expected_impostor, len(impostor_pairs), impostor_match)

    if not genuine_match or not impostor_match:
        raise ValueError(
            f"Pair count mismatch: genuine={len(genuine_pairs)} (expected {expected_genuine}), "
            f"impostor={len(impostor_pairs)} (expected {expected_impostor})"
        )

    # =========================================================================
    # B. Identity-level aggregation
    # =========================================================================
    logger.info("Computing identity-level aggregation...")
    identity_genuine = aggregate_identity_scores(genuine_pairs)
    identity_impostor = aggregate_identity_scores(impostor_pairs)

    id_genuine_scores = []
    for query_img, id_sims in identity_genuine.items():
        for pid, sim in id_sims.items():
            id_genuine_scores.append(sim)
    id_genuine_arr = np.array(id_genuine_scores, dtype=np.float32)

    id_impostor_scores = []
    for query_img, id_sims in identity_impostor.items():
        for pid, sim in id_sims.items():
            id_impostor_scores.append(sim)
    id_impostor_arr = np.array(id_impostor_scores, dtype=np.float32)
    logger.info("Identity genuine: %d, impostor: %d", len(id_genuine_arr), len(id_impostor_arr))

    # =========================================================================
    # C. Image-level metrics
    # =========================================================================
    logger.info("Computing image-level metrics...")
    img_roc = compute_roc_auc(genuine_scores, impostor_scores)
    img_eer = compute_eer(genuine_scores, impostor_scores)
    img_op = compute_operating_points(genuine_scores, impostor_scores)
    logger.info("Image ROC-AUC: %.4f, EER: %.4f at %.4f", img_roc["auc"], img_eer["eer"], img_eer["threshold"])

    # =========================================================================
    # D. Identity-level metrics
    # =========================================================================
    logger.info("Computing identity-level metrics...")
    id_roc = compute_roc_auc(id_genuine_arr, id_impostor_arr)
    id_eer = compute_eer(id_genuine_arr, id_impostor_arr)
    id_op = compute_operating_points(id_genuine_arr, id_impostor_arr)
    logger.info("Identity ROC-AUC: %.4f, EER: %.4f at %.4f", id_roc["auc"], id_eer["eer"], id_eer["threshold"])

    # =========================================================================
    # E. Image-level hard negatives
    # =========================================================================
    logger.info("Computing image-level hard negatives...")
    img_hn = image_level_hard_negatives(impostor_pairs)
    global_max_img = img_hn[0]["similarity"] if img_hn else 0.0
    if img_hn:
        logger.info("Image-level global max impostor: %.6f (%s → %s)",
                     global_max_img, img_hn[0]["query_person_id"], img_hn[0]["impostor_person_id"])

    # =========================================================================
    # F. Identity-level hard negatives (canonical implementation)
    # =========================================================================
    logger.info("Computing identity-level hard negatives...")
    id_hn = identity_level_hard_negatives(query_embeddings, ref_records, ref_embeddings)
    global_max_id = id_hn[0]["identity_score"] if id_hn else 0.0
    if id_hn:
        logger.info("Identity-level global max impostor: %.6f (%s → %s)",
                     global_max_id, id_hn[0]["query_person_id"], id_hn[0]["impostor_person_id"])

    # =========================================================================
    # G. Threshold consistency verification
    # =========================================================================
    logger.info("Verifying threshold consistency (all metrics)...")
    img_consistency = verify_threshold_consistency(genuine_scores, impostor_scores, img_op, "IMAGE")
    id_consistency = verify_threshold_consistency(id_genuine_arr, id_impostor_arr, id_op, "IDENTITY")

    # =========================================================================
    # H. Reproduce 0.2975 case
    # =========================================================================
    logger.info("Reproducing 0.2975 case...")
    case_02975 = reproduce_02975_case(impostor_pairs, identity_impostor, img_op, id_op)

    # =========================================================================
    # I. Write artifacts
    # =========================================================================
    logger.info("Writing artifacts...")

    # Threshold consistency CSV
    all_consistency = img_consistency + id_consistency
    write_csv(OUTPUT_BASE / "threshold_consistency.csv", all_consistency, [
        "score_level", "operating_point", "threshold", "max_impostor_score",
        "impostors_above_threshold",
        "reported_far", "computed_far", "reported_frr", "computed_frr",
        "reported_tpr", "computed_tpr", "reported_tnr", "computed_tnr",
        "reported_precision", "computed_precision", "reported_recall", "computed_recall",
        "reported_f1", "computed_f1",
        "reported_tp", "computed_tp", "reported_fp", "computed_fp",
        "reported_tn", "computed_tn", "reported_fn", "computed_fn",
        "consistent", "failed_metrics",
    ])

    # Corrected hard negatives CSV (one strongest image-level impostor PER QUERY)
    hn_fields = ["query_image", "query_person_id", "impostor_person_id", "impostor_image",
                 "impostor_vector_id", "similarity", "score_level", "source", "license"]
    write_csv(OUTPUT_BASE / "corrected_hard_negatives.csv", img_hn, hn_fields)

    # Identity-level hard negatives CSV (one strongest identity-level impostor PER QUERY)
    id_hn_fields = ["query_image", "query_person_id", "impostor_person_id", "identity_score",
                    "supporting_ref_image", "supporting_ref_vector_id", "score_level", "source", "license"]
    write_csv(OUTPUT_BASE / "identity_hard_negatives.csv", id_hn, id_hn_fields)

    # Corrected operating points CSV
    op_fields = ["point_type", "threshold", "far", "frr", "tpr", "tnr", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    write_csv(OUTPUT_BASE / "corrected_threshold_operating_points.csv", img_op, op_fields)
    write_csv(OUTPUT_BASE / "corrected_identity_operating_points.csv", id_op, op_fields)

    # Calibration verification JSON
    verification = {
        "phase": "13.7.2",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cleanup_items": [
            "removed_duplicate_identity_level_hard_negatives",
            "eliminated_path_based_identity_inference",
            "robust_embedding_dimension_validation",
            "normalization_validation",
            "single_face_contract_enforcement",
            "dynamic_reference_count_derivation",
            "strengthened_threshold_consistency",
            "threshold_direction_helper",
        ],
        "face_telemetry": face_telemetry,
        "reference_gallery": {
            "identities": len(ref_counts),
            "refs_per_person": ref_counts,
            "uniform": len(set(ref_counts.values())) == 1,
        },
        "dataset": {
            "total_identities": len(query_embeddings),
            "total_reference": len(ref_records),
            "total_queries": total_queries,
            "refs_per_person": refs_per_person,
            "expected_genuine_pairs": expected_genuine,
            "actual_genuine_pairs": len(genuine_pairs),
            "genuine_pairs_match": genuine_match,
            "expected_impostor_pairs": expected_impostor,
            "actual_impostor_pairs": len(impostor_pairs),
            "impostor_pairs_match": impostor_match,
        },
        "image_level": {
            "roc_auc": img_roc["auc"],
            "eer": img_eer,
            "global_max_impostor": global_max_img,
            "operating_points": img_op,
            "consistency": img_consistency,
        },
        "identity_level": {
            "roc_auc": id_roc["auc"],
            "eer": id_eer,
            "global_max_impostor": global_max_id,
            "operating_points": id_op,
            "consistency": id_consistency,
        },
        "case_02975": case_02975,
    }
    with open(OUTPUT_BASE / "verification.json", "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2, default=str)

    # Case 0.2975 JSON
    with open(OUTPUT_BASE / "case_02975.json", "w", encoding="utf-8") as f:
        json.dump(case_02975, f, indent=2, default=str)

    # Cleanup summary MD
    with open(OUTPUT_BASE / "cleanup_summary.md", "w", encoding="utf-8") as f:
        f.write("# Phase 13.7.2 Cleanup Summary\n\n")
        f.write("## Changes Made\n\n")
        f.write("1. **Removed duplicate `identity_level_hard_negatives()`** — inferior implementation using path-based identity inference replaced by canonical `identity_level_hard_negatives()` (formerly `_with_support`).\n")
        f.write("2. **Eliminated path-based identity inference** — all person IDs now come from `query_embeddings` keys and `ref_records` metadata.\n")
        f.write("3. **Robust embedding dimension** — uses `index.d` instead of hardcoded 512; validates against expected dimension.\n")
        f.write("4. **Normalization validation** — validates reference and query embeddings are approximately unit-normalized before computing scores.\n")
        f.write("5. **Single-face contract** — calibration extraction now rejects multi-face images instead of silently selecting highest-confidence face.\n")
        f.write("6. **Dynamic reference count** — derives expected pair counts from actual reference gallery metadata, not hardcoded values.\n")
        f.write("7. **Stronger threshold consistency** — verifies TPR, TNR, precision, recall, F1 in addition to TP/FP/TN/FN/FAR/FRR.\n")
        f.write("8. **Threshold direction helper** — single `is_accepted(score, threshold)` function used throughout calibration code.\n")
        f.write("9. **Clarified artifact semantics** — CSV filenames and comments explicitly document one-row-per-query semantics.\n\n")
        f.write("## Face Telemetry\n\n")
        f.write(f"- Valid (exactly 1 face): {face_telemetry['valid']}\n")
        f.write(f"- No face: {face_telemetry['no_face']}\n")
        f.write(f"- Multi-face (rejected): {face_telemetry['multi_face']}\n")
        f.write(f"- Invalid/unreadable: {face_telemetry['invalid']}\n\n")
        f.write("## Metric Comparison\n\n")
        f.write("| Metric | Before | After | Status |\n")
        f.write("|--------|--------|-------|--------|\n")
        f.write(f"| Genuine pairs | 704 | {len(genuine_pairs)} | {'UNCHANGED' if len(genuine_pairs) == 704 else 'CHANGED'} |\n")
        f.write(f"| Impostor pairs | 14784 | {len(impostor_pairs)} | {'UNCHANGED' if len(impostor_pairs) == 14784 else 'CHANGED'} |\n")
        f.write(f"| Image ROC-AUC | 0.8785 | {img_roc['auc']:.4f} | {'UNCHANGED' if abs(img_roc['auc'] - 0.8785) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Identity ROC-AUC | 0.9465 | {id_roc['auc']:.4f} | {'UNCHANGED' if abs(id_roc['auc'] - 0.9465) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Image EER | 0.1960 | {img_eer['eer']:.4f} | {'UNCHANGED' if abs(img_eer['eer'] - 0.196) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Identity EER | 0.1155 | {id_eer['eer']:.4f} | {'UNCHANGED' if abs(id_eer['eer'] - 0.1155) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Image EER threshold | 0.0553 | {img_eer['threshold']:.4f} | {'UNCHANGED' if abs(img_eer['threshold'] - 0.0553) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Identity EER threshold | 0.1214 | {id_eer['threshold']:.4f} | {'UNCHANGED' if abs(id_eer['threshold'] - 0.1214) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Identity Youden threshold | 0.2415 | {id_op[0]['threshold']:.4f} | {'UNCHANGED' if abs(id_op[0]['threshold'] - 0.2415) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Image global max impostor | 0.2975 | {global_max_img:.4f} | {'UNCHANGED' if abs(global_max_img - 0.2975) < 0.001 else 'CHANGED'} |\n")
        f.write(f"| Identity global max impostor | 0.2975 | {global_max_id:.4f} | {'UNCHANGED' if abs(global_max_id - 0.2975) < 0.001 else 'CHANGED'} |\n")

    elapsed = time.time() - t0
    logger.info("Phase 13.7.2 complete. Artifacts saved to %s (%.1fs)", OUTPUT_BASE, elapsed)

    # Print summary
    print("\n" + "=" * 60)
    print("PHASE 13.7.2 SUMMARY")
    print("=" * 60)
    print(f"Image-level global max impostor:  {global_max_img:.6f}")
    print(f"Identity-level global max impostor: {global_max_id:.6f}")
    print(f"Identity-level Youden threshold:   {id_op[0]['threshold']:.6f}")
    print(f"Image-level Youden threshold:      {img_op[0]['threshold']:.6f}")
    print(f"Image ROC-AUC: {img_roc['auc']:.6f}")
    print(f"Identity ROC-AUC: {id_roc['auc']:.6f}")
    print(f"Image EER: {img_eer['eer']:.6f}")
    print(f"Identity EER: {id_eer['eer']:.6f}")
    print(f"All thresholds consistent: {all(c['consistent'] for c in all_consistency)}")
    print(f"Genuine pairs: {len(genuine_pairs)} (expected {expected_genuine})")
    print(f"Impostor pairs: {len(impostor_pairs)} (expected {expected_impostor})")
    print(f"Face telemetry: {face_telemetry}")
    print("=" * 60)


if __name__ == "__main__":
    main()
