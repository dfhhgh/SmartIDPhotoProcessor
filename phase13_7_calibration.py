"""Phase 13.7 — Reverse Search Calibration & Threshold Selection

Audit Phase 13.6.3 evaluation, construct proper query→reference calibration
pairs, compute score distributions, ROC-AUC, EER, operating points, hard
negative analysis, identity error analysis, and threshold stability.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_BASE = Path("outputs/phase13_7")
DATASET_BASE = Path("datasets/celebrity-v3")
MANIFEST_PATH = Path("dataset_acquisition/people_v3_scaled.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_people() -> list[dict[str, Any]]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["people"]


def load_metadata(metadata_path: Path) -> dict[str, Any]:
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_embedding(model, img_path: Path) -> npt.NDArray[np.float32] | None:
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    faces = model.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
    emb = face.normed_embedding
    if emb is None:
        return None
    return emb.ravel().astype(np.float32)


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


# ---------------------------------------------------------------------------
# 1. Load embeddings for reference and query
# ---------------------------------------------------------------------------

def load_reference_embeddings(
    metadata: dict[str, Any],
    index_size: int,
) -> dict[str, dict[str, Any]]:
    """Load reference embeddings from metadata.

    Returns dict mapping vector_id -> {person_id, image_path, embedding}.
    """
    ref_map: dict[str, dict[str, Any]] = {}
    metadata_list = metadata.get("metadata", metadata.get("records", []))

    for i, entry in enumerate(metadata_list):
        if i >= index_size:
            break
        ref_map[str(i)] = {
            "vector_id": i,
            "person_id": entry.get("person_id", entry.get("label", "")),
            "image_path": entry.get("image", entry.get("local_path", "")),
            "sha256": entry.get("sha256", ""),
        }
    return ref_map


def load_reference_embeddings_with_faiss(
    index_path: Path,
    metadata_path: Path,
    model,
) -> tuple[list[dict[str, Any]], npt.NDArray[np.float32]]:
    """Load FAISS index and extract all reference embeddings."""
    import faiss

    index = faiss.read_index(str(index_path))

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_list = metadata.get("metadata", metadata.get("records", []))

    # Extract embeddings from index
    n_vectors = index.ntotal
    embeddings = np.zeros((n_vectors, 512), dtype=np.float32)
    index.reconstruct_n(0, n_vectors, embeddings)

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


# ---------------------------------------------------------------------------
# 2. Build query→reference calibration pairs
# ---------------------------------------------------------------------------

def build_query_reference_pairs(
    query_embeddings: dict[str, list[tuple[str, npt.NDArray[np.float32]]]],
    ref_records: list[dict[str, Any]],
    ref_embeddings: npt.NDArray[np.float32],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build genuine and impostor query→reference pairs.

    For each query image, compute cosine similarity against all reference images.
    Label as genuine if query.identity == reference.identity, else impostor.
    """
    genuine_pairs: list[dict[str, Any]] = []
    impostor_pairs: list[dict[str, Any]] = []

    for person_id, items in query_embeddings.items():
        for img_path, query_emb in items:
            # Compute similarities against all references
            sims = ref_embeddings @ query_emb  # (n_refs,)

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
# 3. Identity-level aggregation
# ---------------------------------------------------------------------------

def aggregate_identity_scores(
    pairs: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """For each query, aggregate scores by reference identity (max similarity per identity)."""
    from collections import defaultdict

    query_groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for p in pairs:
        key = p["query_image"]
        query_groups[key][p["ref_person_id"]].append(p["similarity"])

    # Max per identity per query
    result: dict[str, dict[str, float]] = {}
    for query_img, identity_sims in query_groups.items():
        result[query_img] = {
            pid: max(sims) for pid, sims in identity_sims.items()
        }
    return result


# ---------------------------------------------------------------------------
# 4. ROC / AUC / EER
# ---------------------------------------------------------------------------

def compute_roc_auc(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
    num_thresholds: int = 10000,
) -> dict[str, Any]:
    """Compute ROC curve and AUC."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    all_labels = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores)),
    ])

    thresholds = np.sort(np.unique(all_scores))[::-1]
    if len(thresholds) > num_thresholds:
        indices = np.linspace(0, len(thresholds) - 1, num_thresholds, dtype=int)
        thresholds = thresholds[indices]

    tprs = []
    fprs = []
    for t in thresholds:
        tp = np.sum((genuine_scores >= t))
        fn = np.sum((genuine_scores < t))
        fp = np.sum((impostor_scores >= t))
        tn = np.sum((impostor_scores < t))
        tpr = tp / len(genuine_scores) if len(genuine_scores) > 0 else 0.0
        fpr = fp / len(impostor_scores) if len(impostor_scores) > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    tprs = np.array(tprs)
    fprs = np.array(fprs)

    # AUC via trapezoidal rule
    sorted_idx = np.argsort(fprs)
    fprs_sorted = fprs[sorted_idx]
    tprs_sorted = tprs[sorted_idx]
    auc = float(np.trapezoid(tprs_sorted, fprs_sorted))

    return {
        "auc": round(auc, 6),
        "thresholds": [round(float(t), 6) for t in thresholds],
        "tprs": [round(float(t), 6) for t in tprs],
        "fprs": [round(float(t), 6) for t in fprs],
    }


def compute_eer(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
    num_thresholds: int = 10000,
) -> dict[str, Any]:
    """Compute Equal Error Rate."""
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


# ---------------------------------------------------------------------------
# 5. Operating points
# ---------------------------------------------------------------------------

def compute_operating_points(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
) -> list[dict[str, Any]]:
    """Compute operating points for various FAR/FRR targets."""
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.sort(np.unique(all_scores))[::-1]

    targets = [
        ("youden_j", None, None),
        ("eer", None, None),
        ("far_5pct", 0.05, None),
        ("far_1pct", 0.01, None),
        ("far_0_5pct", 0.005, None),
        ("far_0_1pct", 0.001, None),
        ("frr_5pct", None, 0.05),
        ("frr_10pct", None, 0.10),
    ]

    n_genuine = len(genuine_scores)
    n_impostor = len(impostor_scores)

    points = []
    for name, target_far, target_frr in targets:
        best_t = 0.0
        best_metric = float("inf")

        for t in thresholds:
            far = float(np.sum(impostor_scores >= t)) / n_impostor if n_impostor > 0 else 0.0
            frr = float(np.sum(genuine_scores < t)) / n_genuine if n_genuine > 0 else 0.0
            tpr = 1.0 - frr
            fnr = frr

            if name == "youden_j":
                metric = -(tpr - far)  # maximize Youden's J
            elif name == "eer":
                metric = abs(far - frr)
            elif target_far is not None:
                if far <= target_far:
                    metric = frr  # minimize FRR while meeting FAR target
                else:
                    metric = float("inf")
            elif target_frr is not None:
                if frr <= target_frr:
                    metric = far  # minimize FAR while meeting FRR target
                else:
                    metric = float("inf")
            else:
                metric = float("inf")

            if metric < best_metric:
                best_metric = metric
                best_t = float(t)

        # Evaluate at best threshold
        far = float(np.sum(impostor_scores >= best_t)) / n_impostor if n_impostor > 0 else 0.0
        frr = float(np.sum(genuine_scores < best_t)) / n_genuine if n_genuine > 0 else 0.0
        tpr = 1.0 - frr
        tp = int(np.sum(genuine_scores >= best_t))
        fn = int(np.sum(genuine_scores < best_t))
        fp = int(np.sum(impostor_scores >= best_t))
        tn = int(np.sum(impostor_scores < best_t))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        points.append({
            "point_type": name,
            "threshold": round(best_t, 6),
            "far": round(far, 6),
            "frr": round(frr, 6),
            "tpr": round(tpr, 6),
            "tnr": round(tn / n_impostor if n_impostor > 0 else 0.0, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    return points


# ---------------------------------------------------------------------------
# 6. Hard negative analysis
# ---------------------------------------------------------------------------

def analyze_hard_negatives(
    query_embeddings: dict[str, list[tuple[str, npt.NDArray[np.float32]]]],
    ref_records: list[dict[str, Any]],
    ref_embeddings: npt.NDArray[np.float32],
) -> list[dict[str, Any]]:
    """Find the strongest impostor for each query image."""
    hard_negatives = []

    for person_id, items in query_embeddings.items():
        for img_path, query_emb in items:
            sims = ref_embeddings @ query_emb

            # Mask out same-identity references
            for i, ref in enumerate(ref_records):
                if ref["person_id"] == person_id:
                    continue
                sim = float(sims[i])
                hard_negatives.append({
                    "query_person_id": person_id,
                    "query_image": img_path,
                    "impostor_person_id": ref["person_id"],
                    "impostor_image": ref["image_path"],
                    "impostor_vector_id": ref["vector_id"],
                    "similarity": round(sim, 6),
                    "source": ref.get("source", ""),
                    "license": ref.get("license", ""),
                })

    # Group by query, keep strongest impostor per query
    from collections import defaultdict
    by_query: dict[str, list[dict]] = defaultdict(list)
    for hn in hard_negatives:
        by_query[hn["query_image"]].append(hn)

    strongest = []
    for query_img, pairs in by_query.items():
        best = max(pairs, key=lambda p: p["similarity"])
        strongest.append(best)

    strongest.sort(key=lambda p: p["similarity"], reverse=True)
    return strongest


# ---------------------------------------------------------------------------
# 7. Identity error analysis
# ---------------------------------------------------------------------------

def identity_error_analysis(
    genuine_pairs: list[dict[str, Any]],
    impostor_pairs: list[dict[str, Any]],
    threshold: float,
) -> list[dict[str, Any]]:
    """Analyze errors per identity at a given threshold."""
    from collections import defaultdict

    # Genuine stats per identity
    genuine_by_id: dict[str, list[float]] = defaultdict(list)
    for p in genuine_pairs:
        genuine_by_id[p["query_person_id"]].append(p["similarity"])

    # Strongest impostor against each identity
    impostor_by_target: dict[str, list[float]] = defaultdict(list)
    for p in impostor_pairs:
        impostor_by_target[p["ref_person_id"]].append(p["similarity"])

    results = []
    for person_id in sorted(genuine_by_id.keys()):
        genuine_sims = genuine_by_id[person_id]
        n_queries = len(set(
            p["query_image"] for p in genuine_pairs if p["query_person_id"] == person_id
        ))
        false_rejects = sum(1 for s in genuine_sims if s < threshold)
        strongest_impostor = max(impostor_by_target.get(person_id, [0.0]))
        n_above_threshold = sum(1 for s in impostor_by_target.get(person_id, []) if s >= threshold)

        results.append({
            "person_id": person_id,
            "n_queries": n_queries,
            "genuine_mean": round(float(np.mean(genuine_sims)), 6),
            "genuine_min": round(float(np.min(genuine_sims)), 6),
            "genuine_max": round(float(np.max(genuine_sims)), 6),
            "false_rejects": false_rejects,
            "strongest_impostor": round(strongest_impostor, 6),
            "impostors_above_threshold": n_above_threshold,
        })

    return results


# ---------------------------------------------------------------------------
# 8. Cross-validation / threshold stability
# ---------------------------------------------------------------------------

def cross_validation_stability(
    query_embeddings: dict[str, list[tuple[str, npt.NDArray[np.float32]]]],
    ref_records: list[dict[str, Any]],
    ref_embeddings: npt.NDArray[np.float32],
    n_folds: int = 5,
) -> dict[str, Any]:
    """Evaluate threshold stability via cross-validation.

    Split queries into folds, compute EER threshold per fold.
    """
    rng = np.random.RandomState(SEED)

    # Flatten queries
    all_queries = []
    for person_id, items in query_embeddings.items():
        for img_path, emb in items:
            all_queries.append((person_id, img_path, emb))

    rng.shuffle(all_queries)
    fold_size = len(all_queries) // n_folds

    eer_thresholds = []
    far_1pct_thresholds = []
    far_0_5pct_thresholds = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(all_queries)
        fold_queries = all_queries[start:end]

        # Build scores for this fold
        genuine_scores = []
        impostor_scores = []

        for person_id, img_path, emb in fold_queries:
            sims = ref_embeddings @ emb
            for i, ref in enumerate(ref_records):
                sim = float(sims[i])
                if ref["person_id"] == person_id:
                    genuine_scores.append(sim)
                else:
                    impostor_scores.append(sim)

        genuine_arr = np.array(genuine_scores, dtype=np.float32)
        impostor_arr = np.array(impostor_scores, dtype=np.float32)

        # EER
        eer = compute_eer(genuine_arr, impostor_arr)
        eer_thresholds.append(eer["threshold"])

        # FAR <= 1%
        thresholds_sorted = np.sort(np.unique(impostor_arr))[::-1]
        far_1t = 0.0
        for t in thresholds_sorted:
            far = float(np.sum(impostor_arr >= t)) / len(impostor_arr) if len(impostor_arr) > 0 else 0.0
            if far <= 0.01:
                far_1t = float(t)
                break
        far_1pct_thresholds.append(far_1t)

        # FAR <= 0.5%
        far_05t = 0.0
        for t in thresholds_sorted:
            far = float(np.sum(impostor_arr >= t)) / len(impostor_arr) if len(impostor_arr) > 0 else 0.0
            if far <= 0.005:
                far_05t = float(t)
                break
        far_0_5pct_thresholds.append(far_05t)

    def stability_stats(vals: list[float]) -> dict[str, Any]:
        arr = np.array(vals)
        return {
            "mean": round(float(arr.mean()), 6),
            "median": round(float(np.median(arr)), 6),
            "std": round(float(arr.std()), 6),
            "min": round(float(arr.min()), 6),
            "max": round(float(arr.max()), 6),
            "values": [round(float(v), 6) for v in arr],
        }

    return {
        "n_folds": n_folds,
        "fold_size": fold_size,
        "total_queries": len(all_queries),
        "eer_stability": stability_stats(eer_thresholds),
        "far_1pct_stability": stability_stats(far_1pct_thresholds),
        "far_0_5pct_stability": stability_stats(far_0_5pct_thresholds),
    }


# ---------------------------------------------------------------------------
# 9. Plot generation
# ---------------------------------------------------------------------------

def generate_plots(
    genuine_scores: npt.NDArray[np.float32],
    impostor_scores: npt.NDArray[np.float32],
    roc_data: dict[str, Any],
    operating_points: list[dict[str, Any]],
    eer_result: dict[str, Any],
    plots_dir: Path,
) -> None:
    """Generate calibration plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1. Genuine vs impostor score distributions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(genuine_scores, bins=50, alpha=0.6, label="Genuine", density=True, color="green")
    ax.hist(impostor_scores, bins=50, alpha=0.6, label="Impostor", density=True, color="red")
    ax.axvline(x=eer_result["threshold"], color="black", linestyle="--", label=f'EER threshold={eer_result["threshold"]:.4f}')
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Genuine vs Impostor Score Distributions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "score_distributions.png", dpi=150)
    plt.close(fig)

    # 2. ROC curve
    fig, ax = plt.subplots(figsize=(8, 8))
    fprs = roc_data["fprs"]
    tprs = roc_data["tprs"]
    ax.plot(fprs, tprs, label=f'ROC (AUC={roc_data["auc"]:.4f})')
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # 3. FAR vs threshold and FRR vs threshold
    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.sort(np.unique(all_scores))[::-1]
    n_genuine = len(genuine_scores)
    n_impostor = len(impostor_scores)

    far_vals = []
    frr_vals = []
    for t in thresholds:
        far = float(np.sum(impostor_scores >= t)) / n_impostor if n_impostor > 0 else 0.0
        frr = float(np.sum(genuine_scores < t)) / n_genuine if n_genuine > 0 else 0.0
        far_vals.append(far)
        frr_vals.append(frr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, far_vals, label="FAR", color="red")
    ax.plot(thresholds, frr_vals, label="FRR", color="blue")
    ax.axvline(x=eer_result["threshold"], color="black", linestyle="--", alpha=0.5, label=f'EER={eer_result["threshold"]:.4f}')
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Error Rate")
    ax.set_title("FAR and FRR vs Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "far_frr_vs_threshold.png", dpi=150)
    plt.close(fig)

    # 4. Impostor tail distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_impostor = np.sort(impostor_scores)[::-1]
    ax.plot(range(len(sorted_impostor)), sorted_impostor, color="red")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Similarity")
    ax.set_title("Impostor Score Tail Distribution (sorted descending)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "impostor_tail.png", dpi=150)
    plt.close(fig)

    logger.info("Plots saved to %s", plots_dir)


# ---------------------------------------------------------------------------
# 10. Write CSV artifacts
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    # Load people manifest
    people = load_people()
    logger.info("Loaded %d people from manifest", len(people))

    # Load face model
    from services.face_service import FaceService
    face_service = FaceService()
    model = face_service.get_model()
    logger.info("FaceService loaded.")

    # Load reference embeddings from FAISS index
    index_path = DATASET_BASE / "search_index" / "reference_index.faiss"
    metadata_path = DATASET_BASE / "search_index" / "metadata.json"

    logger.info("Loading reference embeddings from FAISS index...")
    ref_records, ref_embeddings = load_reference_embeddings_with_faiss(index_path, metadata_path, model)
    logger.info("Loaded %d reference embeddings.", len(ref_records))

    # Load metadata for source/license info
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata_raw = json.load(f)
    metadata_list = metadata_raw.get("metadata", metadata_raw.get("records", []))

    # Enrich ref_records with source/license from metadata
    for i, ref in enumerate(ref_records):
        if i < len(metadata_list):
            entry = metadata_list[i]
            ref["source"] = entry.get("source", ref.get("source", ""))
            ref["license"] = entry.get("license", ref.get("license", ""))

    # Extract query embeddings
    logger.info("Extracting query embeddings...")
    query_dir = DATASET_BASE / "query"
    query_embeddings: dict[str, list[tuple[str, npt.NDArray[np.float32]]]] = {}

    for person in people:
        person_id = person["person_id"]
        person_query_dir = query_dir / person_id
        if not person_query_dir.exists():
            continue
        embeddings = []
        for img_path in sorted(person_query_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            emb = extract_embedding(model, img_path)
            if emb is not None:
                embeddings.append((str(img_path), emb))
        if embeddings:
            query_embeddings[person_id] = embeddings

    total_queries = sum(len(v) for v in query_embeddings.values())
    logger.info("Extracted embeddings for %d queries across %d identities.", total_queries, len(query_embeddings))

    # Save outputs directory
    t0 = time.time()

    # =========================================================================
    # A. Build query→reference calibration pairs
    # =========================================================================
    logger.info("Building query→reference calibration pairs...")
    genuine_pairs, impostor_pairs = build_query_reference_pairs(
        query_embeddings, ref_records, ref_embeddings
    )
    logger.info("Genuine pairs: %d, Impostor pairs: %d", len(genuine_pairs), len(impostor_pairs))

    genuine_scores = np.array([p["similarity"] for p in genuine_pairs], dtype=np.float32)
    impostor_scores = np.array([p["similarity"] for p in impostor_pairs], dtype=np.float32)

    # =========================================================================
    # B. Identity-level aggregation
    # =========================================================================
    logger.info("Computing identity-level score aggregation...")
    identity_genuine = aggregate_identity_scores(genuine_pairs)
    identity_impostor = aggregate_identity_scores(impostor_pairs)

    # Identity-level genuine scores (max per query)
    id_genuine_scores = []
    for query_img, id_sims in identity_genuine.items():
        for pid, sim in id_sims.items():
            id_genuine_scores.append(sim)
    id_genuine_arr = np.array(id_genuine_scores, dtype=np.float32) if id_genuine_scores else np.array([], dtype=np.float32)

    # Identity-level impostor scores (max per query per identity)
    id_impostor_scores = []
    for query_img, id_sims in identity_impostor.items():
        for pid, sim in id_sims.items():
            id_impostor_scores.append(sim)
    id_impostor_arr = np.array(id_impostor_scores, dtype=np.float32) if id_impostor_scores else np.array([], dtype=np.float32)

    # =========================================================================
    # C. ROC-AUC and EER
    # =========================================================================
    logger.info("Computing ROC-AUC and EER (image-level)...")
    roc_data = compute_roc_auc(genuine_scores, impostor_scores)
    eer_result = compute_eer(genuine_scores, impostor_scores)

    # Identity-level ROC-AUC and EER
    roc_id = compute_roc_auc(id_genuine_arr, id_impostor_arr) if len(id_genuine_arr) > 0 and len(id_impostor_arr) > 0 else {}
    eer_id = compute_eer(id_genuine_arr, id_impostor_arr) if len(id_genuine_arr) > 0 and len(id_impostor_arr) > 0 else {}

    logger.info("Image-level ROC-AUC: %.4f, EER: %.4f at threshold %.4f",
                roc_data["auc"], eer_result["eer"], eer_result["threshold"])
    if roc_id:
        logger.info("Identity-level ROC-AUC: %.4f, EER: %.4f at threshold %.4f",
                     roc_id["auc"], eer_id["eer"], eer_id["threshold"])

    # =========================================================================
    # D. Operating points
    # =========================================================================
    logger.info("Computing operating points...")
    operating_points = compute_operating_points(genuine_scores, impostor_scores)
    operating_points_id = compute_operating_points(id_genuine_arr, id_impostor_arr) if len(id_genuine_arr) > 0 and len(id_impostor_arr) > 0 else []

    # =========================================================================
    # E. Hard negatives
    # =========================================================================
    logger.info("Analyzing hard negatives...")
    hard_negatives = analyze_hard_negatives(query_embeddings, ref_records, ref_embeddings)
    logger.info("Found %d queries with strongest impostors.", len(hard_negatives))

    # =========================================================================
    # F. Identity error analysis
    # =========================================================================
    logger.info("Running identity error analysis...")
    id_errors = identity_error_analysis(genuine_pairs, impostor_pairs, eer_result["threshold"])

    # =========================================================================
    # G. Cross-validation stability
    # =========================================================================
    logger.info("Running cross-validation stability analysis...")
    cv_stability = cross_validation_stability(query_embeddings, ref_records, ref_embeddings, n_folds=5)
    logger.info("CV stability: EER threshold mean=%.4f std=%.4f", 
                cv_stability["eer_stability"]["mean"], cv_stability["eer_stability"]["std"])

    # =========================================================================
    # H. Generate plots
    # =========================================================================
    logger.info("Generating plots...")
    plots_dir = OUTPUT_BASE / "plots"
    generate_plots(genuine_scores, impostor_scores, roc_data, operating_points, eer_result, plots_dir)

    # =========================================================================
    # I. Write artifacts
    # =========================================================================
    logger.info("Writing artifacts...")

    # Genuine scores CSV
    write_csv(OUTPUT_BASE / "genuine_scores.csv", genuine_pairs,
              ["query_person_id", "query_image", "ref_person_id", "ref_image", "ref_vector_id", "similarity", "source", "license"])

    # Impostor scores CSV
    write_csv(OUTPUT_BASE / "impostor_scores.csv", impostor_pairs,
              ["query_person_id", "query_image", "ref_person_id", "ref_image", "ref_vector_id", "similarity", "source", "license"])

    # Hard negatives CSV
    hn_fields = ["query_person_id", "query_image", "impostor_person_id", "impostor_image",
                 "impostor_vector_id", "similarity", "source", "license"]
    write_csv(OUTPUT_BASE / "hard_negatives.csv", hard_negatives, hn_fields)

    # Operating points CSV
    op_fields = ["point_type", "threshold", "far", "frr", "tpr", "tnr", "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    write_csv(OUTPUT_BASE / "threshold_operating_points.csv", operating_points, op_fields)

    # Identity error analysis CSV
    ie_fields = ["person_id", "n_queries", "genuine_mean", "genuine_min", "genuine_max",
                 "false_rejects", "strongest_impostor", "impostors_above_threshold"]
    write_csv(OUTPUT_BASE / "identity_error_analysis.csv", id_errors, ie_fields)

    # Score distribution CSV
    all_sims = np.concatenate([genuine_scores, impostor_scores])
    labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
    score_dist_rows = [{"similarity": round(float(s), 6), "label": "genuine" if l == 1 else "impostor"}
                       for s, l in zip(all_sims, labels)]
    write_csv(OUTPUT_BASE / "score_distribution.csv", score_dist_rows, ["similarity", "label"])

    # Retrieval results CSV (from hard negatives - per query)
    retrieval_rows = []
    for hn in hard_negatives:
        retrieval_rows.append({
            "query_person_id": hn["query_person_id"],
            "query_image": hn["query_image"],
            "strongest_impostor_person": hn["impostor_person_id"],
            "strongest_impostor_image": hn["impostor_image"],
            "strongest_impostor_similarity": hn["similarity"],
        })
    write_csv(OUTPUT_BASE / "retrieval_results.csv", retrieval_rows,
              ["query_person_id", "query_image", "strongest_impostor_person", "strongest_impostor_image", "strongest_impostor_similarity"])

    # Hard negatives MD
    with open(OUTPUT_BASE / "hard_negatives.md", "w", encoding="utf-8") as f:
        f.write("# Hard Negative Analysis\n\n")
        f.write(f"- Total queries analyzed: {len(hard_negatives)}\n")
        f.write(f"- Global maximum impostor similarity: {hard_negatives[0]['similarity']:.6f}\n")
        if hard_negatives:
            top = hard_negatives[0]
            f.write(f"- Pair: {top['query_person_id']} → {top['impostor_person_id']} (similarity={top['similarity']:.6f})\n")
        f.write("\n## Top 20 Hard Negatives\n\n")
        f.write("| Rank | Query | Impostor | Similarity |\n")
        f.write("|------|-------|----------|------------|\n")
        for i, hn in enumerate(hard_negatives[:20]):
            f.write(f"| {i+1} | {hn['query_person_id']} | {hn['impostor_person_id']} | {hn['similarity']:.6f} |\n")

    # Hard negatives JSON
    with open(OUTPUT_BASE / "hard_negatives.json", "w", encoding="utf-8") as f:
        json.dump(hard_negatives[:50], f, indent=2)

    # Threshold selection MD
    with open(OUTPUT_BASE / "threshold_selection.md", "w", encoding="utf-8") as f:
        f.write("# Threshold Selection\n\n")
        f.write("## Operating Points (Image-Level)\n\n")
        f.write("| Point | Threshold | FAR | FRR | TPR | Precision | Recall | F1 |\n")
        f.write("|-------|-----------|-----|-----|-----|-----------|--------|----|\n")
        for p in operating_points:
            f.write(f"| {p['point_type']} | {p['threshold']:.6f} | {p['far']:.6f} | {p['frr']:.6f} | {p['tpr']:.6f} | {p['precision']:.6f} | {p['recall']:.6f} | {p['f1']:.6f} |\n")
        if operating_points_id:
            f.write("\n## Operating Points (Identity-Level)\n\n")
            f.write("| Point | Threshold | FAR | FRR | TPR | Precision | Recall | F1 |\n")
            f.write("|-------|-----------|-----|-----|-----|-----------|--------|----|\n")
            for p in operating_points_id:
                f.write(f"| {p['point_type']} | {p['threshold']:.6f} | {p['far']:.6f} | {p['frr']:.6f} | {p['tpr']:.6f} | {p['precision']:.6f} | {p['recall']:.6f} | {p['f1']:.6f} |\n")

        f.write("\n## Threshold Stability (5-Fold CV)\n\n")
        for key, label in [("eer_stability", "EER"), ("far_1pct_stability", "FAR<=1%"), ("far_0_5pct_stability", "FAR<=0.5%")]:
            stats = cv_stability[key]
            f.write(f"- **{label}**: mean={stats['mean']:.6f}, std={stats['std']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}\n")

        f.write("\n## Recommendation\n\n")
        f.write("**NOT production-approved** — calibration candidate only.\n")
        f.write(f"- EER threshold: {eer_result['threshold']:.6f} (EER={eer_result['eer']:.6f})\n")
        f.write(f"- Youden's J threshold: {operating_points[0]['threshold']:.6f} (FAR={operating_points[0]['far']:.6f}, FRR={operating_points[0]['frr']:.6f})\n")
        f.write("- This calibration estimates behavior on the current celebrity/reference dataset and is not sufficient by itself to establish production threshold performance.\n")

    # Score distribution summary
    with open(OUTPUT_BASE / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "phase": "13.7",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seed": SEED,
            "image_level": {
                "genuine_pairs": len(genuine_pairs),
                "impostor_pairs": len(impostor_pairs),
                "genuine_scores": score_stats(genuine_scores),
                "impostor_scores": score_stats(impostor_scores),
                "roc_auc": roc_data["auc"],
                "eer": eer_result,
                "operating_points": operating_points,
            },
            "identity_level": {
                "genuine_pairs": len(id_genuine_arr),
                "impostor_pairs": len(id_impostor_arr),
                "genuine_scores": score_stats(id_genuine_arr) if len(id_genuine_arr) > 0 else {},
                "impostor_scores": score_stats(id_impostor_arr) if len(id_impostor_arr) > 0 else {},
                "roc_auc": roc_id.get("auc", None),
                "eer": eer_id if eer_id else None,
                "operating_points": operating_points_id,
            },
            "hard_negatives": {
                "count": len(hard_negatives),
                "global_max_impostor": hard_negatives[0]["similarity"] if hard_negatives else 0.0,
                "top_20": hard_negatives[:20],
            },
            "identity_errors": id_errors,
            "cv_stability": cv_stability,
            "dataset_info": {
                "total_reference": len(ref_records),
                "total_queries": total_queries,
                "n_identities": len(query_embeddings),
                "source_distribution": {},
            },
        }, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("Phase 13.7 complete. Artifacts saved to %s (%.1fs)", OUTPUT_BASE, elapsed)


if __name__ == "__main__":
    main()
