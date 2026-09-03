"""Phase 13.10 — Realistic Stress & Robustness Validation."""
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
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
DATASET_DIR = Path("datasets/non_celebrity-v1")
REF_DIR = DATASET_DIR / "reference"
HELD_DIR = DATASET_DIR / "held_out"
INDEX_DIR = DATASET_DIR / "search_index"
RESULTS_DIR = Path("outputs/phase13_10")
VISUAL_DIR = RESULTS_DIR / "visual_cases"
STRESS_DIR = Path("datasets/non_celebrity-v1/stress_variants")

CELEBRITY_THRESHOLDS = {
    "eer": 0.0457, "youden_j": 0.2301, "far_5pct": 0.1039,
    "far_1pct": 0.1554, "far_0_5pct": 0.1816, "far_0_1pct": 0.2301,
}

STRESS_CONDITIONS = {
    # A. Glasses
    "glasses_rimless": "glasses",
    # B. Pose
    "pose_yaw_left_10": "pose",
    # C. Lighting
    "lighting_underexpose": "lighting",
    # D. Quality
    "quality_jpeg_q30": "quality",
    # E. Appearance
    "appearance_color_jitter": "appearance",
    # F. Combined
    "combined_mild": "combined",
}


def _load_face_service():
    sys.path.insert(0, ".")
    from services.face_service import FaceService
    return FaceService()


# ============================================================
# TRANSFORMATIONS
# ============================================================

def _pil_to_cv2(pil_img):
    arr = np.array(pil_img)
    if len(arr.shape) == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr

def _cv2_to_pil(cv2_img):
    if len(cv2_img.shape) == 3 and cv2_img.shape[2] == 3:
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))
    return Image.fromarray(cv2_img)


def _apply_glasses_rimless(img):
    """Subtle semi-transparent rimless glasses overlay."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = w // 2, int(h * 0.42)
    lens_w, lens_h = int(w * 0.18), int(h * 0.12)
    left_cx = cx - int(w * 0.13)
    right_cx = cx + int(w * 0.13)
    for lc in [left_cx, right_cx]:
        draw.ellipse([lc - lens_w // 2, cy - lens_h // 2, lc + lens_w // 2, cy + lens_h // 2],
                      outline=(80, 80, 80, 90), width=2)
    draw.line([left_cx + lens_w // 2, cy, right_cx - lens_w // 2, cy], fill=(80, 80, 80, 70), width=2)
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    return composited.convert("RGB")


def _apply_glasses_rimmed(img):
    """Thick-rimmed glasses overlay."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = w // 2, int(h * 0.42)
    lens_w, lens_h = int(w * 0.19), int(h * 0.13)
    left_cx = cx - int(w * 0.14)
    right_cx = cx + int(w * 0.14)
    for lc in [left_cx, right_cx]:
        draw.ellipse([lc - lens_w // 2, cy - lens_h // 2, lc + lens_w // 2, cy + lens_h // 2],
                      outline=(30, 30, 30, 200), width=4)
    draw.line([left_cx + lens_w // 2, cy, right_cx - lens_w // 2, cy], fill=(30, 30, 30, 180), width=3)
    draw.line([left_cx - lens_w // 2, cy - 2, left_cx - lens_w // 2 - int(w * 0.04), cy - 5],
              fill=(30, 30, 30, 180), width=3)
    draw.line([right_cx + lens_w // 2, cy - 2, right_cx + lens_w // 2 + int(w * 0.04), cy - 5],
              fill=(30, 30, 30, 180), width=3)
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    return composited.convert("RGB")


def _apply_glasses_reflection(img):
    """Glasses with glare spots on lenses."""
    w, h = img.size
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = w // 2, int(h * 0.42)
    lens_w, lens_h = int(w * 0.18), int(h * 0.12)
    left_cx = cx - int(w * 0.13)
    right_cx = cx + int(w * 0.13)
    for lc in [left_cx, right_cx]:
        draw.ellipse([lc - lens_w // 2, cy - lens_h // 2, lc + lens_w // 2, cy + lens_h // 2],
                      outline=(80, 80, 80, 120), width=2)
        rx = lc + lens_w // 6
        ry = cy - lens_h // 6
        draw.ellipse([rx - 8, ry - 6, rx + 8, ry + 6], fill=(255, 255, 255, 140))
    draw.line([left_cx + lens_w // 2, cy, right_cx - lens_w // 2, cy], fill=(80, 80, 80, 80), width=2)
    img_rgba = img.convert("RGBA")
    composited = Image.alpha_composite(img_rgba, overlay)
    return composited.convert("RGB")


def _apply_eye_degradation(img):
    """Mild blur + brightness reduction on eye region to simulate occlusion."""
    w, h = img.size
    arr = np.array(img).copy()
    cx, cy = w // 2, int(h * 0.42)
    roi_x1 = max(0, cx - int(w * 0.25))
    roi_x2 = min(w, cx + int(w * 0.25))
    roi_y1 = max(0, cy - int(h * 0.1))
    roi_y2 = min(h, cy + int(h * 0.1))
    roi = arr[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_blur = cv2.GaussianBlur(roi, (7, 7), 2.0)
    roi_dark = np.clip(roi_blur.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)
    arr[roi_y1:roi_y2, roi_x1:roi_x2] = roi_dark
    return Image.fromarray(arr)


def _apply_pose_yaw(img, degrees):
    """Simulate yaw rotation via affine transform."""
    w, h = img.size
    shift = int(w * degrees / 180.0 * 0.3)
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[shift, 0], [w + shift, 0], [shift, h]])
    M = cv2.getAffineTransform(pts1, pts2)
    arr = cv2.warpAffine(np.array(img), M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(arr)


def _apply_pose_pitch(img, degrees):
    """Simulate pitch rotation via affine transform."""
    w, h = img.size
    shift = int(h * degrees / 180.0 * 0.3)
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[0, shift], [w, shift], [0, h + shift]])
    M = cv2.getAffineTransform(pts1, pts2)
    arr = cv2.warpAffine(np.array(img), M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(arr)


def _apply_underexpose(img):
    """Reduce brightness to simulate underexposure."""
    arr = np.array(img).astype(np.float32) * 0.6
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_overexpose(img):
    """Increase brightness to simulate overexposure."""
    arr = np.array(img).astype(np.float32) * 1.4
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_uneven(img):
    """Simulate uneven illumination with a horizontal gradient."""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    grad = np.linspace(0.7, 1.3, w).reshape(1, w, 1)
    arr = arr * grad
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_shadow(img):
    """Simulate a mild shadow on the left side."""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    grad = np.linspace(0.7, 1.0, w).reshape(1, w, 1)
    arr = arr * grad
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_jpeg_q30(img):
    """Realistic JPEG compression at quality 30."""
    buf = cv2.imencode(".jpg", np.array(img), [cv2.IMWRITE_JPEG_QUALITY, 30])[1]
    return Image.fromarray(cv2.imdecode(buf, cv2.IMREAD_COLOR))


def _apply_downscale_50(img):
    """Downscale to 50% and upscale back."""
    w, h = img.size
    small = img.resize((w // 2, h // 2), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def _apply_blur_mild(img):
    """Mild Gaussian blur (sigma=1.5)."""
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))


def _apply_noise_mild(img):
    """Mild Gaussian noise (sigma=5)."""
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(SEED).normal(0, 5, arr.shape)
    arr = arr + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _apply_color_jitter(img):
    """Random brightness/contrast/saturation jitter (controlled seed)."""
    rng = random.Random(SEED)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.15))
    return img


def _apply_warm_shift(img):
    """Warm color temperature shift."""
    arr = np.array(img).astype(np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 1.08, 0, 255)  # R
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 0.92, 0, 255)  # B
    return Image.fromarray(arr.astype(np.uint8))


def _apply_cool_shift(img):
    """Cool color temperature shift."""
    arr = np.array(img).astype(np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * 0.92, 0, 255)  # R
    arr[:, :, 2] = np.clip(arr[:, :, 2] * 1.08, 0, 255)  # B
    return Image.fromarray(arr.astype(np.uint8))


def _apply_combined_mild(img):
    """Combined mild: JPEG Q50 + slight blur + slight noise."""
    # JPEG Q50
    buf = cv2.imencode(".jpg", np.array(img), [cv2.IMWRITE_JPEG_QUALITY, 50])[1]
    img = Image.fromarray(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    # Slight blur
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    # Slight noise
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(SEED).normal(0, 3, arr.shape)
    arr = arr + noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


TRANSFORMS = {
    "glasses_rimless": _apply_glasses_rimless,
    "glasses_rimmed": _apply_glasses_rimmed,
    "glasses_reflection": _apply_glasses_reflection,
    "eye_degradation": _apply_eye_degradation,
    "pose_yaw_left_10": lambda img: _apply_pose_yaw(img, -10),
    "pose_yaw_right_10": lambda img: _apply_pose_yaw(img, 10),
    "pose_pitch_up_5": lambda img: _apply_pose_pitch(img, -5),
    "pose_pitch_down_5": lambda img: _apply_pose_pitch(img, 5),
    "lighting_underexpose": _apply_underexpose,
    "lighting_overexpose": _apply_overexpose,
    "lighting_uneven": _apply_uneven,
    "lighting_shadow": _apply_shadow,
    "quality_jpeg_q30": _apply_jpeg_q30,
    "quality_downscale_50": _apply_downscale_50,
    "quality_blur_mild": _apply_blur_mild,
    "quality_noise_mild": _apply_noise_mild,
    "appearance_color_jitter": _apply_color_jitter,
    "appearance_warm_shift": _apply_warm_shift,
    "appearance_cool_shift": _apply_cool_shift,
    "combined_mild": _apply_combined_mild,
}


# ============================================================
# PIPELINE
# ============================================================

def apply_stress_transforms() -> dict[str, Path]:
    """Apply all stress transforms to held-out images. Returns {condition: output_dir}."""
    STRESS_DIR.mkdir(parents=True, exist_ok=True)
    (VISUAL_DIR / "hard_negatives").mkdir(parents=True, exist_ok=True)
    (VISUAL_DIR / "false_rejects").mkdir(parents=True, exist_ok=True)
    (VISUAL_DIR / "false_accepts").mkdir(parents=True, exist_ok=True)

    cond_dirs = {}
    for cond_name in STRESS_CONDITIONS:
        out_dir = STRESS_DIR / cond_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cond_dirs[cond_name] = out_dir

    # For each held-out image, apply each transform
    held_identities = sorted([d.name for d in HELD_DIR.iterdir() if d.is_dir()])
    total_images = 0
    for pid in held_identities:
        pid_dir = HELD_DIR / pid
        for img_file in sorted(pid_dir.glob("*.jpg")):
            img = Image.open(img_file).convert("RGB")
            for cond_name in STRESS_CONDITIONS:
                transform_fn = TRANSFORMS[cond_name]
                out_dir = cond_dirs[cond_name]
                out_pid_dir = out_dir / pid
                out_pid_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_pid_dir / img_file.name
                if not out_path.exists():
                    transformed = transform_fn(img)
                    transformed.save(out_path, quality=95)
            total_images += 1

    logger.info("Applied %d transforms to %d images", len(STRESS_CONDITIONS), total_images)
    return cond_dirs


def extract_embeddings_for_condition(cond_dir: Path, face_model) -> dict[str, list]:
    """Extract embeddings for all images in a stress condition directory.
    Returns {person_id: [(img_path, embedding), ...]}."""
    embeddings = defaultdict(list)
    identities = sorted([d.name for d in cond_dir.iterdir() if d.is_dir()])
    total = 0
    skipped = 0
    for pid in identities:
        pid_dir = cond_dir / pid
        for img_file in sorted(pid_dir.glob("*.jpg")):
            face = face_model.get(cv2.imread(str(img_file)))
            if face is None or len(face) == 0:
                skipped += 1
                continue
            if len(face) != 1:
                skipped += 1
                continue
            emb = face[0].normed_embedding.ravel().astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            embeddings[pid].append((str(img_file), emb))
            total += 1
    logger.info("  Extracted %d embeddings, skipped %d", total, skipped)
    return dict(embeddings)


def run_evaluation_for_condition(cond_name, cond_embeddings, ref_metadata, ref_embeddings):
    """Run full evaluation for a single stress condition."""
    from phase13_7_2_calibration import (
        build_all_pairs, aggregate_identity_scores,
        image_level_hard_negatives, identity_level_hard_negatives,
        compute_operating_points, compute_eer, compute_roc_auc,
        score_stats,
    )

    ref_records = []
    for r in ref_metadata:
        ref_records.append({
            "person_id": r["person_id"],
            "vector_id": r["vector_id"],
            "image_path": r["image_path"],
        })

    genuines, impostors = build_all_pairs(cond_embeddings, ref_records, ref_embeddings)
    gen_scores = np.array([p["similarity"] for p in genuines], dtype=np.float32) if genuines else np.array([], dtype=np.float32)
    imp_scores = np.array([p["similarity"] for p in impostors], dtype=np.float32) if impostors else np.array([], dtype=np.float32)

    if len(gen_scores) == 0 or len(imp_scores) == 0:
        return None

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
    id_hn = identity_level_hard_negatives(cond_embeddings, ref_records, ref_embeddings)

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

    return {
        "condition": cond_name,
        "dataset": {
            "identities": len(cond_embeddings),
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
    }


def run_gallery_size_analysis(cond_name, cond_embeddings, ref_metadata, ref_embeddings):
    """Gallery-size sensitivity for a stress condition."""
    from phase13_7_2_calibration import build_all_pairs, compute_roc_auc, compute_eer

    ref_records = []
    for r in ref_metadata:
        ref_records.append({
            "person_id": r["person_id"],
            "vector_id": r["vector_id"],
            "image_path": r["image_path"],
        })

    all_pids = sorted(set(r["person_id"] for r in ref_records))
    refs_by_pid = defaultdict(list)
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
        g, im = build_all_pairs(cond_embeddings, recs, sub_emb)
        gs_gen = np.array([p["similarity"] for p in g], dtype=np.float32) if g else np.array([], dtype=np.float32)
        gs_imp = np.array([p["similarity"] for p in im], dtype=np.float32) if im else np.array([], dtype=np.float32)
        gallery_sizes[str(gs)] = {
            "reference_vectors": len(recs),
            "genuine_pairs": len(g), "impostor_pairs": len(im),
            "image_roc_auc": compute_roc_auc(gs_gen, gs_imp)["auc"],
            "image_eer": compute_eer(gs_gen, gs_imp)["eer"],
            "global_max_impostor": float(gs_imp.max()) if len(gs_imp) > 0 else 0.0,
        }
    return gallery_sizes


def save_visual_cases(cond_name, cond_embeddings, ref_metadata, ref_embeddings, results):
    """Save image pairs for strongest hard negatives and representative false rejects."""
    from phase13_7_2_calibration import build_all_pairs, aggregate_identity_scores

    ref_records = []
    for r in ref_metadata:
        ref_records.append({
            "person_id": r["person_id"],
            "vector_id": r["vector_id"],
            "image_path": r["image_path"],
        })

    genuines, impostors = build_all_pairs(cond_embeddings, ref_records, ref_embeddings)
    imp_sorted = sorted(impostors, key=lambda x: x["similarity"], reverse=True)

    # Save top hard negative pairs
    case_dir = VISUAL_DIR / "hard_negatives" / cond_name
    case_dir.mkdir(parents=True, exist_ok=True)
    for i, pair in enumerate(imp_sorted[:5]):
        q_path = Path(pair["query_image"])
        r_path = Path(pair["ref_image"])
        if q_path.exists():
            shutil.copy2(q_path, case_dir / f"hn{i}_query_{q_path.name}")
        if r_path.exists():
            shutil.copy2(r_path, case_dir / f"hn{i}_ref_{r_path.name}")

    # Save false rejects (if any)
    threshold = CELEBRITY_THRESHOLDS["youden_j"]
    false_rejects = [p for p in genuines if p["similarity"] < threshold]
    if false_rejects:
        fr_dir = VISUAL_DIR / "false_rejects" / cond_name
        fr_dir.mkdir(parents=True, exist_ok=True)
        fr_sorted = sorted(false_rejects, key=lambda x: x["similarity"])
        for i, pair in enumerate(fr_sorted[:5]):
            q_path = Path(pair["query_image"])
            r_path = Path(pair["ref_image"])
            if q_path.exists():
                shutil.copy2(q_path, fr_dir / f"fr{i}_query_{q_path.name}")
            if r_path.exists():
                shutil.copy2(r_path, fr_dir / f"fr{i}_ref_{r_path.name}")


def compute_degradation(baseline, stress_result):
    """Compute delta metrics between baseline and stress condition."""
    if stress_result is None:
        return None

    base_img = baseline["image_level"]
    stress_img = stress_result["image_level"]
    base_fixed = baseline["fixed_thresholds"]
    stress_fixed = stress_result["fixed_thresholds"]

    deg = {
        "delta_roc_auc": stress_img["roc_auc"] - base_img["roc_auc"],
        "delta_eer": stress_img["eer"]["eer"] - base_img["eer"]["eer"],
        "delta_max_impostor": stress_img["global_max_impostor"] - base_img["global_max_impostor"],
    }

    for name in CELEBRITY_THRESHOLDS:
        if name in stress_fixed and name in base_fixed:
            deg[f"delta_far_{name}"] = stress_fixed[name]["far"] - base_fixed[name]["far"]
            deg[f"delta_frr_{name}"] = stress_fixed[name]["frr"] - base_fixed[name]["frr"]
            deg[f"delta_f1_{name}"] = stress_fixed[name]["f1"] - base_fixed[name]["f1"]

    return deg


def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 13.10 — REALISTIC STRESS & ROBUSTNESS VALIDATION")
    logger.info("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ["hard_negatives", "false_accepts", "false_rejects"]:
        (VISUAL_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Load existing reference data
    logger.info("\n--- Loading Reference Data ---")
    model = _load_face_service().get_model()

    import faiss
    ref_index = faiss.read_index(str(INDEX_DIR / "reference_index.faiss"))
    ref_embeddings = ref_index.reconstruct_n(0, ref_index.ntotal)
    with open(INDEX_DIR / "metadata.json") as f:
        ref_metadata = json.load(f)
    logger.info("Reference gallery: %d vectors", ref_embeddings.shape[0])

    # Load baseline results
    with open("outputs/phase13_9/calibration_results.json") as f:
        baseline = json.load(f)
    logger.info("Baseline loaded: ROC-AUC=%.4f, EER=%.6f, max_imp=%.4f",
                baseline["image_level"]["roc_auc"],
                baseline["image_level"]["eer"]["eer"],
                baseline["image_level"]["global_max_impostor"])

    # Apply stress transforms
    logger.info("\n--- Applying Stress Transforms ---")
    cond_dirs = apply_stress_transforms()

    # Extract embeddings and evaluate each condition
    logger.info("\n--- Evaluating Stress Conditions ---")
    stress_results = {}
    gallery_size_results = {}
    integrity_results = {}

    for cond_name in sorted(STRESS_CONDITIONS.keys()):
        cond_dir = cond_dirs[cond_name]
        logger.info("\n  [%s] Extracting embeddings...", cond_name)
        cond_embeddings = extract_embeddings_for_condition(cond_dir, model)

        if not cond_embeddings:
            logger.warning("  [%s] No embeddings extracted, skipping", cond_name)
            continue

        # Integrity check: no reference/query leakage
        ref_paths = set(r["image_path"] for r in ref_metadata)
        cond_paths = set()
        for pid, pairs in cond_embeddings.items():
            for path, _ in pairs:
                cond_paths.add(path)
        leakage = ref_paths & cond_paths
        integrity_results[cond_name] = {
            "total_images": sum(len(v) for v in cond_embeddings.values()),
            "identities": len(cond_embeddings),
            "leakage": len(leakage),
        }

        logger.info("  [%s] Evaluating...", cond_name)
        result = run_evaluation_for_condition(cond_name, cond_embeddings, ref_metadata, ref_embeddings)
        if result is None:
            logger.warning("  [%s] Evaluation returned None", cond_name)
            continue
        stress_results[cond_name] = result

        # Gallery size analysis
        logger.info("  [%s] Gallery-size analysis...", cond_name)
        gallery_size_results[cond_name] = run_gallery_size_analysis(
            cond_name, cond_embeddings, ref_metadata, ref_embeddings
        )

        # Visual cases
        save_visual_cases(cond_name, cond_embeddings, ref_metadata, ref_embeddings, result)

        logger.info("  [%s] ROC-AUC=%.4f, EER=%.6f, max_imp=%.4f",
                     cond_name,
                     result["image_level"]["roc_auc"],
                     result["image_level"]["eer"]["eer"],
                     result["image_level"]["global_max_impostor"])

    # Degradation analysis
    logger.info("\n--- Computing Degradation ---")
    degradation = {}
    for cond_name, result in stress_results.items():
        degradation[cond_name] = compute_degradation(baseline, result)

    # Save all results
    logger.info("\n--- Saving Results ---")
    with open(RESULTS_DIR / "baseline_results.json", "w") as f:
        json.dump(baseline, f, indent=2, default=str)
    with open(RESULTS_DIR / "stress_results.json", "w") as f:
        json.dump(stress_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "gallery_size_results.json", "w") as f:
        json.dump(gallery_size_results, f, indent=2, default=str)
    with open(RESULTS_DIR / "hard_negative_analysis.json", "w") as f:
        # Collect all hard negatives across conditions
        all_hn = {}
        for cond_name, result in stress_results.items():
            all_hn[cond_name] = {
                "image_level_top5": result["hard_negatives"]["image_level_top5"],
                "identity_level_top5": result["hard_negatives"]["identity_level_top5"],
            }
        json.dump(all_hn, f, indent=2, default=str)
    with open(RESULTS_DIR / "failure_analysis.json", "w") as f:
        json.dump(degradation, f, indent=2, default=str)
    with open(RESULTS_DIR / "integrity_checks.json", "w") as f:
        json.dump(integrity_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE in %.0fs", elapsed)

    # Summary
    logger.info("\n--- SUMMARY ---")
    logger.info("Baseline: ROC-AUC=%.4f, EER=%.6f, max_imp=%.4f",
                baseline["image_level"]["roc_auc"],
                baseline["image_level"]["eer"]["eer"],
                baseline["image_level"]["global_max_impostor"])
    for cond_name in sorted(stress_results.keys()):
        r = stress_results[cond_name]
        d = degradation.get(cond_name, {})
        logger.info("  %s: ROC-AUC=%.4f (Δ%.4f), EER=%.6f (Δ%.6f), max_imp=%.4f (Δ%.4f)",
                     cond_name,
                     r["image_level"]["roc_auc"],
                     d.get("delta_roc_auc", 0),
                     r["image_level"]["eer"]["eer"],
                     d.get("delta_eer", 0),
                     r["image_level"]["global_max_impostor"],
                     d.get("delta_max_impostor", 0))

    # Find worst conditions
    if degradation:
        worst_roc = min(degradation.items(), key=lambda x: x[1].get("delta_roc_auc", 0) if x[1] else 0)
        worst_eer = max(degradation.items(), key=lambda x: x[1].get("delta_eer", 0) if x[1] else 0)
        worst_imp = max(degradation.items(), key=lambda x: x[1].get("delta_max_impostor", 0) if x[1] else 0)
        logger.info("\nWorst ROC-AUC: %s (Δ%.4f)", worst_roc[0], worst_roc[1].get("delta_roc_auc", 0))
        logger.info("Worst EER: %s (Δ%.6f)", worst_eer[0], worst_eer[1].get("delta_eer", 0))
        logger.info("Worst max impostor: %s (Δ%.4f)", worst_imp[0], worst_imp[1].get("delta_max_impostor", 0))


if __name__ == "__main__":
    main()
