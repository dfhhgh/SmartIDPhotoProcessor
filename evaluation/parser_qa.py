"""
Parser QA evaluation: runs production ORIGINAL and FUSED parser modes
on the same aligned face and generates human-reviewable comparison reports.

This is a qualitative + behavioral production QA tool.
It does NOT claim IoU/Dice accuracy since no ground-truth masks exist
for the production-like test images.

Recovery heuristics are QA signals, not segmentation accuracy metrics.
All verdicts require human review of comparison visualizations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.parser_mode import ParserMode
from config.settings import Settings
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from pipeline.aligner import FaceAligner
from pipeline.cropper import FaceCropper
from pipeline.detector import FaceDetector
from pipeline.face_coordinate_transformer import FaceCoordinateTransformer
from pipeline.selector import FaceSelector
from services.face_parser_service import FaceParserService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_CLASSES = {
    FacePart.LEFT_BROW: "LEFT_BROW",
    FacePart.RIGHT_BROW: "RIGHT_BROW",
    FacePart.LEFT_EYE: "LEFT_EYE",
    FacePart.RIGHT_EYE: "RIGHT_EYE",
    FacePart.EYE_GLASS: "EYE_GLASS",
}

TARGET_CLASS_IDS = {2, 3, 4, 5, 6}

_TARGET_COLORS: dict[int, tuple[int, int, int]] = {
    FacePart.LEFT_BROW: (40, 40, 200),
    FacePart.RIGHT_BROW: (200, 40, 40),
    FacePart.LEFT_EYE: (255, 0, 0),
    FacePart.RIGHT_EYE: (0, 0, 255),
    FacePart.EYE_GLASS: (0, 255, 255),
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _sha256_prefix(path: Path, length: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def _class_bbox(mask: np.ndarray, class_id: int) -> dict | None:
    ys, xs = np.where(mask == class_id)
    if len(xs) == 0:
        return None
    x_min, y_min, x_max, y_max = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
            "width": x_max - x_min + 1, "height": y_max - y_min + 1}


def _connected_component_analysis(mask: np.ndarray, class_id: int) -> dict:
    binary = (mask == class_id).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    n_components = n_labels - 1
    if n_components == 0:
        return {"component_count": 0, "largest_component_area": 0, "largest_component_ratio": 0.0}
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(areas.max())
    total_area = int(binary.sum())
    largest_ratio = largest_area / total_area if total_area > 0 else 0.0
    return {"component_count": n_components, "largest_component_area": largest_area,
            "largest_component_ratio": round(largest_ratio, 4)}


def _suspicious_fragmentation(mask: np.ndarray, class_id: int) -> bool:
    cc = _connected_component_analysis(mask, class_id)
    pixel_count = int(np.sum(mask == class_id))
    if pixel_count < 5:
        return False
    if cc["component_count"] > 6:
        return True
    if cc["largest_component_ratio"] < 0.3 and cc["component_count"] > 2:
        return True
    return False


def _eye_glass_interaction(mask: np.ndarray, eye_id: int) -> dict:
    eye_mask = mask == eye_id
    glass_mask = mask == FacePart.EYE_GLASS
    eye_area = int(eye_mask.sum())
    glass_area = int(glass_mask.sum())
    intersection = int((eye_mask & glass_mask).sum())
    eye_in_glass = intersection / eye_area if eye_area > 0 else 0.0
    glass_covers_eye = intersection / glass_area if glass_area > 0 else 0.0
    return {"eye_area": eye_area, "glass_area": glass_area,
            "intersection": intersection, "eye_inside_glasses_ratio": round(eye_in_glass, 4),
            "glass_covers_eye_ratio": round(glass_covers_eye, 4)}


def _change_analysis(orig_mask: np.ndarray, fused_mask: np.ndarray) -> dict:
    results = {}
    total_gained = 0
    total_lost = 0
    for cls_id in TARGET_CLASS_IDS:
        orig_px = orig_mask == cls_id
        fused_px = fused_mask == cls_id
        gained = int((fused_px & ~orig_px).sum())
        lost = int((orig_px & ~fused_px).sum())
        unchanged = int((orig_px & fused_px).sum())
        orig_count = int(orig_px.sum())
        fused_count = int(fused_px.sum())
        gain_ratio = gained / orig_count if orig_count > 0 else (1.0 if fused_count > 0 else 0.0)
        loss_ratio = lost / orig_count if orig_count > 0 else 0.0
        results[cls_id] = {"gained": gained, "lost": lost, "unchanged": unchanged,
                           "gain_ratio": round(gain_ratio, 4), "loss_ratio": round(loss_ratio, 4)}
        total_gained += gained
        total_lost += lost
    return {"per_class": results, "total_gained": total_gained, "total_lost": total_lost}


def _target_mask_colored(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in _TARGET_COLORS.items():
        out[mask == cls_id] = color
    return out


def _change_mask(orig_mask: np.ndarray, fused_mask: np.ndarray) -> np.ndarray:
    h, w = orig_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in TARGET_CLASS_IDS:
        orig_px = orig_mask == cls_id
        fused_px = fused_mask == cls_id
        out[fused_px & ~orig_px] = (0, 255, 0)
        out[orig_px & ~fused_px] = (0, 0, 255)
    return out


def _compute_recovery_status(
    orig_pixels: int, fused_pixels: int, total_pixels: int,
    orig_cc: dict | None = None, fused_cc: dict | None = None,
    orig_bbox: dict | None = None, fused_bbox: dict | None = None,
) -> str:
    orig_ratio = orig_pixels / total_pixels if total_pixels > 0 else 0
    fused_ratio = fused_pixels / total_pixels if total_pixels > 0 else 0
    delta = fused_pixels - orig_pixels

    if orig_pixels < 10 and fused_pixels > 50:
        if fused_ratio > 0.20:
            return "POSSIBLE_OVERSEGMENTATION"
        return "RECOVERED"

    if orig_pixels > 10 and fused_pixels > 10:
        if delta > 20 and delta / max(orig_pixels, 1) > 0.15:
            return "IMPROVED"
        if delta < -20 and abs(delta) / max(orig_pixels, 1) > 0.15:
            return "REDUCED"
        return "UNCHANGED"

    if orig_pixels > 10 and fused_pixels < 10:
        return "REDUCED"

    if fused_ratio > 0.15:
        return "POSSIBLE_OVERSEGMENTATION"

    return "UNCHANGED"


def _parse_both_modes(aligned: np.ndarray) -> tuple[FaceParsingResult, float, FaceParsingResult, float]:
    FaceParserService._instance = None
    FaceParserService._initialized = False
    orig_svc = FaceParserService(parser_mode=ParserMode.ORIGINAL)
    t0 = time.perf_counter()
    orig_result = orig_svc.parse(aligned)
    orig_time = (time.perf_counter() - t0) * 1000.0

    FaceParserService._instance = None
    FaceParserService._initialized = False
    fused_svc = FaceParserService(parser_mode=ParserMode.FUSED)
    t0 = time.perf_counter()
    fused_result = fused_svc.parse(aligned)
    fused_time = (time.perf_counter() - t0) * 1000.0

    return orig_result, orig_time, fused_result, fused_time


# ---------------------------------------------------------------------------
# Per-class geometric analysis
# ---------------------------------------------------------------------------

@dataclass
class ClassMetrics:
    pixel_count: int = 0
    area_ratio: float = 0.0
    bbox: dict | None = None
    component_count: int = 0
    largest_component_area: int = 0
    largest_component_ratio: float = 0.0


def _compute_class_metrics(mask: np.ndarray, class_id: int, total_pixels: int) -> ClassMetrics:
    pixel_count = int(np.sum(mask == class_id))
    area_ratio = round(pixel_count / total_pixels, 6) if total_pixels > 0 else 0.0
    bbox = _class_bbox(mask, class_id)
    cc = _connected_component_analysis(mask, class_id)
    return ClassMetrics(
        pixel_count=pixel_count, area_ratio=area_ratio, bbox=bbox,
        component_count=cc["component_count"],
        largest_component_area=cc["largest_component_area"],
        largest_component_ratio=cc["largest_component_ratio"],
    )


# ---------------------------------------------------------------------------
# ImageResult
# ---------------------------------------------------------------------------

@dataclass
class ImageResult:
    image_name: str
    processing_status: str = "SUCCESS"
    error_message: str = ""

    original_left_brow_pixels: int = 0
    fused_left_brow_pixels: int = 0
    original_right_brow_pixels: int = 0
    fused_right_brow_pixels: int = 0
    original_left_eye_pixels: int = 0
    fused_left_eye_pixels: int = 0
    original_right_eye_pixels: int = 0
    fused_right_eye_pixels: int = 0
    original_eye_glass_pixels: int = 0
    fused_eye_glass_pixels: int = 0

    original_total_target_pixels: int = 0
    fused_total_target_pixels: int = 0

    original_time_ms: float = 0.0
    fused_time_ms: float = 0.0

    eye_recovery_status: str = "UNKNOWN"
    brow_recovery_status: str = "UNKNOWN"

    eye_visibility: str = "UNKNOWN"
    occlusion_type: str = "UNKNOWN"

    qa_verdict: str = "PENDING_REVIEW"
    qa_notes: str = ""

    comparison_path: str = ""

    original_metrics: dict = field(default_factory=dict)
    fused_metrics: dict = field(default_factory=dict)
    eye_glass_interaction: dict = field(default_factory=dict)
    change_analysis: dict = field(default_factory=dict)
    suggested_qa_label: str = "PENDING_REVIEW"


# ---------------------------------------------------------------------------
# ParserQA
# ---------------------------------------------------------------------------

class ParserQA:
    def __init__(self, output_dir: str) -> None:
        self._output_dir = Path(output_dir)
        self._comparisons_dir = self._output_dir / "comparisons"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._comparisons_dir.mkdir(parents=True, exist_ok=True)

        self._detector = FaceDetector()
        self._selector = FaceSelector()
        self._cropper = FaceCropper()
        self._transformer = FaceCoordinateTransformer()
        self._aligner = FaceAligner()

    def _get_aligned_face(self, image: np.ndarray) -> np.ndarray | None:
        faces = self._detector.detect(image)
        if not faces:
            return None
        selection = self._selector.select(faces, image.shape)
        if selection.selected_face is None:
            return None
        crop = self._cropper.crop(image, selection.selected_face)
        transformed = self._transformer.transform(selection.selected_face, crop.crop_x, crop.crop_y)
        alignment = self._aligner.align(crop.image, transformed)
        return alignment.aligned_image

    def _compute_per_class_metrics(self, orig_mask: np.ndarray, fused_mask: np.ndarray, total_pixels: int) -> dict:
        orig_metrics = {}
        fused_metrics = {}
        for cls_id, cls_name in TARGET_CLASSES.items():
            orig_m = _compute_class_metrics(orig_mask, cls_id, total_pixels)
            fused_m = _compute_class_metrics(fused_mask, cls_id, total_pixels)
            orig_metrics[cls_name] = asdict(orig_m)
            fused_metrics[cls_name] = asdict(fused_m)
        return {"original": orig_metrics, "fused": fused_metrics}

    def _compute_eye_glass_interaction(self, orig_mask: np.ndarray, fused_mask: np.ndarray) -> dict:
        return {
            "original_left": _eye_glass_interaction(orig_mask, FacePart.LEFT_EYE),
            "original_right": _eye_glass_interaction(orig_mask, FacePart.RIGHT_EYE),
            "fused_left": _eye_glass_interaction(fused_mask, FacePart.LEFT_EYE),
            "fused_right": _eye_glass_interaction(fused_mask, FacePart.RIGHT_EYE),
        }

    def _suggest_qa_label(self, result: ImageResult) -> str:
        notes = []
        eye = result.eye_recovery_status
        brow = result.brow_recovery_status

        if "POSSIBLE_OVERSEGMENTATION" in eye or "POSSIBLE_OVERSEGMENTATION" in brow:
            return "SUSPECTED_OVERSEGMENTATION"

        if "RECOVERED" in eye or "IMPROVED" in eye or "RECOVERED" in brow or "IMPROVED" in brow:
            return "IMPROVEMENT"

        if "REDUCED" in eye or "REDUCED" in brow:
            return "SUSPECTED_FAILURE"

        return "PENDING_REVIEW"

    def _save_comparison(self, image_name: str, aligned: np.ndarray,
                         orig_result: FaceParsingResult, fused_result: FaceParsingResult,
                         result: ImageResult) -> str:
        from evaluation.overlay_renderer import OverlayRenderer

        base = Path(image_name).stem
        panels = []

        panel_aligned = cv2.resize(aligned, (224, 224), interpolation=cv2.INTER_LINEAR)
        panels.append(("Aligned Face", panel_aligned))

        orig_colored = OverlayRenderer.render_colored_mask(orig_result.mask)
        panels.append(("ORIGINAL 19-class", cv2.resize(orig_colored, (224, 224), interpolation=cv2.INTER_NEAREST)))

        fused_colored = OverlayRenderer.render_colored_mask(fused_result.mask)
        panels.append(("FUSED 19-class", cv2.resize(fused_colored, (224, 224), interpolation=cv2.INTER_NEAREST)))

        orig_target = _target_mask_colored(orig_result.mask)
        panels.append(("ORIGINAL target", cv2.resize(orig_target, (224, 224), interpolation=cv2.INTER_NEAREST)))

        fused_target = _target_mask_colored(fused_result.mask)
        panels.append(("FUSED target", cv2.resize(fused_target, (224, 224), interpolation=cv2.INTER_NEAREST)))

        change = _change_mask(orig_result.mask, fused_result.mask)
        panels.append(("FUSED-ORIGINAL change", cv2.resize(change, (224, 224), interpolation=cv2.INTER_NEAREST)))

        row1 = np.hstack([p[1] for p in panels[:3]])
        row2 = np.hstack([p[1] for p in panels[3:]])
        canvas = np.vstack([row1, row2])

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, (label, _) in enumerate(panels[:3]):
            x = i * 224 + 5
            cv2.putText(canvas, label, (x, 18), font, 0.45, (255, 255, 255), 2)
            cv2.putText(canvas, label, (x, 18), font, 0.45, (0, 0, 0), 1)
        for i, (label, _) in enumerate(panels[3:]):
            x = i * 224 + 5
            cv2.putText(canvas, label, (x, 224 + 18), font, 0.45, (255, 255, 255), 2)
            cv2.putText(canvas, label, (x, 224 + 18), font, 0.45, (0, 0, 0), 1)

        info_y = 224 * 2 - 5
        eye_d = result.fused_left_eye_pixels - result.original_left_eye_pixels
        brow_d = result.fused_left_brow_pixels - result.original_left_brow_pixels
        info_text = f"Eye delta: {eye_d:+d}  Brow delta: {brow_d:+d}  {result.eye_recovery_status}"
        cv2.putText(canvas, info_text, (5, info_y), font, 0.4, (255, 255, 255), 2)
        cv2.putText(canvas, info_text, (5, info_y), font, 0.4, (0, 200, 0), 1)

        out_path = str(self._comparisons_dir / f"{base}_parser_comparison.png")
        cv2.imwrite(out_path, canvas)
        return out_path

    def process_image(self, image_path: str) -> ImageResult:
        name = os.path.basename(image_path)
        result = ImageResult(image_name=name)

        image = cv2.imread(image_path)
        if image is None:
            result.processing_status = "ERROR"
            result.error_message = "Failed to load image"
            return result

        aligned = self._get_aligned_face(image)
        if aligned is None:
            result.processing_status = "ERROR"
            result.error_message = "No face detected"
            return result

        try:
            orig_res, orig_t, fused_res, fused_t = _parse_both_modes(aligned)
        except Exception as exc:
            result.processing_status = "ERROR"
            result.error_message = str(exc)
            return result

        h, w = aligned.shape[:2]
        total_px = h * w

        result.original_left_brow_pixels = int(np.sum(orig_res.mask == FacePart.LEFT_BROW))
        result.fused_left_brow_pixels = int(np.sum(fused_res.mask == FacePart.LEFT_BROW))
        result.original_right_brow_pixels = int(np.sum(orig_res.mask == FacePart.RIGHT_BROW))
        result.fused_right_brow_pixels = int(np.sum(fused_res.mask == FacePart.RIGHT_BROW))
        result.original_left_eye_pixels = int(np.sum(orig_res.mask == FacePart.LEFT_EYE))
        result.fused_left_eye_pixels = int(np.sum(fused_res.mask == FacePart.LEFT_EYE))
        result.original_right_eye_pixels = int(np.sum(orig_res.mask == FacePart.RIGHT_EYE))
        result.fused_right_eye_pixels = int(np.sum(fused_res.mask == FacePart.RIGHT_EYE))
        result.original_eye_glass_pixels = int(np.sum(orig_res.mask == FacePart.EYE_GLASS))
        result.fused_eye_glass_pixels = int(np.sum(fused_res.mask == FacePart.EYE_GLASS))

        result.original_total_target_pixels = (
            result.original_left_brow_pixels + result.original_right_brow_pixels
            + result.original_left_eye_pixels + result.original_right_eye_pixels
            + result.original_eye_glass_pixels
        )
        result.fused_total_target_pixels = (
            result.fused_left_brow_pixels + result.fused_right_brow_pixels
            + result.fused_left_eye_pixels + result.fused_right_eye_pixels
            + result.fused_eye_glass_pixels
        )

        result.original_time_ms = orig_t
        result.fused_time_ms = fused_t

        metrics = self._compute_per_class_metrics(orig_res.mask, fused_res.mask, total_px)
        result.original_metrics = metrics["original"]
        result.fused_metrics = metrics["fused"]

        result.eye_glass_interaction = self._compute_eye_glass_interaction(orig_res.mask, fused_res.mask)
        result.change_analysis = _change_analysis(orig_res.mask, fused_res.mask)

        left_eye_status = _compute_recovery_status(
            result.original_left_eye_pixels, result.fused_left_eye_pixels, total_px)
        right_eye_status = _compute_recovery_status(
            result.original_right_eye_pixels, result.fused_right_eye_pixels, total_px)
        result.eye_recovery_status = f"L:{left_eye_status} R:{right_eye_status}"

        left_brow_status = _compute_recovery_status(
            result.original_left_brow_pixels, result.fused_left_brow_pixels, total_px)
        right_brow_status = _compute_recovery_status(
            result.original_right_brow_pixels, result.fused_right_brow_pixels, total_px)
        result.brow_recovery_status = f"L:{left_brow_status} R:{right_brow_status}"

        result.suggested_qa_label = self._suggest_qa_label(result)

        notes = []
        if "POSSIBLE_OVERSEGMENTATION" in result.eye_recovery_status or "POSSIBLE_OVERSEGMENTATION" in result.brow_recovery_status:
            notes.append("Possible over-segmentation detected")
        if "RECOVERED" in result.eye_recovery_status or "RECOVERED" in result.brow_recovery_status:
            notes.append("Candidate recovery detected")
        if "IMPROVED" in result.eye_recovery_status or "IMPROVED" in result.brow_recovery_status:
            notes.append("Possible improvement detected")
        if "REDUCED" in result.eye_recovery_status or "REDUCED" in result.brow_recovery_status:
            notes.append("Reduction detected")
        if not notes:
            notes.append("Minimal change between modes")
        result.qa_notes = "; ".join(notes)

        try:
            result.comparison_path = self._save_comparison(name, aligned, orig_res, fused_res, result)
        except Exception as exc:
            result.comparison_path = f"ERROR: {exc}"

        return result

    def run(self, image_dir: str) -> list[ImageResult]:
        image_dir_path = Path(image_dir)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(p for p in image_dir_path.iterdir() if p.is_file() and p.suffix.lower() in exts)

        logger.info("Found %d images in %s", len(image_paths), image_dir)
        results: list[ImageResult] = []
        for i, img_path in enumerate(image_paths, 1):
            logger.info("[%d/%d] Processing %s", i, len(image_paths), img_path.name)
            try:
                r = self.process_image(str(img_path))
            except Exception as exc:
                r = ImageResult(image_name=img_path.name, processing_status="ERROR", error_message=str(exc))
            results.append(r)

        self._write_csv(results)
        self._write_per_image_json(results)
        self._write_summary_json(results)
        self._write_summary_md(results)
        self._write_human_review_csv(results)
        return results

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def _write_csv(self, results: list[ImageResult]) -> None:
        path = self._output_dir / "per_image_results.csv"
        fieldnames = [
            "image_name", "processing_status",
            "original_left_brow_pixels", "fused_left_brow_pixels",
            "original_right_brow_pixels", "fused_right_brow_pixels",
            "original_left_eye_pixels", "fused_left_eye_pixels",
            "original_right_eye_pixels", "fused_right_eye_pixels",
            "original_eye_glass_pixels", "fused_eye_glass_pixels",
            "original_total_target_pixels", "fused_total_target_pixels",
            "original_time_ms", "fused_time_ms",
            "eye_recovery_status", "brow_recovery_status",
            "suggested_qa_label", "qa_verdict", "qa_notes",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                d = asdict(r)
                writer.writerow({k: d[k] for k in fieldnames})
        logger.info("CSV written to %s", path)

    def _write_per_image_json(self, results: list[ImageResult]) -> None:
        path = self._output_dir / "per_image_results.json"
        data = [asdict(r) for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Per-image JSON written to %s", path)

    def _write_summary_json(self, results: list[ImageResult]) -> None:
        ok = [r for r in results if r.processing_status == "SUCCESS"]
        err = [r for r in results if r.processing_status == "ERROR"]

        def _stats(vals: list[float]) -> dict:
            a = np.array(vals, dtype=float)
            if len(a) == 0:
                return {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
            return {"mean": float(a.mean()), "median": float(np.median(a)),
                    "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}

        recovery_eye = [r for r in ok if "RECOVERED" in r.eye_recovery_status or "IMPROVED" in r.eye_recovery_status]
        recovery_brow = [r for r in ok if "RECOVERED" in r.brow_recovery_status or "IMPROVED" in r.brow_recovery_status]
        overseg = [r for r in ok if "POSSIBLE_OVERSEGMENTATION" in r.eye_recovery_status or "POSSIBLE_OVERSEGMENTATION" in r.brow_recovery_status]
        reduced = [r for r in ok if "REDUCED" in r.eye_recovery_status or "REDUCED" in r.brow_recovery_status]

        label_dist = {}
        for r in ok:
            label_dist[r.suggested_qa_label] = label_dist.get(r.suggested_qa_label, 0) + 1

        summary = {
            "experiment": "Phase 4 Parser QA - ORIGINAL vs FUSED",
            "total_images": len(results),
            "successful": len(ok),
            "failed": len(err),
            "failed_images": [r.image_name for r in err],
            "per_class_pixel_stats": {},
            "recovery_eye_candidates": len(recovery_eye),
            "recovery_eye_cases": [r.image_name for r in recovery_eye],
            "recovery_brow_candidates": len(recovery_brow),
            "recovery_brow_cases": [r.image_name for r in recovery_brow],
            "oversegmentation_cases_count": len(overseg),
            "oversegmentation_cases": [r.image_name for r in overseg],
            "reduced_cases_count": len(reduced),
            "reduced_cases": [r.image_name for r in reduced],
            "original_avg_time_ms": float(np.mean([r.original_time_ms for r in ok])) if ok else 0.0,
            "fused_avg_time_ms": float(np.mean([r.fused_time_ms for r in ok])) if ok else 0.0,
            "suggested_label_distribution": label_dist,
            "protected_artifacts": {
                "onnx_sha256": _sha256_prefix(ROOT / "ai_models" / "bisenet" / "bisenet_resnet18.onnx"),
                "aux_checkpoint_sha256": _sha256_prefix(ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "training_aux_eye_brow_phase1" / "checkpoints" / "best.pt"),
            },
            "fusion_config": {
                "strategy": Settings.EYE_BROW_FUSION_STRATEGY,
                "threshold": Settings.EYE_BROW_FUSION_THRESHOLD,
                "min_component_size": Settings.EYE_BROW_FUSION_MIN_COMPONENT_SIZE,
            },
            "no_ground_truth_claim": "No ground-truth masks. IoU/Dice claims NOT made.",
        }

        for cls_name in TARGET_CLASSES.values():
            orig_vals = [r.original_metrics.get(cls_name, {}).get("pixel_count", 0) for r in ok]
            fused_vals = [r.fused_metrics.get(cls_name, {}).get("pixel_count", 0) for r in ok]
            deltas = [f - o for o, f in zip(orig_vals, fused_vals)]
            summary["per_class_pixel_stats"][cls_name] = {
                "original": _stats([float(v) for v in orig_vals]),
                "fused": _stats([float(v) for v in fused_vals]),
                "delta": _stats([float(v) for v in deltas]),
            }

        path = self._output_dir / "parser_qa_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Summary JSON written to %s", path)

    def _write_summary_md(self, results: list[ImageResult]) -> None:
        ok = [r for r in results if r.processing_status == "SUCCESS"]
        err = [r for r in results if r.processing_status == "ERROR"]

        def _avg(vals):
            return float(np.mean(vals)) if vals else 0.0

        recovery_eye = [r for r in ok if "RECOVERED" in r.eye_recovery_status or "IMPROVED" in r.eye_recovery_status]
        recovery_brow = [r for r in ok if "RECOVERED" in r.brow_recovery_status or "IMPROVED" in r.brow_recovery_status]
        overseg = [r for r in ok if "POSSIBLE_OVERSEGMENTATION" in r.eye_recovery_status or "POSSIBLE_OVERSEGMENTATION" in r.brow_recovery_status]
        reduced = [r for r in ok if "REDUCED" in r.eye_recovery_status or "REDUCED" in r.brow_recovery_status]

        lines = [
            "# Phase 4 Parser QA Report",
            "",
            "## 1. Experiment Objective",
            "",
            "Evaluate whether the production FUSED parser (BiSeNet + FFM + Auxiliary Eye/Brow Head + Phase 3 Strategy 1 fusion) "
            "visibly improves eye/brow parsing on production-like images compared to the ORIGINAL parser, "
            "especially under eyeglasses and partial occlusion.",
            "",
            "**This is qualitative + behavioral production QA. No ground-truth masks exist for these images.** "
            "IoU/Dice claims are NOT made. Quantitative evidence remains in Phase 2/Phase 3 held-out evaluation.",
            "",
            "## 2. Production Architecture",
            "",
            "- ParserMode.ORIGINAL: ONNX BiSeNet (ResNet-18 backbone + ARM + FFM + conv_out)",
            "- ParserMode.FUSED: PyTorch BiSeNet (same backbone) + FFM + Auxiliary Eye/Brow Head + Phase 3 Strategy 1 fusion",
            "- Same aligned face fed to both modes",
            "- Non-target safety guard prevents auxiliary from overwriting non-target classes",
            "",
            f"- ONNX model: `{_sha256_prefix(ROOT / 'ai_models' / 'bisenet' / 'bisenet_resnet18.onnx')}`",
            f"- Aux checkpoint: `{_sha256_prefix(ROOT / 'dataset_builder' / 'dataset' / 'parser_finetune_current' / 'training_aux_eye_brow_phase1' / 'checkpoints' / 'best.pt')}`",
            f"- Fusion strategy: {Settings.EYE_BROW_FUSION_STRATEGY}, threshold: {Settings.EYE_BROW_FUSION_THRESHOLD}, min_component_size: {Settings.EYE_BROW_FUSION_MIN_COMPONENT_SIZE}",
            "",
            "## 3. Dataset / Image Count",
            "",
            f"- Total images: {len(results)}",
            f"- Successful parser runs: {len(ok)}",
            f"- Failed parser runs: {len(err)}",
            "",
        ]

        if err:
            lines.append("**Failed images:**")
            for r in err:
                lines.append(f"- {r.image_name}: {r.error_message}")
            lines.append("")

        lines += [
            "## 4. ORIGINAL vs FUSED Per-Class Pixel Statistics",
            "",
            "| Class | ORIGINAL mean | FUSED mean | Delta mean | Delta median |",
            "|-------|--------------|-----------|-----------|-------------|",
        ]
        for cls_name in TARGET_CLASSES.values():
            orig_vals = [float(r.original_metrics.get(cls_name, {}).get("pixel_count", 0)) for r in ok]
            fused_vals = [float(r.fused_metrics.get(cls_name, {}).get("pixel_count", 0)) for r in ok]
            deltas = [f - o for o, f in zip(orig_vals, fused_vals)]
            lines.append(
                f"| {cls_name} | {_avg(orig_vals):.0f} | {_avg(fused_vals):.0f} "
                f"| {_avg(deltas):+.0f} | {float(np.median(deltas)) if deltas else 0:+.0f} |"
            )
        lines.append("")

        lines += ["## 5. Eye Recovery Candidates", ""]
        if recovery_eye:
            for r in recovery_eye:
                lines.append(f"- **{r.image_name}**: {r.eye_recovery_status}")
        else:
            lines.append("No eye recovery candidates detected.")
        lines.append("")

        lines += ["## 6. Brow Recovery Candidates", ""]
        if recovery_brow:
            for r in recovery_brow:
                lines.append(f"- **{r.image_name}**: {r.brow_recovery_status}")
        else:
            lines.append("No brow recovery candidates detected.")
        lines.append("")

        lines += ["## 7. Eye/Glasses Interaction Analysis", ""]
        lines.append("| Image | Eye | Orig eye_in_glass | Fused eye_in_glass | Orig glass_covers_eye | Fused glass_covers_eye |")
        lines.append("|-------|-----|-------------------|--------------------|-----------------------|------------------------|")
        for r in ok:
            ig = r.eye_glass_interaction
            for side, key in [("L", "left"), ("R", "right")]:
                o = ig.get(f"original_{key}", {})
                f = ig.get(f"fused_{key}", {})
                if o.get("eye_area", 0) > 0 or f.get("eye_area", 0) > 0:
                    lines.append(
                        f"| {r.image_name} | {side} "
                        f"| {o.get('eye_inside_glasses_ratio', 0):.3f} "
                        f"| {f.get('eye_inside_glasses_ratio', 0):.3f} "
                        f"| {o.get('glass_covers_eye_ratio', 0):.3f} "
                        f"| {f.get('glass_covers_eye_ratio', 0):.3f} |"
                    )
        lines.append("")

        lines += ["## 8. Possible Over-Segmentation Cases", ""]
        if overseg:
            for r in overseg:
                lines.append(f"- **{r.image_name}**: {r.eye_recovery_status} | {r.brow_recovery_status}")
        else:
            lines.append("No possible over-segmentation cases detected.")
        lines.append("")

        lines += ["## 9. Reduction Cases", ""]
        if reduced:
            for r in reduced:
                lines.append(f"- **{r.image_name}**: {r.eye_recovery_status} | {r.brow_recovery_status}")
        else:
            lines.append("No reduction cases detected.")
        lines.append("")

        lines += ["## 10. Fragmentation / Component Analysis", ""]
        lines.append("Suspicious fragmentation heuristic: >6 components, or largest component <30% with >2 components.")
        lines.append("")
        suspicious = []
        for r in ok:
            for cls_name in ["LEFT_EYE", "RIGHT_EYE"]:
                orig_cc = r.original_metrics.get(cls_name, {})
                fused_cc = r.fused_metrics.get(cls_name, {})
                if orig_cc.get("component_count", 0) > 6 or fused_cc.get("component_count", 0) > 6:
                    suspicious.append((r.image_name, cls_name, orig_cc, fused_cc))
        if suspicious:
            for name, cls, o, f in suspicious:
                lines.append(f"- **{name}** {cls}: orig {o.get('component_count',0)} components (largest {o.get('largest_component_ratio',0):.2f}), fused {f.get('component_count',0)} components (largest {f.get('largest_component_ratio',0):.2f})")
        else:
            lines.append("No suspicious fragmentation detected.")
        lines.append("")

        lines += ["## 11. Human Review Status", ""]
        label_dist = {}
        for r in ok:
            label_dist[r.suggested_qa_label] = label_dist.get(r.suggested_qa_label, 0) + 1
        lines.append("| Suggested Label | Count |")
        lines.append("|----------------|-------|")
        for label, count in sorted(label_dist.items()):
            lines.append(f"| {label} | {count} |")
        lines.append("")

        lines += [
            "## 12. QA Verdict Definitions",
            "",
            "- **PASS**: Visible anatomical eye/brow is represented appropriately by FUSED.",
            "- **EXPECTED_CONSERVATIVE**: The eye is genuinely not visible or heavily occluded and FUSED does not hallucinate a large eye region.",
            "- **IMPROVEMENT**: FUSED clearly represents visible eye/brow information that ORIGINAL missed or represented poorly.",
            "- **SUSPECTED_FAILURE**: FUSED appears to miss clearly visible eye/brow information.",
            "- **SUSPECTED_OVERSEGMENTATION**: FUSED appears to create eye/brow regions unsupported by the image.",
            "- **PENDING_REVIEW**: Insufficient evidence (default).",
            "",
            "## 13. Limitations",
            "",
            "1. No ground-truth masks exist for these production-like images.",
            "2. This experiment is qualitative + behavioral production QA only.",
            "3. Quantitative IoU evidence remains in Phase 2/Phase 3 held-out evaluation.",
            "4. Eye visibility and occlusion type fields require manual human review.",
            "5. Recovery analysis is pixel-count + geometric based, not anatomically validated.",
            "6. Recovery heuristics are QA signals, not segmentation accuracy metrics.",
            "",
            "## 14. Final Conclusion",
            "",
        ]
        if recovery_eye:
            lines.append(
                f"FUSED mode shows candidate eye recovery or improvement in {len(recovery_eye)} of {len(ok)} images. "
                "Human review of comparison visualizations is required to confirm these are genuine improvements."
            )
        else:
            lines.append(
                "No clear eye recovery cases were detected in pixel-count analysis. "
                "Human review of comparison visualizations is required for final determination."
            )
        lines.append("")
        lines.append("**All suggested labels are PENDING_REVIEW until human examination of the comparison PNGs.**")

        path = self._output_dir / "parser_qa_summary.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Summary MD written to %s", path)

    def _write_human_review_csv(self, results: list[ImageResult]) -> None:
        path = self._output_dir / "human_review.csv"
        existing: dict[str, dict] = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    existing[row["image_name"]] = row

        fieldnames = [
            "image_name", "comparison_path", "processing_status",
            "eye_recovery_status", "brow_recovery_status",
            "original_left_eye_pixels", "fused_left_eye_pixels",
            "original_right_eye_pixels", "fused_right_eye_pixels",
            "original_left_brow_pixels", "fused_left_brow_pixels",
            "original_right_brow_pixels", "fused_right_brow_pixels",
            "original_eye_glass_pixels", "fused_eye_glass_pixels",
            "eye_glasses_interaction_summary",
            "suggested_qa_label", "human_verdict", "human_notes",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                if r.processing_status != "SUCCESS":
                    continue
                existing_row = existing.get(r.image_name, {})
                ig_summary = (
                    f"L:eye_in_glass={r.eye_glass_interaction.get('original_left',{}).get('eye_inside_glasses_ratio',0):.3f}"
                    f"->{r.eye_glass_interaction.get('fused_left',{}).get('eye_inside_glasses_ratio',0):.3f} "
                    f"R:eye_in_glass={r.eye_glass_interaction.get('original_right',{}).get('eye_inside_glasses_ratio',0):.3f}"
                    f"->{r.eye_glass_interaction.get('fused_right',{}).get('eye_inside_glasses_ratio',0):.3f}"
                )
                writer.writerow({
                    "image_name": r.image_name,
                    "comparison_path": r.comparison_path,
                    "processing_status": r.processing_status,
                    "eye_recovery_status": r.eye_recovery_status,
                    "brow_recovery_status": r.brow_recovery_status,
                    "original_left_eye_pixels": r.original_left_eye_pixels,
                    "fused_left_eye_pixels": r.fused_left_eye_pixels,
                    "original_right_eye_pixels": r.original_right_eye_pixels,
                    "fused_right_eye_pixels": r.fused_right_eye_pixels,
                    "original_left_brow_pixels": r.original_left_brow_pixels,
                    "fused_left_brow_pixels": r.fused_left_brow_pixels,
                    "original_right_brow_pixels": r.original_right_brow_pixels,
                    "fused_right_brow_pixels": r.fused_right_brow_pixels,
                    "original_eye_glass_pixels": r.original_eye_glass_pixels,
                    "fused_eye_glass_pixels": r.fused_eye_glass_pixels,
                    "eye_glasses_interaction_summary": ig_summary,
                    "suggested_qa_label": r.suggested_qa_label,
                    "human_verdict": existing_row.get("human_verdict", "PENDING_REVIEW"),
                    "human_notes": existing_row.get("human_notes", ""),
                })
        logger.info("Human review CSV written to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Parser QA: ORIGINAL vs FUSED comparison")
    ap.add_argument("--image-dir", default=str(ROOT / "test_images" / "experiments"),
                     help="Directory of test images")
    ap.add_argument("--output-dir", default=str(ROOT / "reports" / "experiments" / "phase4_parser_qa"),
                     help="Output directory")
    args = ap.parse_args()

    qa = ParserQA(output_dir=args.output_dir)
    results = qa.run(args.image_dir)

    ok = [r for r in results if r.processing_status == "SUCCESS"]
    err = [r for r in results if r.processing_status == "ERROR"]
    recovery = [r for r in ok if "RECOVERED" in r.eye_recovery_status or "IMPROVED" in r.eye_recovery_status
                or "RECOVERED" in r.brow_recovery_status or "IMPROVED" in r.brow_recovery_status]
    overseg = [r for r in ok if "POSSIBLE_OVERSEGMENTATION" in r.eye_recovery_status or "POSSIBLE_OVERSEGMENTATION" in r.brow_recovery_status]

    print(f"\nParser QA Complete: {len(results)} images, {len(ok)} succeeded, {len(err)} failed")
    print(f"Recovery/improvement candidates: {len(recovery)} cases")
    print(f"Possible over-segmentation: {len(overseg)} cases")
    print(f"Reports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
