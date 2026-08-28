#!/usr/bin/env python3
"""
Final End-to-End Pipeline Integration Experiment.

Runs REAL images through the CURRENT production pipeline from input image
to final ValidationResult. Exercises actual components; does not recreate
validation logic.

Usage:
    python scripts/run_full_pipeline_experiment.py <image_or_directory>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np

from config.parser_mode import ParserMode
from models.parsing.face_part import FacePart
from models.validation_type import ValidationType
from pipeline.photo_validation_pipeline import PhotoValidationPipeline
from pipeline.validation_orchestrator import ValidationOrchestrator
from services.face_parser_service import FaceParserService
from reasoning.semantic_engine import SemanticEvidenceEngine

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

ALL_VALIDATOR_TYPES = [
    ValidationType.BLUR,
    ValidationType.BRIGHTNESS,
    ValidationType.CONTRAST,
    ValidationType.FACE_SIZE,
    ValidationType.HEAD_POSE,
    ValidationType.FACE_VISIBILITY,
    ValidationType.OCCLUSION,
]

ALL_VALIDATOR_NAMES = {
    ValidationType.BLUR: "BLUR",
    ValidationType.BRIGHTNESS: "BRIGHTNESS",
    ValidationType.CONTRAST: "CONTRAST",
    ValidationType.FACE_SIZE: "FACE_SIZE",
    ValidationType.HEAD_POSE: "HEAD_POSE",
    ValidationType.FACE_VISIBILITY: "FACE_VISIBILITY",
    ValidationType.OCCLUSION: "OCCLUSION",
}


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_model_hashes() -> dict[str, str]:
    onnx_path = Path("ai_models/bisenet/bisenet_resnet18.onnx")
    hashes = {}
    if onnx_path.exists():
        hashes["bisenet_resnet18.onnx"] = _hash_file(onnx_path)
    return hashes


def collect_image_paths(target: Path) -> list[Path]:
    paths: list[Path] = []
    if target.is_file():
        if target.suffix.lower() in SUPPORTED_EXTENSIONS:
            paths.append(target)
    elif target.is_dir():
        for p in sorted(target.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                paths.append(p)
    return paths


def _compute_semantic_parts(
    parsing_result,
    face,
) -> dict:
    engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
    parts_info = {}

    for part in [FacePart.LEFT_EYE, FacePart.RIGHT_EYE]:
        ev = engine.compute_eye_evidence(part)
        px = parsing_result.part_area(part)
        ratio = parsing_result.part_ratio(part)
        parts_info[part.name] = {
            "pixels": int(px),
            "area_ratio": round(float(ratio), 6),
            "parser_confidence": round(ev.parser_confidence, 4),
            "landmark_confidence": round(ev.landmark_confidence, 4),
            "pose_confidence": round(ev.pose_confidence, 4),
            "occlusion_confidence": round(ev.occlusion_confidence, 4),
            "eye_support_confidence": round(ev.eye_support_confidence, 4),
            "final_confidence": round(ev.final_confidence, 4),
            "passed": ev.passed,
        }

    for part in [FacePart.NOSE]:
        ev = engine.compute_part_evidence(
            part, 0.0050
        )
        px = parsing_result.part_area(part)
        ratio = parsing_result.part_ratio(part)
        parts_info[part.name] = {
            "pixels": int(px),
            "area_ratio": round(float(ratio), 6),
            "parser_confidence": round(ev.parser_confidence, 4),
            "landmark_confidence": round(ev.landmark_confidence, 4),
            "pose_confidence": round(ev.pose_confidence, 4),
            "occlusion_confidence": round(ev.occlusion_confidence, 4),
            "eye_support_confidence": round(ev.eye_support_confidence, 4),
            "final_confidence": round(ev.final_confidence, 4),
            "passed": ev.passed,
        }

    for part in [FacePart.LEFT_BROW, FacePart.RIGHT_BROW]:
        eye = FacePart.LEFT_EYE if part == FacePart.LEFT_BROW else FacePart.RIGHT_EYE
        ev = engine.compute_eyebrow_evidence(part, eye)
        px = parsing_result.part_area(part)
        ratio = parsing_result.part_ratio(part)
        parts_info[part.name] = {
            "pixels": int(px),
            "area_ratio": round(float(ratio), 6),
            "parser_confidence": round(ev.parser_confidence, 4),
            "landmark_confidence": round(ev.landmark_confidence, 4),
            "pose_confidence": round(ev.pose_confidence, 4),
            "occlusion_confidence": round(ev.occlusion_confidence, 4),
            "eye_support_confidence": round(ev.eye_support_confidence, 4),
            "final_confidence": round(ev.final_confidence, 4),
            "passed": ev.passed,
        }

    for part in [FacePart.MOUTH, FacePart.UPPER_LIP, FacePart.LOWER_LIP]:
        px = parsing_result.part_area(part)
        ratio = parsing_result.part_ratio(part)
        parts_info[part.name] = {
            "pixels": int(px),
            "area_ratio": round(float(ratio), 6),
        }

    return parts_info


def _compute_parser_statistics(parsing_result) -> dict:
    total = parsing_result.total_pixels()
    stats = {}
    for part in FacePart:
        px = parsing_result.part_area(part)
        pct = (px / total) * 100.0 if total > 0 else 0.0
        stats[part.name] = {"pixels": int(px), "percentage": round(pct, 2)}
    return stats


def process_single_image(
    img_path: Path,
    pipeline: PhotoValidationPipeline,
    output_base: Path,
) -> dict:
    result = {
        "image_name": img_path.name,
        "image_path": str(img_path),
        "resolution": [],
        "execution_time_ms": 0,
        "pipeline_status": "PROCESSING_ERROR",
        "face_detection": {},
        "face_selection": {},
        "face_metrics": {},
        "validators": [],
        "semantic_parts": {},
        "parser_statistics": {},
        "root_cause": None,
        "overall_passed": False,
        "output_paths": {},
        "error": None,
    }

    start = time.perf_counter()

    try:
        image = cv2.imread(str(img_path))
        if image is None:
            result["error"] = f"Failed to load image: {img_path}"
            return result

        h, w = image.shape[:2]
        result["resolution"] = [w, h]

        proc = pipeline.validate(image)
        elapsed = (time.perf_counter() - start) * 1000
        result["execution_time_ms"] = round(elapsed, 1)

        face = proc.selected_face
        if face is not None:
            bbox = [round(float(x), 2) for x in face.bbox[:4]]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_ratio = area / max(w * h, 1)
            cx = round((bbox[0] + bbox[2]) / 2, 2)
            cy = round((bbox[1] + bbox[3]) / 2, 2)
            pose = getattr(face, "pose", (0.0, 0.0, 0.0))
            pitch, yaw, roll = (float(pose[i]) for i in range(3)) if pose is not None else (0.0, 0.0, 0.0)

            result["face_detection"] = {
                "bbox": bbox,
                "area_ratio": round(area_ratio, 6),
                "center": [cx, cy],
                "detection_confidence": round(float(getattr(face, "det_score", 0.0)), 4),
            }
            result["face_metrics"] = {
                "yaw": round(yaw, 4),
                "pitch": round(pitch, 4),
                "roll": round(roll, 4),
                "area_ratio": round(area_ratio, 6),
            }

        result["face_selection"] = {
            "selected": face is not None,
        }

        vr = proc.validation_result
        executed_types = {m.type for m in vr.metrics}

        for vtype in ALL_VALIDATOR_TYPES:
            metric = next((m for m in vr.metrics if m.type == vtype), None)
            if metric is not None:
                result["validators"].append({
                    "validator_type": vtype.value,
                    "validator_name": ALL_VALIDATOR_NAMES[vtype],
                    "executed": True,
                    "passed": metric.passed,
                    "score": round(metric.score, 4),
                    "message": metric.message,
                })
            else:
                result["validators"].append({
                    "validator_type": vtype.value,
                    "validator_name": ALL_VALIDATOR_NAMES[vtype],
                    "executed": False,
                    "passed": None,
                    "score": None,
                    "message": "SKIPPED_DUE_TO_CHEAP_FAILURE",
                })

        result["overall_passed"] = vr.is_valid
        result["pipeline_status"] = "VALID" if vr.is_valid else "INVALID"

        if proc.aligned_image is not None:
            aligned_dir = output_base / "aligned"
            aligned_dir.mkdir(parents=True, exist_ok=True)
            aligned_path = aligned_dir / f"{img_path.stem}_aligned.png"
            cv2.imwrite(str(aligned_path), proc.aligned_image)
            result["output_paths"]["aligned"] = str(aligned_path)

        if proc.cropped_image is not None:
            cropped_dir = output_base / "cropped"
            cropped_dir.mkdir(parents=True, exist_ok=True)
            cropped_path = cropped_dir / f"{img_path.stem}_cropped.png"
            cv2.imwrite(str(cropped_path), proc.cropped_image)
            result["output_paths"]["cropped"] = str(cropped_path)

        if proc.aligned_image is not None and face is not None:
            try:
                parser_svc = FaceParserService(parser_mode=ParserMode.FUSED)
                parsing_result = parser_svc.parse(proc.aligned_image)

                masks_dir = output_base / "masks"
                masks_dir.mkdir(parents=True, exist_ok=True)
                mask_path = masks_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), parsing_result.mask.astype(np.uint8))
                result["output_paths"]["mask"] = str(mask_path)

                overlays_dir = output_base / "overlays"
                overlays_dir.mkdir(parents=True, exist_ok=True)
                overlay = proc.aligned_image.copy()
                mask_colored = np.zeros_like(overlay)
                mask_colored[parsing_result.mask == FacePart.EYE_GLASS] = [0, 255, 255]
                mask_colored[parsing_result.mask == FacePart.LEFT_EYE] = [255, 0, 0]
                mask_colored[parsing_result.mask == FacePart.RIGHT_EYE] = [0, 0, 255]
                overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
                overlay_path = overlays_dir / f"{img_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                result["output_paths"]["overlay"] = str(overlay_path)

                result["semantic_parts"] = _compute_semantic_parts(parsing_result, face)
                result["parser_statistics"] = _compute_parser_statistics(parsing_result)
            except Exception as e:
                result["error"] = f"Parser/semantic analysis error: {e}"

        if not vr.is_valid and face is not None:
            from evaluation.root_cause import RootCauseAnalyzer
            semantic_parts_for_rc = {}
            for pname, pdata in result["semantic_parts"].items():
                if "final_confidence" in pdata:
                    from evaluation.models import FacePartEvaluation
                    semantic_parts_for_rc[pname] = FacePartEvaluation(
                        part_name=pname,
                        pixels=pdata.get("pixels", 0),
                        area_ratio=pdata.get("area_ratio", 0.0),
                        parser_confidence=pdata.get("parser_confidence", 0.0),
                        landmark_confidence=pdata.get("landmark_confidence", 0.0),
                        pose_confidence=pdata.get("pose_confidence", 0.0),
                        occlusion_confidence=pdata.get("occlusion_confidence", 0.0),
                        eye_support_confidence=pdata.get("eye_support_confidence", 0.0),
                        final_confidence=pdata.get("final_confidence", 0.0),
                        passed=pdata.get("passed", False),
                    )
            validators_eval = []
            for v in result["validators"]:
                if v["executed"]:
                    from evaluation.models import ValidatorEvaluation
                    validators_eval.append(ValidatorEvaluation(
                        validator_type=ValidationType(v["validator_type"]),
                        validator_name=v["validator_name"],
                        passed=v["passed"],
                        score=v["score"],
                        message=v["message"],
                    ))
            rc = RootCauseAnalyzer.analyze(validators_eval, semantic_parts_for_rc)
            if rc is not None:
                result["root_cause"] = {
                    "cause": rc.cause,
                    "confidence": round(rc.confidence, 4),
                    "explanation": rc.explanation,
                    "evidence": rc.evidence,
                }

    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        result["execution_time_ms"] = round(elapsed, 1)
        result["error"] = f"{type(e).__name__}: {e}"
        result["pipeline_status"] = "PROCESSING_ERROR"

    return result


def print_image_report(result: dict) -> None:
    name = result["image_name"]
    status = result["pipeline_status"]
    elapsed = result["execution_time_ms"]

    print("=" * 56)
    print(f"Image Name    : {name}")
    print(f"Overall Result: {status}")
    print("=" * 56)
    print()

    if result.get("error"):
        print(f"  Error: {result['error']}")
        print()
    else:
        for v in result["validators"]:
            if v["executed"]:
                status_str = "PASS" if v["passed"] else "FAIL"
                score_str = f"Score: {v['score']:.2f}" if v["score"] is not None else ""
                msg = v.get("message", "")
                line = f"  {v['validator_name']:<18} {status_str:<6} {score_str}"
                if not v["passed"] and msg:
                    line += f"   {msg}"
                print(line)
            else:
                print(f"  {v['validator_name']:<18} SKIPPED")
        print()

    if result.get("root_cause"):
        rc = result["root_cause"]
        print(f"  Root Cause: {rc['cause']}")
        print(f"  Explanation: {rc['explanation']}")
        print()

    print(f"  Processing Time: {elapsed:.0f} ms")
    print()


def print_summary(all_results: list[dict]) -> None:
    total = len(all_results)
    valid = sum(1 for r in all_results if r["pipeline_status"] == "VALID")
    invalid = sum(1 for r in all_results if r["pipeline_status"] == "INVALID")
    errors = sum(1 for r in all_results if r["pipeline_status"] == "PROCESSING_ERROR")
    times = [r["execution_time_ms"] for r in all_results]

    print()
    print("=" * 56)
    print("FINAL PIPELINE SUMMARY")
    print("=" * 56)
    print()
    print(f"Images Processed : {total}")
    print(f"Valid Images     : {valid}")
    print(f"Invalid Images   : {invalid}")
    print(f"Processing Errors: {errors}")
    print()

    if times:
        times_sorted = sorted(times)
        p95_idx = max(0, int(len(times_sorted) * 0.95) - 1)
        print(f"Average Time     : {statistics.mean(times):.0f} ms")
        print(f"Median Time      : {statistics.median(times):.0f} ms")
        print(f"P95 Time         : {times_sorted[p95_idx]:.0f} ms")
        print(f"Min Time         : {min(times):.0f} ms")
        print(f"Max Time         : {max(times):.0f} ms")
    print()

    print("Validator Results:")
    print()
    for vtype in ALL_VALIDATOR_TYPES:
        vname = ALL_VALIDATOR_NAMES[vtype]
        pass_count = 0
        fail_count = 0
        skip_count = 0
        for r in all_results:
            v = next((v for v in r["validators"] if v["validator_type"] == vtype.value), None)
            if v is None:
                skip_count += 1
            elif not v["executed"]:
                skip_count += 1
            elif v["passed"]:
                pass_count += 1
            else:
                fail_count += 1
        print(f"  {vname}")
        print(f"    PASS: {pass_count}")
        print(f"    FAIL: {fail_count}")
        print(f"    SKIPPED: {skip_count}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final End-to-End Pipeline Integration Experiment"
    )
    parser.add_argument("path", type=str, help="Image file or directory")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        print(f"Error: Path does not exist: {target}", file=sys.stderr)
        sys.exit(1)

    image_paths = collect_image_paths(target)
    if not image_paths:
        print(f"No supported images found at: {target}", file=sys.stderr)
        sys.exit(1)

    output_base = Path("outputs") / "full_pipeline_experiment"
    output_base.mkdir(parents=True, exist_ok=True)
    json_dir = output_base / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(image_paths)} image(s) to process.")
    print(f"Output directory: {output_base}")
    print()

    pre_hashes = verify_model_hashes()
    print("Pre-experiment model hashes:")
    for name, h in pre_hashes.items():
        print(f"  {name}: {h[:16]}")
    print()

    pipeline = PhotoValidationPipeline(parser_mode=ParserMode.FUSED)

    all_results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] Processing: {img_path.name}")
        result = process_single_image(img_path, pipeline, output_base)
        all_results.append(result)

        json_path = json_dir / f"{img_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print_image_report(result)

    post_hashes = verify_model_hashes()
    print("Post-experiment model hashes:")
    for name, h in post_hashes.items():
        print(f"  {name}: {h[:16]}")
    print()

    hashes_match = pre_hashes == post_hashes
    if not hashes_match:
        print("WARNING: MODEL HASHES CHANGED DURING EXPERIMENT!")
        print("This should not happen. Experiment may be invalid.")
    else:
        print("Model hashes verified: UNCHANGED")
    print()

    print_summary(all_results)

    report = {
        "objective": "Final End-to-End Pipeline Integration Experiment",
        "architecture_tested": "PhotoValidationPipeline (PRODUCTION mode)",
        "dataset": str(target),
        "total_images": len(all_results),
        "valid_images": sum(1 for r in all_results if r["pipeline_status"] == "VALID"),
        "invalid_images": sum(1 for r in all_results if r["pipeline_status"] == "INVALID"),
        "processing_errors": sum(1 for r in all_results if r["pipeline_status"] == "PROCESSING_ERROR"),
        "model_hashes_before": pre_hashes,
        "model_hashes_after": post_hashes,
        "model_hashes_match": hashes_match,
        "image_results": all_results,
    }

    report_path = output_base / "experiment_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
