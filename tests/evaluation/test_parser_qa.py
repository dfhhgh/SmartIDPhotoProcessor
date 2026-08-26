"""Tests for the Parser QA evaluation tool."""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.parser_mode import ParserMode
from evaluation.parser_qa import (
    ParserQA,
    ImageResult,
    ClassMetrics,
    _target_mask_colored,
    _change_mask,
    _compute_recovery_status,
    _class_bbox,
    _connected_component_analysis,
    _eye_glass_interaction,
    _change_analysis,
    _compute_class_metrics,
    _parse_both_modes,
    TARGET_CLASSES,
    TARGET_CLASS_IDS,
)
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from services.face_parser_service import FaceParserService


def _reset():
    FaceParserService._instance = None
    FaceParserService._initialized = False


def _make_aligned():
    images_dir = ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "images"
    for i in range(100):
        p = images_dir / f"sample_{i:04d}.png"
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
                return img
    return np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Recovery status heuristics
# ---------------------------------------------------------------------------

class TestRecoveryStatus:
    def test_recovered(self):
        assert _compute_recovery_status(0, 100, 224 * 224) == "RECOVERED"

    def test_improved(self):
        assert _compute_recovery_status(50, 100, 224 * 224) == "IMPROVED"

    def test_unchanged_similar(self):
        assert _compute_recovery_status(50, 55, 224 * 224) == "UNCHANGED"

    def test_reduced(self):
        assert _compute_recovery_status(100, 5, 224 * 224) == "REDUCED"

    def test_oversegmentation(self):
        assert _compute_recovery_status(0, 50000, 224 * 224) == "POSSIBLE_OVERSEGMENTATION"

    def test_both_low(self):
        assert _compute_recovery_status(0, 0, 224 * 224) == "UNCHANGED"

    def test_recovered_not_overseg(self):
        result = _compute_recovery_status(0, 100, 224 * 224)
        assert result == "RECOVERED"

    def test_high_fused_ratio_is_overseg(self):
        result = _compute_recovery_status(0, 60000, 224 * 224)
        assert result == "POSSIBLE_OVERSEGMENTATION"


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

class TestClassBbox:
    def test_present_class(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[20:40, 30:60] = FacePart.LEFT_EYE
        bbox = _class_bbox(mask, FacePart.LEFT_EYE)
        assert bbox is not None
        assert bbox["x_min"] == 30
        assert bbox["y_min"] == 20
        assert bbox["x_max"] == 59
        assert bbox["y_max"] == 39
        assert bbox["width"] == 30
        assert bbox["height"] == 20

    def test_absent_class(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        bbox = _class_bbox(mask, FacePart.LEFT_EYE)
        assert bbox is None


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

class TestConnectedComponents:
    def test_single_component(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:20, 10:20] = FacePart.LEFT_EYE
        cc = _connected_component_analysis(mask, FacePart.LEFT_EYE)
        assert cc["component_count"] == 1
        assert cc["largest_component_area"] == 100
        assert cc["largest_component_ratio"] == 1.0

    def test_two_components(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:15, 10:15] = FacePart.LEFT_EYE
        mask[50:55, 50:55] = FacePart.LEFT_EYE
        cc = _connected_component_analysis(mask, FacePart.LEFT_EYE)
        assert cc["component_count"] == 2
        assert cc["largest_component_area"] == 25
        assert cc["largest_component_ratio"] == 0.5

    def test_no_components(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        cc = _connected_component_analysis(mask, FacePart.LEFT_EYE)
        assert cc["component_count"] == 0
        assert cc["largest_component_area"] == 0
        assert cc["largest_component_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Eye/glasses interaction
# ---------------------------------------------------------------------------

class TestEyeGlassInteraction:
    def test_eye_inside_glasses(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[20:40, 20:40] = FacePart.EYE_GLASS
        mask[25:35, 25:35] = FacePart.LEFT_EYE
        result = _eye_glass_interaction(mask, FacePart.LEFT_EYE)
        assert result["eye_area"] == 100
        # Glass pixels are overwritten by eye assignment, so glass_area = 300
        assert result["glass_area"] == 300
        assert result["intersection"] == 0
        assert result["eye_inside_glasses_ratio"] == 0.0

    def test_eye_outside_glasses(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[20:40, 20:40] = FacePart.EYE_GLASS
        mask[60:80, 60:80] = FacePart.LEFT_EYE
        result = _eye_glass_interaction(mask, FacePart.LEFT_EYE)
        assert result["intersection"] == 0
        assert result["eye_inside_glasses_ratio"] == 0.0

    def test_no_eye(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[20:40, 20:40] = FacePart.EYE_GLASS
        result = _eye_glass_interaction(mask, FacePart.LEFT_EYE)
        assert result["eye_area"] == 0
        assert result["eye_inside_glasses_ratio"] == 0.0

    def test_no_glasses(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[20:40, 20:40] = FacePart.LEFT_EYE
        result = _eye_glass_interaction(mask, FacePart.LEFT_EYE)
        assert result["glass_area"] == 0
        assert result["eye_inside_glasses_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Change analysis
# ---------------------------------------------------------------------------

class TestChangeAnalysis:
    def test_gained_and_lost(self):
        orig = np.zeros((50, 50), dtype=np.int64)
        orig[0:10, 0:10] = FacePart.LEFT_EYE
        fused = np.zeros((50, 50), dtype=np.int64)
        fused[5:15, 5:15] = FacePart.LEFT_EYE
        result = _change_analysis(orig, fused)
        cls_result = result["per_class"][FacePart.LEFT_EYE]
        assert cls_result["unchanged"] == 25
        assert cls_result["gained"] == 75
        assert cls_result["lost"] == 75
        assert result["total_gained"] > 0
        assert result["total_lost"] > 0

    def test_all_unchanged(self):
        orig = np.zeros((50, 50), dtype=np.int64)
        orig[0:10, 0:10] = FacePart.LEFT_EYE
        fused = orig.copy()
        result = _change_analysis(orig, fused)
        cls_result = result["per_class"][FacePart.LEFT_EYE]
        assert cls_result["gained"] == 0
        assert cls_result["lost"] == 0
        assert cls_result["unchanged"] == 100

    def test_all_lost(self):
        orig = np.zeros((50, 50), dtype=np.int64)
        orig[0:10, 0:10] = FacePart.LEFT_EYE
        fused = np.zeros((50, 50), dtype=np.int64)
        result = _change_analysis(orig, fused)
        cls_result = result["per_class"][FacePart.LEFT_EYE]
        assert cls_result["lost"] == 100
        assert cls_result["unchanged"] == 0


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------

class TestClassMetrics:
    def test_absent_class(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        m = _compute_class_metrics(mask, FacePart.LEFT_EYE, 10000)
        assert m.pixel_count == 0
        assert m.area_ratio == 0.0
        assert m.bbox is None
        assert m.component_count == 0

    def test_present_class(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:20, 10:20] = FacePart.LEFT_EYE
        m = _compute_class_metrics(mask, FacePart.LEFT_EYE, 10000)
        assert m.pixel_count == 100
        assert m.area_ratio == 0.01
        assert m.bbox is not None
        assert m.component_count == 1
        assert m.largest_component_ratio == 1.0


# ---------------------------------------------------------------------------
# Independent LEFT/RIGHT aggregation
# ---------------------------------------------------------------------------

class TestIndependentLeftRight:
    def test_left_and_right_are_independent(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:20, 10:20] = FacePart.LEFT_EYE
        mask[50:60, 50:60] = FacePart.RIGHT_EYE
        left_m = _compute_class_metrics(mask, FacePart.LEFT_EYE, 10000)
        right_m = _compute_class_metrics(mask, FacePart.RIGHT_EYE, 10000)
        assert left_m.pixel_count == 100
        assert right_m.pixel_count == 100
        assert left_m.bbox != right_m.bbox

    def test_asymmetric_counts(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:30, 10:30] = FacePart.LEFT_EYE
        mask[50:55, 50:55] = FacePart.RIGHT_EYE
        left_m = _compute_class_metrics(mask, FacePart.LEFT_EYE, 10000)
        right_m = _compute_class_metrics(mask, FacePart.RIGHT_EYE, 10000)
        assert left_m.pixel_count == 400
        assert right_m.pixel_count == 25
        assert left_m.pixel_count != right_m.pixel_count


# ---------------------------------------------------------------------------
# Target classes
# ---------------------------------------------------------------------------

class TestTargetClasses:
    def test_target_class_ids(self):
        assert TARGET_CLASS_IDS == {2, 3, 4, 5, 6}

    def test_target_classes_match_face_part(self):
        assert FacePart.LEFT_BROW in TARGET_CLASSES
        assert FacePart.RIGHT_BROW in TARGET_CLASSES
        assert FacePart.LEFT_EYE in TARGET_CLASSES
        assert FacePart.RIGHT_EYE in TARGET_CLASSES
        assert FacePart.EYE_GLASS in TARGET_CLASSES

    def test_no_non_target_classes(self):
        for cls_id in TARGET_CLASS_IDS:
            assert 0 <= cls_id <= 18


# ---------------------------------------------------------------------------
# Target mask colored
# ---------------------------------------------------------------------------

class TestTargetMaskColored:
    def test_output_shape(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:20, 10:20] = FacePart.LEFT_EYE
        colored = _target_mask_colored(mask)
        assert colored.shape == (100, 100, 3)
        assert colored.dtype == np.uint8

    def test_colors_match(self):
        mask = np.zeros((100, 100), dtype=np.int64)
        mask[10:20, 10:20] = FacePart.LEFT_EYE
        colored = _target_mask_colored(mask)
        assert tuple(colored[15, 15]) == (255, 0, 0)


# ---------------------------------------------------------------------------
# Change mask
# ---------------------------------------------------------------------------

class TestChangeMask:
    def test_gained_is_green(self):
        orig = np.zeros((100, 100), dtype=np.int64)
        fused = np.zeros((100, 100), dtype=np.int64)
        fused[10:20, 10:20] = FacePart.LEFT_EYE
        change = _change_mask(orig, fused)
        assert tuple(change[15, 15]) == (0, 255, 0)

    def test_lost_is_red(self):
        orig = np.zeros((100, 100), dtype=np.int64)
        orig[10:20, 10:20] = FacePart.LEFT_EYE
        fused = np.zeros((100, 100), dtype=np.int64)
        change = _change_mask(orig, fused)
        assert tuple(change[15, 15]) == (0, 0, 255)

    def test_unchanged_is_black(self):
        orig = np.zeros((100, 100), dtype=np.int64)
        orig[10:20, 10:20] = FacePart.LEFT_EYE
        fused = orig.copy()
        change = _change_mask(orig, fused)
        assert tuple(change[15, 15]) == (0, 0, 0)


# ---------------------------------------------------------------------------
# Parser modes
# ---------------------------------------------------------------------------

class TestParserModes:
    def test_original_mode_uses_onnx(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        assert fp._parser_mode == ParserMode.ORIGINAL

    def test_fused_mode_uses_pytorch(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        assert fp._parser_mode == ParserMode.FUSED
        assert fp._refinement_service is not None

    def test_both_produce_valid_mask(self):
        img = _make_aligned()
        _reset()
        orig = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        r1 = orig.parse(img)
        _reset()
        fused = FaceParserService(parser_mode=ParserMode.FUSED)
        r2 = fused.parse(img)
        assert r1.mask.shape == r2.mask.shape
        assert set(np.unique(r1.mask)).issubset(set(range(19)))
        assert set(np.unique(r2.mask)).issubset(set(range(19)))


# ---------------------------------------------------------------------------
# Production safety
# ---------------------------------------------------------------------------

class TestQAToolSafety:
    def test_onnx_hash_unchanged(self):
        import hashlib
        path = ROOT / "ai_models" / "bisenet" / "bisenet_resnet18.onnx"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        assert h.hexdigest()[:16] == "2218b6183c26ca5c"

    def test_aux_checkpoint_hash_unchanged(self):
        import hashlib
        path = ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "training_aux_eye_brow_phase1" / "checkpoints" / "best.pt"
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        assert h.hexdigest()[:16] == "961e08bf64fdd0b8"

    def test_fusion_config_unchanged(self):
        from config.settings import Settings
        assert Settings.EYE_BROW_FUSION_STRATEGY == 1
        assert Settings.EYE_BROW_FUSION_THRESHOLD == 0.0
        assert Settings.EYE_BROW_FUSION_MIN_COMPONENT_SIZE == 10

    def test_default_parser_mode_unchanged(self):
        from config.settings import Settings
        assert Settings().PARSER_MODE == ParserMode.ORIGINAL


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestParserQAIntegration:
    def test_process_image_with_real_face(self, tmp_path):
        images_dir = ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "images"
        test_img = None
        for i in range(100):
            p = images_dir / f"sample_{i:04d}.png"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None and img.shape[0] > 100:
                    test_img = img
                    break
        if test_img is None:
            pytest.skip("No suitable test image")

        qa = ParserQA(output_dir=str(tmp_path / "qa_out"))
        img_path = str(tmp_path / "test_input.png")
        cv2.imwrite(img_path, test_img)

        result = qa.process_image(img_path)
        assert isinstance(result, ImageResult)
        assert result.processing_status in ("SUCCESS", "ERROR")
        if result.processing_status == "SUCCESS":
            assert result.original_time_ms > 0
            assert result.fused_time_ms > 0
            assert result.eye_recovery_status != ""
            assert result.qa_verdict == "PENDING_REVIEW"
            assert "original_metrics" in result.__dict__
            assert "fused_metrics" in result.__dict__
            assert "eye_glass_interaction" in result.__dict__
            assert "change_analysis" in result.__dict__

    def test_one_error_does_not_stop_batch(self, tmp_path):
        qa = ParserQA(output_dir=str(tmp_path / "qa_out"))
        result = qa.process_image(str(tmp_path / "nonexistent.png"))
        assert result.processing_status == "ERROR"

    def test_deterministic_output(self, tmp_path):
        images_dir = ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "images"
        test_img = None
        for i in range(100):
            p = images_dir / f"sample_{i:04d}.png"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None and img.shape[0] > 100:
                    test_img = img
                    break
        if test_img is None:
            pytest.skip("No suitable test image")

        qa1 = ParserQA(output_dir=str(tmp_path / "out1"))
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, test_img)
        r1 = qa1.process_image(img_path)

        qa2 = ParserQA(output_dir=str(tmp_path / "out2"))
        r2 = qa2.process_image(img_path)

        if r1.processing_status == "SUCCESS" and r2.processing_status == "SUCCESS":
            assert r1.original_left_eye_pixels == r2.original_left_eye_pixels
            assert r1.fused_left_eye_pixels == r2.fused_left_eye_pixels
            assert r1.original_left_brow_pixels == r2.original_left_brow_pixels
            assert r1.fused_left_brow_pixels == r2.fused_left_brow_pixels

    def test_human_review_csv_created(self, tmp_path):
        images_dir = ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "images"
        test_img = None
        for i in range(100):
            p = images_dir / f"sample_{i:04d}.png"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None and img.shape[0] > 100:
                    test_img = img
                    break
        if test_img is None:
            pytest.skip("No suitable test image")

        qa = ParserQA(output_dir=str(tmp_path / "qa_csv"))
        img_path = str(tmp_path / "test.png")
        cv2.imwrite(img_path, test_img)
        qa.run(str(tmp_path))

        csv_path = tmp_path / "qa_csv" / "human_review.csv"
        assert csv_path.exists()

    def test_human_review_preserves_existing_verdicts(self, tmp_path):
        out_dir = tmp_path / "qa_preserve"
        out_dir.mkdir()
        csv_path = out_dir / "human_review.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "comparison_path", "processing_status",
                             "eye_recovery_status", "brow_recovery_status",
                             "original_left_eye_pixels", "fused_left_eye_pixels",
                             "original_right_eye_pixels", "fused_right_eye_pixels",
                             "original_left_brow_pixels", "fused_left_brow_pixels",
                             "original_right_brow_pixels", "fused_right_brow_pixels",
                             "original_eye_glass_pixels", "fused_eye_glass_pixels",
                             "eye_glasses_interaction_summary",
                             "suggested_qa_label", "human_verdict", "human_notes"])
            writer.writerow(["test.png", "", "SUCCESS", "L:UNCHANGED R:UNCHANGED", "L:UNCHANGED R:UNCHANGED",
                             "100", "110", "90", "95", "50", "55", "45", "48", "200", "190",
                             "", "PENDING_REVIEW", "PASS", "Looks good"])

        qa = ParserQA(output_dir=str(out_dir))
        results = [ImageResult(image_name="test.png", processing_status="SUCCESS")]
        qa._write_human_review_csv(results)

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["human_verdict"] == "PASS"
        assert rows[0]["human_notes"] == "Looks good"

    def test_suggested_qa_label_generation(self, tmp_path):
        r = ImageResult(image_name="test.png", processing_status="SUCCESS")
        r.eye_recovery_status = "L:RECOVERED R:UNCHANGED"
        r.brow_recovery_status = "L:UNCHANGED R:UNCHANGED"
        qa = ParserQA(output_dir=str(tmp_path / "tmp"))
        label = qa._suggest_qa_label(r)
        assert label == "IMPROVEMENT"

    def test_suggested_label_oversegmentation(self, tmp_path):
        r = ImageResult(image_name="test.png", processing_status="SUCCESS")
        r.eye_recovery_status = "L:POSSIBLE_OVERSEGMENTATION R:UNCHANGED"
        r.brow_recovery_status = "L:UNCHANGED R:UNCHANGED"
        qa = ParserQA(output_dir=str(tmp_path / "tmp"))
        label = qa._suggest_qa_label(r)
        assert label == "SUSPECTED_OVERSEGMENTATION"

    def test_suggested_label_failure(self, tmp_path):
        r = ImageResult(image_name="test.png", processing_status="SUCCESS")
        r.eye_recovery_status = "L:REDUCED R:UNCHANGED"
        r.brow_recovery_status = "L:UNCHANGED R:UNCHANGED"
        qa = ParserQA(output_dir=str(tmp_path / "tmp"))
        label = qa._suggest_qa_label(r)
        assert label == "SUSPECTED_FAILURE"

    def test_no_ground_truth_limitation(self):
        assert "No ground-truth" in "No ground-truth masks exist for these production-like images."

    def test_parse_both_modes_same_aligned(self):
        img = _make_aligned()
        orig, orig_t, fused, fused_t = _parse_both_modes(img)
        assert orig.mask.shape == fused.mask.shape
        assert orig_t > 0
        assert fused_t > 0
