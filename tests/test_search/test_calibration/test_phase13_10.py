"""Focused tests for Phase 13.10 — Stress & Robustness Validation."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

RESULTS_DIR = Path("outputs/phase13_10")
STRESS_DIR = Path("datasets/non_celebrity-v1/stress_variants")
BASELINE_PATH = RESULTS_DIR / "baseline_results.json"
STRESS_PATH = RESULTS_DIR / "stress_results.json"
GALLERY_PATH = RESULTS_DIR / "gallery_size_results.json"
HN_PATH = RESULTS_DIR / "hard_negative_analysis.json"
DEGRADATION_PATH = RESULTS_DIR / "failure_analysis.json"
INTEGRITY_PATH = RESULTS_DIR / "integrity_checks.json"

CELEBRITY_THRESHOLDS = {
    "eer": 0.0457, "youden_j": 0.2301, "far_5pct": 0.1039,
    "far_1pct": 0.1554, "far_0_5pct": 0.1816, "far_0_1pct": 0.2301,
}

EXPECTED_CONDITIONS = [
    "appearance_color_jitter", "combined_mild", "glasses_rimless",
    "lighting_underexpose", "pose_yaw_left_10", "quality_jpeg_q30",
]


def _load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# 1. FILE EXISTENCE
# ============================================================

class TestFileExistence:
    def test_baseline_results_exist(self):
        assert BASELINE_PATH.exists()

    def test_stress_results_exist(self):
        assert STRESS_PATH.exists()

    def test_gallery_size_results_exist(self):
        assert GALLERY_PATH.exists()

    def test_hard_negative_analysis_exist(self):
        assert HN_PATH.exists()

    def test_degradation_analysis_exist(self):
        assert DEGRADATION_PATH.exists()

    def test_integrity_checks_exist(self):
        assert INTEGRITY_PATH.exists()


# ============================================================
# 2. BASELINE REPRODUCTION
# ============================================================

class TestBaselineReproduction:
    def setup_method(self):
        self.baseline = _load_json(BASELINE_PATH)

    def test_baseline_roc_auc(self):
        assert self.baseline["image_level"]["roc_auc"] == 1.0

    def test_baseline_eer_near_zero(self):
        assert self.baseline["image_level"]["eer"]["eer"] < 0.001

    def test_baseline_identity_roc_auc(self):
        assert self.baseline["identity_level"]["roc_auc"] == 1.0

    def test_baseline_max_impostor(self):
        assert 0.3 < self.baseline["image_level"]["global_max_impostor"] < 0.4

    def test_baseline_genuine_min_above_impostor_max(self):
        gen_min = self.baseline["image_level"]["genuine_stats"]["min"]
        imp_max = self.baseline["image_level"]["global_max_impostor"]
        assert gen_min > imp_max, f"genuine_min={gen_min} <= impostor_max={imp_max}"

    def test_baseline_fixed_thresholds_no_fn(self):
        for name, tv in self.baseline["fixed_thresholds"].items():
            assert tv["fn"] == 0, f"{name}: FN={tv['fn']}"

    def test_baseline_identities(self):
        assert self.baseline["dataset"]["identities"] == 102

    def test_baseline_reference_vectors(self):
        assert self.baseline["dataset"]["reference_vectors"] == 612


# ============================================================
# 3. STRESS RESULTS STRUCTURE
# ============================================================

class TestStressResultsStructure:
    def setup_method(self):
        self.stress = _load_json(STRESS_PATH)

    def test_all_conditions_present(self):
        for cond in EXPECTED_CONDITIONS:
            assert cond in self.stress, f"Missing condition: {cond}"

    def test_condition_count(self):
        assert len(self.stress) == len(EXPECTED_CONDITIONS)

    def test_each_condition_has_image_level(self):
        for cond, result in self.stress.items():
            assert "image_level" in result, f"{cond}: missing image_level"
            assert "roc_auc" in result["image_level"]
            assert "eer" in result["image_level"]
            assert "global_max_impostor" in result["image_level"]
            assert "genuine_stats" in result["image_level"]
            assert "impostor_stats" in result["image_level"]

    def test_each_condition_has_identity_level(self):
        for cond, result in self.stress.items():
            assert "identity_level" in result, f"{cond}: missing identity_level"

    def test_each_condition_has_fixed_thresholds(self):
        for cond, result in self.stress.items():
            assert "fixed_thresholds" in result, f"{cond}: missing fixed_thresholds"
            for tname in CELEBRITY_THRESHOLDS:
                assert tname in result["fixed_thresholds"], f"{cond}: missing threshold {tname}"

    def test_each_condition_has_hard_negatives(self):
        for cond, result in self.stress.items():
            assert "hard_negatives" in result, f"{cond}: missing hard_negatives"

    def test_each_condition_has_gallery_sizes(self):
        gallery = _load_json(GALLERY_PATH)
        for cond in EXPECTED_CONDITIONS:
            assert cond in gallery, f"{cond}: missing from gallery_size_results.json"


# ============================================================
# 4. DETERMINISTIC TRANSFORMS
# ============================================================

class TestDeterministicTransforms:
    def test_stress_dirs_exist(self):
        for cond in EXPECTED_CONDITIONS:
            assert (STRESS_DIR / cond).exists(), f"Missing dir: {cond}"

    def test_stress_image_count(self):
        for cond in EXPECTED_CONDITIONS:
            cond_dir = STRESS_DIR / cond
            total = sum(1 for _ in cond_dir.rglob("*.jpg"))
            assert total == 408, f"{cond}: {total} images (expected 408)"

    def test_stress_identity_count(self):
        for cond in EXPECTED_CONDITIONS:
            cond_dir = STRESS_DIR / cond
            ids = [d.name for d in cond_dir.iterdir() if d.is_dir()]
            assert len(ids) == 102, f"{cond}: {len(ids)} identities (expected 102)"

    def test_no_original_images_modified(self):
        ref_dir = Path("datasets/non_celebrity-v1/reference")
        held_dir = Path("datasets/non_celebrity-v1/held_out")
        for img in ref_dir.rglob("*.jpg"):
            assert img.exists()
        for img in held_dir.rglob("*.jpg"):
            assert img.exists()


# ============================================================
# 5. ROC-AUC = 1.0 ACROSS ALL CONDITIONS
# ============================================================

class TestPerfectSeparability:
    def setup_method(self):
        self.stress = _load_json(STRESS_PATH)

    def test_image_roc_auc_perfect(self):
        for cond, result in self.stress.items():
            assert result["image_level"]["roc_auc"] == 1.0, f"{cond}: ROC-AUC={result['image_level']['roc_auc']}"

    def test_identity_roc_auc_perfect(self):
        for cond, result in self.stress.items():
            assert result["identity_level"]["roc_auc"] == 1.0, f"{cond}: identity ROC-AUC={result['identity_level']['roc_auc']}"


# ============================================================
# 6. ZERO FALSE REJECTIONS AT YOUDEN-J
# ============================================================

class TestZeroFalseRejections:
    def setup_method(self):
        self.stress = _load_json(STRESS_PATH)

    def test_youden_j_zero_fn(self):
        for cond, result in self.stress.items():
            tv = result["fixed_thresholds"]["youden_j"]
            assert tv["fn"] == 0, f"{cond}: youden_j FN={tv['fn']}"

    def test_youden_j_zero_frr(self):
        for cond, result in self.stress.items():
            tv = result["fixed_thresholds"]["youden_j"]
            assert tv["frr"] == 0.0, f"{cond}: youden_j FRR={tv['frr']}"

    def test_all_thresholds_zero_fn(self):
        for cond, result in self.stress.items():
            for tname in CELEBRITY_THRESHOLDS:
                tv = result["fixed_thresholds"][tname]
                assert tv["fn"] == 0, f"{cond}/{tname}: FN={tv['fn']}"


# ============================================================
# 7. THRESHOLD IMMUTABILITY
# ============================================================

class TestThresholdImmutability:
    def test_thresholds_match_celebrity(self):
        for tname, expected in CELEBRITY_THRESHOLDS.items():
            actual = CELEBRITY_THRESHOLDS[tname]
            assert actual == expected, f"{tname}: {actual} != {expected}"


# ============================================================
# 8. GALLERY SIZE CORRECTNESS
# ============================================================

class TestGallerySize:
    def setup_method(self):
        self.gallery = _load_json(GALLERY_PATH)

    def test_gallery_sizes_present(self):
        for cond in EXPECTED_CONDITIONS:
            gs = self.gallery[cond]
            assert "2" in gs, f"{cond}: missing gallery size 2"
            assert "4" in gs, f"{cond}: missing gallery size 4"
            assert "6" in gs, f"{cond}: missing gallery size 6"

    def test_gallery_size_2_has_204_vectors(self):
        for cond in EXPECTED_CONDITIONS:
            assert self.gallery[cond]["2"]["reference_vectors"] == 204

    def test_gallery_size_4_has_408_vectors(self):
        for cond in EXPECTED_CONDITIONS:
            assert self.gallery[cond]["4"]["reference_vectors"] == 408

    def test_gallery_size_6_has_612_vectors(self):
        for cond in EXPECTED_CONDITIONS:
            assert self.gallery[cond]["6"]["reference_vectors"] == 612

    def test_gallery_size_roc_auc_all_1(self):
        for cond in EXPECTED_CONDITIONS:
            for sz in ["2", "4", "6"]:
                assert self.gallery[cond][sz]["image_roc_auc"] == 1.0, \
                    f"{cond}/size_{sz}: ROC-AUC={self.gallery[cond][sz]['image_roc_auc']}"


# ============================================================
# 9. GLOBAL HARD NEGATIVE CALCULATION
# ============================================================

class TestHardNegatives:
    def setup_method(self):
        self.hn = _load_json(HN_PATH)

    def test_all_conditions_have_hard_negatives(self):
        for cond in EXPECTED_CONDITIONS:
            assert cond in self.hn, f"Missing condition in HN: {cond}"
            assert len(self.hn[cond]["image_level_top5"]) > 0

    def test_hard_negatives_are_impostors(self):
        for cond in EXPECTED_CONDITIONS:
            for pair in self.hn[cond]["image_level_top5"]:
                assert pair["query_person_id"] != pair["impostor_person_id"], \
                    f"{cond}: query={pair['query_person_id']} == impostor={pair['impostor_person_id']}"

    def test_hard_negative_similarities_positive(self):
        for cond in EXPECTED_CONDITIONS:
            for pair in self.hn[cond]["image_level_top5"]:
                assert pair["similarity"] > 0, f"{cond}: sim={pair['similarity']}"


# ============================================================
# 10. METRIC CONSISTENCY
# ============================================================

class TestMetricConsistency:
    def setup_method(self):
        self.stress = _load_json(STRESS_PATH)

    def test_frr_plus_tpr_equals_one(self):
        for cond, result in self.stress.items():
            for tname in CELEBRITY_THRESHOLDS:
                tv = result["fixed_thresholds"][tname]
                assert abs(tv["frr"] + tv["tpr"] - 1.0) < 1e-6, \
                    f"{cond}/{tname}: FRR+FPR={tv['frr']+tv['tpr']}"

    def test_far_plus_tnr_equals_one(self):
        for cond, result in self.stress.items():
            for tname in CELEBRITY_THRESHOLDS:
                tv = result["fixed_thresholds"][tname]
                assert abs(tv["far"] + tv["tnr"] - 1.0) < 1e-6, \
                    f"{cond}/{tname}: FAR+TNR={tv['far']+tv['tnr']}"

    def test_tp_plus_fn_equals_genuine_pairs(self):
        for cond, result in self.stress.items():
            tg = result["dataset"]["genuine_pairs"]
            for tname in CELEBRITY_THRESHOLDS:
                tv = result["fixed_thresholds"][tname]
                assert tv["tp"] + tv["fn"] == tg, \
                    f"{cond}/{tname}: TP+FN={tv['tp']+tv['fn']} != {tg}"

    def test_fp_plus_tn_equals_impostor_pairs(self):
        for cond, result in self.stress.items():
            ti = result["dataset"]["impostor_pairs"]
            for tname in CELEBRITY_THRESHOLDS:
                tv = result["fixed_thresholds"][tname]
                assert tv["fp"] + tv["tn"] == ti, \
                    f"{cond}/{tname}: FP+TN={tv['fp']+tv['tn']} != {ti}"


# ============================================================
# 11. IDENTITY-LEVEL AGGREGATION
# ============================================================

class TestIdentityAggregation:
    def setup_method(self):
        self.stress = _load_json(STRESS_PATH)

    def test_identity_eer_near_zero(self):
        for cond, result in self.stress.items():
            eer = result["identity_level"]["eer"].get("eer", 1.0)
            assert eer < 0.01, f"{cond}: identity EER={eer}"

    def test_identity_genuine_mean_higher_than_image(self):
        for cond, result in self.stress.items():
            id_mean = result["identity_level"]["genuine_stats"]["mean"]
            img_mean = result["image_level"]["genuine_stats"]["mean"]
            assert id_mean >= img_mean, f"{cond}: id_mean={id_mean} < img_mean={img_mean}"


# ============================================================
# 12. BASELINE REPRODUCTION
# ============================================================

class TestBaselineReproduction:
    def setup_method(self):
        self.baseline = _load_json(BASELINE_PATH)

    def test_baseline_roc_auc(self):
        assert self.baseline["image_level"]["roc_auc"] == 1.0

    def test_baseline_eer_near_zero(self):
        assert self.baseline["image_level"]["eer"]["eer"] < 0.001

    def test_baseline_max_impostor_range(self):
        mi = self.baseline["image_level"]["global_max_impostor"]
        assert 0.3 < mi < 0.4

    def test_baseline_genuine_min_above_impostor_max(self):
        gen_min = self.baseline["image_level"]["genuine_stats"]["min"]
        imp_max = self.baseline["image_level"]["global_max_impostor"]
        assert gen_min > imp_max

    def test_baseline_gallery_sizes(self):
        gs = self.baseline.get("gallery_sizes", {})
        if gs:
            assert "2" in gs and "4" in gs and "6" in gs
            for sz in ["2", "4", "6"]:
                assert gs[sz]["image_roc_auc"] == 1.0


# ============================================================
# 13. DEGRADATION BOUNDS
# ============================================================

class TestDegradationBounds:
    def setup_method(self):
        self.deg = _load_json(DEGRADATION_PATH)

    def test_roc_auc_delta_zero(self):
        for cond, d in self.deg.items():
            if d:
                assert d["delta_roc_auc"] == 0.0, f"{cond}: delta_roc_auc={d['delta_roc_auc']}"

    def test_eer_delta_bounded(self):
        for cond, d in self.deg.items():
            if d:
                assert abs(d["delta_eer"]) < 0.001, f"{cond}: delta_eer={d['delta_eer']}"

    def test_max_impostor_delta_bounded(self):
        for cond, d in self.deg.items():
            if d:
                assert abs(d["delta_max_impostor"]) < 0.05, \
                    f"{cond}: delta_max_imp={d['delta_max_impostor']}"


# ============================================================
# 14. INTEGRITY CHECKS
# ============================================================

class TestIntegrity:
    def setup_method(self):
        self.integrity = _load_json(INTEGRITY_PATH)

    def test_no_leakage(self):
        for cond, check in self.integrity.items():
            assert check["leakage"] == 0, f"{cond}: leakage={check['leakage']}"

    def test_correct_image_count(self):
        for cond, check in self.integrity.items():
            assert check["total_images"] == 408, f"{cond}: images={check['total_images']}"

    def test_correct_identity_count(self):
        for cond, check in self.integrity.items():
            assert check["identities"] == 102, f"{cond}: identities={check['identities']}"
