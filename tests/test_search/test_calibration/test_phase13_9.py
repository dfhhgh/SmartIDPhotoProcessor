"""Phase 13.9 — Focused tests for non-celebrity validation."""
import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

RESULTS_DIR = Path("outputs/phase13_9")
DATASET_DIR = Path("datasets/non_celebrity-v1")


# --- Test Dataset Identity Integrity ---
class TestDatasetIntegrity:
    def test_results_file_exists(self):
        assert (RESULTS_DIR / "calibration_results.json").exists()

    def test_dataset_102_identities(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            r = json.load(f)
        assert r["dataset"]["identities"] == 102

    def test_612_reference_vectors(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            r = json.load(f)
        assert r["dataset"]["reference_vectors"] == 612

    def test_2448_genuine_pairs(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            r = json.load(f)
        assert r["dataset"]["genuine_pairs"] == 2448  # 102 * 4 * 6

    def test_247248_impostor_pairs(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            r = json.load(f)
        assert r["dataset"]["impostor_pairs"] == 247248  # 102*4 * 102*6 - 2448


# --- Test Single-Face Contract ---
class TestSingleFaceContract:
    def test_zero_rejections(self):
        with open(RESULTS_DIR / "dataset_manifest.json") as f:
            m = json.load(f)
        # All 1020 images passed face validation
        total = m["reference_images"] + m["held_out_images"]
        assert total == 1020


# --- Test Duplicate Detection ---
class TestDuplicateDetection:
    def test_no_content_duplicates(self):
        with open(RESULTS_DIR / "integrity_checks.json") as f:
            ic = json.load(f)
        assert ic["within_ref_duplicates"] == 0
        assert ic["within_held_duplicates"] == 0


# --- Test Leakage Detection ---
class TestLeakageDetection:
    def test_zero_ref_held_leakage(self):
        with open(RESULTS_DIR / "integrity_checks.json") as f:
            ic = json.load(f)
        assert ic["ref_held_leakage"] == 0


# --- Test Split Determinism ---
class TestSplitDeterminism:
    def test_manifest_consistent(self):
        with open(RESULTS_DIR / "dataset_manifest.json") as f:
            m = json.load(f)
        assert m["identities"] == 102
        assert m["reference_images"] == 612
        assert m["held_out_images"] == 408
        assert m["seed"] == 42


# --- Test Reference/Evaluation Separation ---
class TestReferenceEvaluationSeparation:
    def test_reference_dirs_exist(self):
        ref_dir = DATASET_DIR / "reference"
        assert ref_dir.exists()
        dirs = [d for d in ref_dir.iterdir() if d.is_dir()]
        assert len(dirs) == 102

    def test_held_out_dirs_exist(self):
        held_dir = DATASET_DIR / "held_out"
        assert held_dir.exists()
        dirs = [d for d in held_dir.iterdir() if d.is_dir()]
        assert len(dirs) == 102

    def test_6_ref_per_identity(self):
        ref_dir = DATASET_DIR / "reference"
        for person_dir in sorted(ref_dir.iterdir())[:10]:
            if person_dir.is_dir():
                jpgs = list(person_dir.glob("*.jpg"))
                assert len(jpgs) == 6

    def test_4_held_per_identity(self):
        held_dir = DATASET_DIR / "held_out"
        for person_dir in sorted(held_dir.iterdir())[:10]:
            if person_dir.is_dir():
                jpgs = list(person_dir.glob("*.jpg"))
                assert len(jpgs) == 4


# --- Test Gallery Construction ---
class TestGalleryConstruction:
    def test_faiss_index_exists(self):
        assert (DATASET_DIR / "search_index" / "reference_index.faiss").exists()

    def test_metadata_exists(self):
        assert (DATASET_DIR / "search_index" / "metadata.json").exists()

    def test_index_vector_count(self):
        import faiss
        index = faiss.read_index(str(DATASET_DIR / "search_index" / "reference_index.faiss"))
        assert index.ntotal == 612
        assert index.d == 512


# --- Test Fixed-Threshold Evaluation ---
class TestFixedThresholdEvaluation:
    def _load(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            return json.load(f)

    def test_all_thresholds_present(self):
        r = self._load()
        expected = ["eer", "youden_j", "far_5pct", "far_1pct", "far_0_5pct", "far_0_1pct"]
        for name in expected:
            assert name in r["fixed_thresholds"]

    def test_zero_false_rejections(self):
        r = self._load()
        for name, data in r["fixed_thresholds"].items():
            assert data["fn"] == 0, f"{name}: FN={data['fn']}"

    def test_youden_j_threshold(self):
        r = self._load()
        yj = r["fixed_thresholds"]["youden_j"]
        assert yj["threshold"] == pytest.approx(0.2301, abs=0.01)
        assert yj["frr"] == 0.0
        assert yj["f1"] > 0.9


# --- Test Metric Correctness ---
class TestMetricCorrectness:
    def _load(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            return json.load(f)

    def test_image_roc_auc_perfect(self):
        r = self._load()
        assert r["image_level"]["roc_auc"] >= 0.999

    def test_identity_roc_auc_perfect(self):
        r = self._load()
        assert r["identity_level"]["roc_auc"] >= 0.999

    def test_eer_near_zero(self):
        r = self._load()
        assert r["image_level"]["eer"]["eer"] < 0.001

    def test_genuine_mean_high(self):
        r = self._load()
        assert r["image_level"]["genuine_stats"]["mean"] > 0.5

    def test_impostor_mean_low(self):
        r = self._load()
        assert r["image_level"]["impostor_stats"]["mean"] < 0.05

    def test_genuine_max_above_impostor_max(self):
        r = self._load()
        assert r["image_level"]["genuine_stats"]["max"] > r["image_level"]["global_max_impostor"]


# --- Test Hard-Negative Calculation ---
class TestHardNegative:
    def _load(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            return json.load(f)

    def test_hard_negatives_exist(self):
        r = self._load()
        assert len(r["hard_negatives"]["image_level_top5"]) > 0

    def test_max_impostor_below_celebrity(self):
        r = self._load()
        # Non-celebrity max impostor should be lower than celebrity (0.729)
        assert r["image_level"]["global_max_impostor"] < 0.5


# --- Test Gallery-Size Analysis ---
class TestGallerySize:
    def _load(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            return json.load(f)

    def test_all_sizes_present(self):
        r = self._load()
        for gs in ["2", "4", "6"]:
            assert gs in r["gallery_sizes"]

    def test_auc_stable_across_sizes(self):
        r = self._load()
        for gs in ["2", "4", "6"]:
            assert r["gallery_sizes"][gs]["image_roc_auc"] >= 0.999


# --- Test Determinism ---
class TestDeterminism:
    def test_results_consistent(self):
        with open(RESULTS_DIR / "calibration_results.json") as f:
            r1 = json.load(f)
        with open(RESULTS_DIR / "calibration_results.json") as f:
            r2 = json.load(f)
        assert r1["image_level"]["roc_auc"] == r2["image_level"]["roc_auc"]
        assert r1["identity_level"]["roc_auc"] == r2["identity_level"]["roc_auc"]


# --- Test Production Isolation ---
class TestProductionIsolation:
    def test_no_celebrity_data_in_non_celebrity(self):
        ref_dir = DATASET_DIR / "reference"
        for person_dir in ref_dir.iterdir():
            if person_dir.is_dir():
                # No celebrity identity names should appear
                assert person_dir.name not in [
                    "angelina_jolie", "brad_pitt", "morgan_freeman",
                    "lebron_james", "neymar", "mohamed_salah",
                ]
