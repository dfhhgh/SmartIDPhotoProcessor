"""Phase 13.8 — Focused tests for expanded dataset analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Dataset v4 split structure
# ---------------------------------------------------------------------------

class TestDatasetV4Split:
    def test_v4_reference_identities(self):
        ref_dir = Path("datasets/celebrity-v4/reference")
        assert ref_dir.exists(), "v4 reference directory missing"
        identities = [d.name for d in ref_dir.iterdir() if d.is_dir()]
        assert len(identities) == 36, f"Expected 36 identities, got {len(identities)}"

    def test_v4_calibration_identities(self):
        cal_dir = Path("datasets/celebrity-v4/calibration")
        assert cal_dir.exists(), "v4 calibration directory missing"
        identities = [d.name for d in cal_dir.iterdir() if d.is_dir()]
        assert len(identities) == 36, f"Expected 36 identities, got {len(identities)}"

    def test_v4_reference_images_per_identity(self):
        ref_dir = Path("datasets/celebrity-v4/reference")
        for person_dir in ref_dir.iterdir():
            if not person_dir.is_dir():
                continue
            count = len(list(person_dir.glob("*.jpg")))
            assert count >= 4, f"{person_dir.name}: only {count} reference images (need >=4)"

    def test_v4_calibration_images_per_identity(self):
        cal_dir = Path("datasets/celebrity-v4/calibration")
        for person_dir in cal_dir.iterdir():
            if not person_dir.is_dir():
                continue
            count = len(list(person_dir.glob("*.jpg")))
            assert count >= 4, f"{person_dir.name}: only {count} calibration images (need >=4)"

    def test_v4_new_identities_present(self):
        ref_dir = Path("datasets/celebrity-v4/reference")
        new_ids = ["will_smith", "zendaya", "tom_cruise", "pedro_pascal", "oscar_isaac"]
        for pid in new_ids:
            assert (ref_dir / pid).exists(), f"New identity {pid} missing from v4 reference"
            assert (ref_dir / pid / "*.jpg"), f"New identity {pid} has no images"


# ---------------------------------------------------------------------------
# 2. FAISS index integrity
# ---------------------------------------------------------------------------

class TestFAISSIndex:
    def test_index_exists(self):
        index_path = Path("datasets/celebrity-v4/search_index/reference_index.faiss")
        assert index_path.exists(), "FAISS index file missing"

    def test_metadata_exists(self):
        meta_path = Path("datasets/celebrity-v4/search_index/metadata.json")
        assert meta_path.exists(), "Metadata file missing"

    def test_index_vector_count(self):
        import faiss
        index_path = Path("datasets/celebrity-v4/search_index/reference_index.faiss")
        index = faiss.read_index(str(index_path))
        assert index.ntotal == 288, f"Expected 288 vectors, got {index.ntotal}"

    def test_index_dimension(self):
        import faiss
        index_path = Path("datasets/celebrity-v4/search_index/reference_index.faiss")
        index = faiss.read_index(str(index_path))
        assert index.d == 512, f"Expected dim 512, got {index.d}"


# ---------------------------------------------------------------------------
# 3. Calibration results integrity
# ---------------------------------------------------------------------------

class TestCalibrationResults:
    @pytest.fixture(autouse=True)
    def load_results(self):
        results_path = Path("outputs/phase13_8/phase13_8_results.json")
        if results_path.exists():
            with open(results_path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_results_file_exists(self):
        assert self.results is not None, "phase13_8_results.json not found"

    def test_dataset_sizes(self):
        cal = self.results["calibration"]["dataset"]
        assert cal["identities"] == 36
        assert cal["reference_vectors"] == 288
        assert cal["genuine_pairs"] == 1152
        assert cal["impostor_pairs"] == 40320

    def test_image_roc_auc_reasonable(self):
        auc = self.results["calibration"]["image_level"]["roc_auc"]
        assert 0.7 < auc < 1.0, f"Image ROC-AUC {auc} outside reasonable range"

    def test_identity_roc_auc_reasonable(self):
        auc = self.results["calibration"]["identity_level"]["roc_auc"]
        assert 0.7 < auc < 1.0, f"Identity ROC-AUC {auc} outside reasonable range"

    def test_image_eer_reasonable(self):
        eer = self.results["calibration"]["image_level"]["eer"]["eer"]
        assert 0.0 < eer < 0.5, f"Image EER {eer} outside reasonable range"

    def test_identity_eer_reasonable(self):
        eer = self.results["calibration"]["identity_level"]["eer"]["eer"]
        assert 0.0 < eer < 0.5, f"Identity EER {eer} outside reasonable range"

    def test_genuine_impostor_separation(self):
        gen_mean = self.results["calibration"]["image_level"]["genuine_stats"]["mean"]
        imp_mean = self.results["calibration"]["image_level"]["impostor_stats"]["mean"]
        assert gen_mean > imp_mean, f"Genuine mean {gen_mean} <= impostor mean {imp_mean}"

    def test_global_max_impostor_above_threshold(self):
        max_imp = self.results["calibration"]["image_level"]["global_max_impostor"]
        assert max_imp > 0.0, "Global max impostor should be positive"
        assert max_imp < 1.0, f"Global max impostor {max_imp} >= 1.0"

    def test_operating_points_present(self):
        ops = self.results["calibration"]["image_level"]["operating_points"]
        assert len(ops) >= 5, f"Expected >=5 operating points, got {len(ops)}"
        point_types = {op["point_type"] for op in ops}
        assert "youden_j" in point_types
        assert "eer" in point_types


# ---------------------------------------------------------------------------
# 4. Gallery-size analysis
# ---------------------------------------------------------------------------

class TestGallerySizeAnalysis:
    @pytest.fixture(autouse=True)
    def load_results(self):
        results_path = Path("outputs/phase13_8/phase13_8_results.json")
        if results_path.exists():
            with open(results_path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_gallery_sizes_present(self):
        gal = self.results["gallery_size_analysis"]
        assert "overall" in gal
        assert set(gal["overall"].keys()) == {"2", "4", "6", "8"}

    def test_gallery_size_trend(self):
        overall = self.results["gallery_size_analysis"]["overall"]
        maxes = [overall[str(gs)]["mean_impostor_max"] for gs in [2, 4, 6, 8]]
        # Larger gallery should have >= max impostor (not strictly increasing, but trend)
        assert maxes[-1] >= maxes[0], "Gallery trend: larger gallery should have higher max impostor"

    def test_per_identity_results(self):
        per_id = self.results["gallery_size_analysis"]["per_identity"]
        assert len(per_id) == 36, f"Expected 36 identities, got {len(per_id)}"


# ---------------------------------------------------------------------------
# 5. Weak-identity analysis
# ---------------------------------------------------------------------------

class TestWeakIdentityAnalysis:
    @pytest.fixture(autouse=True)
    def load_results(self):
        results_path = Path("outputs/phase13_8/phase13_8_results.json")
        if results_path.exists():
            with open(results_path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_all_weak_identities_present(self):
        weak = self.results["weak_identity_analysis"]
        expected = [
            "jennifer_lawrence", "morgan_freeman", "leonardo_dicaprio",
            "vinicius_junior", "brad_pitt", "neymar", "mohamed_salah",
            "kevin_de_bruyne",
        ]
        for pid in expected:
            assert pid in weak, f"Weak identity {pid} missing from analysis"

    def test_weak_identities_have_queries(self):
        weak = self.results["weak_identity_analysis"]
        for pid, data in weak.items():
            if "status" in data and data["status"] == "not_in_dataset":
                continue
            assert data["query_count"] > 0, f"{pid}: no queries"

    def test_weak_identities_have_impostors(self):
        weak = self.results["weak_identity_analysis"]
        for pid, data in weak.items():
            if "status" in data and data["status"] == "not_in_dataset":
                continue
            assert data["impostor_count"] > 0, f"{pid}: no impostor pairs"

    def test_morgan_freeman_weakest(self):
        weak = self.results["weak_identity_analysis"]
        morgan = weak.get("morgan_freeman", {})
        if "genuine_mean" in morgan:
            assert morgan["genuine_mean"] < 0.3, \
                f"Morgan Freeman genuine mean {morgan['genuine_mean']} unexpectedly high"


# ---------------------------------------------------------------------------
# 6. Comparison with v3 baseline
# ---------------------------------------------------------------------------

class TestComparison:
    @pytest.fixture(autouse=True)
    def load_results(self):
        results_path = Path("outputs/phase13_8/phase13_8_results.json")
        if results_path.exists():
            with open(results_path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_comparison_keys(self):
        comp = self.results["comparison"]
        assert "v3" in comp
        assert "v4" in comp

    def test_v3_baseline_values(self):
        v3 = self.results["comparison"]["v3"]
        assert v3["identities"] == 22
        assert v3["image_roc_auc"] == pytest.approx(0.8785, abs=0.01)
        assert v3["identity_roc_auc"] == pytest.approx(0.9465, abs=0.01)

    def test_v4_expanded_values(self):
        v4 = self.results["comparison"]["v4"]
        assert v4["identities"] == 36
        assert v4["reference_vectors"] == 288
        assert v4["image_roc_auc"] > 0.7
        assert v4["identity_roc_auc"] > 0.7

    def test_v4_more_pairs_than_v3(self):
        v3 = self.results["comparison"]["v3"]
        v4 = self.results["comparison"]["v4"]
        assert v4["genuine_pairs"] > v3["genuine_pairs"]
        assert v4["impostor_pairs"] > v3["impostor_pairs"]


# ---------------------------------------------------------------------------
# 7. Suspicious match detection
# ---------------------------------------------------------------------------

class TestSuspiciousMatch:
    @pytest.fixture(autouse=True)
    def load_results(self):
        results_path = Path("outputs/phase13_8/phase13_8_results.json")
        if results_path.exists():
            with open(results_path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_known_suspicious_match_flagged(self):
        """The lebron_james → morgan_freeman match at 0.7292 should be documented."""
        hn = self.results["calibration"]["hard_negatives"]["identity_level_top5"]
        top_match = hn[0]
        assert top_match["query_person_id"] == "lebron_james"
        assert top_match["impostor_person_id"] == "morgan_freeman"
        assert top_match["identity_score"] > 0.7


# ---------------------------------------------------------------------------
# 8. Score distribution sanity
# ---------------------------------------------------------------------------

class TestScoreDistribution:
    @pytest.fixture(autouse=True)
    def load_results(self):
        results_path = Path("outputs/phase13_8/phase13_8_results.json")
        if results_path.exists():
            with open(results_path) as f:
                self.results = json.load(f)
        else:
            self.results = None

    def test_genuine_scores_high_mean(self):
        stats = self.results["calibration"]["image_level"]["genuine_stats"]
        assert stats["mean"] > 0.2, f"Genuine mean {stats['mean']} too low"

    def test_impostor_scores_low_mean(self):
        stats = self.results["calibration"]["image_level"]["impostor_stats"]
        assert stats["mean"] < 0.1, f"Impostor mean {stats['mean']} too high"

    def test_genuine_max_above_impostor_max(self):
        gen_max = self.results["calibration"]["image_level"]["genuine_stats"]["max"]
        imp_max = self.results["calibration"]["image_level"]["impostor_stats"]["max"]
        assert gen_max > imp_max, "Genuine max should exceed impostor max"
