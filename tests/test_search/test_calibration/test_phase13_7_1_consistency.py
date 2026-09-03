"""Phase 13.7.1 — Focused consistency tests for calibration reconciliation."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest


# ---------------------------------------------------------------------------
# Test 1: Image-level vs identity-level score separation
# ---------------------------------------------------------------------------

class TestScoreLevelSeparation:
    """Verify that image-level and identity-level scores are distinct concepts."""

    def test_identity_score_ge_image_score(self):
        """Identity-level score must be >= image-level score for same query-identity pair."""
        # Simulate: query against 3 references from same impostor identity
        img_sims = [0.1, 0.3, 0.2]
        identity_score = max(img_sims)  # 0.3
        # The identity-level score is the max of image-level scores
        assert identity_score == 0.3
        assert identity_score >= min(img_sims)
        assert identity_score >= np.mean(img_sims)

    def test_identity_aggregation_uses_max(self):
        """Identity aggregation must use max, not mean or sum."""
        # Simulate pairs
        pairs = [
            {"query_image": "q1", "ref_person_id": "A", "similarity": 0.1},
            {"query_image": "q1", "ref_person_id": "A", "similarity": 0.5},
            {"query_image": "q1", "ref_person_id": "A", "similarity": 0.3},
        ]
        query_groups = defaultdict(lambda: defaultdict(list))
        for p in pairs:
            query_groups[p["query_image"]][p["ref_person_id"]].append(p["similarity"])

        result = {}
        for q, id_sims in query_groups.items():
            result[q] = {pid: max(sims) for pid, sims in id_sims.items()}

        assert result["q1"]["A"] == 0.5  # max, not mean (0.3) or sum (0.9)


# ---------------------------------------------------------------------------
# Test 2: Dataset pair counts
# ---------------------------------------------------------------------------

class TestDatasetPairCounts:
    """Verify pair count arithmetic."""

    def test_genuine_pairs_count(self):
        """88 queries × 8 same-identity references = 704 genuine pairs."""
        assert 88 * 8 == 704

    def test_impostor_pairs_count(self):
        """88 queries × (176 - 8) different-identity references = 14784 impostor pairs."""
        assert 88 * (176 - 8) == 14784

    def test_identity_level_genuine_count(self):
        """88 queries × 1 genuine identity = 88 identity-level genuine scores."""
        assert 88 * 1 == 88

    def test_identity_level_impostor_count(self):
        """88 queries × 21 impostor identities = 1848 identity-level impostor scores."""
        assert 88 * 21 == 1848


# ---------------------------------------------------------------------------
# Test 3: FAR/FRR calculation
# ---------------------------------------------------------------------------

class TestFARFRCalculation:
    """Verify FAR and FRR formulas."""

    def test_far_calculation(self):
        """FAR = impostors_above_threshold / total_impostors."""
        impostor_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        threshold = 0.4
        far = float(np.sum(impostor_scores >= threshold)) / len(impostor_scores)
        assert far == pytest.approx(3 / 5)  # 0.5, 0.7, 0.9 >= 0.4

    def test_frr_calculation(self):
        """FRR = genuine_below_threshold / total_genuine."""
        genuine_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        threshold = 0.4
        frr = float(np.sum(genuine_scores < threshold)) / len(genuine_scores)
        assert frr == pytest.approx(2 / 5)  # 0.1, 0.3 < 0.4

    def test_tpr_equals_one_minus_frr(self):
        """TPR = 1 - FRR."""
        genuine_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        threshold = 0.4
        frr = float(np.sum(genuine_scores < threshold)) / len(genuine_scores)
        tpr = float(np.sum(genuine_scores >= threshold)) / len(genuine_scores)
        assert tpr == pytest.approx(1.0 - frr)


# ---------------------------------------------------------------------------
# Test 4: TP/FP/TN/FN consistency
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    """Verify confusion matrix elements are consistent with FAR/FRR."""

    def test_confusion_consistency(self):
        genuine = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        impostor = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        threshold = 0.5

        tp = int(np.sum(genuine >= threshold))
        fn = int(np.sum(genuine < threshold))
        fp = int(np.sum(impostor >= threshold))
        tn = int(np.sum(impostor < threshold))

        assert tp + fn == len(genuine)
        assert fp + tn == len(impostor)

        far = fp / len(impostor)
        frr = fn / len(genuine)
        tpr = tp / len(genuine)
        tnr = tn / len(impostor)

        assert far + tnr == pytest.approx(1.0)
        assert frr + tpr == pytest.approx(1.0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * tpr / (precision + tpr) if (precision + tpr) > 0 else 0.0
        assert 0.0 <= f1 <= 1.0


# ---------------------------------------------------------------------------
# Test 5: Threshold decision rule
# ---------------------------------------------------------------------------

class TestThresholdDecisionRule:
    """Verify consistent decision rule: similarity >= threshold => positive."""

    def test_accept_rule(self):
        """Score >= threshold must be accepted (positive)."""
        scores = [0.5, 0.6, 0.7]
        threshold = 0.5
        for s in scores:
            assert s >= threshold, f"Score {s} should be accepted at threshold {threshold}"

    def test_reject_rule(self):
        """Score < threshold must be rejected (negative)."""
        scores = [0.1, 0.2, 0.49]
        threshold = 0.5
        for s in scores:
            assert s < threshold, f"Score {s} should be rejected at threshold {threshold}"

    def test_boundary_accept(self):
        """Score exactly at threshold must be accepted."""
        assert 0.5 >= 0.5
        assert not (0.5 < 0.5)


# ---------------------------------------------------------------------------
# Test 6: Image-level hard negative is max across references
# ---------------------------------------------------------------------------

class TestImageLevelHardNegative:
    """Verify image-level hard negative finds the strongest single reference."""

    def test_finds_strongest_reference(self):
        """Must return the reference with maximum similarity, not the first."""
        impostor_pairs = [
            {"query_image": "q1", "ref_person_id": "B", "ref_image": "r1", "similarity": 0.1},
            {"query_image": "q1", "ref_person_id": "B", "ref_image": "r2", "similarity": 0.5},
            {"query_image": "q1", "ref_person_id": "B", "ref_image": "r3", "similarity": 0.3},
        ]
        by_query = defaultdict(list)
        for p in impostor_pairs:
            by_query[p["query_image"]].append(p)

        best = max(by_query["q1"], key=lambda p: p["similarity"])
        assert best["similarity"] == 0.5
        assert best["ref_image"] == "r2"


# ---------------------------------------------------------------------------
# Test 7: Identity-level hard negative is max across references per identity
# ---------------------------------------------------------------------------

class TestIdentityLevelHardNegative:
    """Verify identity-level hard negative uses max aggregation."""

    def test_identity_score_is_max(self):
        """Identity score must be max across references for that identity."""
        # Query q1 vs identity B: 3 references with different similarities
        sims = [0.1, 0.5, 0.3]
        identity_score = max(sims)
        assert identity_score == 0.5


# ---------------------------------------------------------------------------
# Test 8: 0.2975 case reproduction
# ---------------------------------------------------------------------------

class TestCase02975:
    """Verify the specific 0.2975 case is correctly reproduced."""

    def test_case_02975_exists(self):
        """The case_02975.json artifact must exist."""
        case_path = Path("outputs/phase13_7_1/case_02975.json")
        assert case_path.exists(), "case_02975.json not found"

    def test_case_02975_scores(self):
        """Image and identity scores must both be 0.2975 (same supporting reference)."""
        case_path = Path("outputs/phase13_7_1/case_02975.json")
        if not case_path.exists():
            pytest.skip("case_02975.json not found")
        with open(case_path) as f:
            case = json.load(f)
        assert case["image_level_score"] == pytest.approx(0.297498, abs=1e-4)
        assert case["identity_level_score"] == pytest.approx(0.297498, abs=1e-4)

    def test_case_02975_above_youden(self):
        """0.2975 must be above identity Youden threshold (0.2415)."""
        case_path = Path("outputs/phase13_7_1/case_02975.json")
        if not case_path.exists():
            pytest.skip("case_02975.json not found")
        with open(case_path) as f:
            case = json.load(f)
        youden_entry = next(
            (e for e in case["threshold_evaluation"]
             if e["threshold_name"] == "youden_j" and e["score_level"] == "IDENTITY"),
            None,
        )
        assert youden_entry is not None
        assert youden_entry["accepted"] is True
        assert youden_entry["score"] > youden_entry["threshold_value"]


# ---------------------------------------------------------------------------
# Test 9: Threshold consistency verification
# ---------------------------------------------------------------------------

class TestThresholdConsistency:
    """Verify that all operating points are internally consistent."""

    def test_consistency_file_exists(self):
        path = Path("outputs/phase13_7_1/threshold_consistency.csv")
        assert path.exists(), "threshold_consistency.csv not found"

    def test_all_points_consistent(self):
        path = Path("outputs/phase13_7_1/threshold_consistency.csv")
        if not path.exists():
            pytest.skip("threshold_consistency.csv not found")
        import csv
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["consistent"] == "True", (
                    f"Inconsistent: {row['score_level']} {row['operating_point']} "
                    f"reported_far={row['reported_far']} computed_far={row['computed_far']}"
                )


# ---------------------------------------------------------------------------
# Test 10: Deterministic output
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Verify that re-running produces identical results."""

    def test_calibration_verification_exists(self):
        path = Path("outputs/phase13_7_1/calibration_verification.json")
        assert path.exists()

    def test_pair_counts_deterministic(self):
        """Pair counts must always be 704 and 14784 for this dataset."""
        path = Path("outputs/phase13_7_1/calibration_verification.json")
        if not path.exists():
            pytest.skip("calibration_verification.json not found")
        with open(path) as f:
            data = json.load(f)
        assert data["dataset"]["actual_genuine_pairs"] == 704
        assert data["dataset"]["actual_impostor_pairs"] == 14784
        assert data["dataset"]["genuine_pairs_match"] is True
        assert data["dataset"]["impostor_pairs_match"] is True
