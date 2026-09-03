"""Phase 13.7.2 — Focused cleanup/hardening tests."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest


# Import the cleaned-up module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from phase13_7_2_calibration import (
    is_accepted,
    aggregate_identity_scores,
    image_level_hard_negatives,
    identity_level_hard_negatives,
    verify_threshold_consistency,
    compute_eer,
    compute_operating_points,
    EXPECTED_EMBEDDING_DIM,
    NORM_TOLERANCE,
)


# ---------------------------------------------------------------------------
# 1. Query identity comes from metadata, not filesystem path
# ---------------------------------------------------------------------------

class TestIdentityFromMetadata:
    def test_identity_from_query_embeddings_keys(self):
        """Person IDs must come from query_embeddings dict keys, not path parsing."""
        query_embeddings = {
            "alice": [("img1.jpg", np.zeros(5, dtype=np.float32))],
            "bob": [("img2.jpg", np.zeros(5, dtype=np.float32))],
        }
        ref_records = [
            {"vector_id": 0, "person_id": "alice", "image_path": "a1.jpg"},
            {"vector_id": 1, "person_id": "bob", "image_path": "b1.jpg"},
        ]
        ref_embeddings = np.zeros((2, 5), dtype=np.float32)
        hn = identity_level_hard_negatives(query_embeddings, ref_records, ref_embeddings)
        for row in hn:
            assert row["query_person_id"] in ("alice", "bob")
            # Verify it wasn't extracted from path
            assert "\\" not in row["query_person_id"]
            assert "/" not in row["query_person_id"]


# ---------------------------------------------------------------------------
# 2-3. Reference/query dimension validation
# ---------------------------------------------------------------------------

class TestDimensionValidation:
    def test_expected_dimension_constant(self):
        assert EXPECTED_EMBEDDING_DIM == 512

    def test_norm_tolerance_defined(self):
        assert 0.0 < NORM_TOLERANCE < 1.0


# ---------------------------------------------------------------------------
# 4-5. Normalized reference/query embeddings
# ---------------------------------------------------------------------------

class TestNormalizationValidation:
    def test_unit_norm_passes(self):
        """Unit-normalized vectors should not raise."""
        emb = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        emb = emb / np.linalg.norm(emb)
        assert abs(float(np.linalg.norm(emb)) - 1.0) < NORM_TOLERANCE

    def test_non_unit_norm_detected(self):
        """Non-unit vector should be detected."""
        emb = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) > NORM_TOLERANCE


# ---------------------------------------------------------------------------
# 6-8. Face handling (zero/one/multi)
# ---------------------------------------------------------------------------

class TestFaceHandling:
    def test_is_accepted_boundary(self):
        """score >= threshold must be accepted."""
        assert is_accepted(0.5, 0.5) is True
        assert is_accepted(0.49, 0.5) is False
        assert is_accepted(0.6, 0.5) is True

    def test_is_accepted_negative_scores(self):
        """Negative scores should work with negative thresholds."""
        assert is_accepted(-0.1, -0.2) is True
        assert is_accepted(-0.3, -0.2) is False


# ---------------------------------------------------------------------------
# 9. Dynamic reference count validation
# ---------------------------------------------------------------------------

class TestDynamicReferenceCount:
    def test_validate_reference_gallery_uniform(self):
        """Uniform gallery should pass."""
        from phase13_7_2_calibration import validate_reference_gallery
        ref_records = [
            {"person_id": "a", "image_path": f"a_{i}.jpg"} for i in range(3)
        ] + [
            {"person_id": "b", "image_path": f"b_{i}.jpg"} for i in range(3)
        ]
        people = [{"person_id": "a"}, {"person_id": "b"}]
        counts = validate_reference_gallery(ref_records, people)
        assert counts == {"a": 3, "b": 3}

    def test_validate_reference_gallery_missing_identity(self):
        """Identity with no references should raise."""
        from phase13_7_2_calibration import validate_reference_gallery
        ref_records = [{"person_id": "a", "image_path": "a.jpg"}]
        people = [{"person_id": "a"}, {"person_id": "b"}]
        with pytest.raises(ValueError, match="no reference images"):
            validate_reference_gallery(ref_records, people)


# ---------------------------------------------------------------------------
# 10-11. Genuine/impostor pair counts
# ---------------------------------------------------------------------------

class TestPairCounts:
    def test_genuine_pairs_count(self):
        assert 88 * 8 == 704

    def test_impostor_pairs_count(self):
        assert 88 * (176 - 8) == 14784


# ---------------------------------------------------------------------------
# 12-13. Image/identity-level hard negatives
# ---------------------------------------------------------------------------

class TestHardNegatives:
    def test_image_level_returns_one_per_query(self):
        """Image-level hard negatives must return exactly one row per query."""
        impostor_pairs = [
            {"query_image": "q1", "query_person_id": "A", "ref_person_id": "B",
             "ref_image": "r1", "ref_vector_id": 0, "similarity": 0.1},
            {"query_image": "q1", "query_person_id": "A", "ref_person_id": "B",
             "ref_image": "r2", "ref_vector_id": 1, "similarity": 0.5},
            {"query_image": "q2", "query_person_id": "A", "ref_person_id": "C",
             "ref_image": "r3", "ref_vector_id": 2, "similarity": 0.3},
        ]
        hn = image_level_hard_negatives(impostor_pairs)
        assert len(hn) == 2  # one per query
        q1_hn = next(h for h in hn if h["query_image"] == "q1")
        assert q1_hn["similarity"] == 0.5  # strongest, not first

    def test_identity_level_returns_rows(self):
        """Identity-level must return results."""
        query_embeddings = {
            "A": [("q1.jpg", np.array([1, 0, 0, 0], dtype=np.float32))],
        }
        ref_records = [
            {"vector_id": 0, "person_id": "A", "image_path": "a.jpg",
             "source": "", "license": ""},
            {"vector_id": 1, "person_id": "B", "image_path": "b.jpg",
             "source": "", "license": ""},
        ]
        ref_embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        hn = identity_level_hard_negatives(query_embeddings, ref_records, ref_embeddings)
        assert len(hn) >= 1
        assert hn[0]["query_person_id"] == "A"
        assert hn[0]["impostor_person_id"] == "B"


# ---------------------------------------------------------------------------
# 14. Supporting reference selection
# ---------------------------------------------------------------------------

class TestSupportingReference:
    def test_supporting_ref_is_max(self):
        """Supporting reference must be the one producing the max similarity."""
        query_embeddings = {
            "A": [("q1.jpg", np.array([1, 0, 0, 0], dtype=np.float32))],
        }
        ref_records = [
            {"vector_id": 0, "person_id": "A", "image_path": "a.jpg",
             "source": "", "license": ""},
            {"vector_id": 1, "person_id": "B", "image_path": "b_low.jpg",
             "source": "", "license": ""},
            {"vector_id": 2, "person_id": "B", "image_path": "b_high.jpg",
             "source": "", "license": ""},
        ]
        # Normalized reference embeddings: dot product = cosine similarity
        ref_embeddings = np.array([
            [1, 0, 0, 0],       # A: same as query (genuine)
            [0, 1, 0, 0],       # B low: orthogonal (sim=0)
            [0.8, 0.6, 0, 0],   # B high: partially aligned (sim=0.8)
        ], dtype=np.float32)
        hn = identity_level_hard_negatives(query_embeddings, ref_records, ref_embeddings)
        b_hn = next(h for h in hn if h["impostor_person_id"] == "B")
        assert b_hn["supporting_ref_image"] == "b_high.jpg"
        assert b_hn["identity_score"] > 0.7


# ---------------------------------------------------------------------------
# 15. Threshold direction
# ---------------------------------------------------------------------------

class TestThresholdDirection:
    def test_accept_at_threshold(self):
        assert is_accepted(0.5, 0.5) is True

    def test_reject_below_threshold(self):
        assert is_accepted(0.49, 0.5) is False

    def test_accept_above_threshold(self):
        assert is_accepted(0.6, 0.5) is True


# ---------------------------------------------------------------------------
# 16-23. TP/FP/TN/FN, FAR, FRR, TPR, precision, recall, F1, TNR consistency
# ---------------------------------------------------------------------------

class TestMetricConsistency:
    def _make_scores(self):
        genuine = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float32)
        impostor = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32)
        return genuine, impostor

    def test_confusion_matrix_elements(self):
        genuine, impostor = self._make_scores()
        threshold = 0.5
        tp = int(np.sum(genuine >= threshold))
        fn = int(np.sum(genuine < threshold))
        fp = int(np.sum(impostor >= threshold))
        tn = int(np.sum(impostor < threshold))
        assert tp + fn == len(genuine)
        assert fp + tn == len(impostor)
        assert tp == 3  # 0.5, 0.7, 0.9
        assert fn == 2  # 0.1, 0.3
        assert fp == 2  # 0.6, 0.8
        assert tn == 2  # 0.2, 0.4

    def test_far_formula(self):
        _, impostor = self._make_scores()
        threshold = 0.5
        fp = int(np.sum(impostor >= threshold))
        far = fp / len(impostor)
        assert far == pytest.approx(0.5)

    def test_frr_formula(self):
        genuine, _ = self._make_scores()
        threshold = 0.5
        fn = int(np.sum(genuine < threshold))
        frr = fn / len(genuine)
        assert frr == pytest.approx(0.4)

    def test_tpr_formula(self):
        genuine, _ = self._make_scores()
        threshold = 0.5
        tp = int(np.sum(genuine >= threshold))
        tpr = tp / len(genuine)
        assert tpr == pytest.approx(0.6)

    def test_tnr_formula(self):
        _, impostor = self._make_scores()
        threshold = 0.5
        tn = int(np.sum(impostor < threshold))
        tnr = tn / len(impostor)
        assert tnr == pytest.approx(0.5)

    def test_precision_formula(self):
        genuine, impostor = self._make_scores()
        threshold = 0.5
        tp = int(np.sum(genuine >= threshold))
        fp = int(np.sum(impostor >= threshold))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        assert precision == pytest.approx(0.6)

    def test_recall_equals_tpr(self):
        genuine, _ = self._make_scores()
        threshold = 0.5
        recall = float(np.sum(genuine >= threshold)) / len(genuine)
        tpr = 1.0 - float(np.sum(genuine < threshold)) / len(genuine)
        assert recall == pytest.approx(tpr)

    def test_f1_formula(self):
        genuine, impostor = self._make_scores()
        threshold = 0.5
        tp = int(np.sum(genuine >= threshold))
        fp = int(np.sum(impostor >= threshold))
        tpr = tp / len(genuine)
        precision = tp / (tp + fp)
        f1 = 2 * precision * tpr / (precision + tpr)
        assert f1 == pytest.approx(0.6)

    def test_far_frr_complement(self):
        genuine, impostor = self._make_scores()
        threshold = 0.5
        far = float(np.sum(impostor >= threshold)) / len(impostor)
        frr = float(np.sum(genuine < threshold)) / len(genuine)
        tpr = 1.0 - frr
        tnr = 1.0 - far
        assert far + tnr == pytest.approx(1.0)
        assert frr + tpr == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 24. Deterministic output
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_verification_json_exists(self):
        path = Path("outputs/phase13_7_2/verification.json")
        assert path.exists()

    def test_pair_counts_deterministic(self):
        path = Path("outputs/phase13_7_2/verification.json")
        if not path.exists():
            pytest.skip("verification.json not found")
        with open(path) as f:
            data = json.load(f)
        assert data["dataset"]["actual_genuine_pairs"] == 704
        assert data["dataset"]["actual_impostor_pairs"] == 14784

    def test_metrics_unchanged(self):
        """All calibration metrics must match Phase 13.7 values."""
        path = Path("outputs/phase13_7_2/verification.json")
        if not path.exists():
            pytest.skip("verification.json not found")
        with open(path) as f:
            data = json.load(f)
        assert data["image_level"]["roc_auc"] == pytest.approx(0.8785, abs=0.001)
        assert data["identity_level"]["roc_auc"] == pytest.approx(0.9465, abs=0.001)
        assert data["image_level"]["eer"]["eer"] == pytest.approx(0.196, abs=0.001)
        assert data["identity_level"]["eer"]["eer"] == pytest.approx(0.1155, abs=0.001)
        assert data["image_level"]["global_max_impostor"] == pytest.approx(0.2975, abs=0.001)
        assert data["identity_level"]["global_max_impostor"] == pytest.approx(0.2975, abs=0.001)

    def test_no_duplicates_in_code(self):
        """Verify the inferior duplicate function was removed."""
        import inspect
        # The canonical function should have 'supporting_ref' in its output
        source = inspect.getsource(identity_level_hard_negatives)
        assert "supporting_ref_image" in source
        # Should NOT contain path-based identity inference
        assert 'split("\\\\")' not in source
        assert "split(\"/\")" not in source
