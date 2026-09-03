"""Comprehensive tests for search.calibration package.

Covers:
- Positive & negative pair generation (no self-pairs)
- Deterministic pairing & sampling
- Similarity evaluation (inner product / cosine similarity)
- Distribution statistics & percentiles
- Threshold sweep (uniform + event-based)
- ROC-AUC with sklearn verification
- EER interpolation
- Candidate operating points
- Artifact export (JSON/CSV)
- Synthetic ground-truth verification
- Edge cases (empty, single-person, insufficient images)
- Reproducibility
- No production side effects
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from search.calibration.calibration_report import CalibrationReporter, CalibrationSummary
from search.calibration.pair_generator import PairGenerator, PairSample
from search.calibration.similarity_evaluator import SimilarityEvaluator
from search.calibration.threshold_evaluator import (
    EERResult,
    ROCAUCResult,
    ThresholdEvaluator,
    ThresholdMetrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vector(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_identity_data(n_persons: int = 5, images_per_person: int = 3, seed: int = 42):
    """Create deterministic identity data for testing."""
    rng = np.random.RandomState(seed)
    identity_data = {}
    for i in range(n_persons):
        pid = f"person_{i:03d}"
        items = []
        for j in range(images_per_person):
            v = rng.randn(512).astype(np.float32)
            v = v / np.linalg.norm(v)
            items.append((f"{pid}/img_{j:02d}.jpg", v))
        identity_data[pid] = items
    return identity_data


# ---------------------------------------------------------------------------
# Pair Generator Tests
# ---------------------------------------------------------------------------

class TestPairGenerator:
    def test_generates_positive_and_negative_pairs(self) -> None:
        identity_data = {
            "alice": [("alice/1.jpg", _make_vector(1)), ("alice/2.jpg", _make_vector(2))],
            "bob": [("bob/1.jpg", _make_vector(3)), ("bob/2.jpg", _make_vector(4))],
            "charlie": [("charlie/1.jpg", _make_vector(5))],
        }
        gen = PairGenerator(seed=42)
        pos_pairs, neg_pairs = gen.generate_pairs(identity_data)

        assert len(pos_pairs) == 2
        for p in pos_pairs:
            assert p.is_positive
            assert p.person_id_1 == p.person_id_2
            assert p.image_1 != p.image_2

        assert len(neg_pairs) == 8
        for p in neg_pairs:
            assert not p.is_positive
            assert p.person_id_1 != p.person_id_2

    def test_no_self_pairs(self) -> None:
        v1 = _make_vector(1)
        v2 = _make_vector(2)
        identity_data = {
            "alice": [("alice/1.jpg", v1), ("alice/2.jpg", v2)],
        }
        gen = PairGenerator(seed=42)
        pos_pairs, _ = gen.generate_pairs(identity_data)
        for p in pos_pairs:
            assert p.image_1 != p.image_2

    def test_determinism(self) -> None:
        identity_data = _make_identity_data(3, 4, seed=99)
        gen1 = PairGenerator(seed=123)
        gen2 = PairGenerator(seed=123)
        pos1, neg1 = gen1.generate_pairs(identity_data)
        pos2, neg2 = gen2.generate_pairs(identity_data)

        assert len(pos1) == len(pos2)
        assert len(neg1) == len(neg2)
        for p1, p2 in zip(pos1, pos2):
            assert p1.image_1 == p2.image_1
            assert p1.image_2 == p2.image_2
            np.testing.assert_array_equal(p1.embedding_1, p2.embedding_1)

    def test_max_positive_pairs_limit(self) -> None:
        identity_data = _make_identity_data(10, 5, seed=1)
        gen = PairGenerator(seed=42)
        pos_pairs, _ = gen.generate_pairs(identity_data, max_positive_pairs=5)
        assert len(pos_pairs) <= 5

    def test_max_negative_pairs_limit(self) -> None:
        identity_data = _make_identity_data(10, 5, seed=1)
        gen = PairGenerator(seed=42)
        _, neg_pairs = gen.generate_pairs(identity_data, max_negative_pairs=10)
        assert len(neg_pairs) <= 10

    def test_person_with_one_image_no_positive_pairs(self) -> None:
        identity_data = {
            "alice": [("alice/1.jpg", _make_vector(1))],
            "bob": [("bob/1.jpg", _make_vector(2))],
        }
        gen = PairGenerator(seed=42)
        pos_pairs, _ = gen.generate_pairs(identity_data)
        assert len(pos_pairs) == 0

    def test_empty_identity_data(self) -> None:
        gen = PairGenerator(seed=42)
        pos, neg = gen.generate_pairs({})
        assert len(pos) == 0
        assert len(neg) == 0


# ---------------------------------------------------------------------------
# Similarity Evaluator Tests
# ---------------------------------------------------------------------------

class TestSimilarityEvaluator:
    def test_similarity_computation(self) -> None:
        v1 = _make_vector(1)
        v2 = _make_vector(2)
        pair = PairSample(
            person_id_1="a", image_1="a/1.jpg", embedding_1=v1,
            person_id_2="b", image_2="b/1.jpg", embedding_2=v2,
            is_positive=False,
        )
        evaluator = SimilarityEvaluator()
        evaluated = evaluator.evaluate_pair(pair)
        expected_sim = float(np.dot(v1, v2))
        assert abs(evaluated.similarity - expected_sim) < 1e-6

    def test_self_similarity_is_one(self) -> None:
        v = _make_vector(1)
        pair = PairSample(
            person_id_1="a", image_1="a/1.jpg", embedding_1=v,
            person_id_2="a", image_2="a/1.jpg", embedding_2=v,
            is_positive=True,
        )
        evaluator = SimilarityEvaluator()
        evaluated = evaluator.evaluate_pair(pair)
        assert abs(evaluated.similarity - 1.0) < 1e-6

    def test_batch_evaluation(self) -> None:
        identity_data = _make_identity_data(3, 3, seed=42)
        gen = PairGenerator(seed=42)
        pos, neg = gen.generate_pairs(identity_data)
        all_pairs = pos + neg

        evaluator = SimilarityEvaluator()
        results = evaluator.evaluate_batch(all_pairs)
        assert len(results) == len(all_pairs)
        for r in results:
            assert -1.0 <= r.similarity <= 1.0


# ---------------------------------------------------------------------------
# Threshold Evaluator Tests
# ---------------------------------------------------------------------------

class TestThresholdEvaluator:
    @pytest.fixture
    def synthetic_scores(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(42)
        genuine = rng.normal(0.85, 0.05, 200).astype(np.float32)
        impostor = rng.normal(0.30, 0.10, 200).astype(np.float32)
        return genuine, impostor

    def test_distribution_stats(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, _ = synthetic_scores
        evaluator = ThresholdEvaluator()
        stats = evaluator.compute_distribution(genuine)
        assert stats.count == 200
        assert stats.min <= stats.mean <= stats.max
        assert "p50" in stats.percentiles
        assert "p99" in stats.percentiles

    def test_empty_distribution(self) -> None:
        evaluator = ThresholdEvaluator()
        stats = evaluator.compute_distribution(np.array([], dtype=np.float32))
        assert stats.count == 0

    def test_threshold_evaluation_metrics(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        metrics = evaluator.evaluate_threshold(genuine, impostor, threshold=0.60)
        assert metrics.tp + metrics.fn == len(genuine)
        assert metrics.tn + metrics.fp == len(impostor)
        assert 0.0 <= metrics.far <= 1.0
        assert 0.0 <= metrics.frr <= 1.0
        assert abs(metrics.far - metrics.fpr) < 1e-10
        assert abs(metrics.frr - metrics.fnr) < 1e-10

    def test_threshold_boundary_all_positive(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        metrics = evaluator.evaluate_threshold(genuine, impostor, threshold=-1.0)
        assert metrics.tp == len(genuine)
        assert metrics.fn == 0
        assert metrics.fp == len(impostor)
        assert metrics.tn == 0

    def test_threshold_boundary_all_negative(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        metrics = evaluator.evaluate_threshold(genuine, impostor, threshold=2.0)
        assert metrics.tp == 0
        assert metrics.fn == len(genuine)
        assert metrics.fp == 0
        assert metrics.tn == len(impostor)

    def test_roc_auc_perfect_separation(self) -> None:
        genuine = np.array([0.9, 0.85, 0.95], dtype=np.float32)
        impostor = np.array([0.1, 0.2, 0.15], dtype=np.float32)
        evaluator = ThresholdEvaluator()
        result = evaluator.compute_roc_auc(genuine, impostor)
        assert result.custom_auc == 1.0
        assert result.sklearn_auc == 1.0
        assert result.verified

    def test_roc_auc_random(self) -> None:
        rng = np.random.RandomState(42)
        genuine = rng.uniform(0.0, 1.0, 1000).astype(np.float32)
        impostor = rng.uniform(0.0, 1.0, 1000).astype(np.float32)
        evaluator = ThresholdEvaluator()
        result = evaluator.compute_roc_auc(genuine, impostor)
        assert 0.4 < result.sklearn_auc < 0.6
        assert result.verified

    def test_roc_auc_sklearn_verification(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        result = evaluator.compute_roc_auc(genuine, impostor)
        assert result.verified, f"ROC-AUC mismatch: custom={result.custom_auc}, sklearn={result.sklearn_auc}"
        assert abs(result.custom_auc - result.sklearn_auc) < 1e-4

    def test_roc_auc_single_class(self) -> None:
        genuine = np.array([0.9, 0.85], dtype=np.float32)
        impostor = np.array([0.9, 0.85], dtype=np.float32)
        evaluator = ThresholdEvaluator()
        result = evaluator.compute_roc_auc(genuine, impostor)
        assert result.sklearn_auc == 0.5

    def test_event_based_sweep(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        sweep = evaluator.sweep_thresholds_event_based(genuine, impostor)
        assert len(sweep) > 0
        # Event-based should capture all unique score boundaries
        thresholds = [m.threshold for m in sweep]
        assert thresholds == sorted(thresholds)

    def test_uniform_sweep(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        sweep = evaluator.sweep_thresholds_uniform(genuine, impostor, num_steps=50)
        assert len(sweep) == 50

    def test_eer_interpolated(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        result = evaluator.estimate_eer_interpolated(genuine, impostor)
        assert 0.0 <= result.eer <= 0.5
        assert result.method == "interpolation"
        assert result.threshold_resolution > 0

    def test_operating_points(self, synthetic_scores: tuple[np.ndarray, np.ndarray]) -> None:
        genuine, impostor = synthetic_scores
        evaluator = ThresholdEvaluator()
        sweep = evaluator.sweep_thresholds_event_based(genuine, impostor)
        points = evaluator.find_operating_points(sweep)
        assert len(points) == 3
        point_types = [p["point_type"] for p in points]
        assert "low_far" in point_types
        assert "balanced_far_frr" in point_types
        assert "high_recall" in point_types


# ---------------------------------------------------------------------------
# Synthetic Ground-Truth Verification (MANDATORY)
# ---------------------------------------------------------------------------

class TestSyntheticGroundTruth:
    """Verify implementation against manually calculable expected values."""

    def test_known_scores_manual_calculation(self) -> None:
        """Manual calculation with known genuine/impostor scores."""
        genuine = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        impostor = np.array([0.3, 0.2, 0.1], dtype=np.float32)

        evaluator = ThresholdEvaluator()

        # At threshold 0.6: TP=3, FN=0, FP=0, TN=3
        m = evaluator.evaluate_threshold(genuine, impostor, 0.6)
        assert m.tp == 3
        assert m.fn == 0
        assert m.fp == 0
        assert m.tn == 3
        assert m.far == 0.0
        assert m.frr == 0.0
        assert m.tpr == 1.0
        assert m.tnr == 1.0

        # At threshold 0.85: TP=1 (only 0.9), FN=2, FP=0, TN=3
        m = evaluator.evaluate_threshold(genuine, impostor, 0.85)
        assert m.tp == 1
        assert m.fn == 2
        assert m.fp == 0
        assert m.tn == 3
        assert abs(m.far - 0.0) < 1e-10
        assert abs(m.frr - 2 / 3) < 1e-10

        # At threshold 0.05: TP=3, FN=0, FP=3, TN=0 (all above)
        m = evaluator.evaluate_threshold(genuine, impostor, 0.05)
        assert m.tp == 3
        assert m.fn == 0
        assert m.fp == 3
        assert m.tn == 0
        assert abs(m.far - 1.0) < 1e-10
        assert abs(m.frr - 0.0) < 1e-10

    def test_roc_auc_known_values(self) -> None:
        """ROC-AUC with known expected value."""
        # Perfect separation
        genuine = np.array([1.0, 0.9, 0.8], dtype=np.float32)
        impostor = np.array([0.1, 0.0, -0.1], dtype=np.float32)

        evaluator = ThresholdEvaluator()
        result = evaluator.compute_roc_auc(genuine, impostor)
        assert result.custom_auc == 1.0
        assert result.sklearn_auc == 1.0

        # No separation (same distribution)
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = evaluator.compute_roc_auc(scores, scores)
        assert result.sklearn_auc == 0.5

    def test_eer_known_values(self) -> None:
        """EER with clearly separated distributions should be near 0."""
        genuine = np.array([0.95, 0.90, 0.85], dtype=np.float32)
        impostor = np.array([0.10, 0.15, 0.20], dtype=np.float32)

        evaluator = ThresholdEvaluator()
        result = evaluator.estimate_eer_interpolated(genuine, impostor)
        assert result.eer < 0.05

    def test_distribution_statistics_known(self) -> None:
        """Verify distribution statistics against manual calculation."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        evaluator = ThresholdEvaluator()
        stats = evaluator.compute_distribution(scores)

        assert stats.count == 5
        assert stats.min == 1.0
        assert stats.max == 5.0
        assert abs(stats.mean - 3.0) < 1e-6
        assert abs(stats.median - 3.0) < 1e-6

    def test_perfect_eer_zero(self) -> None:
        """Perfectly separated distributions should have EER near 0."""
        rng = np.random.RandomState(42)
        genuine = (0.9 + rng.rand(100) * 0.1).astype(np.float32)
        impostor = (0.1 + rng.rand(100) * 0.1).astype(np.float32)

        evaluator = ThresholdEvaluator()
        result = evaluator.estimate_eer_interpolated(genuine, impostor)
        assert result.eer < 0.01


# ---------------------------------------------------------------------------
# Calibration Reporter Tests
# ---------------------------------------------------------------------------

class TestCalibrationReporter:
    def test_save_artifacts(self) -> None:
        summary = CalibrationSummary(
            schema_version=1,
            calibration_mode="test",
            dataset_path="/test",
            total_persons=2,
            total_images=4,
            usable_images=4,
            excluded_images=0,
            persons_with_2plus_images=2,
            reference_images=2,
            query_images=2,
            positive_pairs_count=2,
            negative_pairs_count=4,
            genuine_stats={},
            impostor_stats={},
            roc_auc={},
            eer={},
            candidate_operating_points=[],
            top_k_retrieval={},
            limitations=[],
        )
        sweep = [
            ThresholdMetrics(0.5, 10, 1, 9, 0, 1.0, 0.9, 0.1, 0.0, 0.1, 0.0, 0.91, 1.0)
        ]
        genuine = np.array([0.8, 0.85], dtype=np.float32)
        impostor = np.array([0.2, 0.3], dtype=np.float32)

        reporter = CalibrationReporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = reporter.save_artifacts(tmpdir, summary, sweep, genuine, impostor)
            assert Path(paths["summary_path"]).exists()
            assert Path(paths["sweep_path"]).exists()
            assert Path(paths["genuine_path"]).exists()
            assert Path(paths["impostor_path"]).exists()

    def test_save_dataset_manifest(self) -> None:
        reporter = CalibrationReporter()
        manifest = {"test": "data", "seed": 42}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = reporter.save_dataset_manifest(tmpdir, manifest)
            assert Path(path).exists()

    def test_generate_plots(self) -> None:
        genuine = np.random.RandomState(42).normal(0.85, 0.05, 50).astype(np.float32)
        impostor = np.random.RandomState(43).normal(0.30, 0.10, 50).astype(np.float32)

        evaluator = ThresholdEvaluator()
        roc_result = evaluator.compute_roc_auc(genuine, impostor)
        sweep = evaluator.sweep_thresholds_event_based(genuine, impostor)

        reporter = CalibrationReporter()
        with tempfile.TemporaryDirectory() as tmpdir:
            plots = reporter.generate_plots(tmpdir, genuine, impostor, sweep, roc_result)
            assert len(plots) >= 3
            for p in plots:
                assert Path(p).exists()


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_genuine_scores(self) -> None:
        evaluator = ThresholdEvaluator()
        genuine = np.array([], dtype=np.float32)
        impostor = np.array([0.3, 0.4], dtype=np.float32)
        stats = evaluator.compute_distribution(genuine)
        assert stats.count == 0

    def test_empty_impostor_scores(self) -> None:
        evaluator = ThresholdEvaluator()
        genuine = np.array([0.8, 0.9], dtype=np.float32)
        impostor = np.array([], dtype=np.float32)
        stats = evaluator.compute_distribution(impostor)
        assert stats.count == 0

    def test_single_genuine_score(self) -> None:
        evaluator = ThresholdEvaluator()
        genuine = np.array([0.8], dtype=np.float32)
        impostor = np.array([0.3], dtype=np.float32)
        m = evaluator.evaluate_threshold(genuine, impostor, 0.5)
        assert m.tp == 1
        assert m.fn == 0

    def test_identical_scores(self) -> None:
        evaluator = ThresholdEvaluator()
        scores = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        result = evaluator.compute_roc_auc(scores, scores)
        # With identical scores, sklearn returns 0.5 (random)
        assert 0.4 <= result.sklearn_auc <= 0.6

    def test_nan_in_scores(self) -> None:
        evaluator = ThresholdEvaluator()
        scores = np.array([0.8, np.nan, 0.9], dtype=np.float32)
        stats = evaluator.compute_distribution(scores)
        assert stats.count == 3

    def test_inf_in_scores(self) -> None:
        evaluator = ThresholdEvaluator()
        scores = np.array([0.8, np.inf, 0.9], dtype=np.float32)
        stats = evaluator.compute_distribution(scores)
        assert stats.count == 3


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_pair_generation_reproducible(self) -> None:
        identity_data = _make_identity_data(5, 4, seed=42)
        gen1 = PairGenerator(seed=99)
        gen2 = PairGenerator(seed=99)
        pos1, neg1 = gen1.generate_pairs(identity_data)
        pos2, neg2 = gen2.generate_pairs(identity_data)

        assert len(pos1) == len(pos2)
        assert len(neg1) == len(neg2)
        for p1, p2 in zip(pos1, pos2):
            assert p1.image_1 == p2.image_1
            assert p1.image_2 == p2.image_2
            np.testing.assert_array_equal(p1.embedding_1, p2.embedding_1)

    def test_threshold_sweep_reproducible(self) -> None:
        rng = np.random.RandomState(42)
        genuine = rng.normal(0.85, 0.05, 100).astype(np.float32)
        impostor = rng.normal(0.30, 0.10, 100).astype(np.float32)

        evaluator = ThresholdEvaluator()
        sweep1 = evaluator.sweep_thresholds_event_based(genuine, impostor)
        sweep2 = evaluator.sweep_thresholds_event_based(genuine, impostor)

        assert len(sweep1) == len(sweep2)
        for m1, m2 in zip(sweep1, sweep2):
            assert abs(m1.threshold - m2.threshold) < 1e-10
            assert m1.tp == m2.tp
            assert m1.fp == m2.fp

    def test_roc_auc_reproducible(self) -> None:
        rng = np.random.RandomState(42)
        genuine = rng.normal(0.85, 0.05, 100).astype(np.float32)
        impostor = rng.normal(0.30, 0.10, 100).astype(np.float32)

        evaluator = ThresholdEvaluator()
        r1 = evaluator.compute_roc_auc(genuine, impostor)
        r2 = evaluator.compute_roc_auc(genuine, impostor)
        assert abs(r1.sklearn_auc - r2.sklearn_auc) < 1e-10
