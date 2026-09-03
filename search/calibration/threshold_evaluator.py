"""Threshold evaluation, distribution statistics, ROC, and FAR/FRR calculations.

Provides:
- Distribution statistics with percentiles
- ROC-AUC with sklearn verification
- Event-based threshold sweep using observed score boundaries
- Interpolation-based EER estimation
- Full confusion matrix metrics at every evaluated threshold
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score as sklearn_roc_auc


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DistributionStats:
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    percentiles: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "percentiles": self.percentiles,
        }


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    tpr: float
    tnr: float
    fpr: float
    fnr: float
    far: float
    frr: float
    precision: float
    recall: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "tpr": self.tpr,
            "tnr": self.tnr,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "far": self.far,
            "frr": self.frr,
            "precision": self.precision,
            "recall": self.recall,
        }


@dataclass(frozen=True, slots=True)
class ROCAUCResult:
    custom_auc: float
    sklearn_auc: float
    absolute_difference: float
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "custom_auc": self.custom_auc,
            "sklearn_auc": self.sklearn_auc,
            "absolute_difference": self.absolute_difference,
            "verified": self.verified,
        }


@dataclass(frozen=True, slots=True)
class EERResult:
    eer: float
    threshold: float
    far_at_eer: float
    frr_at_eer: float
    method: str
    threshold_resolution: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "eer": self.eer,
            "threshold": self.threshold,
            "far_at_eer": self.far_at_eer,
            "frr_at_eer": self.frr_at_eer,
            "method": self.method,
            "threshold_resolution": self.threshold_resolution,
        }


# ---------------------------------------------------------------------------
# ThresholdEvaluator
# ---------------------------------------------------------------------------

class ThresholdEvaluator:
    """Computes distribution statistics, ROC-AUC, FAR, FRR, and threshold sweeps."""

    def compute_distribution(self, scores: npt.NDArray[np.float32]) -> DistributionStats:
        """Compute statistical summary and percentiles for a score array."""
        if scores.size == 0:
            return DistributionStats(
                count=0, min=0.0, max=0.0, mean=0.0, median=0.0, std=0.0, percentiles={}
            )

        flat = scores.ravel()
        p_vals = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        p_computed = np.percentile(flat, p_vals)
        percentiles = {f"p{p}": float(val) for p, val in zip(p_vals, p_computed)}

        return DistributionStats(
            count=int(flat.size),
            min=float(flat.min()),
            max=float(flat.max()),
            mean=float(flat.mean()),
            median=float(np.median(flat)),
            std=float(flat.std()),
            percentiles=percentiles,
        )

    def evaluate_threshold(
        self,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
        threshold: float,
    ) -> ThresholdMetrics:
        """Evaluate confusion matrix, FAR, and FRR at a given similarity threshold.

        Convention: similarity >= threshold is predicted POSITIVE (same person).
        """
        tp = int(np.sum(genuine_scores >= threshold))
        fn = int(np.sum(genuine_scores < threshold))
        fp = int(np.sum(impostor_scores >= threshold))
        tn = int(np.sum(impostor_scores < threshold))

        total_genuine = len(genuine_scores)
        total_impostor = len(impostor_scores)

        tpr = tp / total_genuine if total_genuine > 0 else 0.0
        fnr = fn / total_genuine if total_genuine > 0 else 0.0
        tnr = tn / total_impostor if total_impostor > 0 else 0.0
        fpr = fp / total_impostor if total_impostor > 0 else 0.0

        far = fpr
        frr = fnr
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr

        return ThresholdMetrics(
            threshold=float(threshold),
            tp=tp, fp=fp, tn=tn, fn=fn,
            tpr=float(tpr), tnr=float(tnr), fpr=float(fpr), fnr=float(fnr),
            far=float(far), frr=float(frr),
            precision=float(precision), recall=float(recall),
        )

    def sweep_thresholds_uniform(
        self,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
        num_steps: int = 200,
    ) -> list[ThresholdMetrics]:
        """Sweep thresholds uniformly across the observed score range."""
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        if all_scores.size == 0:
            return []

        min_s = float(all_scores.min())
        max_s = float(all_scores.max())
        thresholds = np.linspace(min_s, max_s, num_steps)
        return [self.evaluate_threshold(genuine_scores, impostor_scores, t) for t in thresholds]

    def sweep_thresholds_event_based(
        self,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
    ) -> list[ThresholdMetrics]:
        """Sweep thresholds at exact observed score boundaries.

        Methodology:
        1. Collect all unique scores from genuine and impostor.
        2. Sort ascending.
        3. Evaluate at each unique score AND at midpoints between adjacent unique scores.
        4. Also evaluate just below min and just above max to capture edge cases.

        This ensures every meaningful classification change is captured.
        """
        all_scores = np.concatenate([genuine_scores, impostor_scores])
        if all_scores.size == 0:
            return []

        unique_scores = np.unique(all_scores)
        unique_scores = np.sort(unique_scores)

        # Build threshold set: midpoints + unique boundaries
        thresholds: list[float] = []

        # Just below minimum to capture all-positive case
        thresholds.append(float(unique_scores[0]) - 1e-9)

        # Midpoints between adjacent unique scores
        for i in range(len(unique_scores) - 1):
            mid = (unique_scores[i] + unique_scores[i + 1]) / 2.0
            thresholds.append(float(mid))

        # At each unique score boundary
        for s in unique_scores:
            thresholds.append(float(s))

        # Just above maximum to capture all-negative case
        thresholds.append(float(unique_scores[-1]) + 1e-9)

        thresholds.sort()
        return [self.evaluate_threshold(genuine_scores, impostor_scores, t) for t in thresholds]

    def compute_roc_auc(
        self,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
    ) -> ROCAUCResult:
        """Compute ROC-AUC using both custom trapezoidal and sklearn, then verify."""
        custom = self._compute_roc_auc_custom(genuine_scores, impostor_scores)
        sklearn_val = self._compute_roc_auc_sklearn(genuine_scores, impostor_scores)

        abs_diff = abs(custom - sklearn_val)
        verified = abs_diff < 1e-4

        return ROCAUCResult(
            custom_auc=custom,
            sklearn_auc=sklearn_val,
            absolute_difference=abs_diff,
            verified=verified,
        )

    def _compute_roc_auc_custom(
        self,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
    ) -> float:
        """Custom ROC-AUC using trapezoidal rule over sorted scores."""
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(impostor_scores)])

        if len(np.unique(labels)) < 2:
            return 0.0

        # Sort descending by score
        sort_idx = np.argsort(-scores)
        scores = scores[sort_idx]
        labels = labels[sort_idx]

        tps = np.cumsum(labels)
        fps = np.cumsum(1 - labels)

        total_pos = int(np.sum(labels))
        total_neg = int(np.sum(1 - labels))

        tpr_arr = tps / total_pos if total_pos > 0 else np.zeros_like(tps, dtype=float)
        fpr_arr = fps / total_neg if total_neg > 0 else np.zeros_like(fps, dtype=float)

        # Prepend (0, 0)
        tpr_arr = np.concatenate([[0.0], tpr_arr])
        fpr_arr = np.concatenate([[0.0], fpr_arr])

        return float(np.trapezoid(tpr_arr, fpr_arr))

    def _compute_roc_auc_sklearn(
        self,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
    ) -> float:
        """Reference ROC-AUC using sklearn.metrics.roc_auc_score."""
        scores = np.concatenate([genuine_scores, impostor_scores])
        labels = np.concatenate([
            np.ones(len(genuine_scores), dtype=int),
            np.zeros(len(impostor_scores), dtype=int),
        ])

        if len(np.unique(labels)) < 2:
            return 0.0

        return float(sklearn_roc_auc(labels, scores))

    def estimate_eer_interpolated(
        self,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
    ) -> EERResult:
        """Estimate EER using interpolation between observed thresholds.

        Methodology:
        1. Compute event-based threshold sweep.
        2. For each consecutive pair of thresholds where FAR-FRR changes sign,
           linearly interpolate to find the exact crossing point.
        3. Report the EER as the average of FAR and FRR at the crossing point.
        """
        sweep = self.sweep_thresholds_event_based(genuine_scores, impostor_scores)
        if len(sweep) < 2:
            return EERResult(
                eer=0.5, threshold=0.0, far_at_eer=0.5, frr_at_eer=0.5,
                method="insufficient_data", threshold_resolution=len(sweep),
            )

        # Find sign change in (FAR - FRR)
        diffs = [(m.far - m.frr, m) for m in sweep]

        best_diff = float("inf")
        best_eer = 0.5
        best_thresh = 0.0
        best_far = 0.5
        best_frr = 0.5

        for i in range(len(diffs) - 1):
            d1, m1 = diffs[i]
            d2, m2 = diffs[i + 1]

            if d1 == 0.0:
                best_eer = (m1.far + m1.frr) / 2.0
                best_thresh = m1.threshold
                best_far = m1.far
                best_frr = m1.frr
                break

            if d1 * d2 < 0:
                # Sign change — interpolate
                frac = d1 / (d1 - d2)
                interp_thresh = m1.threshold + frac * (m2.threshold - m1.threshold)
                interp_far = m1.far + frac * (m2.far - m1.far)
                interp_frr = m1.frr + frac * (m2.frr - m1.frr)
                interp_eer = (interp_far + interp_frr) / 2.0

                if abs(interp_eer - 0.5) < abs(best_eer - 0.5) or best_eer == 0.5:
                    best_eer = interp_eer
                    best_thresh = interp_thresh
                    best_far = interp_far
                    best_frr = interp_frr

        # Fallback: closest to equal error
        if best_eer == 0.5:
            for m in sweep:
                diff = abs(m.far - m.frr)
                if diff < best_diff:
                    best_diff = diff
                    best_eer = (m.far + m.frr) / 2.0
                    best_thresh = m.threshold
                    best_far = m.far
                    best_frr = m.frr

        return EERResult(
            eer=float(best_eer),
            threshold=float(best_thresh),
            far_at_eer=float(best_far),
            frr_at_eer=float(best_frr),
            method="interpolation",
            threshold_resolution=len(sweep),
        )

    def estimate_eer_from_sweep(
        self,
        metrics_sweep: list[ThresholdMetrics],
    ) -> EERResult:
        """Estimate EER from a pre-computed sweep (legacy interface)."""
        if not metrics_sweep:
            return EERResult(
                eer=0.5, threshold=0.0, far_at_eer=0.5, frr_at_eer=0.5,
                method="empty_sweep", threshold_resolution=0,
            )

        best_diff = float("inf")
        best_eer = 0.5
        best_thresh = 0.0
        best_far = 0.5
        best_frr = 0.5

        for m in metrics_sweep:
            diff = abs(m.far - m.frr)
            if diff < best_diff:
                best_diff = diff
                best_eer = (m.far + m.frr) / 2.0
                best_thresh = m.threshold
                best_far = m.far
                best_frr = m.frr

        return EERResult(
            eer=float(best_eer),
            threshold=float(best_thresh),
            far_at_eer=float(best_far),
            frr_at_eer=float(best_frr),
            method="discrete_sweep_minimize_abs_diff",
            threshold_resolution=len(metrics_sweep),
        )

    def find_operating_points(
        self,
        sweep: list[ThresholdMetrics],
    ) -> list[dict[str, Any]]:
        """Identify candidate operating points from event-based sweep.

        Returns three candidates:
        A. Low-FAR point: lowest threshold where FAR <= 0.01 (or closest to it)
        B. Balanced point: threshold minimizing |FAR - FRR|
        C. High-recall point: lowest threshold where TPR >= 0.95 (or closest)
        """
        if not sweep:
            return []

        candidates: list[dict[str, Any]] = []

        # A. Low-FAR point
        low_far = [m for m in sweep if m.far <= 0.01]
        if low_far:
            # Among those with FAR <= 0.01, pick the one with highest TPR
            best = max(low_far, key=lambda m: m.tpr)
        else:
            # Find the threshold with minimum FAR
            best = min(sweep, key=lambda m: m.far)
        candidates.append({"point_type": "low_far", **best.to_dict()})

        # B. Balanced point
        balanced = min(sweep, key=lambda m: abs(m.far - m.frr))
        candidates.append({"point_type": "balanced_far_frr", **balanced.to_dict()})

        # C. High-recall point
        high_recall = [m for m in sweep if m.tpr >= 0.95]
        if high_recall:
            # Among those with TPR >= 0.95, pick the one with lowest FAR
            best = min(high_recall, key=lambda m: m.far)
        else:
            # Find the threshold with maximum TPR
            best = max(sweep, key=lambda m: m.tpr)
        candidates.append({"point_type": "high_recall", **best.to_dict()})

        return candidates
