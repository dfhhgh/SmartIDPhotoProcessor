"""Calibration summary report generator and exporter.

Exports:
- calibration_summary.json
- threshold_metrics.csv
- genuine_scores.csv
- impostor_scores.csv
- dataset_manifest.json
- ROC-AUC verification report
- EER report
- Candidate operating points
- Plots (distribution, ROC, FAR/FRR)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from search.calibration.threshold_evaluator import (
    DistributionStats,
    EERResult,
    ROCAUCResult,
    ThresholdMetrics,
)


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    schema_version: int
    calibration_mode: str  # "full" | "impostor_only" | "exploratory"
    dataset_path: str
    total_persons: int
    total_images: int
    usable_images: int
    excluded_images: int
    persons_with_2plus_images: int
    reference_images: int
    query_images: int
    positive_pairs_count: int
    negative_pairs_count: int
    genuine_stats: dict[str, Any]
    impostor_stats: dict[str, Any]
    roc_auc: dict[str, Any]
    eer: dict[str, Any]
    candidate_operating_points: list[dict[str, Any]]
    top_k_retrieval: dict[str, Any]
    limitations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_mode": self.calibration_mode,
            "dataset_path": self.dataset_path,
            "total_persons": self.total_persons,
            "total_images": self.total_images,
            "usable_images": self.usable_images,
            "excluded_images": self.excluded_images,
            "persons_with_2plus_images": self.persons_with_2plus_images,
            "reference_images": self.reference_images,
            "query_images": self.query_images,
            "positive_pairs_count": self.positive_pairs_count,
            "negative_pairs_count": self.negative_pairs_count,
            "genuine_stats": self.genuine_stats,
            "impostor_stats": self.impostor_stats,
            "roc_auc": self.roc_auc,
            "eer": self.eer,
            "candidate_operating_points": self.candidate_operating_points,
            "top_k_retrieval": self.top_k_retrieval,
            "limitations": self.limitations,
        }


class CalibrationReporter:
    """Exports calibration metrics, summaries, and plots to disk."""

    def save_artifacts(
        self,
        output_dir: Path | str,
        summary: CalibrationSummary,
        threshold_sweep: list[ThresholdMetrics],
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
    ) -> dict[str, str]:
        """Save all calibration artifacts to *output_dir*."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        summary_path = out / "calibration_summary.json"
        sweep_path = out / "threshold_metrics.csv"
        genuine_path = out / "genuine_scores.csv"
        impostor_path = out / "impostor_scores.csv"

        summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

        sweep_lines = [
            "threshold,tp,fp,tn,fn,tpr,tnr,fpr,fnr,far,frr,precision,recall"
        ]
        for m in threshold_sweep:
            sweep_lines.append(
                f"{m.threshold},{m.tp},{m.fp},{m.tn},{m.fn},{m.tpr},{m.tnr},"
                f"{m.fpr},{m.fnr},{m.far},{m.frr},{m.precision},{m.recall}"
            )
        sweep_path.write_text("\n".join(sweep_lines), encoding="utf-8")

        gen_lines = ["similarity"] + [str(float(s)) for s in genuine_scores.ravel()]
        genuine_path.write_text("\n".join(gen_lines), encoding="utf-8")

        imp_lines = ["similarity"] + [str(float(s)) for s in impostor_scores.ravel()]
        impostor_path.write_text("\n".join(imp_lines), encoding="utf-8")

        return {
            "summary_path": str(summary_path),
            "sweep_path": str(sweep_path),
            "genuine_path": str(genuine_path),
            "impostor_path": str(impostor_path),
        }

    def save_dataset_manifest(
        self,
        output_dir: Path | str,
        manifest: dict[str, Any],
    ) -> str:
        """Save the dataset manifest JSON."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest_path = out / "PHASE_13_5_DATASET_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return str(manifest_path)

    def generate_plots(
        self,
        output_dir: Path | str,
        genuine_scores: npt.NDArray[np.float32],
        impostor_scores: npt.NDArray[np.float32],
        sweep: list[ThresholdMetrics],
        roc_auc_result: ROCAUCResult,
    ) -> list[str]:
        """Generate and save calibration plots. Returns list of saved file paths."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []

        gen = genuine_scores.ravel()
        imp = impostor_scores.ravel()

        # 1. Genuine vs Impostor distribution
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        if gen.size > 0:
            ax.hist(gen, bins=50, alpha=0.6, label="Genuine (same person)", color="green", density=True)
        if imp.size > 0:
            ax.hist(imp, bins=50, alpha=0.6, label="Impostor (different person)", color="red", density=True)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title("Genuine vs Impostor Similarity Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = str(out / "genuine_vs_impostor_distribution.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)

        # 2. ROC curve
        if gen.size > 0 and imp.size > 0:
            all_scores = np.concatenate([gen, imp])
            labels = np.concatenate([np.ones(gen.size), np.zeros(imp.size)])
            sort_idx = np.argsort(-all_scores)
            sorted_labels = labels[sort_idx]
            tps = np.cumsum(sorted_labels)
            fps = np.cumsum(1 - sorted_labels)
            total_pos = int(np.sum(labels))
            total_neg = int(np.sum(1 - labels))
            tpr_arr = np.concatenate([[0.0], tps / total_pos]) if total_pos > 0 else np.zeros(len(tps) + 1)
            fpr_arr = np.concatenate([[0.0], fps / total_neg]) if total_neg > 0 else np.zeros(len(fps) + 1)

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.plot(fpr_arr, tpr_arr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc_result.sklearn_auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
            ax.set_xlabel("False Positive Rate (FPR)")
            ax.set_ylabel("True Positive Rate (TPR)")
            ax.set_title("Receiver Operating Characteristic (ROC) Curve")
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            p = str(out / "roc_curve.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

        # 3. FAR/FRR vs threshold
        if sweep:
            thresholds = [m.threshold for m in sweep]
            fars = [m.far for m in sweep]
            frrs = [m.frr for m in sweep]

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(thresholds, fars, "r-", linewidth=2, label="FAR (False Acceptance Rate)")
            ax.plot(thresholds, frrs, "b-", linewidth=2, label="FRR (False Rejection Rate)")
            ax.set_xlabel("Similarity Threshold")
            ax.set_ylabel("Error Rate")
            ax.set_title("FAR and FRR vs Similarity Threshold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Mark EER crossing
            min_diff_idx = min(range(len(sweep)), key=lambda i: abs(sweep[i].far - sweep[i].frr))
            ax.axvline(x=sweep[min_diff_idx].threshold, color="gray", linestyle="--", alpha=0.5,
                       label=f"~EER threshold ({sweep[min_diff_idx].threshold:.4f})")
            ax.legend()
            p = str(out / "far_frr_vs_threshold.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

        # 4. Genuine distribution (separate)
        if gen.size > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.hist(gen, bins=50, color="green", alpha=0.7, edgecolor="black")
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Count")
            ax.set_title("Genuine (Same-Person) Similarity Distribution")
            ax.grid(True, alpha=0.3)
            p = str(out / "genuine_distribution.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

        # 5. Impostor distribution (separate)
        if imp.size > 0:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.hist(imp, bins=50, color="red", alpha=0.7, edgecolor="black")
            ax.set_xlabel("Cosine Similarity")
            ax.set_ylabel("Count")
            ax.set_title("Impostor (Different-Person) Similarity Distribution")
            ax.grid(True, alpha=0.3)
            p = str(out / "impostor_distribution.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(p)

        return saved
