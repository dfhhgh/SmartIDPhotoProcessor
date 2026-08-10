"""
Numerical comparison between ONNX and PyTorch model outputs.

Computes quantitative metrics to verify reproduction fidelity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonResult:
    output_shape_match: bool
    onnx_shape: tuple[int, ...]
    pytorch_shape: tuple[int, ...]
    max_abs_diff: float
    mean_abs_diff: float
    mse: float
    cosine_similarity: float
    argmax_agreement: float
    mask_iou_per_class: dict[int, float]
    mean_mask_iou: float


def compare_outputs(
    onnx_logits: npt.NDArray[np.float32],
    pytorch_logits: npt.NDArray[np.float32],
    n_classes: int = 19,
) -> ComparisonResult:
    assert onnx_logits.shape == pytorch_logits.shape, (
        f"Shape mismatch: ONNX={onnx_logits.shape}, PyTorch={pytorch_logits.shape}"
    )

    # Remove batch dimension for comparison if present
    if onnx_logits.ndim == 4:
        onnx_logits = onnx_logits[0]
        pytorch_logits = pytorch_logits[0]

    shape_match = True
    onnx_shape = tuple(onnx_logits.shape)
    pytorch_shape = tuple(pytorch_logits.shape)

    diff = onnx_logits.astype(np.float64) - pytorch_logits.astype(np.float64)
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))

    onnx_flat = onnx_logits.ravel().astype(np.float64)
    pt_flat = pytorch_logits.ravel().astype(np.float64)
    dot = np.dot(onnx_flat, pt_flat)
    norm_onnx = np.linalg.norm(onnx_flat)
    norm_pt = np.linalg.norm(pt_flat)
    cosine = float(dot / (norm_onnx * norm_pt + 1e-8))

    onnx_mask = np.argmax(onnx_logits, axis=1).ravel()
    pt_mask = np.argmax(pytorch_logits, axis=1).ravel()
    agreement = float(np.mean(onnx_mask == pt_mask))

    ious: dict[int, float] = {}
    for c in range(n_classes):
        onnx_pixels = onnx_mask == c
        pt_pixels = pt_mask == c
        intersection = np.sum(onnx_pixels & pt_pixels)
        union = np.sum(onnx_pixels | pt_pixels)
        ious[c] = float(intersection / union) if union > 0 else 1.0

    mean_iou = float(np.mean(list(ious.values())))

    return ComparisonResult(
        output_shape_match=shape_match,
        onnx_shape=onnx_shape,
        pytorch_shape=pytorch_shape,
        max_abs_diff=max_abs,
        mean_abs_diff=mean_abs,
        mse=mse,
        cosine_similarity=cosine,
        argmax_agreement=agreement,
        mask_iou_per_class=ious,
        mean_mask_iou=mean_iou,
    )
