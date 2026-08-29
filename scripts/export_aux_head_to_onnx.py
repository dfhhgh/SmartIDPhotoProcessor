"""
Script to export the trained EyeBrowRefinementHead to ONNX format.

Target: ai_models/bisenet/aux_head.onnx
Source: dataset_builder/dataset/parser_finetune_current/training_aux_eye_brow_phase1/checkpoints/best.pt
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "training_aux_eye_brow_phase1"))

import torch
import torch.onnx
import onnx
import onnx.checker

from auxiliary_head import EyeBrowRefinementHead

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune_current"
    / "training_aux_eye_brow_phase1"
    / "checkpoints"
    / "best.pt"
)
OUTPUT_ONNX_PATH = PROJECT_ROOT / "ai_models" / "bisenet" / "aux_head.onnx"

EXPECTED_SHA256 = "961e08bf64fdd0b8ae044ac6bf0d30ecbed13a22364301903a2c42c0c99e6e00"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def export_aux_head() -> None:
    logger.info("Inspecting auxiliary checkpoint at %s", CHECKPOINT_PATH)
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    file_hash = _sha256(CHECKPOINT_PATH)
    logger.info("Checkpoint SHA256: %s", file_hash)
    if not file_hash.startswith(EXPECTED_SHA256[:16]):
        raise ValueError(f"Checkpoint hash mismatch! Expected prefix {EXPECTED_SHA256[:16]}, got {file_hash[:16]}")

    device = torch.device("cpu")
    head = EyeBrowRefinementHead(ffm_channels=256, mid_channels=(128, 64), n_classes=6)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint["head_state_dict"]
    head.load_state_dict(state_dict)
    head.eval()

    total_params = sum(p.numel() for p in head.parameters())
    logger.info("Loaded EyeBrowRefinementHead successfully. Parameter count: %d", total_params)
    if total_params != 369408:
        raise ValueError(f"Expected 369408 parameters, got {total_params}")

    OUTPUT_ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Dummy input: (1, 256, 64, 64) for FFM output at 512x512 input resolution
    dummy_input = torch.randn(1, 256, 64, 64, dtype=torch.float32)
    # The forward signature is forward(self, ffm_features, target_h, target_w)
    # In ONNX export, target_h and target_w are static integers passed to interpolate.
    # We can wrap the head to bake in target_h=512, target_w=512.
    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x, 512, 512)

    wrapped_model = ExportWrapper(head)
    wrapped_model.eval()

    logger.info("Exporting EyeBrowRefinementHead to ONNX at %s ...", OUTPUT_ONNX_PATH)
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        str(OUTPUT_ONNX_PATH),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["ffm_features"],
        output_names=["logits_aux"],
        dynamic_axes=None,
    )

    logger.info("Verifying exported ONNX model with onnx.checker...")
    onnx_model = onnx.load(str(OUTPUT_ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model check passed successfully.")

    onnx_hash = _sha256(OUTPUT_ONNX_PATH)
    onnx_size = OUTPUT_ONNX_PATH.stat().st_size
    logger.info("Exported ONNX SHA256: %s (size: %d bytes)", onnx_hash, onnx_size)


if __name__ == "__main__":
    export_aux_head()
