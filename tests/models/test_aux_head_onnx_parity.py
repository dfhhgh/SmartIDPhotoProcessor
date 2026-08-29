"""
Phase 7: Auxiliary Head ONNX export and comprehensive numerical parity validation.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import onnx
import onnx.checker
import onnxruntime as ort
import pytest
import torch

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.parser_mode import ParserMode
from models.parsing.face_part import FacePart
from services.face_parser_service import (
    EyeBrowRefinementFusion,
    FaceParserService,
)
from experiments.parser_reproduction.bisenet_model import BiSeNet
from experiments.parser_reproduction.weight_mapping import load_onnx_to_pytorch
from dataset_builder.dataset.parser_finetune_current.training_aux_eye_brow_phase1.auxiliary_head import (
    EyeBrowRefinementHead,
)

logger = logging.getLogger(__name__)

ONNX_BACKBONE_PATH = PROJECT_ROOT / "ai_models" / "bisenet" / "bisenet_resnet18.onnx"
AUX_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune_current"
    / "training_aux_eye_brow_phase1"
    / "checkpoints"
    / "best.pt"
)
AUX_ONNX_PATH = PROJECT_ROOT / "ai_models" / "bisenet" / "aux_head.onnx"
IMAGES_DIR = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune_current"
    / "images"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


MAX_ABS_TOL = 1e-4


class TestPhase7AuxHeadOnnxParity:
    """Rigorous Phase 7 parity suite comparing PyTorch vs ONNX auxiliary head and full fusion.

    All auxiliary-head parity comparisons run CPU PyTorch vs CPU ONNX on identical
    input tensors to isolate export fidelity from execution-provider variation.
    """

    def test_1_checkpoint_integrity(self):
        assert AUX_CHECKPOINT_PATH.exists()
        assert AUX_CHECKPOINT_PATH.stat().st_size == 4470853
        assert _sha256(AUX_CHECKPOINT_PATH).startswith("961e08bf64fdd0b8")

    def test_2_onnx_model_exists_and_valid(self):
        assert AUX_ONNX_PATH.exists()
        onnx_model = onnx.load(str(AUX_ONNX_PATH))
        onnx.checker.check_model(onnx_model)

        sess = ort.InferenceSession(str(AUX_ONNX_PATH), providers=["CPUExecutionProvider"])
        assert [i.name for i in sess.get_inputs()] == ["ffm_features"]
        assert [o.name for o in sess.get_outputs()] == ["logits_aux"]

    def test_3_synthetic_logits_parity(self):
        device = torch.device("cpu")
        head = EyeBrowRefinementHead(ffm_channels=256, mid_channels=(128, 64), n_classes=6)
        ckpt = torch.load(AUX_CHECKPOINT_PATH, map_location=device, weights_only=False)
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval()

        sess = ort.InferenceSession(str(AUX_ONNX_PATH), providers=["CPUExecutionProvider"])

        for name, test_tensor in [
            ("random", torch.randn(1, 256, 64, 64, dtype=torch.float32)),
            ("zeros", torch.zeros(1, 256, 64, 64, dtype=torch.float32)),
            ("ones", torch.ones(1, 256, 64, 64, dtype=torch.float32)),
        ]:
            with torch.no_grad():
                pt_out = head(test_tensor, 512, 512).numpy()

            onnx_out = sess.run(None, {"ffm_features": test_tensor.numpy()})[0]

            max_abs_diff = float(np.abs(pt_out - onnx_out).max())
            mean_abs_diff = float(np.abs(pt_out - onnx_out).mean())
            rmse = float(np.sqrt(np.mean((pt_out - onnx_out) ** 2)))

            pt_argmax = np.argmax(pt_out, axis=1)
            onnx_argmax = np.argmax(onnx_out, axis=1)
            argmax_equal = bool(np.array_equal(pt_argmax, onnx_argmax))

            logger.info(
                "Synthetic [%s] max=%.2e mean=%.2e RMSE=%.2e argmax_eq=%s",
                name, max_abs_diff, mean_abs_diff, rmse, argmax_equal,
            )

            assert max_abs_diff <= MAX_ABS_TOL, (
                f"Logits diff {max_abs_diff:.2e} > {MAX_ABS_TOL} on {name}"
            )
            assert argmax_equal, f"Argmax mismatch on {name}"

    def test_4_real_feature_and_fusion_parity(self):
        image_paths = sorted(IMAGES_DIR.glob("sample_000*.png"))
        if not image_paths:
            pytest.skip("No dataset images found for real parity test")

        bisenet_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        bisenet = BiSeNet(n_classes=len(FacePart))
        bisenet = load_onnx_to_pytorch(ONNX_BACKBONE_PATH, bisenet).to(bisenet_device)
        bisenet.eval()
        for p in bisenet.parameters():
            p.requires_grad = False

        cpu_device = torch.device("cpu")
        pt_head_cpu = EyeBrowRefinementHead(ffm_channels=256, mid_channels=(128, 64), n_classes=6)
        ckpt = torch.load(AUX_CHECKPOINT_PATH, map_location=cpu_device, weights_only=False)
        pt_head_cpu.load_state_dict(ckpt["head_state_dict"])
        pt_head_cpu.eval()
        for p in pt_head_cpu.parameters():
            p.requires_grad = False

        onnx_sess = ort.InferenceSession(str(AUX_ONNX_PATH), providers=["CPUExecutionProvider"])

        fusion = EyeBrowRefinementFusion(strategy=1, threshold=0.0)

        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        total_tested = 0
        max_seen_diff = 0.0

        for img_path in image_paths[:20]:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
            normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
            tensor_np = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
            tensor_bisenet = torch.from_numpy(
                np.ascontiguousarray(tensor_np, dtype=np.float32)
            ).to(bisenet_device)

            with torch.no_grad():
                feat_res8, feat_cp8, _ = bisenet.cp(tensor_bisenet)
                feat_fuse_cuda = bisenet.ffm(feat_res8, feat_cp8)
                logits_19_cuda = bisenet.conv_out(feat_fuse_cuda, 512, 512)

            feat_fuse_cpu = feat_fuse_cuda.detach().cpu().contiguous()
            logits_19_cpu = logits_19_cuda.detach().cpu().contiguous()

            with torch.no_grad():
                logits_aux_pt_cpu = pt_head_cpu(feat_fuse_cpu, 512, 512)

            logits_aux_onnx_np = onnx_sess.run(
                None, {"ffm_features": feat_fuse_cpu.numpy()}
            )[0]
            logits_aux_onnx = torch.from_numpy(logits_aux_onnx_np)

            max_abs_diff = float(torch.abs(logits_aux_pt_cpu - logits_aux_onnx).max())
            mean_abs_diff = float(torch.abs(logits_aux_pt_cpu - logits_aux_onnx).mean())
            rmse_val = float(
                torch.sqrt(torch.mean((logits_aux_pt_cpu - logits_aux_onnx) ** 2))
            )
            max_seen_diff = max(max_seen_diff, max_abs_diff)

            pt_argmax = np.argmax(logits_aux_pt_cpu.numpy(), axis=1)
            onnx_argmax = np.argmax(logits_aux_onnx.numpy(), axis=1)
            argmax_equal = bool(np.array_equal(pt_argmax, onnx_argmax))
            total_pixels = pt_argmax.size
            n_diff = int(np.sum(pt_argmax != onnx_argmax))
            pct_class_diff = 100.0 * n_diff / total_pixels

            logger.info(
                "[%s] max=%.2e mean=%.2e RMSE=%.2e argmax_eq=%s pct_diff=%.6f%%",
                img_path.name, max_abs_diff, mean_abs_diff, rmse_val,
                argmax_equal, pct_class_diff,
            )

            assert max_abs_diff <= MAX_ABS_TOL, (
                f"Aux logits diff {max_abs_diff:.2e} > {MAX_ABS_TOL} on {img_path.name}"
            )
            assert argmax_equal, (
                f"Argmax mismatch ({n_diff} pixels) on {img_path.name}"
            )

            mask_pt, diag_pt = fusion.apply(logits_19_cpu, logits_aux_pt_cpu)
            mask_onnx, diag_onnx = fusion.apply(logits_19_cpu, logits_aux_onnx)

            assert diag_pt.corrections_attempted == diag_onnx.corrections_attempted
            assert diag_pt.corrections_accepted == diag_onnx.corrections_accepted
            assert diag_pt.corrections_rejected == diag_onnx.corrections_rejected
            assert diag_pt.roi_pixels == diag_onnx.roi_pixels
            assert np.array_equal(mask_pt, mask_onnx), (
                f"Final mask mismatch on {img_path.name}"
            )
            total_tested += 1

        logger.info(
            "Tested %d images. Max CPU aux logit diff: %.2e. 100%% exact mask equality.",
            total_tested, max_seen_diff,
        )
        assert total_tested > 0
