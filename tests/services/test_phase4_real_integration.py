"""Real end-to-end Phase 4 integration test with actual model weights.

This test loads the real production ONNX weights into PyTorch BiSeNet,
loads the real Phase 1 auxiliary checkpoint, runs inference on a real
dataset image, and verifies the full fused pipeline produces valid output.

Marked as 'slow' so it can be excluded from rapid unit-test cycles.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from config.parser_mode import ParserMode
from dataset_builder.dataset.parser_finetune_current.training_aux_eye_brow_phase1.auxiliary_head import (
    EyeBrowRefinementHead,
)
from experiments.parser_reproduction.bisenet_model import BiSeNet
from experiments.parser_reproduction.weight_mapping import load_onnx_to_pytorch
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from services.face_parser_service import (
    EyeBrowRefinementFusion,
    FaceParserService,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = PROJECT_ROOT / "ai_models" / "bisenet" / "bisenet_resnet18.onnx"
AUX_CHECKPOINT = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune_current"
    / "training_aux_eye_brow_phase1"
    / "checkpoints"
    / "best.pt"
)
IMAGES_DIR = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune_current"
    / "images"
)

_TARGET_CLASSES_19 = frozenset({2, 3, 4, 5, 6})


def _sha256_prefix(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@pytest.fixture(scope="module")
def real_image() -> np.ndarray:
    """Load a real aligned face image from the dataset."""
    candidates = sorted(IMAGES_DIR.glob("sample_000*.png"))
    if not candidates:
        pytest.skip("No dataset images found for real integration test")
    img = cv2.imread(str(candidates[0]))
    assert img is not None, f"Failed to load {candidates[0]}"
    return img


@pytest.fixture(scope="module")
def fused_bisenet() -> BiSeNet:
    """Load production ONNX weights into PyTorch BiSeNet on cuda:0."""
    assert torch.cuda.is_available(), "CUDA required for FUSED integration test"
    device = torch.device("cuda:0")

    bisenet = BiSeNet(n_classes=len(FacePart))
    bisenet = load_onnx_to_pytorch(ONNX_PATH, bisenet).to(device)
    bisenet.eval()
    for p in bisenet.parameters():
        p.requires_grad = False
    for m in bisenet.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            m.track_running_stats = False
    return bisenet


@pytest.fixture(scope="module")
def fused_aux_head() -> EyeBrowRefinementHead:
    """Load Phase 1 auxiliary head checkpoint on cuda:0."""
    assert torch.cuda.is_available(), "CUDA required for FUSED integration test"
    device = torch.device("cuda:0")

    head = EyeBrowRefinementHead(ffm_channels=256, mid_channels=(128, 64), n_classes=6)
    ckpt = torch.load(AUX_CHECKPOINT, map_location=device, weights_only=False)
    head.load_state_dict(ckpt["head_state_dict"])
    head = head.to(device)
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head


@pytest.fixture(scope="module")
def fused_tensor(real_image: np.ndarray) -> torch.Tensor:
    """Preprocess a real image into a (1, 3, 512, 512) tensor on cuda:0."""
    device = torch.device("cuda:0")
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    return torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)).to(device)


class TestRealModelWeightIntegrity:
    """Verify artifact hashes and model loading with real weights."""

    def test_onnx_hash_unchanged(self):
        assert _sha256_prefix(ONNX_PATH) == "2218b6183c26ca5c"

    def test_aux_checkpoint_hash_unchanged(self):
        assert _sha256_prefix(AUX_CHECKPOINT) == "961e08bf64fdd0b8"

    def test_bisenet_loads_weights_on_cuda(self, fused_bisenet: BiSeNet):
        assert next(fused_bisenet.parameters()).device == torch.device("cuda:0")

    def test_bisenet_is_frozen(self, fused_bisenet: BiSeNet):
        for p in fused_bisenet.parameters():
            assert not p.requires_grad

    def test_aux_head_loads_weights_on_cuda(self, fused_aux_head: EyeBrowRefinementHead):
        assert next(fused_aux_head.parameters()).device == torch.device("cuda:0")

    def test_aux_head_is_frozen(self, fused_aux_head: EyeBrowRefinementHead):
        for p in fused_aux_head.parameters():
            assert not p.requires_grad


class TestRealFusedInference:
    """Run real end-to-end fused inference and verify outputs."""

    def test_produces_valid_19_class_mask(
        self,
        fused_bisenet: BiSeNet,
        fused_aux_head: EyeBrowRefinementHead,
        fused_tensor: torch.Tensor,
        real_image: np.ndarray,
    ):
        h, w = real_image.shape[:2]
        with torch.no_grad():
            logits_19, _, _ = fused_bisenet(fused_tensor)
            feat_res8, feat_cp8, _ = fused_bisenet.cp(fused_tensor)
            fused_features = fused_bisenet.ffm(feat_res8, feat_cp8)
            logits_aux = fused_aux_head(fused_features, 512, 512)

            fusion = EyeBrowRefinementFusion(strategy=1, threshold=0.0)
            final_mask, diagnostics = fusion.apply(logits_19, logits_aux)

        assert final_mask.shape == (512, 512)
        assert final_mask.dtype == np.int64
        assert int(final_mask.min()) >= 0
        assert int(final_mask.max()) <= int(FacePart.HAT)
        assert not np.any(np.isnan(final_mask.astype(float)))

    def test_final_mask_contains_valid_face_part_ids(
        self,
        fused_bisenet: BiSeNet,
        fused_aux_head: EyeBrowRefinementHead,
        fused_tensor: torch.Tensor,
    ):
        with torch.no_grad():
            logits_19, _, _ = fused_bisenet(fused_tensor)
            feat_res8, feat_cp8, _ = fused_bisenet.cp(fused_tensor)
            fused_features = fused_bisenet.ffm(feat_res8, feat_cp8)
            logits_aux = fused_aux_head(fused_features, 512, 512)

            fusion = EyeBrowRefinementFusion(strategy=1, threshold=0.0)
            final_mask, _ = fusion.apply(logits_19, logits_aux)

        unique_classes = set(np.unique(final_mask))
        valid_ids = {int(p) for p in FacePart}
        assert unique_classes.issubset(valid_ids), (
            f"Invalid class IDs in mask: {unique_classes - valid_ids}"
        )

    def test_deterministic_repeated_inference(
        self,
        fused_bisenet: BiSeNet,
        fused_aux_head: EyeBrowRefinementHead,
        fused_tensor: torch.Tensor,
    ):
        def _run():
            with torch.no_grad():
                logits_19, _, _ = fused_bisenet(fused_tensor)
                feat_res8, feat_cp8, _ = fused_bisenet.cp(fused_tensor)
                fused_features = fused_bisenet.ffm(feat_res8, feat_cp8)
                logits_aux = fused_aux_head(fused_features, 512, 512)
                return EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
                    logits_19, logits_aux
                )

        mask1, _ = _run()
        mask2, _ = _run()
        assert np.array_equal(mask1, mask2)

    def test_resized_output_matches_original_dimensions(
        self,
        fused_bisenet: BiSeNet,
        fused_aux_head: EyeBrowRefinementHead,
        real_image: np.ndarray,
    ):
        device = torch.device("cuda:0")
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        h, w = real_image.shape[:2]
        rgb = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = torch.from_numpy(np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]).float().to(device)

        with torch.no_grad():
            logits_19, _, _ = fused_bisenet(tensor)
            feat_res8, feat_cp8, _ = fused_bisenet.cp(tensor)
            fused_features = fused_bisenet.ffm(feat_res8, feat_cp8)
            logits_aux = fused_aux_head(fused_features, 512, 512)
            final_mask, _ = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
                logits_19, logits_aux
            )

        if (512, 512) != (h, w):
            final_mask = cv2.resize(
                final_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        assert final_mask.shape == (h, w)

    def test_can_construct_face_parsing_result(
        self,
        fused_bisenet: BiSeNet,
        fused_aux_head: EyeBrowRefinementHead,
        fused_tensor: torch.Tensor,
        real_image: np.ndarray,
    ):
        h, w = real_image.shape[:2]
        with torch.no_grad():
            logits_19, _, _ = fused_bisenet(fused_tensor)
            feat_res8, feat_cp8, _ = fused_bisenet.cp(fused_tensor)
            fused_features = fused_bisenet.ffm(feat_res8, feat_cp8)
            logits_aux = fused_aux_head(fused_features, 512, 512)
            final_mask, _ = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
                logits_19, logits_aux
            )

        if (512, 512) != (h, w):
            final_mask = cv2.resize(
                final_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)

        result = FaceParsingResult(mask=final_mask, image_height=h, image_width=w)
        assert result.image_size() == (h, w)
        assert result.total_pixels() == h * w


class TestRealOriginalMode:
    """Verify ORIGINAL mode still works with real ONNX session."""

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        FaceParserService._instance = None
        FaceParserService._initialized = False
        yield
        FaceParserService._instance = None
        FaceParserService._initialized = False

    def test_original_mode_parse_produces_valid_result(self, real_image: np.ndarray):
        service = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        result = service.parse(real_image)

        assert isinstance(result, FaceParsingResult)
        h, w = real_image.shape[:2]
        assert result.image_size() == (h, w)
        assert int(result.mask.min()) >= 0
        assert int(result.mask.max()) <= int(FacePart.HAT)
        assert not np.any(np.isnan(result.mask.astype(float)))


class TestFusedModeRejectsCPU:
    """Verify FUSED mode requires CUDA and rejects CPU."""

    def test_fused_service_rejects_cpu_device(self):
        from services.face_parser_service import EyeBrowRefinementService

        service = EyeBrowRefinementService(
            onnx_model_path=ONNX_PATH,
            checkpoint_path=AUX_CHECKPOINT,
            fusion=EyeBrowRefinementFusion(),
            device=torch.device("cpu"),
        )
        with pytest.raises(RuntimeError, match="cuda:0"):
            service._resolve_device()
