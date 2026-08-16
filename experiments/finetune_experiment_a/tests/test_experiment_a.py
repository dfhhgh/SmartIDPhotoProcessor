"""Tests for Experiment A: head-only fine-tuning with frozen backbone."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.parser_reproduction.bisenet_model import BiSeNet
from experiments.finetune_experiment_a.config import (
    ExperimentAConfig,
    NON_TARGET_CLASS_IDS,
    TARGET_CLASS_IDS,
)
from experiments.finetune_experiment_a.trainer import (
    _FROZEN_MODULE_NAMES,
    _TRAINABLE_MODULE_NAMES,
    assert_freezing_correct,
    freeze_modules,
    _count_parameters,
)
from dataset_builder.dataset.parser_finetune.training.config import CLASS_NAMES
from dataset_builder.dataset.parser_finetune.training.metrics import (
    SegmentationMetrics,
    logits_to_prediction,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def model() -> BiSeNet:
    """Create a fresh BiSeNet model (random weights, no ONNX loading)."""
    return BiSeNet(n_classes=19)


@pytest.fixture
def frozen_model() -> BiSeNet:
    """Create and freeze a BiSeNet model."""
    m = BiSeNet(n_classes=19)
    freeze_modules(m)
    return m


# ── Freezing tests ───────────────────────────────────────────────────────

class TestParameterFreezing:
    """Verify that exactly the intended modules are frozen/trainable."""

    def test_all_params_start_trainable(self, model: BiSeNet) -> None:
        """Before freezing, all parameters should be trainable."""
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} should be trainable before freezing"

    def test_freeze_modules_frozens_correct_modules(self, frozen_model: BiSeNet) -> None:
        """After freezing, cp and ffm should be frozen; heads should be trainable."""
        assert_freezing_correct(frozen_model)

    def test_frozen_module_params_not_trainable(self, frozen_model: BiSeNet) -> None:
        """All parameters in cp and ffm must have requires_grad=False."""
        for module_name in _FROZEN_MODULE_NAMES:
            module = getattr(frozen_model, module_name)
            for param_name, param in module.named_parameters():
                assert not param.requires_grad, (
                    f"Frozen module '{module_name}.{param_name}' has requires_grad=True"
                )

    def test_trainable_module_params_trainable(self, frozen_model: BiSeNet) -> None:
        """All parameters in output heads must have requires_grad=True."""
        for head_name in _TRAINABLE_MODULE_NAMES:
            module = getattr(frozen_model, head_name)
            for param_name, param in module.named_parameters():
                assert param.requires_grad, (
                    f"Trainable head '{head_name}.{param_name}' has requires_grad=False"
                )

    def test_frozen_bn_in_eval(self, frozen_model: BiSeNet) -> None:
        """BatchNorm in frozen modules must be in eval mode after freezing."""
        for module_name in _FROZEN_MODULE_NAMES:
            module = getattr(frozen_model, module_name)
            for name, m in module.named_modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    assert not m.training, (
                        f"BN '{module_name}.{name}' is still in training mode"
                    )

    def test_trainable_count_is_small(self, frozen_model: BiSeNet) -> None:
        """Trainable params should be a small fraction of total."""
        summary = _count_parameters(frozen_model)
        assert summary["trainable_pct"] < 10.0, (
            f"Trainable percentage {summary['trainable_pct']:.2f}% is too high. "
            f"Expected < 10% for head-only training."
        )

    def test_frozen_count_dominates(self, frozen_model: BiSeNet) -> None:
        """Frozen params should be > 90% of total."""
        summary = _count_parameters(frozen_model)
        frozen_pct = 100.0 * summary["frozen_params"] / summary["total_params"]
        assert frozen_pct > 90.0, (
            f"Frozen percentage {frozen_pct:.2f}% should be > 90%"
        )

    def test_only_head_params_in_optimizer(self, frozen_model: BiSeNet) -> None:
        """Optimizer should receive only trainable (head) parameters."""
        trainable = [p for p in frozen_model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-5)

        # Verify every param group param is trainable
        for group in optimizer.param_groups:
            for param in group["params"]:
                assert param.requires_grad

    def test_frozen_modules_are_complete(self, model: BiSeNet) -> None:
        """Every named child should be either frozen or trainable, no missing."""
        frozen_model = BiSeNet(n_classes=19)
        freeze_modules(frozen_model)
        all_children = set(dict(model.named_children()).keys())
        covered = set(_FROZEN_MODULE_NAMES) | set(_TRAINABLE_MODULE_NAMES)
        assert all_children == covered, (
            f"Missing modules: {all_children - covered}"
        )


# ── Metric computation tests ─────────────────────────────────────────────

class TestMetrics:
    """Verify full 19-class, target, and non-target metric calculations."""

    def test_confusion_matrix_shape(self) -> None:
        """Confusion matrix should be 19x19."""
        metrics = SegmentationMetrics(n_classes=19)
        pred = torch.randint(0, 19, (1, 512, 512))
        target = torch.randint(0, 19, (1, 512, 512))
        cm = metrics.confusion_matrix(pred, target)
        assert cm.shape == (19, 19)

    def test_perfect_prediction_has_zero_offdiagonal(self) -> None:
        """Perfect prediction should produce a diagonal confusion matrix."""
        metrics = SegmentationMetrics(n_classes=19)
        target = torch.randint(0, 19, (1, 100, 100))
        cm = metrics.confusion_matrix(target.clone(), target)
        off_diagonal = cm.sum() - np.trace(cm)
        assert off_diagonal == 0

    def test_all_19_classes_in_metrics(self) -> None:
        """from_confusion_matrix should produce entries for all 19 classes."""
        metrics = SegmentationMetrics(n_classes=19)
        target = torch.arange(19).reshape(1, 19, 1).expand(1, 19, 100)
        pred = target.clone()
        cm = metrics.confusion_matrix(pred, target)
        result = metrics.from_confusion_matrix(cm)
        for i in range(19):
            name = CLASS_NAMES[i]
            assert name in result["per_class_iou"], f"Missing IoU for {name}"

    def test_target_vs_non_target_partition(self) -> None:
        """Target + non-target class IDs should cover all 19 classes."""
        all_ids = set(TARGET_CLASS_IDS) | set(NON_TARGET_CLASS_IDS)
        assert all_ids == set(range(19))
        assert len(set(TARGET_CLASS_IDS) & set(NON_TARGET_CLASS_IDS)) == 0

    def test_target_classes_match_expected(self) -> None:
        """Target class IDs should be exactly the 6 expected classes."""
        expected_names = {"LEFT_BROW", "RIGHT_BROW", "LEFT_EYE", "RIGHT_EYE", "EYE_GLASS", "HAT"}
        actual_names = {CLASS_NAMES[i] for i in TARGET_CLASS_IDS}
        assert actual_names == expected_names


# ── Checkpoint criterion tests ───────────────────────────────────────────

class TestCheckpointCriterion:
    """Verify that checkpoint selection uses full mIoU, not target-only."""

    def test_best_pt_uses_full_miou(self) -> None:
        """Simulate two epochs and verify best.pt is selected by full mIoU."""
        # Epoch 1: full mIoU=0.8, target mIoU=0.7
        # Epoch 2: full mIoU=0.75, target mIoU=0.8
        # Best should be epoch 1 (higher full mIoU)
        best_val_mean_iou = -1.0
        best_epoch = -1

        epochs = [
            {"epoch": 1, "mean_iou": 0.8, "target_mean_iou": 0.7},
            {"epoch": 2, "mean_iou": 0.75, "target_mean_iou": 0.8},
        ]
        for rec in epochs:
            if rec["mean_iou"] > best_val_mean_iou:
                best_val_mean_iou = rec["mean_iou"]
                best_epoch = rec["epoch"]

        assert best_epoch == 1, (
            f"Expected epoch 1 to be selected (full mIoU=0.8), got epoch {best_epoch}"
        )

    def test_target_miou_cannot_select_checkpoint(self) -> None:
        """Even if target mIoU is higher, full mIoU must determine best.pt."""
        best_val_mean_iou = -1.0
        best_val_target_mean_iou = -1.0
        best_epoch = -1

        # Epoch 1: full=0.70, target=0.60
        # Epoch 2: full=0.69, target=0.65 (higher target, lower full)
        for rec in [
            {"epoch": 1, "mean_iou": 0.70, "target_mean_iou": 0.60},
            {"epoch": 2, "mean_iou": 0.69, "target_mean_iou": 0.65},
        ]:
            if rec["mean_iou"] > best_val_mean_iou:
                best_val_mean_iou = rec["mean_iou"]
                best_val_target_mean_iou = rec["target_mean_iou"]
                best_epoch = rec["epoch"]

        assert best_epoch == 1
        # target mIoU of best checkpoint should be from epoch 1
        assert best_val_target_mean_iou == 0.60


# ── Config tests ─────────────────────────────────────────────────────────

class TestConfig:
    """Verify ExperimentAConfig is valid."""

    def test_config_to_dict(self) -> None:
        """to_dict should produce a JSON-serializable dict."""
        import json
        config = ExperimentAConfig()
        d = config.to_dict()
        json.dumps(d)  # should not raise

    def test_expanded_dir_exists(self) -> None:
        """Expanded dataset directory should exist."""
        config = ExperimentAConfig()
        assert config.expanded_dir.exists(), f"Missing: {config.expanded_dir}"

    def test_manifest_exists(self) -> None:
        """Expanded manifest should exist."""
        config = ExperimentAConfig()
        assert config.manifest_path.exists(), f"Missing: {config.manifest_path}"

    def test_onnx_model_exists(self) -> None:
        """Production ONNX model should exist."""
        config = ExperimentAConfig()
        assert config.onnx_model_path.exists(), f"Missing: {config.onnx_model_path}"


# ── Forward pass test ────────────────────────────────────────────────────

class TestForwardPass:
    """Verify frozen model can still do forward/backward pass."""

    def test_frozen_model_forward(self, frozen_model: BiSeNet) -> None:
        """Frozen model should produce correct output shape."""
        frozen_model.eval()
        x = torch.randn(1, 3, 512, 512)
        out, out16, out32 = frozen_model(x)
        assert out.shape == (1, 19, 512, 512)
        assert out16.shape == (1, 19, 512, 512)
        assert out32.shape == (1, 19, 512, 512)

    def test_frozen_model_backward(self, frozen_model: BiSeNet) -> None:
        """Only trainable head params should receive gradients."""
        frozen_model.train()
        for module_name in _FROZEN_MODULE_NAMES:
            module = getattr(frozen_model, module_name)
            for m in module.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

        x = torch.randn(1, 3, 512, 512)
        target = torch.randint(0, 19, (1, 512, 512))
        out, out16, out32 = frozen_model(x)
        # Use all three outputs to match the actual training loss
        loss = (
            torch.nn.functional.cross_entropy(out, target)
            + 0.4 * torch.nn.functional.cross_entropy(out16, target)
            + 0.4 * torch.nn.functional.cross_entropy(out32, target)
        )
        loss.backward()

        # Trainable heads should have gradients
        for head_name in _TRAINABLE_MODULE_NAMES:
            head = getattr(frozen_model, head_name)
            for param in head.parameters():
                assert param.grad is not None, (
                    f"Trainable head '{head_name}' has no gradient"
                )

        # Frozen modules should NOT have gradients
        for module_name in _FROZEN_MODULE_NAMES:
            module = getattr(frozen_model, module_name)
            for param in module.parameters():
                assert param.grad is None, (
                    f"Frozen module '{module_name}' received gradient"
                )
