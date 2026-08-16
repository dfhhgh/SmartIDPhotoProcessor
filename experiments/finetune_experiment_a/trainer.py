"""Experiment A trainer: head-only BiSeNet fine-tuning with frozen backbone.

Freezes the ContextPath (backbone + ARM + conv_heads) and FeatureFusionModule.
Trains only the three BiSeNetOutput heads (conv_out, conv_out16, conv_out32).
Selects best checkpoint by full 19-class validation mIoU.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.parser_reproduction.bisenet_model import BiSeNet
from experiments.parser_reproduction.weight_mapping import load_onnx_to_pytorch
from dataset_builder.dataset.parser_finetune.training.config import (
    CLASS_NAMES,
    TARGET_CLASS_IDS,
)
from dataset_builder.dataset.parser_finetune.training.losses import SegmentationLoss
from dataset_builder.dataset.parser_finetune.training.metrics import (
    SegmentationMetrics,
    logits_to_prediction,
)
from dataset_builder.dataset.parser_finetune_expanded.training.expanded_dataset import (
    ExpandedParserDataset,
)
from .config import ExperimentAConfig, NON_TARGET_CLASS_IDS

logger = logging.getLogger(__name__)

# ── Modules to freeze (feature extraction / fusion) ──────────────────────
_FROZEN_MODULE_NAMES: tuple[str, ...] = (
    "cp",   # ContextPath: backbone (ResNet18) + ARM16/ARM32 + conv_heads + conv_avg
    "ffm",  # FeatureFusionModule
)

# ── Modules that remain trainable (segmentation output heads) ────────────
_TRAINABLE_MODULE_NAMES: tuple[str, ...] = (
    "conv_out",    # Main segmentation head
    "conv_out16",  # Auxiliary head @ 1/16
    "conv_out32",  # Auxiliary head @ 1/32
)


# ──────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────

def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name: str | None = None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────
# Model creation & freezing
# ──────────────────────────────────────────────────────────────────────────

def create_model(config: ExperimentAConfig, device: torch.device) -> BiSeNet:
    """Load ONNX weights into BiSeNet, then freeze non-head modules."""
    if not config.onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {config.onnx_model_path}")
    model = BiSeNet(n_classes=config.n_classes)
    model = load_onnx_to_pytorch(config.onnx_model_path, model)
    model = model.to(device)
    return model


def freeze_modules(model: BiSeNet) -> dict[str, Any]:
    """Freeze backbone + fusion modules. Return parameter summary.

    Returns a dict with 'frozen', 'trainable', 'total' counts and
    lists of module names.
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze only the output heads
    for head_name in _TRAINABLE_MODULE_NAMES:
        head_module = getattr(model, head_name)
        for param in head_module.parameters():
            param.requires_grad = True

    # Also keep BatchNorm in eval mode for frozen modules
    for module_name in _FROZEN_MODULE_NAMES:
        module = getattr(model, module_name)
        for m in module.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()

    return _count_parameters(model)


def _count_parameters(model: BiSeNet) -> dict[str, Any]:
    """Count frozen/trainable parameters and verify module states."""
    frozen_params = 0
    trainable_params = 0
    total_params = 0
    frozen_modules: list[str] = []
    trainable_modules: list[str] = []

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()

    # Identify top-level module status
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if module_trainable == 0:
            frozen_modules.append(f"{name} ({module_params:,} params)")
        else:
            trainable_modules.append(
                f"{name} ({module_trainable:,}/{module_params:,} trainable)"
            )

    return {
        "total_params": total_params,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "trainable_pct": 100.0 * trainable_params / total_params if total_params else 0.0,
        "frozen_module_names": list(_FROZEN_MODULE_NAMES),
        "trainable_module_names": list(_TRAINABLE_MODULE_NAMES),
        "frozen_modules_detail": frozen_modules,
        "trainable_modules_detail": trainable_modules,
    }


def assert_freezing_correct(model: BiSeNet) -> None:
    """Runtime assertion that freezing is applied correctly."""
    for head_name in _TRAINABLE_MODULE_NAMES:
        head = getattr(model, head_name)
        for param in head.parameters():
            assert param.requires_grad, (
                f"Trainable head '{head_name}' has frozen parameter!"
            )

    for module_name in _FROZEN_MODULE_NAMES:
        module = getattr(model, module_name)
        for param in module.parameters():
            assert not param.requires_grad, (
                f"Frozen module '{module_name}' has trainable parameter!"
            )


def print_parameter_summary(summary: dict[str, Any]) -> None:
    """Log a clear parameter summary before training."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT A: PARAMETER FREEZING SUMMARY")
    logger.info("=" * 70)
    logger.info("Total parameters:     %s", f"{summary['total_params']:,}")
    logger.info("Frozen parameters:    %s", f"{summary['frozen_params']:,}")
    logger.info("Trainable parameters: %s", f"{summary['trainable_params']:,}")
    logger.info("Trainable %%:          %.2f%%", summary["trainable_pct"])
    logger.info("")
    logger.info("FROZEN modules (feature extraction / fusion):")
    for mod in summary["frozen_modules_detail"]:
        logger.info("  - %s", mod)
    logger.info("")
    logger.info("TRAINABLE modules (segmentation output heads):")
    for mod in summary["trainable_modules_detail"]:
        logger.info("  - %s", mod)
    logger.info("=" * 70)


# ──────────────────────────────────────────────────────────────────────────
# Dataset / DataLoader
# ──────────────────────────────────────────────────────────────────────────

def create_expanded_dataloader(
    config: ExperimentAConfig,
    split: str,
    shuffle: bool = False,
    augment: bool | None = None,
    drop_last: bool = False,
) -> DataLoader:
    dataset = ExpandedParserDataset(config=config, split=split, augment=augment)
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        generator=generator,
        drop_last=drop_last,
    )


# ──────────────────────────────────────────────────────────────────────────
# Evaluation (full 19-class + target + non-target)
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: ExperimentAConfig,
) -> dict[str, Any]:
    """Evaluate on all 19 classes. Returns full, target, and non-target metrics."""
    model.eval()
    # Ensure frozen BN layers stay in eval
    for module_name in _FROZEN_MODULE_NAMES:
        module = getattr(model, module_name, None)
        if module is not None:
            for m in module.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

    metrics = SegmentationMetrics(n_classes=config.n_classes)
    matrix = np.zeros((config.n_classes, config.n_classes), dtype=np.int64)
    sample_ids: list[str] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        outputs = model(images)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        preds = logits_to_prediction(logits)
        matrix += metrics.confusion_matrix(preds, masks)
        sample_ids.extend(str(sid) for sid in batch["sample_id"])

    result = metrics.from_confusion_matrix(matrix)

    # Compute non-target mIoU
    per_class_iou = result["per_class_iou"]
    non_target_ious = [
        per_class_iou[CLASS_NAMES[cid]]
        for cid in NON_TARGET_CLASS_IDS
        if per_class_iou.get(CLASS_NAMES[cid]) is not None
    ]
    result["non_target_mean_iou"] = (
        float(np.mean(non_target_ious)) if non_target_ious else 0.0
    )

    # Compute per-class precision and recall from confusion matrix
    per_class_precision: dict[str, float | None] = {}
    per_class_recall: dict[str, float | None] = {}
    for class_id in range(config.n_classes):
        tp = float(matrix[class_id, class_id])
        fp = float(matrix[:, class_id].sum() - tp)
        fn = float(matrix[class_id, :].sum() - tp)
        name = CLASS_NAMES[class_id]
        per_class_precision[name] = tp / (tp + fp) if (tp + fp) > 0 else None
        per_class_recall[name] = tp / (tp + fn) if (tp + fn) > 0 else None

    result["per_class_precision"] = per_class_precision
    result["per_class_recall"] = per_class_recall
    result["sample_ids"] = sample_ids
    result["sample_count"] = len(sample_ids)
    result["confusion_matrix"] = matrix.astype(int).tolist()
    return result


# ──────────────────────────────────────────────────────────────────────────
# Checkpoint save / load
# ──────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: ExperimentAConfig,
    epoch: int,
    history: list[dict[str, Any]],
    best_val_mean_iou: float,
    best_val_target_mean_iou: float,
    param_summary: dict[str, Any],
    val_metrics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.to_dict(),
        "history": history,
        "best_val_mean_iou": best_val_mean_iou,
        "best_val_target_mean_iou": best_val_target_mean_iou,
        "class_names": CLASS_NAMES,
        "target_class_ids": list(TARGET_CLASS_IDS),
        "non_target_class_ids": list(NON_TARGET_CLASS_IDS),
        "checkpoint_criterion": "best_val_mean_iou",
        "trainable_param_summary": {
            "total_params": param_summary["total_params"],
            "frozen_params": param_summary["frozen_params"],
            "trainable_params": param_summary["trainable_params"],
            "trainable_pct": param_summary["trainable_pct"],
            "frozen_module_names": param_summary["frozen_module_names"],
            "trainable_module_names": param_summary["trainable_module_names"],
        },
    }
    if val_metrics is not None:
        payload["per_class_val_iou"] = val_metrics.get("per_class_iou", {})
        payload["per_class_val_precision"] = val_metrics.get("per_class_precision", {})
        payload["per_class_val_recall"] = val_metrics.get("per_class_recall", {})
    torch.save(payload, path)


# ──────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────

def train_experiment_a(
    config: ExperimentAConfig | None = None,
    device_name: str | None = None,
) -> dict[str, Any]:
    """Run Experiment A: head-only fine-tuning with frozen backbone."""
    config = config or ExperimentAConfig()
    set_reproducible_seed(config.seed)
    device = resolve_device(device_name)

    logger.info("Creating model and loading ONNX weights...")
    model = create_model(config, device)

    logger.info("Freezing backbone and feature-fusion modules...")
    param_summary = freeze_modules(model)
    assert_freezing_correct(model)
    print_parameter_summary(param_summary)

    train_loader = create_expanded_dataloader(
        config, split="train", shuffle=True, augment=True, drop_last=True
    )
    val_loader = create_expanded_dataloader(
        config, split="val", shuffle=False, augment=False
    )
    test_loader = create_expanded_dataloader(
        config, split="test", shuffle=False, augment=False
    )

    # Only pass trainable parameters to the optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = SegmentationLoss(
        ignore_index=config.ignore_index,
        aux16_weight=config.aux16_weight,
        aux32_weight=config.aux32_weight,
        class_weights=config.class_weights,
        n_classes=config.n_classes,
    )

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.report_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []

    # PRIMARY criterion: full 19-class mIoU
    best_val_mean_iou = -1.0
    # Logged but NOT used for selection
    best_val_target_mean_iou = -1.0
    best_epoch = -1

    for epoch in range(1, config.epochs + 1):
        model.train()
        # Keep frozen BN in eval
        for module_name in _FROZEN_MODULE_NAMES:
            module = getattr(model, module_name)
            for m in module.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()

        epoch_loss = 0.0
        batches = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
            batches += 1

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, device, config)

        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_pixel_accuracy": val_metrics["pixel_accuracy"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_target_mean_iou": val_metrics["target_mean_iou"],
            "val_non_target_mean_iou": val_metrics["non_target_mean_iou"],
        }
        history.append(record)

        logger.info(
            "Epoch %d/%d  loss=%.4f  val_mIoU=%.6f  target_mIoU=%.6f  non_target_mIoU=%.6f",
            epoch, config.epochs, record["train_loss"],
            record["val_mean_iou"], record["val_target_mean_iou"],
            record["val_non_target_mean_iou"],
        )

        # PRIMARY: select by full 19-class mIoU
        if val_metrics["mean_iou"] > best_val_mean_iou:
            best_val_mean_iou = float(val_metrics["mean_iou"])
            best_val_target_mean_iou = float(val_metrics["target_mean_iou"])
            best_epoch = epoch
            save_checkpoint(
                config.checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                history,
                best_val_mean_iou,
                best_val_target_mean_iou,
                param_summary,
                val_metrics=val_metrics,
            )

        logger.info(
            "  [best_val_mIoU=%.6f at epoch %d]",
            best_val_mean_iou, best_epoch,
        )

    # Save final checkpoint
    save_checkpoint(
        config.checkpoint_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        config,
        config.epochs,
        history,
        best_val_mean_iou,
        best_val_target_mean_iou,
        param_summary,
    )

    # Evaluate best checkpoint on all splits
    best_ckpt = torch.load(
        config.checkpoint_dir / "best.pt", map_location=device
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    train_eval = evaluate_model(
        model,
        create_expanded_dataloader(config, "train", shuffle=False, augment=False),
        device,
        config,
    )
    val_eval = evaluate_model(model, val_loader, device, config)
    test_eval = evaluate_model(model, test_loader, device, config)

    final_metrics: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "Experiment_A_head_only_finetune",
        "checkpoint_criterion": "best_val_mean_iou",
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_val_mean_iou,
        "best_val_target_mean_iou": best_val_target_mean_iou,
        "train_evaluation": _strip_confusion(train_eval),
        "val_evaluation": _strip_confusion(val_eval),
        "test_evaluation": _strip_confusion(test_eval),
        "history": history,
        "param_summary": param_summary,
        "reproducibility": _reproducibility_record(config, device),
    }

    (config.report_dir / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (config.report_dir / "validation_metrics.json").write_text(
        json.dumps(final_metrics, indent=2, default=str), encoding="utf-8"
    )
    _write_summary(config, final_metrics)
    _write_per_class_json(config, final_metrics)

    logger.info("Training complete. Best epoch: %d", best_epoch)
    logger.info("Best val mIoU (19-class): %.6f", best_val_mean_iou)
    logger.info("Best val target mIoU:     %.6f", best_val_target_mean_iou)
    return final_metrics


def _strip_confusion(metrics: dict[str, Any]) -> dict[str, Any]:
    """Remove large confusion_matrix from metrics for the JSON report."""
    return {k: v for k, v in metrics.items() if k != "confusion_matrix"}


def _reproducibility_record(
    config: ExperimentAConfig, device: torch.device
) -> dict[str, Any]:
    train_count = len(ExpandedParserDataset(config=config, split="train", augment=False))
    val_count = len(ExpandedParserDataset(config=config, split="val", augment=False))
    test_count = len(ExpandedParserDataset(config=config, split="test", augment=False))
    return {
        "seed": config.seed,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "dataset_manifest_path": str(config.manifest_path),
        "train_samples": train_count,
        "val_samples": val_count,
        "test_samples": test_count,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "scheduler": config.scheduler,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "augmentation": config.augmentation,
        "class_weights": config.class_weights,
        "aux16_weight": config.aux16_weight,
        "aux32_weight": config.aux32_weight,
        "ignore_index": config.ignore_index,
        "frozen_module_names": list(_FROZEN_MODULE_NAMES),
        "trainable_module_names": list(_TRAINABLE_MODULE_NAMES),
    }


# ──────────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────────

def _write_summary(config: ExperimentAConfig, metrics: dict[str, Any]) -> None:
    path = config.report_dir / "experiment_summary.md"
    ps = metrics["param_summary"]
    rep = metrics["reproducibility"]
    test = metrics.get("test_evaluation", {})

    best_epoch = metrics["best_epoch"]
    best_val_miou = metrics["best_val_mean_iou"]
    best_val_target = metrics["best_val_target_mean_iou"]

    # Extract per-class IoUs for the best checkpoint (from val)
    best_val_metrics = metrics.get("val_evaluation", {})
    per_class = best_val_metrics.get("per_class_iou", {})
    non_target_miou = best_val_metrics.get("non_target_mean_iou", "N/A")

    target_classes_str = ", ".join(CLASS_NAMES[i] for i in TARGET_CLASS_IDS)

    md = f"""# Experiment A: Head-Only BiSeNet Fine-Tuning

## Hypothesis
Catastrophic forgetting in the previous full-network fine-tuning was caused by
updating the backbone on a small dataset. Freezing feature extraction and
training only the output heads should preserve non-target class performance.

## Architecture Freezing

| Component | Status | Parameters |
|-----------|--------|------------|
| ContextPath (backbone + ARM + conv_heads) | FROZEN | {ps['frozen_modules_detail'][0] if ps['frozen_modules_detail'] else 'N/A'} |
| FeatureFusionModule | FROZEN | {ps['frozen_modules_detail'][1] if len(ps['frozen_modules_detail']) > 1 else 'N/A'} |
| conv_out (main head) | TRAINABLE | {ps['trainable_modules_detail'][0] if ps['trainable_modules_detail'] else 'N/A'} |
| conv_out16 (aux head) | TRAINABLE | {ps['trainable_modules_detail'][1] if len(ps['trainable_modules_detail']) > 1 else 'N/A'} |
| conv_out32 (aux head) | TRAINABLE | {ps['trainable_modules_detail'][2] if len(ps['trainable_modules_detail']) > 2 else 'N/A'} |

## Parameter Counts

| Metric | Value |
|--------|------:|
| Total parameters | {ps['total_params']:,} |
| Frozen parameters | {ps['frozen_params']:,} |
| Trainable parameters | {ps['trainable_params']:,} |
| Trainable % | {ps['trainable_pct']:.2f}% |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | {config.epochs} |
| Batch size | {config.batch_size} |
| Learning rate | {config.learning_rate} |
| Optimizer | {config.optimizer} |
| Scheduler | {config.scheduler} |
| Class weights | {config.class_weights} |
| Aux16 weight | {config.aux16_weight} |
| Aux32 weight | {config.aux32_weight} |
| Checkpoint criterion | **Full 19-class val mIoU** |
| Target classes | {target_classes_str} |

## Results

| Metric | Value |
|--------|------:|
| Best epoch | {best_epoch} |
| Best val mIoU (19-class) | {best_val_miou:.6f} |
| Best val target mIoU | {best_val_target:.6f} |
| Best val non-target mIoU | {non_target_miou if isinstance(non_target_miou, str) else f'{non_target_miou:.6f}'} |

## Per-Class IoU (Best Checkpoint, Validation)

| Class | IoU |
|-------|----:|
"""
    for name in [CLASS_NAMES[i] for i in range(19)]:
        iou = per_class.get(name)
        iou_str = f"{iou:.6f}" if iou is not None else "N/A"
        md += f"| {name} | {iou_str} |\n"

    md += f"""

## Dataset

| Split | Samples |
|-------|--------:|
| Train | {rep['train_samples']} |
| Val | {rep['val_samples']} |
| Test | {rep['test_samples']} |

## Reproducibility

- Seed: {rep['seed']}
- Python: {rep['python_version']}
- PyTorch: {rep['torch_version']}
- CUDA available: {rep['cuda_available']}
- Device: {rep['device']}
"""
    path.write_text(md, encoding="utf-8")


def _write_per_class_json(config: ExperimentAConfig, metrics: dict[str, Any]) -> None:
    """Write per-class metrics separately for easy comparison."""
    val = metrics.get("val_evaluation", {})
    test = metrics.get("test_evaluation", {})
    output: dict[str, Any] = {
        "experiment": "Experiment_A",
        "best_epoch": metrics["best_epoch"],
        "best_val_mean_iou": metrics["best_val_mean_iou"],
        "best_val_target_mean_iou": metrics["best_val_target_mean_iou"],
        "val_per_class_iou": val.get("per_class_iou", {}),
        "val_per_class_precision": val.get("per_class_precision", {}),
        "val_per_class_recall": val.get("per_class_recall", {}),
        "val_non_target_mean_iou": val.get("non_target_mean_iou"),
        "test_per_class_iou": test.get("per_class_iou", {}),
        "test_per_class_precision": test.get("per_class_precision", {}),
        "test_per_class_recall": test.get("per_class_recall", {}),
        "test_non_target_mean_iou": test.get("non_target_mean_iou"),
    }
    (config.report_dir / "per_class_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment A: head-only BiSeNet fine-tuning"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    config = ExperimentAConfig()
    if args.epochs is not None or args.batch_size is not None or args.learning_rate is not None:
        config = ExperimentAConfig(
            epochs=args.epochs if args.epochs is not None else config.epochs,
            batch_size=args.batch_size if args.batch_size is not None else config.batch_size,
            learning_rate=args.learning_rate if args.learning_rate is not None else config.learning_rate,
        )
    train_experiment_a(config, device_name=args.device)


if __name__ == "__main__":
    main()
