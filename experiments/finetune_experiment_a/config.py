"""Configuration for Experiment A: head-only BiSeNet fine-tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Import shared constants to avoid duplication.
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_builder.dataset.parser_finetune.training.config import (  # noqa: E402
    CLASS_NAMES,
    TARGET_CLASS_IDS,
    AugmentationConfig,
)

NON_TARGET_CLASS_IDS: tuple[int, ...] = tuple(
    i for i in range(19) if i not in set(TARGET_CLASS_IDS)
)


@dataclass(frozen=True)
class ExperimentAConfig:
    """Reproducible settings for Experiment A (head-only fine-tuning)."""

    project_root: Path = _PROJECT_ROOT
    expanded_dir: Path = (
        project_root / "dataset_builder" / "dataset" / "parser_finetune_expanded"
    )
    manifest_path: Path = expanded_dir / "reports" / "expanded_manifest.json"
    onnx_model_path: Path = (
        project_root / "ai_models" / "bisenet" / "bisenet_resnet18.onnx"
    )
    output_dir: Path = project_root / "experiments" / "finetune_experiment_a"
    checkpoint_dir: Path = output_dir / "checkpoints"
    report_dir: Path = output_dir / "reports"

    n_classes: int = 19
    image_size: int = 512
    frozen_mask_size: int = 112
    seed: int = 42
    batch_size: int = 4
    epochs: int = 20
    learning_rate: float = 1.0e-5
    weight_decay: float = 1.0e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    ignore_index: int | None = None
    aux16_weight: float = 0.4
    aux32_weight: float = 0.4
    num_workers: int = 0
    augmentation: dict[str, Any] | None = field(
        default_factory=lambda: {
            "enabled": True,
            "horizontal_flip_probability": 0.5,
            "max_rotation_degrees": 8.0,
            "max_translation_fraction": 0.04,
            "min_scale": 0.96,
            "max_scale": 1.04,
            "brightness_delta": 0.08,
            "contrast_delta": 0.08,
        }
    )
    class_weights: dict[int, float] | None = field(
        default_factory=lambda: {4: 2.0, 5: 2.0, 6: 1.0}
    )

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
        return data
