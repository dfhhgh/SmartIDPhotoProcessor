from dataclasses import dataclass
from pathlib import Path
import os

from smartidphotoprocessor.config.parser_mode import ParserMode


def _resolve_base_dir() -> Path:
    """Resolve the project base directory.

    Priority:
    1. SMARTIDPHOTO_BASE_DIR environment variable (production deployment)
    2. Repository root (development: 3 levels up from src/smartidphotoprocessor/config/)
    """
    env = os.environ.get("SMARTIDPHOTO_BASE_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Settings:
    # Project
    PROJECT_NAME: str = "Student Photo Validator"

    # Base directory — resolved via env var or repository root
    BASE_DIR: Path = _resolve_base_dir()

    # InsightFace
    MODEL_NAME: str = "buffalo_l"
    MODEL_ROOT: Path = BASE_DIR / "ai_models"

    # Hardware
    USE_GPU: bool = True
    GPU_ID: int = 0

    # Face parser
    # FUSED is the production default: ONNX backbone + ONNX aux head + Python fusion.
    PARSER_MODE: ParserMode = ParserMode.FUSED
    AUX_EYE_BROW_CHECKPOINT_PATH: Path = (
        BASE_DIR
        / "dataset_builder"
        / "dataset"
        / "parser_finetune_current"
        / "training_aux_eye_brow_phase1"
        / "checkpoints"
        / "best.pt"
    )
    EYE_BROW_FUSION_STRATEGY: int = 1
    EYE_BROW_FUSION_THRESHOLD: float = 0.0
    EYE_BROW_FUSION_MIN_COMPONENT_SIZE: int = 10

    # Detection
    DETECTION_SIZE: tuple[int, int] = (640, 640)
    DETECTION_THRESHOLD: float = 0.5

    # Output directories
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    DEBUG_OUTPUT: Path = OUTPUT_DIR / "debug"
    ALIGNED_OUTPUT: Path = OUTPUT_DIR / "aligned"
    CROPPED_OUTPUT: Path = OUTPUT_DIR / "cropped"

    # Reverse Search
    REVERSE_SEARCH_ENABLED: bool = False
    REVERSE_SEARCH_INDEX_PATH: Path | None = None
    REVERSE_SEARCH_METADATA_PATH: Path | None = None
