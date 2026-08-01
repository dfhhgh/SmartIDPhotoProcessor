from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # Project
    PROJECT_NAME: str = "Student Photo Validator"

    # Base directory
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # InsightFace
    MODEL_NAME: str = "buffalo_l"
    MODEL_ROOT: Path = BASE_DIR / "ai_models"

    # Hardware
    USE_GPU: bool = True
    GPU_ID: int = 0

    # Detection
    DETECTION_SIZE: tuple[int, int] = (640, 640)
    DETECTION_THRESHOLD: float = 0.5

    # Output directories
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    DEBUG_OUTPUT: Path = OUTPUT_DIR / "debug"
    ALIGNED_OUTPUT: Path = OUTPUT_DIR / "aligned"
    CROPPED_OUTPUT: Path = OUTPUT_DIR / "cropped"