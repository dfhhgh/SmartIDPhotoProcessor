"""
Central configuration for the Dataset Builder project.

All settings are defined in a single frozen dataclass to provide
immutability, type safety, and a single source of truth.
No business logic, API calls, or side effects belong here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Immutable configuration for the Dataset Builder.

    Every attribute is documented and has a sensible default.
    API keys are read from environment variables and must never
    be hardcoded.
    """

    # ------------------------------------------------------------------
    # 1. Project
    # ------------------------------------------------------------------

    PROJECT_NAME: str = "Dataset Builder"
    """Human-readable project name used in reports and metadata."""

    PROJECT_VERSION: str = "0.1.0"
    """Semantic version of this dataset builder."""

    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    """Absolute path to the dataset_builder/ root directory."""

    DATASET_DIR: Path = PROJECT_ROOT / "dataset"
    """Root directory for all dataset artifacts."""

    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    """Root directory for generated reports."""

    CONFIG_DIR: Path = PROJECT_ROOT / "config"
    """Directory containing configuration files."""

    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    """Directory for log files."""

    CACHE_DIR: Path = PROJECT_ROOT / "cache"
    """Directory for temporary cache data."""

    TEMP_DIR: Path = PROJECT_ROOT / "temp"
    """Directory for ephemeral working files."""

    DOWNLOADS_DIR: Path = PROJECT_ROOT / "downloads"
    """Directory for raw downloaded archives or files."""

    QUERIES_DIR: Path = PROJECT_ROOT / "queries"
    """Directory containing query definition files."""

    # ------------------------------------------------------------------
    # 2. Dataset folders
    # ------------------------------------------------------------------

    RAW_IMAGES_DIR: Path = DATASET_DIR / "raw"
    """Directory for unprocessed downloaded images."""

    FILTERED_IMAGES_DIR: Path = DATASET_DIR / "filtered"
    """Directory for images that pass quality filters."""

    SELECTED_IMAGES_DIR: Path = DATASET_DIR / "selected"
    """Directory for the final curated dataset."""

    DUPLICATES_DIR: Path = DATASET_DIR / "duplicates_removed"
    """Directory for images identified as duplicates."""

    METADATA_DIR: Path = DATASET_DIR / "metadata"
    """Directory for JSON/CSV metadata files."""

    DATASET_CATEGORIES: tuple[str, ...] = (
        "normal",
        "hijab",
        "eyeglasses",
        "sunglasses",
        "mask",
        "cap",
        "beard",
        "helmet",
        "scarf",
        "hair_occlusion",
    )
    """Canonical dataset categories.  Each becomes a subdirectory under
    raw/, filtered/, and selected/."""

    # ------------------------------------------------------------------
    # 3. API Keys
    # ------------------------------------------------------------------

    PEXELS_API_KEY: str = field(
        default_factory=lambda: os.environ.get("PEXELS_API_KEY", "")
    )
    """Pexels API key.  Read from the PEXELS_API_KEY environment variable."""

    PIXABAY_API_KEY: str = field(
        default_factory=lambda: os.environ.get("PIXABAY_API_KEY", "")
    )
    """Pixabay API key.  Read from the PIXABAY_API_KEY environment variable."""

    ENABLED_SOURCES: tuple[str, ...] = ("pexels", "pixabay", "openverse", "wikimedia_commons")
    """Active image sources.  Toggle sources here without code changes."""

    # ------------------------------------------------------------------
    # 4. Downloader
    # ------------------------------------------------------------------

    DOWNLOAD_TIMEOUT_SECONDS: int = 30
    """Maximum seconds to wait for a single HTTP response."""

    MAX_RETRIES: int = 3
    """Number of retry attempts for transient HTTP failures."""

    BACKOFF_FACTOR: float = 1.5
    """Exponential backoff multiplier between retries."""

    REQUEST_DELAY_SECONDS: float = 0.5
    """Minimum pause between consecutive API requests to respect rate limits."""

    MAX_IMAGES_PER_QUERY: int = 200
    """Upper bound on images fetched for a single search query."""

    MAX_PAGES_PER_QUERY: int = 10
    """Maximum number of paginated result pages to traverse per query."""

    SUPPORTED_IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
    """File extensions treated as valid images during download and filtering."""

    MAX_CONCURRENT_DOWNLOADS: int = 4
    """Maximum number of simultaneous download threads."""

    DEFAULT_USER_AGENT: str = "DatasetBuilder/1.0 (+Research)"
    """User-Agent header sent with every HTTP request."""

    DEFAULT_HEADERS: dict[str, str] = field(
        default_factory=lambda: {"User-Agent": "DatasetBuilder/1.0 (+Research)"}
    )
    """Default HTTP headers applied to all requests."""

    # ------------------------------------------------------------------
    # 5. Image Requirements
    # ------------------------------------------------------------------

    MIN_IMAGE_WIDTH: int = 640
    """Minimum acceptable image width in pixels."""

    MIN_IMAGE_HEIGHT: int = 480
    """Minimum acceptable image height in pixels."""

    MAX_IMAGE_WIDTH: int = 8192
    """Maximum acceptable image width in pixels."""

    MAX_IMAGE_HEIGHT: int = 8192
    """Maximum acceptable image height in pixels."""

    ALLOWED_IMAGE_FORMATS: tuple[str, ...] = ("JPEG", "PNG", "WEBP")
    """Pillow format names accepted during validation."""

    MIN_MEGAPIXELS: float = 1.0
    """Minimum image megapixel count (width * height / 1_000_000)."""

    # ------------------------------------------------------------------
    # 6. Face Filtering
    # ------------------------------------------------------------------

    MIN_FACE_AREA_RATIO: float = 0.02
    """Minimum ratio of face area to total image area (0.0-1.0)."""

    MAX_FACES_PER_IMAGE: int = 1
    """Maximum number of detected faces allowed per image."""

    ALLOW_PROFILE_FACES: bool = True
    """Whether to accept profile (non-frontal) face poses."""

    MAX_PROFILE_YAW_DEGREES: float = 30.0
    """Maximum yaw angle (in degrees) before a face is classified as profile."""

    # ------------------------------------------------------------------
    # 7. Duplicate Detection
    # ------------------------------------------------------------------

    IMAGEHASH_SIZE: int = 16
    """Hash width/height in bits for perceptual hashing."""

    DUPLICATE_DISTANCE_THRESHOLD: int = 5
    """Maximum Hamming distance between hashes to consider images duplicates."""

    # ------------------------------------------------------------------
    # 8. Quality Filtering
    # ------------------------------------------------------------------

    MIN_BLUR_SCORE: float = 100.0
    """Minimum Laplacian variance indicating a sharp image."""

    MIN_BRIGHTNESS: float = 40.0
    """Minimum average pixel intensity (0-255)."""

    MAX_BRIGHTNESS: float = 220.0
    """Maximum average pixel intensity (0-255)."""

    MIN_CONTRAST: float = 35.0
    """Minimum standard deviation of pixel intensities."""

    # ------------------------------------------------------------------
    # 9. Logging
    # ------------------------------------------------------------------

    LOG_LEVEL: str = "INFO"
    """Python logging level name."""

    LOG_TO_FILE: bool = True
    """Whether to write log output to a file."""

    LOG_FILENAME: str = "dataset_builder.log"
    """Filename for the log file, relative to PROJECT_ROOT."""

    # ------------------------------------------------------------------
    # 10. Metadata
    # ------------------------------------------------------------------

    SAVE_JSON_METADATA: bool = True
    """Persist per-image metadata as JSON files."""

    SAVE_CSV_INDEX: bool = True
    """Generate a CSV index of all downloaded images."""

    SAVE_SOURCE_INFORMATION: bool = True
    """Record the originating API source in metadata."""

    SAVE_LICENSE_INFORMATION: bool = True
    """Record the image license type in metadata."""

    SAVE_AUTHOR_INFORMATION: bool = True
    """Record the photographer/author name in metadata."""

    SAVE_SOURCE_URL: bool = True
    """Record the original URL of the image in metadata."""

    SAVE_DOWNLOAD_TIMESTAMP: bool = True
    """Record the UTC timestamp of when the image was downloaded."""
