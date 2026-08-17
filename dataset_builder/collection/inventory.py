"""
Inventory existing raw images for incremental collection.

Validates existing images in the raw dataset directory and provides
counts of valid/invalid/duplicate images per category.

Category is ALWAYS derived from the parent directory name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2

from dataset_builder.config.settings import Settings


@dataclass
class ImageInventory:
    """Inventory status for a single image."""

    path: Path
    """Absolute path to the image file."""

    category: str
    """Category derived from parent directory."""

    valid: bool
    """Whether the image is valid (readable, correct format)."""

    rejection_reason: str | None = None
    """Reason for rejection if invalid."""

    source: str = ""
    """Source provider extracted from filename (e.g. 'pexels')."""

    source_id: str = ""
    """Source ID extracted from filename (e.g. '10173294')."""


@dataclass
class CategoryInventory:
    """Inventory results for a single category."""

    category: str
    """Category name."""

    total_files: int = 0
    """Total image files in the category directory."""

    valid_images: int = 0
    """Number of valid, readable images."""

    invalid_images: int = 0
    """Number of invalid/unreadable images."""

    duplicate_images: int = 0
    """Number of perceptual duplicates within this category."""

    unique_valid: int = 0
    """Number of unique valid images (valid - duplicates)."""

    images: list[ImageInventory] = field(default_factory=list)
    """Detailed inventory of each image."""


@dataclass
class InventoryResult:
    """Complete inventory results across all categories."""

    categories: dict[str, CategoryInventory] = field(default_factory=dict)
    """Per-category inventory results."""

    uncategorized_files: list[Path] = field(default_factory=list)
    """Files directly under raw/ not in any category directory."""

    unknown_dirs: list[str] = field(default_factory=list)
    """Directories under raw/ that are not known categories."""

    total_files: int = 0
    """Total files across all known categories."""

    total_valid: int = 0
    """Total valid images across all known categories."""

    total_invalid: int = 0
    """Total invalid images across all known categories."""

    total_duplicates: int = 0
    """Total duplicate images across all known categories."""

    total_unique_valid: int = 0
    """Total unique valid images across all known categories."""


class Inventory:
    """Inventory existing raw images for incremental collection.

    Validates existing images in the raw dataset directory and provides
    counts of valid/invalid/duplicate images per category.

    Category is ALWAYS derived from the parent directory name.

    Parameters
    ----------
    settings:
        Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._known_categories: set[str] = set(settings.DATASET_CATEGORIES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inventory_category(
        self,
        category: str,
        raw_dir: Path | None = None,
    ) -> CategoryInventory:
        """Inventory images in a single category directory.

        Parameters
        ----------
        category:
            Category name (must match a subdirectory under raw/).
        raw_dir:
            Root raw directory. If None, uses Settings.RAW_IMAGES_DIR.

        Returns
        -------
        CategoryInventory
            Inventory results for this category.
        """
        base_dir = raw_dir or self._settings.RAW_IMAGES_DIR
        cat_dir = base_dir / category

        result = CategoryInventory(category=category)

        if not cat_dir.exists():
            return result

        # Find all image files directly in this directory
        files = [
            p for p in cat_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in self._settings.SUPPORTED_IMAGE_EXTENSIONS
        ]

        result.total_files = len(files)

        for file_path in files:
            inv = self._inventory_image(file_path, category)
            result.images.append(inv)

            if inv.valid:
                result.valid_images += 1
            else:
                result.invalid_images += 1

        # Perceptual duplicate detection within this category
        self._detect_duplicates(result)

        result.unique_valid = result.valid_images - result.duplicate_images

        return result

    def inventory_all(
        self,
        raw_dir: Path | None = None,
    ) -> InventoryResult:
        """Inventory all images across all categories.

        Discovers categories from the directory structure under raw/.
        Reports uncategorized files and unknown directories.

        Parameters
        ----------
        raw_dir:
            Root raw directory. If None, uses Settings.RAW_IMAGES_DIR.

        Returns
        -------
        InventoryResult
            Complete inventory results.
        """
        base_dir = raw_dir or self._settings.RAW_IMAGES_DIR
        result = InventoryResult()

        if not base_dir.exists():
            return result

        # Discover all subdirectories
        known_categories_inv = set()
        for item in sorted(base_dir.iterdir()):
            if not item.is_dir():
                # File directly under raw/ — uncategorized
                if (
                    item.is_file()
                    and item.suffix.lower() in self._settings.SUPPORTED_IMAGE_EXTENSIONS
                ):
                    result.uncategorized_files.append(item)
                continue

            # It's a directory — check if it's a known category
            if item.name in self._known_categories:
                cat_inv = self.inventory_category(item.name, raw_dir)
                result.categories[item.name] = cat_inv
                known_categories_inv.add(item.name)
            else:
                # Unknown directory — report it
                result.unknown_dirs.append(item.name)
                # Still inventory it as an unknown category
                cat_inv = self.inventory_category(item.name, raw_dir)
                result.categories[item.name] = cat_inv

        # Also inventory known categories that don't have a directory yet
        for cat_name in self._known_categories:
            if cat_name not in known_categories_inv:
                cat_inv = self.inventory_category(cat_name, raw_dir)
                result.categories[cat_name] = cat_inv

        # Aggregate totals
        for cat_inv in result.categories.values():
            result.total_files += cat_inv.total_files
            result.total_valid += cat_inv.valid_images
            result.total_invalid += cat_inv.invalid_images
            result.total_duplicates += cat_inv.duplicate_images
            result.total_unique_valid += cat_inv.unique_valid

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inventory_image(
        self, file_path: Path, category: str
    ) -> ImageInventory:
        """Inventory a single image file.

        Category is always the directory-derived category passed in.
        """
        inv = ImageInventory(path=file_path, category=category, valid=False)

        # Extract source and ID from filename
        # Expected format: source_sourceid.ext
        stem = file_path.stem
        parts = stem.split("_", 1)
        if len(parts) == 2:
            inv.source = parts[0]
            inv.source_id = parts[1]

        # Check if file is readable and meets minimum dimensions
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                inv.rejection_reason = "unreadable"
                return inv

            h, w = img.shape[:2]
            if (
                w < self._settings.MIN_IMAGE_WIDTH
                or h < self._settings.MIN_IMAGE_HEIGHT
            ):
                inv.rejection_reason = "too_small"
                return inv

            inv.valid = True

        except Exception as e:
            inv.rejection_reason = f"load_error: {e}"
            return inv

        return inv

    def _detect_duplicates(self, cat_inv: CategoryInventory) -> None:
        """Detect perceptual duplicates within a category inventory.

        Marks duplicate images by incrementing ``duplicate_images``
        and does NOT remove them from ``images``.
        """
        import imagehash
        from PIL import Image as PILImage

        valid_images = [img for img in cat_inv.images if img.valid]
        if len(valid_images) <= 1:
            return

        seen_hashes: list[tuple[Path, object]] = []

        for img_inv in valid_images:
            h = self._compute_phash(img_inv.path)
            if h is None:
                continue

            is_dup = False
            for _, existing_h in seen_hashes:
                if h - existing_h <= self._settings.DUPLICATE_DISTANCE_THRESHOLD:
                    is_dup = True
                    break

            if is_dup:
                cat_inv.duplicate_images += 1
            else:
                seen_hashes.append((img_inv.path, h))

    def _compute_phash(self, file_path: Path) -> object | None:
        """Compute perceptual hash for an image file."""
        import imagehash
        from PIL import Image as PILImage

        if not file_path.exists():
            return None
        if file_path.suffix.lower() not in self._settings.SUPPORTED_IMAGE_EXTENSIONS:
            return None
        try:
            with PILImage.open(file_path) as img:
                return imagehash.phash(
                    img, hash_size=self._settings.IMAGEHASH_SIZE
                )
        except Exception:
            return None
