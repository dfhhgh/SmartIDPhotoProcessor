"""
Perceptual hash duplicate index for cross-source duplicate detection.

Builds and maintains an in-memory index of perceptual hashes
for fast duplicate checking during collection.
"""

from __future__ import annotations

from pathlib import Path

import imagehash
from PIL import Image

from dataset_builder.config.settings import Settings


class DuplicateIndex:
    """In-memory index of perceptual hashes for duplicate detection.

    Uses imagehash.phash with settings from Settings to identify
    visually similar images across sources.

    Parameters
    ----------
    settings:
        Application settings controlling hash size and distance threshold.
    """

    def __init__(self, settings: Settings) -> None:
        self._hash_size: int = settings.IMAGEHASH_SIZE
        self._distance_threshold: int = settings.DUPLICATE_DISTANCE_THRESHOLD
        self._supported_extensions: tuple[str, ...] = settings.SUPPORTED_IMAGE_EXTENSIONS
        self._hashes: list[tuple[Path, imagehash.ImageHash]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_image(self, file_path: Path) -> bool:
        """Compute and add the hash for a single image.

        Parameters
        ----------
        file_path:
            Path to the image file.

        Returns
        -------
        bool
            True if the hash was computed and added, False if the
            image could not be read.
        """
        h = self._compute_hash(file_path)
        if h is not None:
            self._hashes.append((file_path, h))
            return True
        return False

    def add_batch(self, file_paths: list[Path]) -> int:
        """Compute and add hashes for multiple images.

        Parameters
        ----------
        file_paths:
            List of image file paths.

        Returns
        -------
        int
            Number of images successfully hashed.
        """
        count = 0
        for path in file_paths:
            if self.add_image(path):
                count += 1
        return count

    def is_duplicate(self, file_path: Path) -> bool:
        """Check if an image is a duplicate of any indexed image.

        Parameters
        ----------
        file_path:
            Path to the image to check.

        Returns
        -------
        bool
            True if the image is within the distance threshold of
            any indexed image.
        """
        h = self._compute_hash(file_path)
        if h is None:
            return False

        for _, existing_hash in self._hashes:
            if h - existing_hash <= self._distance_threshold:
                return True

        return False

    def is_duplicate_of(
        self, file_path: Path, reference_path: Path
    ) -> bool:
        """Check if an image is a duplicate of a specific reference.

        Parameters
        ----------
        file_path:
            Path to the image to check.
        reference_path:
            Path to the reference image.

        Returns
        -------
        bool
            True if the images are within the distance threshold.
        """
        h1 = self._compute_hash(file_path)
        h2 = self._compute_hash(reference_path)

        if h1 is None or h2 is None:
            return False

        return h1 - h2 <= self._distance_threshold

    def find_duplicates(
        self, file_path: Path
    ) -> list[tuple[Path, int]]:
        """Find all indexed images that are duplicates of the given image.

        Parameters
        ----------
        file_path:
            Path to the image to check.

        Returns
        -------
        list[tuple[Path, int]]
            List of (path, distance) pairs for all duplicates.
        """
        h = self._compute_hash(file_path)
        if h is None:
            return []

        duplicates = []
        for path, existing_hash in self._hashes:
            distance = h - existing_hash
            if distance <= self._distance_threshold:
                duplicates.append((path, distance))

        return duplicates

    def compute_hash(self, file_path: Path) -> imagehash.ImageHash | None:
        """Compute the perceptual hash for a single image.

        Parameters
        ----------
        file_path:
            Path to the image file.

        Returns
        -------
        imagehash.ImageHash or None
            The perceptual hash, or None if the image could not be read.
        """
        return self._compute_hash(file_path)

    def clear(self) -> None:
        """Clear all indexed hashes."""
        self._hashes.clear()

    @property
    def size(self) -> int:
        """Return the number of indexed images."""
        return len(self._hashes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_hash(self, file_path: Path) -> imagehash.ImageHash | None:
        """Compute the perceptual hash for a single image.

        Returns None when the image cannot be read.
        """
        if not file_path.exists():
            return None

        if file_path.suffix.lower() not in self._supported_extensions:
            return None

        try:
            with Image.open(file_path) as img:
                return imagehash.phash(img, hash_size=self._hash_size)
        except Exception:
            return None
