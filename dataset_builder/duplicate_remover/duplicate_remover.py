"""
Perceptual duplicate image detection and removal.

Scans directories for visually similar images using perceptual
hashing and provides utilities for grouping and moving duplicates
while preserving the best representative of each group.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import imagehash
from PIL import Image

from dataset_builder.config.settings import Settings


# ------------------------------------------------------------------
# Immutable dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateImage:
    """A single image involved in duplicate detection."""

    path: Path
    """Absolute path to the image file."""

    hash: imagehash.ImageHash
    """Perceptual hash of the image."""

    width: int
    """Image width in pixels."""

    height: int
    """Image height in pixels."""

    file_size: int
    """File size in bytes."""


@dataclass(frozen=True)
class DuplicateGroup:
    """A set of perceptually identical images.

    The ``original`` is the best representative selected by the
    deduplication algorithm.  All ``duplicates`` are candidates for
    removal.
    """

    original: DuplicateImage
    """The chosen representative image."""

    duplicates: tuple[DuplicateImage, ...]
    """All other images in this group."""

    distance_scores: dict[str, int]
    """Mapping of duplicate path (str) to Hamming distance from original."""


@dataclass(frozen=True)
class DuplicateStatistics:
    """Summary statistics produced by duplicate detection."""

    total_images: int
    """Total number of images scanned."""

    unique_images: int
    """Number of unique (non-duplicate) images."""

    duplicate_images: int
    """Number of duplicate images (candidates for removal)."""

    duplicate_groups: int
    """Number of distinct duplicate groups."""

    duplicate_ratio: float
    """Fraction of images that are duplicates (0.0 - 1.0)."""


# ------------------------------------------------------------------
# Duplicate remover
# ------------------------------------------------------------------


class DuplicateRemover:
    """Detect and manage perceptually duplicate images.

    Uses :func:`imagehash.phash` with settings from
    :class:`Settings` to identify visually similar images.
    No files are modified or deleted until
    :meth:`move_duplicates` is called explicitly.

    Parameters
    ----------
    settings:
        Application settings controlling hash size and distance
        threshold.

    Examples
    --------
    ::

        remover = DuplicateRemover(settings)
        remover.scan(Path("dataset/raw"))
        stats = remover.statistics()
        remover.move_duplicates(Path("dataset/duplicates_removed"))
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._hash_size: int = settings.IMAGEHASH_SIZE
        self._distance_threshold: int = settings.DUPLICATE_DISTANCE_THRESHOLD
        self._supported_extensions: tuple[str, ...] = settings.SUPPORTED_IMAGE_EXTENSIONS
        self._groups: list[DuplicateGroup] = []
        self._scanned: list[DuplicateImage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def duplicates(self) -> list[DuplicateGroup]:
        """Return all detected duplicate groups.

        Returns
        -------
        list[DuplicateGroup]
            A copy of the internal duplicate group list.
        """
        return list(self._groups)

    def scan(self, directory: Path) -> None:
        """Scan a directory for images and detect duplicates.

        Computes perceptual hashes for every supported image file
        and groups those whose Hamming distance is within the
        configured threshold.  No files are modified.

        Parameters
        ----------
        directory:
            Root directory to scan recursively.

        Raises
        ------
        FileNotFoundError
            When *directory* does not exist.
        """
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        images = self._scan_directory(directory)
        self._scanned = images

        self._groups = self._group_duplicates(images)

    def move_duplicates(self, destination: Path) -> int:
        """Move all duplicate images except the chosen originals.

        Each duplicate file is moved into *destination*.  The
        original image remains in its current location.

        Parameters
        ----------
        destination:
            Target directory for moved duplicates.  Created
            automatically if it does not exist.

        Returns
        -------
        int
            Number of files moved.
        """
        destination.mkdir(parents=True, exist_ok=True)

        moved = 0
        for group in self._groups:
            for dup in group.duplicates:
                target = destination / dup.path.name
                target = self._resolve_collision(target)
                shutil.move(str(dup.path), str(target))
                moved += 1

        return moved

    def statistics(self) -> DuplicateStatistics:
        """Return summary statistics for the last scan.

        Returns
        -------
        DuplicateStatistics
            Snapshot of scan results.  If no scan has been performed
            all counters are zero.
        """
        total = len(self._scanned)
        dup_count = sum(len(g.duplicates) for g in self._groups)
        unique = total - dup_count

        ratio = dup_count / total if total > 0 else 0.0

        return DuplicateStatistics(
            total_images=total,
            unique_images=unique,
            duplicate_images=dup_count,
            duplicate_groups=len(self._groups),
            duplicate_ratio=ratio,
        )

    def clear(self) -> None:
        """Clear all internal caches from the last scan."""
        self._scanned.clear()
        self._groups.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scan_directory(self, directory: Path) -> list[DuplicateImage]:
        """Recursively scan *directory* and compute hashes."""
        images: list[DuplicateImage] = []

        for file_path in sorted(directory.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in self._supported_extensions:
                continue

            image = self._compute_hash(file_path)
            if image is not None:
                images.append(image)

        return images

    def _compute_hash(self, file_path: Path) -> DuplicateImage | None:
        """Compute the perceptual hash for a single image.

        Returns ``None`` when the image cannot be read.
        """
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                file_size = file_path.stat().st_size
                h = imagehash.phash(img, hash_size=self._hash_size)

                return DuplicateImage(
                    path=file_path,
                    hash=h,
                    width=width,
                    height=height,
                    file_size=file_size,
                )
        except Exception:
            return None

    def _are_duplicates(self, a: DuplicateImage, b: DuplicateImage) -> bool:
        """Check whether two images are within the distance threshold."""
        distance = a.hash - b.hash
        return distance <= self._distance_threshold

    def _choose_original(self, group: list[DuplicateImage]) -> DuplicateImage:
        """Select the best representative from a group of duplicates.

        Priority order:

        1. Highest resolution (width * height)
        2. Largest file size
        3. Alphabetical filename (ascending)
        """
        return max(
            group,
            key=lambda img: (
                img.width * img.height,
                img.file_size,
                -ord(img.path.name[0]) if img.path.name else 0,
            ),
        )

    def _group_duplicates(
        self, images: list[DuplicateImage]
    ) -> list[DuplicateGroup]:
        """Cluster images into duplicate groups using union-find."""
        parent: dict[int, int] = {i: i for i in range(len(images))}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(len(images)):
            for j in range(i + 1, len(images)):
                if self._are_duplicates(images[i], images[j]):
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(len(images)):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        groups: list[DuplicateGroup] = []
        for indices in clusters.values():
            if len(indices) < 2:
                continue

            group_images = [images[i] for i in indices]
            original = self._choose_original(group_images)
            others = tuple(img for img in group_images if img is not original)

            distance_scores: dict[str, int] = {}
            for dup in others:
                distance_scores[str(dup.path)] = original.hash - dup.hash

            groups.append(
                DuplicateGroup(
                    original=original,
                    duplicates=others,
                    distance_scores=distance_scores,
                )
            )

        return groups

    @staticmethod
    def _resolve_collision(target: Path) -> Path:
        """Return a non-existing path by appending a counter suffix."""
        if not target.exists():
            return target

        stem = target.stem
        suffix = target.suffix
        parent = target.parent

        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
