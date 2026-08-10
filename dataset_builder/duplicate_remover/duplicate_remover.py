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
from typing import Callable

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

    group_id: int
    """Deterministic identifier for this group (scan-scoped)."""

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


@dataclass(frozen=True)
class MoveStatistics:
    """Result of a move_duplicates operation."""

    moved: int
    """Number of files successfully moved."""

    failed: int
    """Number of files that failed to move."""

    skipped: int
    """Number of files skipped (e.g. already at destination)."""

    collisions_resolved: int
    """Number of filename collisions resolved with suffix."""


# ------------------------------------------------------------------
# Type alias for progress callback
# ------------------------------------------------------------------

ProgressCallback = Callable[[int, int], None]
"""Signature: ``(processed: int, total: int) -> None``."""


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

    def scan(
        self,
        directory: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Scan a directory for images and detect duplicates.

        Any previous scan results are cleared before scanning.
        Computes perceptual hashes for every supported image file
        and groups those whose Hamming distance is within the
        configured threshold.  No files are modified.

        Parameters
        ----------
        directory:
            Root directory to scan recursively.
        progress_callback:
            Optional function called with ``(processed, total)``
            after each image is hashed.

        Raises
        ------
        FileNotFoundError
            When *directory* does not exist.
        """
        self.clear()

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        images = self._scan_directory(directory, progress_callback)
        self._scanned = images
        self._groups = self._group_duplicates(images)

    def move_duplicates(self, destination: Path) -> MoveStatistics:
        """Move all duplicate images except the chosen originals.

        Each duplicate file is moved into *destination*.  The
        original image remains in its current location.  Failures
        are handled individually; remaining files are still moved.

        Parameters
        ----------
        destination:
            Target directory for moved duplicates.  Created
            automatically if it does not exist.

        Returns
        -------
        MoveStatistics
            Counts of moved, failed, skipped, and collision-resolved
            files.
        """
        destination.mkdir(parents=True, exist_ok=True)

        moved = 0
        failed = 0
        skipped = 0
        collisions = 0

        for group in self._groups:
            for dup in group.duplicates:
                target = destination / dup.path.name
                target, was_resolved = self._resolve_collision(target)
                if was_resolved:
                    collisions += 1

                if not dup.path.exists():
                    skipped += 1
                    continue

                try:
                    shutil.move(str(dup.path), str(target))
                    moved += 1
                except Exception:
                    failed += 1

        return MoveStatistics(
            moved=moved,
            failed=failed,
            skipped=skipped,
            collisions_resolved=collisions,
        )

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

    def _scan_directory(
        self,
        directory: Path,
        progress_callback: ProgressCallback | None,
    ) -> list[DuplicateImage]:
        """Recursively scan *directory* and compute hashes."""
        files = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in self._supported_extensions
        )

        images: list[DuplicateImage] = []
        total = len(files)

        for idx, file_path in enumerate(files):
            image = self._compute_hash(file_path)
            if image is not None:
                images.append(image)

            if progress_callback is not None:
                progress_callback(idx + 1, total)

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
        3. Alphabetically smallest filename (case-insensitive)
        """
        return max(
            group,
            key=lambda img: (
                img.width * img.height,
                img.file_size,
                tuple(-c for c in img.path.name.lower().encode()),
            ),
        )

    def _group_duplicates(
        self, images: list[DuplicateImage]
    ) -> list[DuplicateGroup]:
        """Cluster images into duplicate groups using union-find.

        Complexity: O(N^2) pairwise comparisons where N is the number
        of images.  For datasets exceeding ~50k images, consider
        replacing this stage with a BK-Tree or locality-sensitive
        hashing (LSH) index to achieve sub-quadratic performance.
        """
        # TODO: For very large datasets, replace O(N^2) pairwise
        # comparison with BK-Tree or LSH-based approximate nearest
        # neighbor search.

        n = len(images)
        parent: list[int] = list(range(n))
        rank: list[int] = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        for i in range(n):
            for j in range(i + 1, n):
                if self._are_duplicates(images[i], images[j]):
                    union(i, j)

        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        groups: list[DuplicateGroup] = []
        for group_idx, indices in enumerate(clusters.values()):
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
                    group_id=group_idx,
                    original=original,
                    duplicates=others,
                    distance_scores=distance_scores,
                )
            )

        return groups

    @staticmethod
    def _resolve_collision(target: Path) -> tuple[Path, bool]:
        """Return a non-existing path, resolving collisions with suffix.

        Returns the resolved path and a boolean indicating whether
        a collision was resolved.
        """
        if not target.exists():
            return target, False

        stem = target.stem
        suffix = target.suffix
        parent = target.parent

        counter = 1
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate, True
            counter += 1
