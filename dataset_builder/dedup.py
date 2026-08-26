"""
Persistent dual-index perceptual hash deduplication system.

Maintains TWO logically separate hash indices:

  RAW index   — hashes of unprocessed downloaded images
  ALIGNED index — hashes of face-aligned images

Design rules:
  - RAW pHash is NEVER compared against ALIGNED pHash
  - ALIGNED pHash is NEVER compared against RAW pHash
  - Indices are persisted to disk as JSON and survive process restarts
  - The index file path is the single source of truth for what has been seen

Usage::

    dedup = DedupIndex(index_dir=Path("..."))
    # Load existing images from disk into the index
    dedup.load_aligned_from_directory(images_dir)

    # During collection
    if not dedup.is_raw_duplicate(raw_download_path):
        # ... align face ...
        if not dedup.is_aligned_duplicate(aligned_path):
            dedup.register_accepted(sample_id, raw_path, aligned_path)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import imagehash
from PIL import Image

log = logging.getLogger(__name__)

HASH_SIZE = 16
THRESHOLD = 5


class DedupIndex:
    """Persistent dual-index perceptual hash deduplication.

    Parameters
    ----------
    index_dir:
        Directory where ``raw_phash_index.json`` and
        ``aligned_phash_index.json`` are stored.
    hash_size:
        pHash hash size (default 16, producing 256-bit hashes).
    threshold:
        Maximum Hamming distance to consider two hashes as duplicates.
    """

    def __init__(
        self,
        index_dir: Path,
        hash_size: int = HASH_SIZE,
        threshold: int = THRESHOLD,
    ) -> None:
        self.index_dir = index_dir
        self.hash_size = hash_size
        self.threshold = threshold
        self.raw_index_path = index_dir / "raw_phash_index.json"
        self.aligned_index_path = index_dir / "aligned_phash_index.json"
        self.raw_hashes: dict[str, str] = {}
        self.aligned_hashes: dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load indices from disk if they exist."""
        if self.raw_index_path.exists():
            try:
                self.raw_hashes = json.loads(
                    self.raw_index_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                log.warning("Corrupt raw index, starting fresh")
                self.raw_hashes = {}
        if self.aligned_index_path.exists():
            try:
                self.aligned_hashes = json.loads(
                    self.aligned_index_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError):
                log.warning("Corrupt aligned index, starting fresh")
                self.aligned_hashes = {}

    def save(self) -> None:
        """Persist both indices to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.raw_index_path.write_text(
            json.dumps(self.raw_hashes, indent=2), encoding="utf-8"
        )
        self.aligned_index_path.write_text(
            json.dumps(self.aligned_hashes, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # Hash computation
    # ------------------------------------------------------------------

    def compute_phash(self, img_path: Path) -> imagehash.ImageHash | None:
        """Compute pHash for an image file.

        Returns None if the image cannot be read.
        """
        if not img_path.exists():
            return None
        try:
            with Image.open(img_path) as img:
                return imagehash.phash(img, hash_size=self.hash_size)
        except Exception:
            return None

    @staticmethod
    def _hex_to_hash(h: str) -> imagehash.ImageHash:
        return imagehash.hex_to_hash(h)

    # ------------------------------------------------------------------
    # RAW index operations
    # ------------------------------------------------------------------

    def is_raw_duplicate(self, img_path: Path) -> bool:
        """Check if a RAW image duplicates any image in the RAW index.

        Compares ONLY against other RAW hashes.
        """
        h = self.compute_phash(img_path)
        if h is None:
            return True  # cannot read → treat as duplicate to be safe
        for existing_hex in self.raw_hashes.values():
            if h - self._hex_to_hash(existing_hex) <= self.threshold:
                return True
        return False

    def add_raw(self, sample_id: str, img_path: Path) -> bool:
        """Compute pHash and add to the RAW index. Persists to disk."""
        h = self.compute_phash(img_path)
        if h is None:
            return False
        self.raw_hashes[sample_id] = str(h)
        self.save()
        return True

    # ------------------------------------------------------------------
    # ALIGNED index operations
    # ------------------------------------------------------------------

    def is_aligned_duplicate(self, img_path: Path) -> bool:
        """Check if an ALIGNED image duplicates any image in the ALIGNED index.

        Compares ONLY against other ALIGNED hashes.
        """
        h = self.compute_phash(img_path)
        if h is None:
            return True
        for existing_hex in self.aligned_hashes.values():
            if h - self._hex_to_hash(existing_hex) <= self.threshold:
                return True
        return False

    def add_aligned(self, sample_id: str, img_path: Path) -> bool:
        """Compute pHash and add to the ALIGNED index. Persists to disk."""
        h = self.compute_phash(img_path)
        if h is None:
            return False
        self.aligned_hashes[sample_id] = str(h)
        self.save()
        return True

    # ------------------------------------------------------------------
    # Bulk loaders
    # ------------------------------------------------------------------

    def load_raw_from_directory(self, directory: Path, glob: str = "*.png") -> int:
        """Load all matching images into the RAW index."""
        count = 0
        for f in sorted(directory.glob(glob)):
            if self.add_raw(f.stem, f):
                count += 1
        return count

    def load_aligned_from_directory(self, directory: Path, glob: str = "*.png") -> int:
        """Load all matching images into the ALIGNED index."""
        count = 0
        for f in sorted(directory.glob(glob)):
            if self.add_aligned(f.stem, f):
                count += 1
        return count

    # ------------------------------------------------------------------
    # Convenience: register an accepted image in both indices
    # ------------------------------------------------------------------

    def register_accepted(
        self,
        sample_id: str,
        raw_path: Path,
        aligned_path: Path,
    ) -> bool:
        """Add an accepted image to BOTH the RAW and ALIGNED indices.

        Call this after an image has passed both dedup gates and been
        written to disk.  The RAW index records the hash of the original
        downloaded file; the ALIGNED index records the hash of the
        face-aligned output.
        """
        raw_ok = self.add_raw(sample_id, raw_path)
        aligned_ok = self.add_aligned(sample_id, aligned_path)
        return raw_ok and aligned_ok

    # ------------------------------------------------------------------
    # Group finding (for quarantine / analysis)
    # ------------------------------------------------------------------

    def find_aligned_duplicate_groups(self) -> dict[str, list[str]]:
        """Find all duplicate groups in the ALIGNED index.

        Returns a dict mapping root_id -> [member_ids].
        Only groups with >= 2 members are included.
        """
        sample_ids = list(self.aligned_hashes.keys())
        n = len(sample_ids)
        parent = list(range(n))
        rank = [0] * n

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

        hash_list = [
            self._hex_to_hash(self.aligned_hashes[sid]) for sid in sample_ids
        ]
        for i in range(n):
            for j in range(i + 1, n):
                if hash_list[i] - hash_list[j] <= self.threshold:
                    union(i, j)

        groups: dict[str, list[str]] = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(sample_ids[i])

        # Filter to only duplicate groups (>= 2 members)
        return {k: v for k, v in groups.items() if len(v) >= 2}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raw_count(self) -> int:
        return len(self.raw_hashes)

    @property
    def aligned_count(self) -> int:
        return len(self.aligned_hashes)
