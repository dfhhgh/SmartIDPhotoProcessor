"""Tests for the fixed dual-index dedup system.

Covers all 7 required test scenarios:
  Test 1 — Same RAW image
  Test 2 — Same image after restart (persistence)
  Test 3 — RAW vs ALIGNED representation separation
  Test 4 — Same ALIGNED image
  Test 5 — Different images
  Test 6 — Protected originals
  Test 7 — Repeated collector execution
"""

from __future__ import annotations

import random as _random
from pathlib import Path

import imagehash
import pytest
from PIL import Image

from dataset_builder.dedup import DedupIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED_COUNTER = 0


def _make_image(path: Path, seed: int | None = None,
                size: tuple[int, int] = (256, 256)) -> Path:
    """Create a test image with deterministic random noise pattern.

    Different seeds produce genuinely different pHashes.
    """
    global _SEED_COUNTER
    if seed is None:
        _SEED_COUNTER += 1
        seed = _SEED_COUNTER * 997 + 13
    rng = _random.Random(seed)
    pixels = bytes([rng.randint(0, 255) for _ in range(size[0] * size[1] * 3)])
    img = Image.frombytes("RGB", size, pixels)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    d = tmp_path / "dedup_index"
    d.mkdir()
    return d


@pytest.fixture
def dedup(index_dir: Path) -> DedupIndex:
    return DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)


# ---------------------------------------------------------------------------
# Test 1 — Same RAW image
# ---------------------------------------------------------------------------

class Test1_SameRawImage:
    """Two copies of the exact same RAW image must be detected as duplicates."""

    def test_identical_raw_files(self, dedup: DedupIndex, tmp_path: Path):
        raw_a = _make_image(tmp_path / "raw" / "photo_a.png", seed=100)
        raw_b = tmp_path / "raw" / "photo_b.png"
        raw_b.write_bytes(raw_a.read_bytes())

        dedup.add_raw("img_a", raw_a)

        assert dedup.is_raw_duplicate(raw_b) is True

    def test_identical_raw_bytes(self, dedup: DedupIndex, tmp_path: Path):
        raw_a = _make_image(tmp_path / "raw" / "photo_a.png", seed=200)
        raw_b = tmp_path / "raw" / "photo_b.png"
        raw_b.write_bytes(raw_a.read_bytes())

        dedup.add_raw("img_a", raw_a)

        assert dedup.is_raw_duplicate(raw_b) is True


# ---------------------------------------------------------------------------
# Test 2 — Same image after restart (persistence)
# ---------------------------------------------------------------------------

class Test2_RestartPersistence:
    """Index must survive process restart and still detect duplicates."""

    def test_index_survives_restart(self, index_dir: Path, tmp_path: Path):
        raw_img = _make_image(tmp_path / "raw" / "photo.png", seed=300)

        # Run 1: create index, add image
        dedup1 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        dedup1.add_raw("sample_0001", raw_img)

        # Run 2: new instance loads from disk
        dedup2 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        assert dedup2.is_raw_duplicate(raw_img) is True
        assert dedup2.raw_count == 1

    def test_aligned_index_survives_restart(self, index_dir: Path, tmp_path: Path):
        aligned_img = _make_image(tmp_path / "aligned" / "sample_0001.png", seed=400)

        dedup1 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        dedup1.add_aligned("sample_0001", aligned_img)

        dedup2 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        assert dedup2.is_aligned_duplicate(aligned_img) is True
        assert dedup2.aligned_count == 1

    def test_both_indices_persist(self, index_dir: Path, tmp_path: Path):
        raw_img = _make_image(tmp_path / "raw" / "r.png", seed=500)
        aligned_img = _make_image(tmp_path / "aligned" / "a.png", seed=600)

        dedup1 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        dedup1.add_raw("s1", raw_img)
        dedup1.add_aligned("s1", aligned_img)

        dedup2 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        assert dedup2.raw_count == 1
        assert dedup2.aligned_count == 1


# ---------------------------------------------------------------------------
# Test 3 — RAW vs ALIGNED representation separation
# ---------------------------------------------------------------------------

class Test3_RawVsAlignedSeparation:
    """RAW and ALIGNED pHashes must NEVER be compared against each other."""

    def test_raw_check_ignores_aligned_index(self, dedup: DedupIndex, tmp_path: Path):
        """A new raw image must NOT be flagged as duplicate based on aligned index content."""
        aligned_img = _make_image(tmp_path / "aligned" / "photo.png", seed=700)
        dedup.add_aligned("s1", aligned_img)

        # New raw image with different seed → genuinely different image
        new_raw = _make_image(tmp_path / "raw" / "new.png", seed=800)
        assert dedup.is_raw_duplicate(new_raw) is False

    def test_aligned_check_ignores_raw_index(self, dedup: DedupIndex, tmp_path: Path):
        """A new aligned image must NOT be flagged as duplicate based on raw index content."""
        raw_img = _make_image(tmp_path / "raw" / "photo.png", seed=900)
        dedup.add_raw("s1", raw_img)

        new_aligned = _make_image(tmp_path / "aligned" / "new.png", seed=1000)
        assert dedup.is_aligned_duplicate(new_aligned) is False

    def test_same_image_both_indices_still_independent(self, dedup: DedupIndex, tmp_path: Path):
        """Same file loaded into both indices — each index only checks its own."""
        img = _make_image(tmp_path / "shared" / "photo.png", seed=1100)

        dedup.add_raw("s1", img)
        dedup.add_aligned("s1", img)

        # Identical copy detected in raw index
        copy = tmp_path / "copy.png"
        copy.write_bytes(img.read_bytes())
        assert dedup.is_raw_duplicate(copy) is True

        # But a DIFFERENT new image must not be detected by either index
        different_raw = _make_image(tmp_path / "diff_raw.png", seed=1200)
        different_aligned = _make_image(tmp_path / "diff_aligned.png", seed=1300)
        assert dedup.is_raw_duplicate(different_raw) is False
        assert dedup.is_aligned_duplicate(different_aligned) is False


# ---------------------------------------------------------------------------
# Test 4 — Same ALIGNED image
# ---------------------------------------------------------------------------

class Test4_SameAlignedImage:
    """Identical aligned images must be detected as duplicates."""

    def test_identical_aligned_files(self, dedup: DedupIndex, tmp_path: Path):
        aligned_a = _make_image(tmp_path / "aligned" / "a.png", seed=1400)
        aligned_b = tmp_path / "aligned" / "b.png"
        aligned_b.write_bytes(aligned_a.read_bytes())

        dedup.add_aligned("s1", aligned_a)
        assert dedup.is_aligned_duplicate(aligned_b) is True

    def test_aligned_group_finding(self, dedup: DedupIndex, tmp_path: Path):
        """find_aligned_duplicate_groups must return correct groups."""
        img = _make_image(tmp_path / "aligned" / "shared.png", seed=1500)

        for sid in ["s1", "s2", "s3"]:
            copy = tmp_path / "aligned" / f"{sid}.png"
            copy.write_bytes(img.read_bytes())
            dedup.add_aligned(sid, copy)

        groups = dedup.find_aligned_duplicate_groups()
        assert len(groups) == 1
        group_members = list(groups.values())[0]
        assert len(group_members) == 3
        assert set(group_members) == {"s1", "s2", "s3"}


# ---------------------------------------------------------------------------
# Test 5 — Different images
# ---------------------------------------------------------------------------

class Test5_DifferentImages:
    """Genuinely different images must NOT be flagged as duplicates."""

    def test_different_raw_images(self, dedup: DedupIndex, tmp_path: Path):
        raw_a = _make_image(tmp_path / "raw" / "a.png", seed=1600)
        raw_b = _make_image(tmp_path / "raw" / "b.png", seed=1700)

        dedup.add_raw("s1", raw_a)
        assert dedup.is_raw_duplicate(raw_b) is False

    def test_different_aligned_images(self, dedup: DedupIndex, tmp_path: Path):
        aligned_a = _make_image(tmp_path / "aligned" / "a.png", seed=1800)
        aligned_b = _make_image(tmp_path / "aligned" / "b.png", seed=1900)

        dedup.add_aligned("s1", aligned_a)
        assert dedup.is_aligned_duplicate(aligned_b) is False

    def test_no_false_positives_across_many(self, dedup: DedupIndex, tmp_path: Path):
        """50 distinct random images must produce 0 duplicate groups."""
        for i in range(50):
            img = _make_image(
                tmp_path / "aligned" / f"s{i:04d}.png",
                seed=2000 + i * 17,
            )
            dedup.add_aligned(f"s{i:04d}", img)

        groups = dedup.find_aligned_duplicate_groups()
        assert len(groups) == 0


# ---------------------------------------------------------------------------
# Test 6 — Protected originals
# ---------------------------------------------------------------------------

class Test6_ProtectedOriginals:
    """The dedup system must compare new candidates against protected originals
    without modifying the originals."""

    def test_protected_not_modified(self, dedup: DedupIndex, tmp_path: Path):
        protected = _make_image(tmp_path / "protected" / "sample_0000.png", seed=2100)
        original_bytes = protected.read_bytes()
        original_hash = imagehash.phash(Image.open(protected), hash_size=16)

        dedup.add_aligned("sample_0000", protected)

        # File must be unchanged
        assert protected.read_bytes() == original_bytes
        # Hash must be unchanged
        assert imagehash.phash(Image.open(protected), hash_size=16) == original_hash

    def test_new_candidate_checked_against_protected(self, dedup: DedupIndex, tmp_path: Path):
        protected = _make_image(tmp_path / "protected" / "sample_0000.png", seed=2200)
        dedup.add_aligned("sample_0000", protected)

        # Identical new candidate must be detected
        new_copy = tmp_path / "new" / "candidate.png"
        new_copy.parent.mkdir(parents=True, exist_ok=True)
        new_copy.write_bytes(protected.read_bytes())
        assert dedup.is_aligned_duplicate(new_copy) is True

        # Different new candidate must pass
        different = _make_image(tmp_path / "new" / "different.png", seed=2300)
        assert dedup.is_aligned_duplicate(different) is False


# ---------------------------------------------------------------------------
# Test 7 — Repeated collector execution
# ---------------------------------------------------------------------------

class Test7_RepeatedExecution:
    """Previously accepted images must NOT be accepted again after restart."""

    def test_second_run_rejectsPreviouslyAccepted(self, index_dir: Path, tmp_path: Path):
        """Simulate: Run1 accepts image A. Run2 tries same image -> must reject."""
        img_a = _make_image(tmp_path / "downloads" / "a.png", seed=2400)

        # Run 1
        dedup1 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        assert dedup1.is_raw_duplicate(img_a) is False  # first time -> not dup
        dedup1.add_raw("sample_0001", img_a)
        dedup1.save()

        # Run 2 (new process)
        dedup2 = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        assert dedup2.is_raw_duplicate(img_a) is True  # already seen -> duplicate
        assert dedup2.raw_count == 1

    def test_multiple_restarts(self, index_dir: Path, tmp_path: Path):
        """Multiple restarts: each new image accepted only once."""
        images = []
        for i in range(10):
            img = _make_image(
                tmp_path / "downloads" / f"img_{i}.png",
                seed=2500 + i * 31,
            )
            images.append(img)

        # Simulate 10 separate runs, each accepting one image
        for run_idx, img in enumerate(images):
            dedup = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
            # This image must NOT be a duplicate (first time seen)
            assert dedup.is_raw_duplicate(img) is False, f"Run {run_idx}: false positive"
            dedup.add_raw(f"sample_{run_idx:04d}", img)

        # Final run: all images must be detected as duplicates
        dedup_final = DedupIndex(index_dir=index_dir, hash_size=16, threshold=5)
        for i, img in enumerate(images):
            assert dedup_final.is_raw_duplicate(img) is True, f"Image {i} not detected after restart"
        assert dedup_final.raw_count == 10
