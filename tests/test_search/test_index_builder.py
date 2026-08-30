"""Tests for search.index_builder — Reference dataset ingestion & FAISS index building.

Covers:
- Dataset discovery & deterministic ordering
- Single and multiple images per person
- Zero-face / unreadable image handling (failure isolation)
- Empty / missing dataset errors
- FAISS index & metadata.json generation
- Artifact consistency (index.ntotal == len(records))
- Reproducible rebuilds
- End-to-end search smoke test
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from search.flat_index import FlatIndex
from search.index_builder import BuildReport, IndexBuilder, MetadataRecord


# ---------------------------------------------------------------------------
# Helpers & Test Data Generation
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REAL_TEST_IMAGE = PROJECT_ROOT / "test_images" / "good" / "glasses.jpg"


def _copy_real_face_image(path: Path) -> None:
    """Copy a real test image containing a detectable face."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if REAL_TEST_IMAGE.exists():
        shutil.copy(REAL_TEST_IMAGE, path)
    else:
        # Fallback if test image missing: create a dummy image (will trigger NO_FACE)
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)


def _create_blank_image(path: Path) -> None:
    """Create a blank image with no face (triggers NO_FACE)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndexBuilderDatasetValidation:
    def test_missing_dataset_dir_raises(self) -> None:
        builder = IndexBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent = Path(tmpdir) / "nonexistent"
            with pytest.raises(ValueError, match="does not exist"):
                builder.build(nonexistent, tmpdir)

    def test_empty_dataset_dir_raises(self) -> None:
        builder = IndexBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset"
            dataset.mkdir()
            with pytest.raises(ValueError, match="No person subdirectories found"):
                builder.build(dataset, tmpdir)

    def test_no_supported_images_raises(self) -> None:
        builder = IndexBuilder()
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset = Path(tmpdir) / "dataset"
            pdir = dataset / "person_001"
            pdir.mkdir(parents=True)
            (pdir / "not_an_image.txt").write_text("hello")
            with pytest.raises(ValueError, match="No supported images found"):
                builder.build(dataset, tmpdir)


class TestIndexBuilderIngestion:
    def test_successful_build_with_real_face_data(self) -> None:
        builder = IndexBuilder(dimension=512)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset"
            output = root / "output"

            p1 = dataset / "person_001"
            p2 = dataset / "person_002"

            _copy_real_face_image(p1 / "img_001.jpg")
            _copy_real_face_image(p1 / "img_002.jpg")
            _copy_real_face_image(p2 / "img_001.jpg")

            # Add a blank image that will be skipped (NO_FACE)
            _create_blank_image(p1 / "bad_blank.jpg")

            index, report = builder.build(dataset, output)

            # Assertions
            assert isinstance(report, BuildReport)
            assert report.total_input_images == 4
            assert report.skipped_images == 1
            assert report.total_persons == 2
            assert report.accepted_embeddings == index.size
            assert index.size == 3

            # Check artifacts exist
            assert (output / "reference_index.faiss").exists()
            assert (output / "metadata.json").exists()

            # Verify metadata content
            with open(output / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)

            assert meta["schema_version"] == 1
            assert meta["embedding_dimension"] == 512
            assert meta["total_vectors"] == 3
            assert meta["total_persons"] == 2
            assert len(meta["records"]) == 3

            for i, rec in enumerate(meta["records"]):
                assert rec["vector_id"] == i
                assert "person_id" in rec
                assert "image" in rec


class TestIndexBuilderReproducibility:
    def test_rebuild_produces_identical_artifacts(self) -> None:
        builder = IndexBuilder(dimension=512)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset"
            out1 = root / "out1"
            out2 = root / "out2"

            p = dataset / "person_001"
            _copy_real_face_image(p / "img1.jpg")
            _copy_real_face_image(p / "img2.jpg")

            idx1, rep1 = builder.build(dataset, out1)
            idx2, rep2 = builder.build(dataset, out2)

            assert idx1.size == idx2.size
            assert rep1.accepted_embeddings == rep2.accepted_embeddings

            meta1 = json.loads(Path(rep1.metadata_path).read_text(encoding="utf-8"))
            meta2 = json.loads(Path(rep2.metadata_path).read_text(encoding="utf-8"))

            assert meta1["total_vectors"] == meta2["total_vectors"]
            for r1, r2 in zip(meta1["records"], meta2["records"]):
                assert r1["vector_id"] == r2["vector_id"]
                assert r1["person_id"] == r2["person_id"]
                assert r1["image"] == r2["image"]


class TestIndexBuilderEndToEndSearch:
    def test_load_and_search_smoke_test(self) -> None:
        builder = IndexBuilder(dimension=512)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset"
            output = root / "output"

            p1 = dataset / "alice"
            p2 = dataset / "bob"

            _copy_real_face_image(p1 / "photo.jpg")
            _copy_real_face_image(p2 / "photo.jpg")

            index, report = builder.build(dataset, output)

            # Load artifacts back
            loaded_index = FlatIndex.load(output / "reference_index.faiss")
            meta = json.loads((output / "metadata.json").read_text(encoding="utf-8"))

            assert loaded_index.size == 2

            # Query with a random normalized vector
            query = np.random.randn(512).astype(np.float32)
            query /= np.linalg.norm(query)

            search_result = loaded_index.search(query, k=1)
            top_id = int(search_result.ids[0])

            record = meta["records"][top_id]
            assert record["vector_id"] == top_id
            assert record["person_id"] in ("alice", "bob")
            assert "photo.jpg" in record["image"]
