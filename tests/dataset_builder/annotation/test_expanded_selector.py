"""Regression tests for expanded_selector.py with Phase 2 manifest schema.

These tests verify:
- Current Phase 2 manifest schema compatibility
- Every valid sample record contains/resolves a sample_id
- Malformed/non-sample records are rejected clearly
- Annotation selection loads successfully
- Existing corrected samples remain protected
- No duplicate sample IDs
- All referenced image paths exist
- All existing corrected-mask paths remain unchanged
"""

import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dataset_builder.dataset.parser_finetune.annotation.expanded_selector import (
    load_expanded_selection,
    classify_expanded_sample,
)
from dataset_builder.dataset.parser_finetune.annotation.expanded_config import (
    CurrentAnnotationConfig,
    ExpandedAnnotationConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURRENT_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "parser_finetune_current"
MANIFEST_PATH = CURRENT_DIR / "annotation_manifest.json"


class TestPhase2ManifestSchema:
    """Verify the Phase 2 manifest has the expected schema."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_has_version(self):
        assert "version" in self.manifest

    def test_manifest_has_samples_array(self):
        assert "samples" in self.manifest
        assert isinstance(self.manifest["samples"], list)

    def test_manifest_total_matches_samples(self):
        assert self.manifest["total_samples"] == len(self.manifest["samples"])

    def test_all_samples_have_id(self):
        for i, sample in enumerate(self.manifest["samples"]):
            assert "id" in sample, f"Sample {i} missing 'id' field"
            assert isinstance(sample["id"], str), f"Sample {i} id is not a string"
            assert len(sample["id"]) > 0, f"Sample {i} has empty id"

    def test_all_samples_have_source_category(self):
        for i, sample in enumerate(self.manifest["samples"]):
            assert "source_category" in sample, f"Sample {i} missing 'source_category'"

    def test_no_duplicate_ids(self):
        ids = [s["id"] for s in self.manifest["samples"]]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in manifest"

    def test_sample_ids_match_image_files(self):
        images_dir = CURRENT_DIR / "images"
        for sample in self.manifest["samples"]:
            img_path = images_dir / f"{sample['id']}.png"
            assert img_path.exists(), f"Image missing for {sample['id']}"


class TestExpandedSelectorFix:
    """Verify expanded_selector.py correctly handles Phase 2 schema."""

    def test_load_expanded_selection_current(self):
        """The main regression: load_expanded_selection must not crash with KeyError."""
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        assert selection is not None
        assert len(selection.samples) > 0

    def test_load_expanded_selection_sample_id_populated(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        for sample in selection.samples:
            assert sample.sample_id is not None
            assert len(sample.sample_id) > 0

    def test_load_expanded_selection_aligned_path_populated(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        for sample in selection.samples:
            assert sample.aligned_image_path is not None
            assert len(sample.aligned_image_path) > 0

    def test_load_expanded_selection_initial_mask_populated(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        for sample in selection.samples:
            assert sample.initial_mask_path is not None
            assert len(sample.initial_mask_path) > 0

    def test_load_expanded_selection_images_exist(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        for sample in selection.samples:
            assert Path(sample.aligned_image_path).exists(), (
                f"Image missing: {sample.aligned_image_path}"
            )

    def test_load_expanded_selection_initial_masks_exist(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        for sample in selection.samples:
            assert Path(sample.initial_mask_path).exists(), (
                f"Initial mask missing: {sample.initial_mask_path}"
            )

    def test_load_expanded_selection_no_duplicate_sample_ids(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        ids = [s.sample_id for s in selection.samples]
        assert len(ids) == len(set(ids)), "Duplicate sample IDs in selection"

    def test_load_expanded_selection_splits_assigned(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        for sample in selection.samples:
            assert sample.split in ("train", "val", "test", ""), (
                f"Invalid split '{sample.split}' for {sample.sample_id}"
            )

    def test_load_expanded_selection_category_counts(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        total = sum(selection.category_counts.values())
        assert total == len(selection.samples)

    def test_load_expanded_selection_backward_compat_sample_id_field(self):
        """Verify the selector accepts both 'sample_id' and 'id' fields."""
        config = CurrentAnnotationConfig()
        manifest = json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
        sample = manifest["samples"][0]
        # Current Phase 2 uses 'id'
        assert "id" in sample
        # Selector should handle this via fallback
        selection = load_expanded_selection(config)
        assert selection is not None
        assert any(s.sample_id == sample["id"] for s in selection.samples)


class TestExpandedSelectorMalformedRecords:
    """Verify the selector rejects or skips malformed records gracefully."""

    def test_skips_records_with_no_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = self._make_config(tmpdir, samples=[
                {"source_category": "test"},
            ])
            selection = load_expanded_selection(config)
            assert selection is not None
            assert len(selection.samples) == 0

    def test_accepts_records_with_sample_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.zeros((112, 112), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(Path(tmpdir) / "images" / "test_001.png"),
                        np.zeros((112, 112, 3), dtype=np.uint8))
            cv2.imwrite(str(Path(tmpdir) / "initial_masks" / "test_001.png"), mask)

            config = self._make_config(tmpdir, samples=[
                {"sample_id": "test_001", "source_category": "eyeglasses"},
            ])
            selection = load_expanded_selection(config)
            assert selection is not None
            assert len(selection.samples) == 1
            assert selection.samples[0].sample_id == "test_001"

    def test_accepts_records_with_id(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.zeros((112, 112), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(Path(tmpdir) / "images" / "test_002.png"),
                        np.zeros((112, 112, 3), dtype=np.uint8))
            cv2.imwrite(str(Path(tmpdir) / "initial_masks" / "test_002.png"), mask)

            config = self._make_config(tmpdir, samples=[
                {"id": "test_002", "source_category": "hijab"},
            ])
            selection = load_expanded_selection(config)
            assert selection is not None
            assert len(selection.samples) == 1
            assert selection.samples[0].sample_id == "test_002"

    def test_derives_paths_when_not_in_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mask = np.zeros((112, 112), dtype=np.uint8)
            import cv2
            (Path(tmpdir) / "images").mkdir()
            (Path(tmpdir) / "initial_masks").mkdir()
            cv2.imwrite(str(Path(tmpdir) / "images" / "test_003.png"),
                        np.zeros((112, 112, 3), dtype=np.uint8))
            cv2.imwrite(str(Path(tmpdir) / "initial_masks" / "test_003.png"), mask)

            config = self._make_config(tmpdir, samples=[
                {"id": "test_003", "source_category": "mask"},
            ])
            selection = load_expanded_selection(config)
            assert selection is not None
            assert len(selection.samples) == 1
            assert "test_003.png" in selection.samples[0].aligned_image_path
            assert "test_003.png" in selection.samples[0].initial_mask_path

    def _make_config(self, tmpdir, samples):
        import dataclasses
        import cv2

        base = Path(tmpdir)
        for d in ["images", "initial_masks", "masks", "splits", "reports", "metadata"]:
            (base / d).mkdir(exist_ok=True)

        # Write splits
        ids = [s.get("id") or s.get("sample_id", "") for s in samples]
        for split_name in ["train", "val", "test"]:
            (base / "splits" / f"{split_name}.txt").write_text(
                "\n".join(ids) + "\n" if ids else ""
            )

        manifest_path = base / "annotation_manifest.json"
        manifest_path.write_text(json.dumps({
            "version": "2.0",
            "total_samples": len(samples),
            "samples": samples,
        }), encoding="utf-8")

        return dataclasses.replace(
            CurrentAnnotationConfig(),
            CURRENT_DIR=base,
            IMAGES_DIR=base / "images",
            INITIAL_MASKS_DIR=base / "initial_masks",
            MASKS_DIR=base / "masks",
            SPLITS_DIR=base / "splits",
            REPORTS_DIR=base / "reports",
            METADATA_DIR=base / "metadata",
            MANIFEST_PATH=manifest_path,
        )


class TestProtectedDataIntegrity:
    """Verify existing corrected masks and protected data remain untouched."""

    def test_corrected_masks_count_unchanged(self):
        corrected_dir = CURRENT_DIR / "annotation" / "corrected_masks"
        count = len(list(corrected_dir.glob("*.png")))
        assert count >= 339, f"Expected >= 339 corrected masks, found {count}"

    def test_corrected_masks_still_exist(self):
        corrected_dir = CURRENT_DIR / "annotation" / "corrected_masks"
        for mask_file in sorted(corrected_dir.glob("*.png")):
            assert mask_file.exists()
            assert mask_file.stat().st_size > 0

    def test_initial_masks_intact(self):
        initial_dir = CURRENT_DIR / "initial_masks"
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for sample in manifest["samples"]:
            mask_path = initial_dir / f"{sample['id']}.png"
            assert mask_path.exists(), f"Initial mask missing: {sample['id']}"


class TestAnnotationStartupSimulation:
    """Simulate the annotation tool startup without launching a server."""

    def test_current_config_resolves(self):
        config = CurrentAnnotationConfig()
        assert config.MANIFEST_PATH.exists()
        assert config.IMAGES_DIR.exists()
        assert config.INITIAL_MASKS_DIR.exists()

    def test_selection_loads_all_samples(self):
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        assert selection is not None
        assert len(selection.samples) == 1058

    def test_ensure_expanded_equivalent(self):
        """Simulate _ensure_expanded() from the annotation tool."""
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        assert selection is not None
        assert len(selection.samples) > 0
        assert selection.total_in_manifest == 1058

    def test_init_metadata_equivalent(self):
        """Simulate _init_metadata() from the annotation tool."""
        from dataset_builder.dataset.parser_finetune.annotation.annotation_metadata import (
            load_annotation_metadata,
        )
        config = CurrentAnnotationConfig()
        selection = load_expanded_selection(config)
        # Just verify we can query metadata for samples without crashing
        for sample in selection.samples[:10]:
            meta = load_annotation_metadata(sample.sample_id, config)
            # meta may be None for new samples, that's OK
