"""Tests for the annotation preparation adapter (scripts/prepare_annotation.py).

Covers:
1. Current manifest generation
2. Split preservation
3. Split exclusivity
4. Manifest/sample consistency
5. Metadata consistency
6. Class mapping validation
7. Idempotent rerun
8. Missing metadata recovery
9. Missing artifact recovery
10. Ground-truth pending status
11. Final READY_FOR_ANNOTATION status
12. Validation failure status
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# Add project root to path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_annotation import (
    PreparationReport,
    PreparationResult,
    _assign_sample_ids,
    _build_manifest,
    _check_artifacts,
    _keep_images,
    _load_manifest,
    _validate_artifacts,
    _validate_class_mapping,
    _validate_manifest,
    _validate_metadata,
    _validate_splits,
    _write_split_files,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_fake_manifest(output_dir: Path, images: list[dict]) -> Path:
    """Create a fake dataset_manifest.json for testing."""
    manifest = {"images": images}
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def _make_image_entry(
    category: str = "normal",
    filename: str = "img_001.jpg",
    decision: str = "KEEP",
    split: str = "train",
) -> dict:
    return {
        "path": f"/fake/raw/{category}/{filename}",
        "filename": filename,
        "category": category,
        "decision": decision,
        "split": split if decision == "KEEP" else "",
    }


def _create_aligned_image(path: Path, sample_id: str) -> None:
    """Create a fake 112x112 aligned image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_mask(path: Path) -> None:
    """Create a fake 112x112 mask with values 0-18."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = np.random.randint(0, 19, (112, 112), dtype=np.uint8)
    cv2.imwrite(str(path), mask)


def _create_metadata(
    path: Path,
    sample_id: str,
    split: str = "train",
    aligned_path: str = "",
    initial_mask_path: str = "",
    ground_truth_mask: str = "",
) -> None:
    """Create a fake metadata JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "sample_id": sample_id,
        "source_image": f"/fake/raw/{sample_id}.jpg",
        "source_category": "normal",
        "aligned_image": aligned_path,
        "initial_mask": initial_mask_path,
        "ground_truth_mask": ground_truth_mask,
        "face_bbox": [0.1, 0.1, 0.5, 0.5],
        "face_kps": [[0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.25, 0.4], [0.35, 0.4]],
        "image_width": 112,
        "image_height": 112,
        "detection_score": 0.99,
        "face_area_ratio": 0.15,
        "selection_reason": "quality_pass",
        "quality_status": "annotation_pending",
        "split": split,
    }
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def _create_face_data(path: Path, sample_id: str) -> None:
    """Create a fake face_data sidecar."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = {
        "face_bbox": [0.1, 0.1, 0.5, 0.5],
        "face_kps": [[0.2, 0.2], [0.3, 0.3], [0.4, 0.4], [0.25, 0.4], [0.35, 0.4]],
        "detection_score": 0.99,
        "face_area_ratio": 0.15,
    }
    with open(path, "w") as f:
        json.dump(fd, f)


def _create_class_mapping(path: Path) -> None:
    """Create a fake class_mapping.json with 19 classes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {
        0: "BACKGROUND", 1: "SKIN", 2: "LEFT_BROW", 3: "RIGHT_BROW",
        4: "LEFT_EYE", 5: "RIGHT_EYE", 6: "EYE_GLASS", 7: "LEFT_EAR",
        8: "RIGHT_EAR", 9: "EAR_RING", 10: "NOSE", 11: "MOUTH",
        12: "UPPER_LIP", 13: "LOWER_LIP", 14: "NECK", 15: "NECKLACE",
        16: "CLOTH", 17: "HAIR", 18: "HAT",
    }
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)


def _setup_sample(
    output_dir: Path,
    sample_id: str,
    split: str = "train",
    include_aligned: bool = True,
    include_initial_mask: bool = True,
    include_mask: bool = True,
    include_metadata: bool = True,
    include_face_data: bool = True,
) -> PreparationResult:
    """Set up a complete sample with all artifacts."""
    aligned_path = output_dir / "images" / f"{sample_id}.png"
    initial_mask_path = output_dir / "initial_masks" / f"{sample_id}.png"
    mask_path = output_dir / "masks" / f"{sample_id}.png"
    metadata_path = output_dir / "metadata" / f"{sample_id}.json"
    face_data_path = output_dir / "face_data" / f"{sample_id}.json"

    if include_aligned:
        _create_aligned_image(aligned_path, sample_id)
    if include_initial_mask:
        _create_mask(initial_mask_path)
    if include_mask:
        _create_mask(mask_path)
    if include_metadata:
        _create_metadata(
            metadata_path,
            sample_id,
            split=split,
            aligned_path=str(aligned_path),
            initial_mask_path=str(initial_mask_path),
            ground_truth_mask=str(mask_path),
        )
    if include_face_data:
        _create_face_data(face_data_path, sample_id)

    return PreparationResult(
        sample_id=sample_id,
        source_path=f"/fake/raw/normal/{sample_id}.jpg",
        source_category="normal",
        filename=f"{sample_id}.jpg",
        split=split,
        aligned_path=str(aligned_path),
        initial_mask_path=str(initial_mask_path),
        mask_path=str(mask_path),
        metadata_path=str(metadata_path),
        aligned=include_aligned,
        masked=include_initial_mask,
    )


# ---------------------------------------------------------------------------
# 1. Current manifest generation
# ---------------------------------------------------------------------------


class TestManifestGeneration:
    """Test that annotation manifest is correctly generated."""

    def test_manifest_contains_required_fields(self, tmp_path: Path) -> None:
        """Manifest must have version, annotation_status, total_samples, splits, samples."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i in range(3):
            r = _setup_sample(output_dir, f"sample_{i:04d}", split="train")
            results.append(r)

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        assert manifest["version"] == "current_v1"
        assert manifest["annotation_status"] == "annotation_pending"
        assert manifest["total_samples"] == 3
        assert "train" in manifest["splits"]
        assert "val" in manifest["splits"]
        assert "test" in manifest["splits"]
        assert len(manifest["samples"]) == 3

    def test_manifest_sample_has_all_required_fields(self, tmp_path: Path) -> None:
        """Each manifest sample must have all required fields."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        results = [r]

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        sample = manifest["samples"][0]
        required = [
            "sample_id", "source_category", "source_filename",
            "aligned_image", "initial_mask", "ground_truth_mask",
            "split", "annotation_status",
        ]
        for field in required:
            assert field in sample, f"Missing field: {field}"

    def test_manifest_split_counts_match(self, tmp_path: Path) -> None:
        """Manifest split counts must match actual sample distribution."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i in range(5):
            results.append(_setup_sample(output_dir, f"sample_{i:04d}", split="train"))
        for i in range(3):
            results.append(
                _setup_sample(output_dir, f"sample_{5+i:04d}", split="val")
            )
        for i in range(2):
            results.append(
                _setup_sample(output_dir, f"sample_{8+i:04d}", split="test")
            )

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        assert manifest["splits"]["train"] == 5
        assert manifest["splits"]["val"] == 3
        assert manifest["splits"]["test"] == 2
        assert manifest["total_samples"] == 10

    def test_manifest_written_to_annotation_manifest_json(
        self, tmp_path: Path
    ) -> None:
        """Manifest must be written to annotation_manifest.json at output root."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        results = [r]

        manifest_dict = {"images": []}
        _build_manifest(results, output_dir, manifest_dict)

        manifest_path = output_dir / "annotation_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            loaded = json.load(f)
        assert loaded["total_samples"] == 1


# ---------------------------------------------------------------------------
# 2. Split preservation
# ---------------------------------------------------------------------------


class TestSplitPreservation:
    """Test that splits from the authoritative manifest are preserved."""

    def test_split_from_manifest_preserved(self, tmp_path: Path) -> None:
        """Split values must come directly from the authoritative manifest."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="val"),
            _setup_sample(output_dir, "sample_0002", split="test"),
        ]

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        splits = {s["sample_id"]: s["split"] for s in manifest["samples"]}
        assert splits["sample_0000"] == "train"
        assert splits["sample_0001"] == "val"
        assert splits["sample_0002"] == "test"

    def test_split_files_contain_correct_ids(self, tmp_path: Path) -> None:
        """Split files must contain the correct sample IDs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="train"),
            _setup_sample(output_dir, "sample_0002", split="val"),
            _setup_sample(output_dir, "sample_0003", split="test"),
        ]

        split_counts = _write_split_files(results, output_dir)

        train_ids = (output_dir / "splits" / "train.txt").read_text().strip().split("\n")
        val_ids = (output_dir / "splits" / "val.txt").read_text().strip().split("\n")
        test_ids = (output_dir / "splits" / "test.txt").read_text().strip().split("\n")

        assert "sample_0000" in train_ids
        assert "sample_0001" in train_ids
        assert "sample_0002" in val_ids
        assert "sample_0003" in test_ids


# ---------------------------------------------------------------------------
# 3. Split exclusivity
# ---------------------------------------------------------------------------


class TestSplitExclusivity:
    """Test that no sample appears in multiple splits."""

    def test_no_sample_in_multiple_splits(self, tmp_path: Path) -> None:
        """No sample_id should appear in more than one split file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="val"),
            _setup_sample(output_dir, "sample_0002", split="test"),
        ]

        _write_split_files(results, output_dir)

        all_ids = set()
        for split_name in ("train", "val", "test"):
            path = output_dir / "splits" / f"{split_name}.txt"
            for line in path.read_text().splitlines():
                sid = line.strip()
                assert sid not in all_ids, f"{sid} appears in multiple splits"
                all_ids.add(sid)

    def test_all_successful_samples_in_split(self, tmp_path: Path) -> None:
        """Every successful sample must appear in exactly one split."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="val"),
            _setup_sample(output_dir, "sample_0002", split="test"),
        ]

        _write_split_files(results, output_dir)

        all_split_ids = set()
        for split_name in ("train", "val", "test"):
            path = output_dir / "splits" / f"{split_name}.txt"
            for line in path.read_text().splitlines():
                sid = line.strip()
                if sid:
                    all_split_ids.add(sid)

        successful_ids = {r.sample_id for r in results}
        assert all_split_ids == successful_ids

    def test_split_validation_pass(self, tmp_path: Path) -> None:
        """_validate_splits should PASS with correct split files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="val"),
            _setup_sample(output_dir, "sample_0002", split="test"),
        ]

        _write_split_files(results, output_dir)
        status, issues = _validate_splits(results, output_dir)

        assert status == "PASS"
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# 4. Manifest/sample consistency
# ---------------------------------------------------------------------------


class TestManifestSampleConsistency:
    """Test manifest and sample set consistency."""

    def test_manifest_ids_match_successful_results(self, tmp_path: Path) -> None:
        """Manifest sample IDs must match the set of successful results."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="val"),
        ]

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        manifest_ids = {s["sample_id"] for s in manifest["samples"]}
        successful_ids = {r.sample_id for r in results if r.aligned and r.masked}
        assert manifest_ids == successful_ids

    def test_no_duplicate_ids_in_manifest(self, tmp_path: Path) -> None:
        """No duplicate sample IDs allowed in the manifest."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="train"),
        ]

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        ids = [s["sample_id"] for s in manifest["samples"]]
        assert len(ids) == len(set(ids))

    def test_manifest_validation_pass(self, tmp_path: Path) -> None:
        """_validate_manifest should PASS with consistent manifest."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="val"),
        ]

        manifest_dict = {"images": []}
        _build_manifest(results, output_dir, manifest_dict)
        _write_split_files(results, output_dir)

        status, issues = _validate_manifest(results, output_dir)
        assert status == "PASS"
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# 5. Metadata consistency
# ---------------------------------------------------------------------------


class TestMetadataConsistency:
    """Test metadata matches manifest and artifacts."""

    def test_metadata_sample_id_matches(self, tmp_path: Path) -> None:
        """Metadata sample_id must match the PreparationResult sample_id."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        results = [r]

        manifest_dict = {"images": []}
        _build_manifest(results, output_dir, manifest_dict)

        status, issues = _validate_metadata(results, output_dir)
        assert status == "PASS"

    def test_metadata_split_matches_manifest(self, tmp_path: Path) -> None:
        """Metadata split must match manifest split."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="val")
        results = [r]

        manifest_dict = {"images": []}
        _build_manifest(results, output_dir, manifest_dict)

        status, issues = _validate_metadata(results, output_dir)
        assert status == "PASS"

    def test_metadata_paths_match(self, tmp_path: Path) -> None:
        """Metadata aligned_image, initial_mask, ground_truth_mask paths must match."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        results = [r]

        manifest_dict = {"images": []}
        _build_manifest(results, output_dir, manifest_dict)

        # Verify paths in metadata
        meta_path = output_dir / "metadata" / "sample_0000.json"
        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["aligned_image"] == r.aligned_path
        assert meta["initial_mask"] == r.initial_mask_path
        assert meta["ground_truth_mask"] == r.mask_path


# ---------------------------------------------------------------------------
# 6. Class mapping validation
# ---------------------------------------------------------------------------


class TestClassMappingValidation:
    """Test class mapping validation."""

    def test_class_mapping_pass(self, tmp_path: Path) -> None:
        """_validate_class_mapping should PASS with valid 19-class mapping."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _create_class_mapping(output_dir / "metadata" / "class_mapping.json")

        status, issues = _validate_class_mapping(output_dir)
        assert status == "PASS"
        assert len(issues) == 0

    def test_class_mapping_missing_file(self, tmp_path: Path) -> None:
        """_validate_class_mapping should FAIL when file is missing."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        status, issues = _validate_class_mapping(output_dir)
        assert status == "FAIL"
        assert any("not found" in i for i in issues)

    def test_class_mapping_wrong_count(self, tmp_path: Path) -> None:
        """_validate_class_mapping should FAIL with wrong number of classes."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        mapping_path = output_dir / "metadata" / "class_mapping.json"
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        with open(mapping_path, "w") as f:
            json.dump({0: "BACKGROUND", 1: "SKIN"}, f)

        status, issues = _validate_class_mapping(output_dir)
        assert status == "FAIL"
        assert any("19" in i for i in issues)


# ---------------------------------------------------------------------------
# 7. Idempotent rerun
# ---------------------------------------------------------------------------


class TestIdempotentRerun:
    """Test that running preparation twice produces same results."""

    def test_check_artifacts_all_present(self, tmp_path: Path) -> None:
        """_check_artifacts should return all True when all artifacts exist."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _setup_sample(output_dir, "sample_0000", split="train")

        artifacts = _check_artifacts("sample_0000", output_dir)
        assert artifacts["aligned"] is True
        assert artifacts["initial_mask"] is True
        assert artifacts["mask"] is True
        assert artifacts["metadata"] is True
        assert artifacts["face_data"] is True

    def test_check_artifacts_missing_metadata(self, tmp_path: Path) -> None:
        """_check_artifacts should detect missing metadata."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _setup_sample(
            output_dir, "sample_0000", split="train", include_metadata=False
        )

        artifacts = _check_artifacts("sample_0000", output_dir)
        assert artifacts["aligned"] is True
        assert artifacts["initial_mask"] is True
        assert artifacts["metadata"] is False

    def test_deterministic_sample_ids(self, tmp_path: Path) -> None:
        """Sample IDs should be deterministic across runs."""
        images = [
            _make_image_entry("normal", "b.jpg"),
            _make_image_entry("normal", "a.jpg"),
            _make_image_entry("eyeglasses", "c.jpg"),
        ]
        sorted_images = _keep_images({"images": images})
        ids1 = _assign_sample_ids(sorted_images)
        ids2 = _assign_sample_ids(sorted_images)
        assert ids1 == ids2

    def test_split_counts_deterministic(self, tmp_path: Path) -> None:
        """Split file counts should be the same on rerun."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, f"sample_{i:04d}", split="train")
            for i in range(5)
        ]

        splits1 = _write_split_files(results, output_dir)
        # Read counts
        counts1 = {}
        for name in ("train", "val", "test"):
            p = output_dir / "splits" / f"{name}.txt"
            counts1[name] = sum(1 for l in p.read_text().splitlines() if l.strip())

        splits2 = _write_split_files(results, output_dir)
        counts2 = {}
        for name in ("train", "val", "test"):
            p = output_dir / "splits" / f"{name}.txt"
            counts2[name] = sum(1 for l in p.read_text().splitlines() if l.strip())

        assert counts1 == counts2


# ---------------------------------------------------------------------------
# 8. Missing metadata recovery
# ---------------------------------------------------------------------------


class TestMissingMetadataRecovery:
    """Test recovery when metadata is missing but other artifacts exist."""

    def test_check_artifacts_detects_missing_metadata(self, tmp_path: Path) -> None:
        """Should detect metadata as missing when aligned and mask exist."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _setup_sample(
            output_dir,
            "sample_0000",
            split="train",
            include_metadata=False,
        )

        artifacts = _check_artifacts("sample_0000", output_dir)
        assert artifacts["aligned"] is True
        assert artifacts["initial_mask"] is True
        assert artifacts["mask"] is True
        assert artifacts["metadata"] is False

    def test_face_data_sidecar_enables_metadata_recovery(self, tmp_path: Path) -> None:
        """face_data sidecar should allow metadata recovery without re-running alignment."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        _setup_sample(
            output_dir,
            "sample_0000",
            split="train",
            include_metadata=False,
            include_face_data=True,
        )

        artifacts = _check_artifacts("sample_0000", output_dir)
        assert artifacts["face_data"] is True

        # Verify face_data content
        face_data_path = output_dir / "face_data" / "sample_0000.json"
        with open(face_data_path) as f:
            fd = json.load(f)
        assert "face_bbox" in fd
        assert "face_kps" in fd
        assert "detection_score" in fd
        assert "face_area_ratio" in fd


# ---------------------------------------------------------------------------
# 9. Missing artifact recovery
# ---------------------------------------------------------------------------


class TestMissingArtifactRecovery:
    """Test recovery when some artifacts are missing."""

    def test_check_artifacts_all_missing(self, tmp_path: Path) -> None:
        """All artifacts should be detected as missing for a new sample."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        artifacts = _check_artifacts("nonexistent_sample", output_dir)
        assert artifacts["aligned"] is False
        assert artifacts["initial_mask"] is False
        assert artifacts["mask"] is False
        assert artifacts["metadata"] is False
        assert artifacts["face_data"] is False

    def test_check_artifacts_partial(self, tmp_path: Path) -> None:
        """Should correctly detect partial artifact presence."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Only create aligned image
        _create_aligned_image(output_dir / "images" / "sample_0000.png", "sample_0000")

        artifacts = _check_artifacts("sample_0000", output_dir)
        assert artifacts["aligned"] is True
        assert artifacts["initial_mask"] is False
        assert artifacts["mask"] is False
        assert artifacts["metadata"] is False


# ---------------------------------------------------------------------------
# 10. Ground-truth pending status
# ---------------------------------------------------------------------------


class TestGroundTruthPendingStatus:
    """Test that ground_truth_status is correctly set."""

    def test_manifest_ground_truth_status(self, tmp_path: Path) -> None:
        """Manifest samples must have ground_truth_status = initial_model_mask_pending_correction."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        results = [r]

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        sample = manifest["samples"][0]
        assert sample["ground_truth_status"] == "initial_model_mask_pending_correction"
        assert sample["annotation_status"] == "annotation_pending"

    def test_manifest_annotation_status_pending(self, tmp_path: Path) -> None:
        """All manifest samples must have annotation_status = annotation_pending."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, f"sample_{i:04d}", split="train")
            for i in range(5)
        ]

        manifest_dict = {"images": []}
        manifest = _build_manifest(results, output_dir, manifest_dict)

        for sample in manifest["samples"]:
            assert sample["annotation_status"] == "annotation_pending"

    def test_manifest_version_current_v1(self, tmp_path: Path) -> None:
        """Manifest version must be 'current_v1'."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        manifest_dict = {"images": []}
        manifest = _build_manifest([r], output_dir, manifest_dict)

        assert manifest["version"] == "current_v1"


# ---------------------------------------------------------------------------
# 11. Final READY_FOR_ANNOTATION status
# ---------------------------------------------------------------------------


class TestFinalStatus:
    """Test final READY_FOR_ANNOTATION / NOT_READY_FOR_ANNOTATION status."""

    def test_all_pass_gives_ready(self, tmp_path: Path) -> None:
        """All validations PASS should give READY_FOR_ANNOTATION."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
            _setup_sample(output_dir, "sample_0001", split="val"),
            _setup_sample(output_dir, "sample_0002", split="test"),
        ]

        manifest_dict = {"images": []}
        _build_manifest(results, output_dir, manifest_dict)
        _write_split_files(results, output_dir)
        _create_class_mapping(output_dir / "metadata" / "class_mapping.json")

        report = PreparationReport(
            manifest_validation="PASS",
            split_validation="PASS",
            metadata_validation="PASS",
            class_mapping_validation="PASS",
            artifact_validation="PASS",
        )

        from scripts.prepare_annotation import _write_reports

        _write_reports(report, output_dir)

        assert report.final_status == "READY_FOR_ANNOTATION"

    def test_any_fail_gives_not_ready(self, tmp_path: Path) -> None:
        """Any validation FAIL should give NOT_READY_FOR_ANNOTATION."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        report = PreparationReport(
            manifest_validation="PASS",
            split_validation="FAIL",
            metadata_validation="PASS",
            class_mapping_validation="PASS",
            artifact_validation="PASS",
        )

        from scripts.prepare_annotation import _write_reports

        _write_reports(report, output_dir)

        assert report.final_status == "NOT_READY_FOR_ANNOTATION"

    def test_report_json_contains_final_status(self, tmp_path: Path) -> None:
        """Report JSON must contain final_status field."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        report = PreparationReport(
            final_status="READY_FOR_ANNOTATION",
            manifest_validation="PASS",
            split_validation="PASS",
            metadata_validation="PASS",
            class_mapping_validation="PASS",
            artifact_validation="PASS",
        )

        from scripts.prepare_annotation import _write_reports

        _write_reports(report, output_dir)

        json_path = output_dir / "reports" / "annotation_preparation_report.json"
        with open(json_path) as f:
            data = json.load(f)
        assert data["final_status"] == "READY_FOR_ANNOTATION"

    def test_report_text_contains_final_status(self, tmp_path: Path) -> None:
        """Report text must contain FINAL STATUS line."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        report = PreparationReport(
            final_status="READY_FOR_ANNOTATION",
            manifest_validation="PASS",
            split_validation="PASS",
            metadata_validation="PASS",
            class_mapping_validation="PASS",
            artifact_validation="PASS",
        )

        from scripts.prepare_annotation import _write_reports

        _write_reports(report, output_dir)

        txt_path = output_dir / "reports" / "annotation_preparation_summary.txt"
        text = txt_path.read_text()
        assert "FINAL STATUS: READY_FOR_ANNOTATION" in text


# ---------------------------------------------------------------------------
# 12. Validation failure status
# ---------------------------------------------------------------------------


class TestValidationFailureStatus:
    """Test that validation failures are properly detected and reported."""

    def test_split_validation_fails_on_missing_file(self, tmp_path: Path) -> None:
        """Split validation should FAIL when split file is missing."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
        ]

        status, issues = _validate_splits(results, output_dir)
        assert status == "FAIL"
        assert any("missing" in i.lower() for i in issues)

    def test_manifest_validation_fails_on_missing_manifest(
        self, tmp_path: Path
    ) -> None:
        """Manifest validation should FAIL when manifest doesn't exist."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        results = [
            _setup_sample(output_dir, "sample_0000", split="train"),
        ]

        status, issues = _validate_manifest(results, output_dir)
        assert status == "FAIL"
        assert any("not found" in i for i in issues)

    def test_metadata_validation_fails_on_missing_metadata(
        self, tmp_path: Path
    ) -> None:
        """Metadata validation should FAIL when metadata file is missing."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(
            output_dir, "sample_0000", split="train", include_metadata=False
        )
        results = [r]

        manifest_dict = {"images": []}
        _build_manifest(results, output_dir, manifest_dict)

        status, issues = _validate_metadata(results, output_dir)
        assert status == "FAIL"
        assert any("missing" in i.lower() for i in issues)

    def test_artifact_validation_fails_on_missing_image(
        self, tmp_path: Path
    ) -> None:
        """Artifact validation should FAIL when aligned image file is missing on disk."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        # Now remove the aligned image to simulate it being missing on disk
        Path(r.aligned_path).unlink(missing_ok=True)

        results = [r]

        status, issues = _validate_artifacts(results, output_dir)
        assert status == "FAIL"
        assert any("missing" in i.lower() for i in issues)

    def test_validation_issues_appear_in_report(self, tmp_path: Path) -> None:
        """Validation issues should appear in the report."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        report = PreparationReport(
            validation_issues=["test_issue_1", "test_issue_2"],
            manifest_validation="FAIL",
            split_validation="PASS",
            metadata_validation="PASS",
            class_mapping_validation="PASS",
            artifact_validation="PASS",
        )

        from scripts.prepare_annotation import _write_reports

        _write_reports(report, output_dir)

        txt_path = output_dir / "reports" / "annotation_preparation_summary.txt"
        text = txt_path.read_text()
        assert "test_issue_1" in text
        assert "test_issue_2" in text

    def test_manifest_validation_fails_on_count_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Manifest validation should FAIL when total_samples doesn't match."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        r = _setup_sample(output_dir, "sample_0000", split="train")
        results = [r]

        # Write manifest with wrong count
        manifest_path = output_dir / "annotation_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump({
                "version": "current_v1",
                "total_samples": 999,
                "splits": {"train": 1, "val": 0, "test": 0},
                "samples": [{"sample_id": "sample_0000", "source_category": "normal",
                             "source_filename": "sample_0000.jpg",
                             "aligned_image": r.aligned_path,
                             "initial_mask": r.initial_mask_path,
                             "ground_truth_mask": r.mask_path,
                             "split": "train",
                             "annotation_status": "annotation_pending"}],
            }, f)

        status, issues = _validate_manifest(results, output_dir)
        assert status == "FAIL"
        assert any("999" in i for i in issues)

    def test_class_mapping_validation_in_report(self, tmp_path: Path) -> None:
        """Class mapping validation result should appear in the report."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        report = PreparationReport(
            class_mapping_validation="FAIL",
            manifest_validation="PASS",
            split_validation="PASS",
            metadata_validation="PASS",
            artifact_validation="PASS",
        )

        from scripts.prepare_annotation import _write_reports

        _write_reports(report, output_dir)

        txt_path = output_dir / "reports" / "annotation_preparation_summary.txt"
        text = txt_path.read_text()
        assert "Class mapping:          FAIL" in text
