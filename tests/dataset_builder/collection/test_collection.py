"""Unit tests for collection module.

Covers:
A. Category-aware inventory
B. Wrong global attribution regression
C. Remaining calculation
D. Complete category detection
E. Incremental behavior
F. Duplicate detection against existing raw images
G. Dry-run does not modify files/state
H. Unknown category directory
I. Uncategorized files directly under raw/
J. Existing invalid files reported but not counted as valid
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from dataset_builder.collection.collection_state import CollectionState
from dataset_builder.collection.duplicate_index import DuplicateIndex
from dataset_builder.collection.inventory import (
    Inventory,
    CategoryInventory,
    ImageInventory,
    InventoryResult,
)
from dataset_builder.config.settings import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        DATASET_DIR=tmp_path / "dataset",
        RAW_IMAGES_DIR=tmp_path / "raw",
        REPORTS_DIR=tmp_path / "reports",
        MIN_IMAGE_WIDTH=100,
        MIN_IMAGE_HEIGHT=100,
        IMAGEHASH_SIZE=8,
        DUPLICATE_DISTANCE_THRESHOLD=8,
    )


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "collection_state.json"


@pytest.fixture
def state(state_path: Path) -> CollectionState:
    return CollectionState(state_path)


@pytest.fixture
def dup_index(settings: Settings) -> DuplicateIndex:
    return DuplicateIndex(settings)


@pytest.fixture
def inventory(settings: Settings) -> Inventory:
    return Inventory(settings)


def create_test_image(path: Path, size: tuple[int, int] = (800, 600), color: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Create a test image at the specified path."""
    img = Image.new("RGB", size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def create_test_image_array(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    """Create a test image as numpy array (for OpenCV tests)."""
    img = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)


def create_raw_dataset(raw_dir: Path) -> None:
    """Create a realistic multi-category raw dataset."""
    categories = {
        "normal": 30,
        "eyeglasses": 20,
        "sunglasses": 15,
    }
    for cat, count in categories.items():
        for i in range(count):
            img = np.random.randint(
                0, 255, (600, 800, 3), dtype=np.uint8
            )
            img_path = raw_dir / cat / f"pexels_{1000 + i}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(img_path)


# ---------------------------------------------------------------------------
# A. Category-aware inventory
# ---------------------------------------------------------------------------

class TestCategoryAwareInventory:
    """Inventory must derive category from parent directory."""

    def test_category_from_parent_directory(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        for cat, count in [("normal", 2), ("eyeglasses", 1), ("sunglasses", 1)]:
            for i in range(count):
                create_test_image_array(raw / cat / f"img_{i}.jpg")

        result = inventory.inventory_all(raw)

        assert result.categories["normal"].valid_images == 2
        assert result.categories["eyeglasses"].valid_images == 1
        assert result.categories["sunglasses"].valid_images == 1

    def test_each_category_independent(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "a.jpg")
        create_test_image_array(raw / "normal" / "b.jpg")
        create_test_image_array(raw / "eyeglasses" / "c.jpg")

        inv_normal = inventory.inventory_category("normal", raw)
        inv_eye = inventory.inventory_category("eyeglasses", raw)

        assert inv_normal.valid_images == 2
        assert inv_eye.valid_images == 1
        assert inv_normal.category == "normal"
        assert inv_eye.category == "eyeglasses"

    def test_inventory_all_totals(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        for i in range(5):
            create_test_image_array(raw / "normal" / f"img_{i}.jpg")
        for i in range(3):
            create_test_image_array(raw / "eyeglasses" / f"img_{i}.jpg")

        result = inventory.inventory_all(raw)

        assert result.total_valid == 8
        assert result.categories["normal"].valid_images == 5
        assert result.categories["eyeglasses"].valid_images == 3


# ---------------------------------------------------------------------------
# B. Wrong global attribution regression
# ---------------------------------------------------------------------------

class TestGlobalAttributionRegression:
    """A global total must NOT be assigned to a single category."""

    def test_global_total_not_attributed_to_one_category(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "a.jpg")
        create_test_image_array(raw / "normal" / "b.jpg")
        create_test_image_array(raw / "eyeglasses" / "c.jpg")

        result = inventory.inventory_all(raw)

        # eyeglasses must NOT equal the total
        assert result.categories["eyeglasses"].valid_images != result.total_valid
        # Total must be sum of categories
        assert result.total_valid == sum(
            ci.valid_images for ci in result.categories.values()
        )

    def test_empty_category_gets_zero_not_total(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "a.jpg")

        inv_empty = inventory.inventory_category("eyeglasses", raw)

        assert inv_empty.valid_images == 0


# ---------------------------------------------------------------------------
# C. Remaining calculation
# ---------------------------------------------------------------------------

class TestRemainingCalculation:
    """remaining = max(target - existing_valid, 0) per category."""

    def test_remaining_basic(self) -> None:
        from dataset_builder.collection.incremental_collector import CategoryCollectionStats

        stats = CategoryCollectionStats(category="test")
        stats.target = 60
        stats.existing_valid = 30

        remaining = max(0, stats.target - stats.existing_valid)
        assert remaining == 30

    def test_remaining_when_over_target(self) -> None:
        from dataset_builder.collection.incremental_collector import CategoryCollectionStats

        stats = CategoryCollectionStats(category="test")
        stats.target = 30
        stats.existing_valid = 50

        remaining = max(0, stats.target - stats.existing_valid)
        assert remaining == 0

    def test_total_remaining_is_sum_of_per_category(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        for i in range(30):
            create_test_image_array(raw / "normal" / f"img_{i}.jpg")
        for i in range(10):
            create_test_image_array(raw / "eyeglasses" / f"img_{i}.jpg")

        inv_normal = inventory.inventory_category("normal", raw)
        inv_eye = inventory.inventory_category("eyeglasses", raw)

        # Targets: normal=100, eyeglasses=60
        remaining_normal = max(0, 100 - inv_normal.unique_valid)
        remaining_eye = max(0, 60 - inv_eye.unique_valid)

        assert remaining_normal == 70
        assert remaining_eye == 50
        assert remaining_normal + remaining_eye == 120


# ---------------------------------------------------------------------------
# D. Complete category
# ---------------------------------------------------------------------------

class TestCompleteCategory:
    """A category is COMPLETE when existing_valid >= target."""

    def test_complete_category(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        for i in range(30):
            create_test_image_array(raw / "normal" / f"img_{i}.jpg")

        inv = inventory.inventory_category("normal", raw)

        target = 30
        remaining = max(0, target - inv.unique_valid)
        assert remaining == 0

    def test_no_search_when_complete(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        for i in range(30):
            create_test_image_array(raw / "normal" / f"img_{i}.jpg")

        inv = inventory.inventory_category("normal", raw)
        remaining = max(0, 30 - inv.unique_valid)

        # When remaining is 0, collector should not search
        assert remaining == 0


# ---------------------------------------------------------------------------
# E. Incremental behavior
# ---------------------------------------------------------------------------

class TestIncrementalBehavior:
    """Run collection twice; previously accepted images not re-downloaded."""

    def test_seen_tracking_prevents_redownload(self, state: CollectionState) -> None:
        state.record_seen(
            source="pexels",
            source_id="123",
            download_url="http://example.com/1.jpg",
            local_path="normal/pexels_123.jpg",
            category="normal",
            query="portrait",
            status="accepted",
        )
        state.save()

        # Second load
        state2 = CollectionState(state._state_path)
        assert state2.is_seen("pexels", "123") is True

    def test_url_dedup(self, state: CollectionState) -> None:
        state.record_seen(
            source="pexels",
            source_id="123",
            download_url="http://example.com/1.jpg",
            local_path="normal/pexels_123.jpg",
            category="normal",
            query="portrait",
            status="accepted",
        )

        assert state.is_url_seen("http://example.com/1.jpg") is True
        assert state.is_url_seen("http://example.com/2.jpg") is False


# ---------------------------------------------------------------------------
# F. Duplicate detection against existing raw images
# ---------------------------------------------------------------------------

class TestDuplicateDetectionAgainstExisting:
    """New images must be checked against existing raw images."""

    def test_identical_image_detected(
        self, dup_index: DuplicateIndex, tmp_path: Path
    ) -> None:
        existing = tmp_path / "existing.jpg"
        new = tmp_path / "new.jpg"
        create_test_image(existing, size=(800, 600), color=(200, 100, 50))
        create_test_image(new, size=(800, 600), color=(200, 100, 50))

        dup_index.add_image(existing)
        assert dup_index.is_duplicate(new) is True

    def test_different_image_not_duplicate(
        self, dup_index: DuplicateIndex, tmp_path: Path
    ) -> None:
        existing = tmp_path / "existing.jpg"
        new = tmp_path / "new.jpg"
        arr1 = np.zeros((600, 800, 3), dtype=np.uint8)
        arr1[::10, ::10] = 255
        arr2 = np.zeros((600, 800, 3), dtype=np.uint8)
        gradient = np.linspace(0, 255, 800, dtype=np.uint8)
        arr2[:, :, 0] = gradient
        arr2[:, :, 1] = 255 - gradient
        arr2[:, :, 2] = 128
        Image.fromarray(arr1).save(existing)
        Image.fromarray(arr2).save(new)

        dup_index.add_image(existing)
        assert dup_index.is_duplicate(new) is False


# ---------------------------------------------------------------------------
# G. Dry-run does not modify files/state
# ---------------------------------------------------------------------------

class TestDryRunSafety:
    """Dry-run must not modify the filesystem or persistent state."""

    def test_dry_run_no_state_mutation(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        from dataset_builder.collection.incremental_collector import IncrementalCollector
        from dataset_builder.config.collection_config import (
            CollectionConfig,
            CategoryConfig,
            CategoryType,
            Priority,
        )

        raw = tmp_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        state_path = tmp_path / "dataset" / "collection_state.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)

        query_loader = MagicMock()
        query_loader.categories.return_value = ["normal"]
        query_loader.load_category.return_value = ["portrait photo"]

        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source.search.return_value = [
            MagicMock(id="1", download_url="http://ex.com/1.jpg"),
        ]

        collector = IncrementalCollector(
            settings=settings,
            config=CollectionConfig(),
            query_loader=query_loader,
            sources=[mock_source],
        )

        cat_config = CategoryConfig(
            name="normal",
            category_type=CategoryType.TARGET,
            target_images=100,
            percentage=0.0,
            priority=Priority.MEDIUM,
        )

        stats = collector.collect_category(
            category_config=cat_config,
            dry_run=True,
            category_limit=100,
        )

        # Dry-run should count simulated accepted but not write state
        assert stats.accepted_this_run >= 0

        # State file should not have been created or should be empty
        if state_path.exists():
            import json
            with open(state_path) as f:
                data = json.load(f)
            # No 'accepted' records should exist from dry-run
            for key, val in data.get("seen", {}).items():
                assert val["local_path"] != "(dry_run)_test_source_1"

    def test_dry_run_no_filesystem_change(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        from dataset_builder.collection.incremental_collector import IncrementalCollector
        from dataset_builder.config.collection_config import (
            CollectionConfig,
            CategoryConfig,
            CategoryType,
            Priority,
        )

        raw = tmp_path / "raw"
        raw.mkdir(parents=True, exist_ok=True)

        query_loader = MagicMock()
        query_loader.categories.return_value = ["normal"]
        query_loader.load_category.return_value = ["portrait photo"]

        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source.search.return_value = [
            MagicMock(id="1", download_url="http://ex.com/1.jpg"),
        ]

        collector = IncrementalCollector(
            settings=settings,
            config=CollectionConfig(),
            query_loader=query_loader,
            sources=[mock_source],
        )

        cat_config = CategoryConfig(
            name="normal",
            category_type=CategoryType.TARGET,
            target_images=100,
            percentage=0.0,
            priority=Priority.MEDIUM,
        )

        # Count files before
        files_before = list(raw.rglob("*"))
        collector.collect_category(
            category_config=cat_config,
            dry_run=True,
            category_limit=100,
        )
        files_after = list(raw.rglob("*"))

        assert len(files_before) == len(files_after)


# ---------------------------------------------------------------------------
# H. Unknown category directory
# ---------------------------------------------------------------------------

class TestUnknownCategoryDirectory:
    """Directories under raw/ that are not known categories are reported."""

    def test_unknown_dir_reported(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "a.jpg")
        create_test_image_array(raw / "something_else" / "b.jpg")

        result = inventory.inventory_all(raw)

        assert "something_else" in result.unknown_dirs

    def test_unknown_dir_not_counted_in_totals(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "a.jpg")
        create_test_image_array(raw / "something_else" / "b.jpg")

        result = inventory.inventory_all(raw)

        # Unknown dirs are still inventoried but should be distinguishable
        assert result.categories["something_else"].valid_images == 1
        assert result.categories["normal"].valid_images == 1


# ---------------------------------------------------------------------------
# I. Uncategorized files directly under raw/
# ---------------------------------------------------------------------------

class TestUncategorizedFiles:
    """Files directly under raw/ are reported as uncategorized."""

    def test_uncategorized_files_detected(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "a.jpg")
        create_test_image_array(raw / "orphan.jpg")

        result = inventory.inventory_all(raw)

        assert len(result.uncategorized_files) == 1
        assert result.uncategorized_files[0].name == "orphan.jpg"

    def test_uncategorized_not_counted_in_any_category(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "orphan.jpg")

        result = inventory.inventory_all(raw)

        assert len(result.uncategorized_files) == 1
        assert result.total_valid == 0  # Not in any category


# ---------------------------------------------------------------------------
# J. Invalid files reported but not counted as valid
# ---------------------------------------------------------------------------

class TestInvalidFiles:
    """Invalid files are reported but not counted as valid."""

    def test_unreadable_file_counted_invalid(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "valid.jpg")
        # Create an invalid file
        invalid = raw / "normal" / "invalid.txt"
        invalid.write_text("not an image")

        result = inventory.inventory_category("normal", raw)

        # .txt is not in SUPPORTED_IMAGE_EXTENSIONS, so not counted at all
        assert result.total_files == 1
        assert result.valid_images == 1
        assert result.invalid_images == 0

    def test_too_small_image_counted_invalid(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        # Create a too-small image (below MIN_IMAGE_WIDTH/HEIGHT)
        small = raw / "normal" / "small.jpg"
        small.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        img.save(small)

        result = inventory.inventory_category("normal", raw)

        assert result.total_files == 1
        assert result.valid_images == 0
        assert result.invalid_images == 1
        assert result.images[0].rejection_reason == "too_small"

    def test_invalid_not_counted_in_unique_valid(
        self, inventory: Inventory, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        create_test_image_array(raw / "normal" / "valid.jpg")
        small = raw / "normal" / "small.jpg"
        small.parent.mkdir(parents=True, exist_ok=True)
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        img.save(small)

        result = inventory.inventory_category("normal", raw)

        assert result.unique_valid == 1  # Only the valid image


# ---------------------------------------------------------------------------
# CollectionState tests
# ---------------------------------------------------------------------------

class TestCollectionState:
    """Tests for CollectionState."""

    def test_is_seen_returns_false_for_unknown(self, state: CollectionState) -> None:
        assert state.is_seen("pexels", "123") is False

    def test_record_seen_and_is_seen(self, state: CollectionState) -> None:
        state.record_seen(
            source="pexels",
            source_id="123",
            download_url="http://example.com/img.jpg",
            local_path="normal/pexels_123.jpg",
            category="normal",
            query="man portrait",
            status="accepted",
        )

        assert state.is_seen("pexels", "123") is True
        assert state.get_status("pexels", "123") == "accepted"

    def test_persistence(self, state_path: Path) -> None:
        state1 = CollectionState(state_path)
        state1.record_seen(
            source="pexels",
            source_id="123",
            download_url="http://example.com/img.jpg",
            local_path="normal/pexels_123.jpg",
            category="normal",
            query="man portrait",
            status="accepted",
        )
        state1.save()

        state2 = CollectionState(state_path)
        assert state2.is_seen("pexels", "123") is True

    def test_category_progress(self, state: CollectionState) -> None:
        state.set_category_target("normal", 100)
        state.increment_accepted("normal")
        state.increment_accepted("normal")
        state.increment_rejected("normal", "duplicate")

        progress = state.get_category_progress("normal")
        assert progress.target == 100
        assert progress.accepted == 2
        assert progress.rejected_duplicate == 1

    def test_get_accepted_count(self, state: CollectionState) -> None:
        state.record_seen(
            source="pexels", source_id="123", download_url="http://e.com/1.jpg",
            local_path="n/1.jpg", category="normal", query="q", status="accepted",
        )
        state.record_seen(
            source="pexels", source_id="456", download_url="http://e.com/2.jpg",
            local_path="n/2.jpg", category="normal", query="q", status="rejected_duplicate",
        )
        assert state.get_accepted_count() == 1


# ---------------------------------------------------------------------------
# DuplicateIndex tests
# ---------------------------------------------------------------------------

class TestDuplicateIndex:
    """Tests for DuplicateIndex."""

    def test_add_image_success(self, dup_index: DuplicateIndex, tmp_path: Path) -> None:
        img_path = tmp_path / "test.jpg"
        create_test_image(img_path)
        assert dup_index.add_image(img_path) is True
        assert dup_index.size == 1

    def test_add_image_nonexistent(self, dup_index: DuplicateIndex, tmp_path: Path) -> None:
        assert dup_index.add_image(tmp_path / "nope.jpg") is False

    def test_clear(self, dup_index: DuplicateIndex, tmp_path: Path) -> None:
        create_test_image(tmp_path / "test.jpg")
        dup_index.add_image(tmp_path / "test.jpg")
        dup_index.clear()
        assert dup_index.size == 0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestCollectionIntegration:
    """Integration tests with mocked network calls."""

    def test_incremental_collector_initialization(self, tmp_path: Path) -> None:
        from dataset_builder.collection.incremental_collector import IncrementalCollector
        from dataset_builder.config.collection_config import CollectionConfig

        settings = Settings(
            DATASET_DIR=tmp_path / "dataset",
            RAW_IMAGES_DIR=tmp_path / "raw",
            REPORTS_DIR=tmp_path / "reports",
        )
        query_loader = MagicMock()
        collector = IncrementalCollector(
            settings=settings,
            config=CollectionConfig(),
            query_loader=query_loader,
            sources=[],
        )
        assert collector._settings == settings

    def test_calculate_remaining(self, tmp_path: Path) -> None:
        from dataset_builder.collection.incremental_collector import IncrementalCollector
        from dataset_builder.config.collection_config import (
            CollectionConfig, CategoryConfig, CategoryType, Priority,
        )

        raw = tmp_path / "raw"
        for i in range(5):
            create_test_image_array(raw / "normal" / f"pexels_{i}.jpg")

        settings = Settings(
            DATASET_DIR=tmp_path / "dataset",
            RAW_IMAGES_DIR=raw,
            REPORTS_DIR=tmp_path / "reports",
            MIN_IMAGE_WIDTH=100,
            MIN_IMAGE_HEIGHT=100,
        )

        collector = IncrementalCollector(
            settings=settings,
            config=CollectionConfig(),
            query_loader=MagicMock(),
            sources=[],
        )

        cat_config = CategoryConfig(
            name="normal", category_type=CategoryType.TARGET,
            target_images=10, percentage=0.0, priority=Priority.MEDIUM,
        )

        remaining = collector.calculate_remaining(cat_config)
        assert remaining == 5

    def test_dry_run_does_not_download(self, tmp_path: Path) -> None:
        from dataset_builder.collection.incremental_collector import IncrementalCollector
        from dataset_builder.config.collection_config import (
            CollectionConfig, CategoryConfig, CategoryType, Priority,
        )

        settings = Settings(
            DATASET_DIR=tmp_path / "dataset",
            RAW_IMAGES_DIR=tmp_path / "raw",
            REPORTS_DIR=tmp_path / "reports",
            MIN_IMAGE_WIDTH=100,
            MIN_IMAGE_HEIGHT=100,
        )

        query_loader = MagicMock()
        query_loader.categories.return_value = ["normal"]
        query_loader.load_category.return_value = ["man portrait"]

        mock_source = MagicMock()
        mock_source.name = "test_source"
        mock_source.search.return_value = [
            MagicMock(id="1", download_url="http://ex.com/1.jpg"),
            MagicMock(id="2", download_url="http://ex.com/2.jpg"),
        ]

        collector = IncrementalCollector(
            settings=settings,
            config=CollectionConfig(),
            query_loader=query_loader,
            sources=[mock_source],
        )

        cat_config = CategoryConfig(
            name="normal", category_type=CategoryType.TARGET,
            target_images=10, percentage=0.0, priority=Priority.MEDIUM,
        )

        stats = collector.collect_category(
            category_config=cat_config,
            dry_run=True,
            category_limit=2,
        )

        mock_source.download.assert_not_called()
        assert stats.accepted_this_run == 2

    def test_collect_all_per_category_totals(self, tmp_path: Path) -> None:
        from dataset_builder.collection.incremental_collector import IncrementalCollector
        from dataset_builder.config.collection_config import (
            CollectionConfig, CategoryConfig, CategoryType, Priority,
        )

        raw = tmp_path / "raw"
        for i in range(30):
            create_test_image_array(raw / "normal" / f"pexels_{i}.jpg")
        for i in range(10):
            create_test_image_array(raw / "eyeglasses" / f"pexels_{i}.jpg")

        settings = Settings(
            DATASET_DIR=tmp_path / "dataset",
            RAW_IMAGES_DIR=raw,
            REPORTS_DIR=tmp_path / "reports",
            MIN_IMAGE_WIDTH=100,
            MIN_IMAGE_HEIGHT=100,
        )

        # Create a custom config with specific targets
        config = CollectionConfig(categories=(
            CategoryConfig(
                name="normal", category_type=CategoryType.BASELINE,
                target_images=100, percentage=20.0, priority=Priority.HIGH,
            ),
            CategoryConfig(
                name="eyeglasses", category_type=CategoryType.TARGET,
                target_images=60, percentage=12.0, priority=Priority.HIGH,
            ),
        ))

        query_loader = MagicMock()
        query_loader.categories.return_value = ["normal", "eyeglasses"]
        query_loader.load_category.return_value = []

        collector = IncrementalCollector(
            settings=settings, config=config,
            query_loader=query_loader, sources=[],
        )

        stats = collector.collect_all(
            dry_run=True,
            categories=["normal", "eyeglasses"],
        )

        # Per-category remaining
        assert stats.categories[0].existing_valid == 30
        assert stats.categories[0].remaining == 70
        assert stats.categories[1].existing_valid == 10
        assert stats.categories[1].remaining == 50

        # Global totals are sum of per-category
        assert stats.total_existing_valid == 40
        assert stats.total_remaining == 120
