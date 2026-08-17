"""
Incremental dataset collector for BiSeNet fine-tuning.

Orchestrates collection with:
- Existing image inventory
- Source-level deduplication (seen results)
- Perceptual deduplication (cross-source)
- Face filtering (single face only)
- Pagination support
- Persistent state
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imagehash
from PIL import Image

from dataset_builder.collection.collection_state import CollectionState
from dataset_builder.collection.duplicate_index import DuplicateIndex
from dataset_builder.collection.inventory import Inventory, CategoryInventory
from dataset_builder.config.collection_config import (
    CategoryConfig,
    CategoryType,
    CollectionConfig,
    Priority,
)
from dataset_builder.config.settings import Settings
from dataset_builder.queries.query_loader import QueryLoader
from dataset_builder.sources.base_source import BaseSource, DownloadResult, SearchResult


# ------------------------------------------------------------------
# Rejection reasons
# ------------------------------------------------------------------

class RejectionReason:
    """Constants for rejection reasons."""

    DUPLICATE = "duplicate"
    MULTIPLE_FACES = "multiple_faces"
    NO_FACE = "no_face"
    FACE_TOO_SMALL = "face_too_small"
    PROFILE_FACE = "profile_face"
    DECODE_FAILED = "decode_failed"
    DOWNLOAD_FAILED = "download_failed"
    SEEN_BEFORE = "seen_before"


# ------------------------------------------------------------------
# Collection statistics
# ------------------------------------------------------------------

@dataclass
class CategoryCollectionStats:
    """Statistics for a single category collection run."""

    category: str
    """Category name."""

    target: int = 0
    """Target number of images."""

    existing_valid: int = 0
    """Number of valid images already existing."""

    existing_duplicates: int = 0
    """Number of duplicates in existing images."""

    existing_invalid: int = 0
    """Number of invalid images in existing."""

    remaining: int = 0
    """Number of images still needed."""

    search_attempts: int = 0
    """Number of search results examined."""

    download_attempts: int = 0
    """Number of download attempts."""

    rejected_seen: int = 0
    """Rejected: already seen."""

    rejected_duplicate: int = 0
    """Rejected: perceptual duplicate."""

    rejected_multiple_faces: int = 0
    """Rejected: multiple faces detected."""

    rejected_no_face: int = 0
    """Rejected: no face detected."""

    rejected_face_too_small: int = 0
    """Rejected: face too small."""

    rejected_profile: int = 0
    """Rejected: profile face."""

    rejected_decode: int = 0
    """Rejected: decode/load failed."""

    rejected_download: int = 0
    """Rejected: download failed."""

    accepted_this_run: int = 0
    """Accepted during this collection run."""

    final_valid_unique: int = 0
    """Final count of valid unique images."""

    queries_executed: int = 0
    """Number of queries executed."""

    pages_searched: int = 0
    """Total pages searched across all queries."""


@dataclass
class CollectionStats:
    """Complete collection statistics."""

    mode: str = "dry_run"
    """Collection mode."""

    generated_at: str = ""
    """Timestamp of report generation."""

    total_target: int = 0
    """Total target across all categories."""

    total_existing_valid: int = 0
    """Total existing valid images."""

    total_remaining: int = 0
    """Total remaining images needed."""

    total_accepted: int = 0
    """Total accepted during this run."""

    categories: list[CategoryCollectionStats] = field(default_factory=list)
    """Per-category statistics."""


# ------------------------------------------------------------------
# Incremental collector
# ------------------------------------------------------------------

class IncrementalCollector:
    """Incremental dataset collector.

    Orchestrates collection with inventory, deduplication, face
    filtering, pagination, and persistent state.

    Parameters
    ----------
    settings:
        Application settings.
    config:
        Collection configuration.
    query_loader:
        Query loader for search queries.
    sources:
        List of validated source instances.
    """

    def __init__(
        self,
        settings: Settings,
        config: CollectionConfig,
        query_loader: QueryLoader,
        sources: list[BaseSource],
    ) -> None:
        self._settings: Settings = settings
        self._config: CollectionConfig = config
        self._query_loader: QueryLoader = query_loader
        self._sources: list[BaseSource] = sources

        # Initialize components
        self._state: CollectionState = CollectionState(
            settings.DATASET_DIR / "collection_state.json"
        )
        self._dup_index: DuplicateIndex = DuplicateIndex(settings)
        self._inventory: Inventory = Inventory(settings)

        # Load existing raw images into duplicate index
        self._load_existing_images()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inventory_category(
        self, category: str
    ) -> CategoryInventory:
        """Get inventory for a single category.

        Parameters
        ----------
        category:
            Category name.

        Returns
        -------
        CategoryInventory
            Inventory results.
        """
        return self._inventory.inventory_category(category)

    def calculate_remaining(
        self,
        category_config: CategoryConfig,
        category_limit: int | None = None,
    ) -> int:
        """Calculate remaining images needed for a category.

        Parameters
        ----------
        category_config:
            Category configuration.
        category_limit:
            Override for max images in this category.

        Returns
        -------
        int
            Number of additional images needed.
        """
        target = category_config.target_images
        if category_limit is not None:
            target = min(target, category_limit)

        cat_inv = self._inventory.inventory_category(category_config.name)
        existing_valid = cat_inv.unique_valid

        remaining = max(0, target - existing_valid)
        return remaining

    def collect_category(
        self,
        category_config: CategoryConfig,
        dry_run: bool = True,
        category_limit: int | None = None,
        max_pages: int | None = None,
    ) -> CategoryCollectionStats:
        """Collect images for a single category.

        Parameters
        ----------
        category_config:
            Category configuration.
        dry_run:
            If True, only simulate collection.
        category_limit:
            Override for max images in this category.
        max_pages:
            Override for max pages per query.

        Returns
        -------
        CategoryCollectionStats
            Collection statistics.
        """
        stats = CategoryCollectionStats(category=category_config.name)

        # Get existing inventory
        cat_inv = self._inventory.inventory_category(category_config.name)
        stats.existing_valid = cat_inv.unique_valid
        stats.existing_duplicates = cat_inv.duplicate_images
        stats.existing_invalid = cat_inv.invalid_images

        # Calculate target and remaining
        target = category_config.target_images
        if category_limit is not None:
            target = min(target, category_limit)

        stats.target = target
        stats.remaining = max(0, target - cat_inv.unique_valid)

        if stats.remaining == 0:
            stats.final_valid_unique = cat_inv.unique_valid
            return stats

        # Set state target
        self._state.set_category_target(category_config.name, target)

        # Get queries
        try:
            queries = self._query_loader.load_category(category_config.name)
        except (FileNotFoundError, ValueError):
            stats.final_valid_unique = cat_inv.unique_valid
            return stats

        # Limit queries
        queries = queries[: self._config.max_queries_per_category]

        # Add existing images to duplicate index
        self._add_existing_to_index(cat_inv)

        collected = 0
        pages_per_query = max_pages or self._settings.MAX_PAGES_PER_QUERY
        per_page = self._config.max_images_per_query

        for query in queries:
            if collected >= stats.remaining:
                break

            for source in self._sources:
                if collected >= stats.remaining:
                    break

                # Check if this query+source was fully processed
                if self._state.is_query_processed(
                    category_config.name, query, source.name
                ):
                    continue

                # Get starting page
                start_page = self._state.get_last_page(
                    category_config.name, query, source.name
                ) + 1

                for page in range(start_page, start_page + pages_per_query):
                    if collected >= stats.remaining:
                        break

                    stats.pages_searched += 1
                    stats.queries_executed += 1

                    try:
                        search_results = source.search(
                            query=query,
                            page=page,
                            per_page=per_page,
                        )
                    except Exception:
                        break

                    if not search_results:
                        # No more results, mark as fully processed
                        self._state.mark_query_processed(
                            category_config.name, query, source.name
                        )
                        break

                    for sr in search_results:
                        if collected >= stats.remaining:
                            break

                        stats.search_attempts += 1

                        # Check source-level dedup
                        if self._state.is_seen(source.name, sr.id):
                            stats.rejected_seen += 1
                            continue

                        # Check URL dedup
                        if self._state.is_url_seen(sr.download_url):
                            stats.rejected_seen += 1
                            self._state.record_seen(
                                source=source.name,
                                source_id=sr.id,
                                download_url=sr.download_url,
                                local_path="",
                                category=category_config.name,
                                query=query,
                                status="rejected_duplicate",
                                rejection_reason="url_seen",
                            )
                            continue

                        if dry_run:
                            # In dry run, just count — do NOT record in state
                            stats.accepted_this_run += 1
                            collected += 1
                            continue

                        # Download to temporary location
                        stats.download_attempts += 1
                        result = self._download_and_validate(
                            source=source,
                            result=sr,
                            category=category_config.name,
                            query=query,
                            stats=stats,
                        )

                        if result is not None:
                            collected += 1

                    # Save state after each page (only in real mode)
                    if not dry_run:
                        self._state.set_last_page(
                            category_config.name, query, source.name, page
                        )
                        self._state.save()

                # Mark query as processed if we exhausted pages (only in real mode)
                if collected < stats.remaining and not dry_run:
                    self._state.mark_query_processed(
                        category_config.name, query, source.name
                    )

        stats.final_valid_unique = cat_inv.unique_valid + collected
        if not dry_run:
            self._state.save()

        return stats

    def collect_all(
        self,
        dry_run: bool = True,
        categories: list[str] | None = None,
        category_limit: int | None = None,
        max_pages: int | None = None,
    ) -> CollectionStats:
        """Collect images for all selected categories.

        Parameters
        ----------
        dry_run:
            If True, only simulate collection.
        categories:
            List of categories to collect. If None, collects all.
        category_limit:
            Override for max images per category.
        max_pages:
            Override for max pages per query.

        Returns
        -------
        CollectionStats
            Complete collection statistics.
        """
        stats = CollectionStats(
            mode="dry_run" if dry_run else "collection",
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Determine categories
        if categories is None:
            categories = self._query_loader.categories()

        # Collect each category and accumulate totals
        for cat_name in categories:
            cat_config = self._config.get_category(cat_name)
            if cat_config is None:
                continue

            cat_stats = self.collect_category(
                category_config=cat_config,
                dry_run=dry_run,
                category_limit=category_limit,
                max_pages=max_pages,
            )

            stats.categories.append(cat_stats)
            stats.total_target += cat_stats.target
            stats.total_existing_valid += cat_stats.existing_valid
            stats.total_remaining += cat_stats.remaining
            stats.total_accepted += cat_stats.accepted_this_run

        return stats

    def save_state(self) -> None:
        """Persist collection state to disk."""
        self._state.save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_existing_images(self) -> None:
        """Load existing raw images into the duplicate index."""
        raw_dir = self._settings.RAW_IMAGES_DIR
        if not raw_dir.exists():
            return

        for cat_dir in raw_dir.iterdir():
            if not cat_dir.is_dir():
                continue

            for file_path in cat_dir.iterdir():
                if (
                    file_path.is_file()
                    and file_path.suffix.lower()
                    in self._settings.SUPPORTED_IMAGE_EXTENSIONS
                ):
                    self._dup_index.add_image(file_path)

    def _add_existing_to_index(self, cat_inv: CategoryInventory) -> None:
        """Add existing valid images to the duplicate index."""
        for img in cat_inv.images:
            if img.valid:
                self._dup_index.add_image(img.path)

    def _download_and_validate(
        self,
        source: BaseSource,
        result: SearchResult,
        category: str,
        query: str,
        stats: CategoryCollectionStats,
    ) -> Path | None:
        """Download, validate, and accept/reject a single image.

        Returns the path if accepted, None if rejected.
        """
        # Create temporary directory for download
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Download
            download_result = source.download(result, tmp_path)

            if not download_result.success:
                stats.rejected_download += 1
                self._state.record_seen(
                    source=source.name,
                    source_id=result.id,
                    download_url=result.download_url,
                    local_path="",
                    category=category,
                    query=query,
                    status="rejected_other",
                    rejection_reason=RejectionReason.DOWNLOAD_FAILED,
                )
                self._state.increment_rejected(category, "other")
                return None

            if download_result.local_path is None:
                stats.rejected_download += 1
                self._state.record_seen(
                    source=source.name,
                    source_id=result.id,
                    download_url=result.download_url,
                    local_path="",
                    category=category,
                    query=query,
                    status="rejected_other",
                    rejection_reason=RejectionReason.DOWNLOAD_FAILED,
                )
                self._state.increment_rejected(category, "other")
                return None

            local_path = download_result.local_path

            # Check perceptual duplicate
            if self._dup_index.is_duplicate(local_path):
                stats.rejected_duplicate += 1
                self._state.record_seen(
                    source=source.name,
                    source_id=result.id,
                    download_url=result.download_url,
                    local_path=str(local_path.relative_to(tmp_path)),
                    category=category,
                    query=query,
                    status="rejected_duplicate",
                    rejection_reason=RejectionReason.DUPLICATE,
                )
                self._state.increment_rejected(category, "duplicate")
                return None

            # Validate image (decode, face check)
            validation = self._validate_image(local_path)

            if not validation[0]:
                reason = validation[1]
                if reason == RejectionReason.NO_FACE:
                    stats.rejected_no_face += 1
                elif reason == RejectionReason.MULTIPLE_FACES:
                    stats.rejected_multiple_faces += 1
                elif reason == RejectionReason.FACE_TOO_SMALL:
                    stats.rejected_face_too_small += 1
                elif reason == RejectionReason.PROFILE_FACE:
                    stats.rejected_profile += 1
                else:
                    stats.rejected_decode += 1

                self._state.record_seen(
                    source=source.name,
                    source_id=result.id,
                    download_url=result.download_url,
                    local_path=str(local_path.relative_to(tmp_path)),
                    category=category,
                    query=query,
                    status="rejected_face",
                    rejection_reason=reason,
                )
                self._state.increment_rejected(category, "face")
                return None

            # Accept the image
            # Move from temp to raw directory
            dest_dir = self._settings.RAW_IMAGES_DIR / category
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Create filename
            ext = local_path.suffix or ".jpg"
            filename = f"{source.name}_{result.id}{ext}"
            dest_path = dest_dir / filename

            # Handle filename collision
            counter = 1
            while dest_path.exists():
                filename = f"{source.name}_{result.id}_{counter}{ext}"
                dest_path = dest_dir / filename
                counter += 1

            # Copy file
            import shutil
            shutil.copy2(str(local_path), str(dest_path))

            # Add to duplicate index
            self._dup_index.add_image(dest_path)

            # Record in state
            self._state.record_seen(
                source=source.name,
                source_id=result.id,
                download_url=result.download_url,
                local_path=str(dest_path.relative_to(self._settings.RAW_IMAGES_DIR)),
                category=category,
                query=query,
                status="accepted",
            )
            self._state.increment_accepted(category)

            stats.accepted_this_run += 1
            stats.rejected_multiple_faces += 0  # Reset for this image

            return dest_path

    def _validate_image(self, file_path: Path) -> tuple[bool, str | None]:
        """Validate an image file.

        Returns (is_valid, rejection_reason).
        """
        # Try to decode
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return False, RejectionReason.DECODE_FAILED
        except Exception:
            return False, RejectionReason.DECODE_FAILED

        # Check dimensions
        h, w = img.shape[:2]
        if w < self._settings.MIN_IMAGE_WIDTH or h < self._settings.MIN_IMAGE_HEIGHT:
            return False, RejectionReason.DECODE_FAILED

        # Face detection
        if self._config.require_face_detection:
            try:
                from pipeline.detector import FaceDetector

                detector = FaceDetector()
                faces = detector.detect(img)

                if len(faces) == 0:
                    return False, RejectionReason.NO_FACE

                if len(faces) > self._config.max_faces_per_image:
                    return False, RejectionReason.MULTIPLE_FACES

                # Check face size
                img_area = h * w
                largest_face = max(
                    faces,
                    key=lambda f: (
                        (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        if hasattr(f, "bbox")
                        else 0
                    ),
                )

                if hasattr(largest_face, "bbox"):
                    x1, y1, x2, y2 = largest_face.bbox[:4]
                    face_area = (x2 - x1) * (y2 - y1)
                    face_ratio = face_area / img_area if img_area > 0 else 0

                    if face_ratio < self._settings.MIN_FACE_AREA_RATIO:
                        return False, RejectionReason.FACE_TOO_SMALL

                # Check profile face
                if not self._settings.ALLOW_PROFILE_FACES:
                    pose = getattr(largest_face, "pose", None)
                    if pose is not None:
                        try:
                            _pitch, yaw, _roll = pose
                            if abs(float(yaw)) > self._settings.MAX_PROFILE_YAW_DEGREES:
                                return False, RejectionReason.PROFILE_FACE
                        except (TypeError, ValueError):
                            pass

            except ImportError:
                # FaceDetector not available, skip face check
                pass
            except Exception:
                # Face detection failed, skip
                pass

        return True, None
