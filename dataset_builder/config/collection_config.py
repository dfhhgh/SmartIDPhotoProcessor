"""
Collection configuration for controlled dataset expansion.

Defines the collection matrix: target counts, source preferences,
and category priorities for the BiSeNet fine-tuning dataset.

This configuration is designed to produce a balanced dataset that:
1. Preserves baseline parser behavior (normal category)
2. Provides sufficient target-condition examples
3. Avoids catastrophic forgetting during fine-tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CategoryType(Enum):
    """Whether a category is a target condition or baseline preservation."""

    TARGET = "target"
    """Target condition for fine-tuning improvement."""

    BASELINE = "baseline"
    """Baseline category to preserve pretrained behavior."""


class Priority(Enum):
    """Collection priority level."""

    HIGH = "high"
    """Critical for model performance."""

    MEDIUM = "medium"
    """Important but not critical."""

    LOW = "low"
    """Nice to have but not essential."""


@dataclass(frozen=True)
class CategoryConfig:
    """Configuration for a single dataset category."""

    name: str
    """Category identifier matching the query .txt filename."""

    category_type: CategoryType
    """Whether this is a target condition or baseline."""

    target_images: int
    """Target number of images to collect for this category."""

    percentage: float
    """Percentage of total dataset this category represents."""

    priority: Priority
    """Collection priority level."""

    preferred_sources: tuple[str, ...] = (
        "pexels",
        "pixabay",
        "openverse",
        "wikimedia_commons",
    )
    """Preferred sources in order of preference. Empty tuple = all sources."""

    description: str = ""
    """Brief description of what this category captures."""


@dataclass(frozen=True)
class CollectionConfig:
    """Master configuration for dataset collection.

    Defines the complete collection matrix with target counts,
    source preferences, and limits for controlled collection.
    """

    # ------------------------------------------------------------------
    # Collection limits
    # ------------------------------------------------------------------

    pilot_images_per_category: int = 30
    """Target images per category for the pilot collection."""

    max_images_per_category: int = 100
    """Maximum images per category for full collection."""

    max_images_per_query: int = 5
    """Maximum images to download per individual query."""

    max_queries_per_category: int = 15
    """Maximum queries to execute per category."""

    # ------------------------------------------------------------------
    # Source configuration
    # ------------------------------------------------------------------

    enabled_sources: tuple[str, ...] = (
        "pexels",
        "pixabay",
        "openverse",
        "wikimedia_commons",
    )
    """Sources to use for collection."""

    # ------------------------------------------------------------------
    # Category definitions
    # ------------------------------------------------------------------

    categories: tuple[CategoryConfig, ...] = field(
        default_factory=lambda: (
            # === BASELINE/PRESERVATION CATEGORIES ===
            CategoryConfig(
                name="normal",
                category_type=CategoryType.BASELINE,
                target_images=100,
                percentage=20.0,
                priority=Priority.HIGH,
                description="Unobstructed faces with all 19 parser classes visible",
            ),
            # === TARGET CONDITION CATEGORIES ===
            CategoryConfig(
                name="eyeglasses",
                category_type=CategoryType.TARGET,
                target_images=60,
                percentage=12.0,
                priority=Priority.HIGH,
                description="Faces with clear eyeglasses covering eyes",
            ),
            CategoryConfig(
                name="sunglasses",
                category_type=CategoryType.TARGET,
                target_images=60,
                percentage=12.0,
                priority=Priority.HIGH,
                description="Faces with dark sunglasses covering eyes",
            ),
            CategoryConfig(
                name="hijab",
                category_type=CategoryType.TARGET,
                target_images=50,
                percentage=10.0,
                priority=Priority.MEDIUM,
                description="Faces with hijab/headscarf covering hair and neck",
            ),
            CategoryConfig(
                name="mask",
                category_type=CategoryType.TARGET,
                target_images=50,
                percentage=10.0,
                priority=Priority.MEDIUM,
                description="Faces with surgical/cloth mask covering nose and mouth",
            ),
            CategoryConfig(
                name="cap",
                category_type=CategoryType.TARGET,
                target_images=50,
                percentage=10.0,
                priority=Priority.MEDIUM,
                description="Faces with cap/hat covering forehead and hair",
            ),
            CategoryConfig(
                name="beard",
                category_type=CategoryType.TARGET,
                target_images=40,
                percentage=8.0,
                priority=Priority.MEDIUM,
                description="Faces with beard/mustache covering lower face",
            ),
            CategoryConfig(
                name="helmet",
                category_type=CategoryType.TARGET,
                target_images=40,
                percentage=8.0,
                priority=Priority.LOW,
                description="Faces with hard helmet covering top of head",
            ),
            CategoryConfig(
                name="scarf",
                category_type=CategoryType.TARGET,
                target_images=30,
                percentage=6.0,
                priority=Priority.LOW,
                description="Faces with scarf covering neck and lower face",
            ),
            CategoryConfig(
                name="hair_occlusion",
                category_type=CategoryType.TARGET,
                target_images=20,
                percentage=4.0,
                priority=Priority.LOW,
                description="Faces with hair partially covering facial features",
            ),
        )
    )
    """Category configurations defining the collection matrix."""

    # ------------------------------------------------------------------
    # Quality gates
    # ------------------------------------------------------------------

    min_image_width: int = 640
    """Minimum image width in pixels."""

    min_image_height: int = 480
    """Minimum image height in pixels."""

    require_face_detection: bool = True
    """Whether to require face detection before accepting an image."""

    max_faces_per_image: int = 1
    """Maximum number of faces allowed per image."""

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    enable_cross_source_duplicates: bool = True
    """Whether to detect duplicates across sources."""

    duplicate_distance_threshold: int = 5
    """Hamming distance threshold for perceptual hash duplicates."""

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def total_target_images(self) -> int:
        """Total target images across all categories."""
        return sum(cat.target_images for cat in self.categories)

    @property
    def target_condition_count(self) -> int:
        """Number of target condition categories."""
        return sum(
            1
            for cat in self.categories
            if cat.category_type == CategoryType.TARGET
        )

    @property
    def baseline_count(self) -> int:
        """Number of baseline preservation categories."""
        return sum(
            1
            for cat in self.categories
            if cat.category_type == CategoryType.BASELINE
        )

    def get_category(self, name: str) -> CategoryConfig | None:
        """Get configuration for a specific category.

        Parameters
        ----------
        name:
            Category name.

        Returns
        -------
        CategoryConfig or None
            Category configuration, or None if not found.
        """
        for cat in self.categories:
            if cat.name == name:
                return cat
        return None

    def get_categories_by_type(
        self, category_type: CategoryType
    ) -> tuple[CategoryConfig, ...]:
        """Get all categories of a specific type.

        Parameters
        ----------
        category_type:
            Type to filter by.

        Returns
        -------
        tuple[CategoryConfig, ...]
            Matching categories.
        """
        return tuple(
            cat for cat in self.categories if cat.category_type == category_type
        )

    def get_categories_by_priority(
        self, priority: Priority
    ) -> tuple[CategoryConfig, ...]:
        """Get all categories with a specific priority.

        Parameters
        ----------
        priority:
            Priority level to filter by.

        Returns
        -------
        tuple[CategoryConfig, ...]
            Matching categories.
        """
        return tuple(
            cat for cat in self.categories if cat.priority == priority
        )

    def get_pilot_limits(self) -> dict[str, int]:
        """Get per-category limits for pilot collection.

        Returns
        -------
        dict[str, int]
            Mapping of category name to pilot image limit.
        """
        return {
            cat.name: min(cat.target_images, self.pilot_images_per_category)
            for cat in self.categories
        }

    def get_full_limits(self) -> dict[str, int]:
        """Get per-category limits for full collection.

        Returns
        -------
        dict[str, int]
            Mapping of category name to full collection image limit.
        """
        return {
            cat.name: min(cat.target_images, self.max_images_per_category)
            for cat in self.categories
        }

    def summary(self) -> str:
        """Return a human-readable summary of the collection plan."""
        lines = [
            "Collection Configuration Summary",
            "=" * 40,
            f"Total target images: {self.total_target_images}",
            f"Target conditions: {self.target_condition_count}",
            f"Baseline categories: {self.baseline_count}",
            f"Enabled sources: {', '.join(self.enabled_sources)}",
            "",
            "Category Distribution:",
        ]

        for cat in self.categories:
            type_label = "BASELINE" if cat.category_type == CategoryType.BASELINE else "TARGET"
            lines.append(
                f"  {cat.name:20s} {type_label:8s} "
                f"{cat.target_images:4d} images ({cat.percentage:5.1f}%) "
                f"[{cat.priority.value}]"
            )

        lines.append("")
        lines.append(f"Pilot limit: {self.pilot_images_per_category} images/category")
        lines.append(f"Max per query: {self.max_images_per_query}")
        lines.append(f"Max queries/category: {self.max_queries_per_category}")

        return "\n".join(lines)
