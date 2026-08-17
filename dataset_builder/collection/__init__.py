"""Collection module for incremental dataset building."""

from dataset_builder.collection.collection_state import CollectionState
from dataset_builder.collection.duplicate_index import DuplicateIndex
from dataset_builder.collection.inventory import (
    Inventory,
    InventoryResult,
    CategoryInventory,
)
from dataset_builder.collection.incremental_collector import (
    IncrementalCollector,
    CollectionStats,
    CategoryCollectionStats,
)

__all__ = [
    "CollectionState",
    "DuplicateIndex",
    "Inventory",
    "InventoryResult",
    "CategoryInventory",
    "IncrementalCollector",
    "CollectionStats",
    "CategoryCollectionStats",
]
