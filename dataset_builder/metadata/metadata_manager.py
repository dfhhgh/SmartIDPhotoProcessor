"""
Metadata persistence manager for downloaded images.

Provides storage, deduplication, and serialization of
:class:`ImageMetadata` records without performing any downloads,
filtering, or HTTP operations.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from dataset_builder.sources.base_source import ImageMetadata


class MetadataManager:
    """Manage an in-memory collection of :class:`ImageMetadata` records.

    Records are stored in insertion order.  Duplicates are detected
    by the composite key ``(source, id)`` and silently ignored.

    Usage
    -----
    ::

        manager = MetadataManager()
        manager.add(metadata)
        manager.save_json(Path("metadata.json"))
    """

    def __init__(self) -> None:
        self._records: list[ImageMetadata] = []
        self._index: dict[tuple[str, str], int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, metadata: ImageMetadata) -> bool:
        """Add a single metadata record.

        Parameters
        ----------
        metadata:
            The record to store.

        Returns
        -------
        bool
            ``True`` if the record was added, ``False`` if a duplicate
            ``(source, id)`` already exists.
        """
        key = (metadata.source, metadata.id)

        if key in self._index:
            return False

        self._index[key] = len(self._records)
        self._records.append(metadata)
        return True

    def add_many(self, metadata: list[ImageMetadata]) -> int:
        """Add multiple metadata records.

        Parameters
        ----------
        metadata:
            Records to store.  Duplicates are silently skipped.

        Returns
        -------
        int
            Number of records actually added (excluding duplicates).
        """
        count = 0
        for record in metadata:
            if self.add(record):
                count += 1
        return count

    def save_json(self, path: Path) -> None:
        """Serialize all records to a JSON file.

        Parameters
        ----------
        path:
            Destination file.  Parent directories are created
            automatically.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for record in self._records:
            record_dict = asdict(record)
            record_dict["local_path"] = str(record_dict["local_path"])
            data.append(record_dict)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_json(self, path: Path) -> int:
        """Load records from a JSON file, merging with existing data.

        Parameters
        ----------
        path:
            Source JSON file.

        Returns
        -------
        int
            Number of new records loaded (excluding duplicates).

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = [self._dict_to_metadata(item) for item in data]
        return self.add_many(records)

    def save_csv(self, path: Path) -> None:
        """Serialize all records to a CSV file.

        Parameters
        ----------
        path:
            Destination file.  Parent directories are created
            automatically.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self._records:
            return

        fieldnames = list(asdict(self._records[0]).keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                row = asdict(record)
                row["local_path"] = str(row["local_path"])
                writer.writerow(row)

    def load_csv(self, path: Path) -> int:
        """Load records from a CSV file, merging with existing data.

        Parameters
        ----------
        path:
            Source CSV file.

        Returns
        -------
        int
            Number of new records loaded (excluding duplicates).

        Raises
        ------
        FileNotFoundError
            When the file does not exist.
        """
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = [self._dict_to_metadata(row) for row in reader]

        return self.add_many(records)

    def clear(self) -> None:
        """Remove all records from the manager."""
        self._records.clear()
        self._index.clear()

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)

    def all(self) -> list[ImageMetadata]:
        """Return all records in insertion order.

        Returns
        -------
        list[ImageMetadata]
            A copy of the internal record list.
        """
        return list(self._records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dict_to_metadata(data: dict[str, object]) -> ImageMetadata:
        """Convert a dictionary back into an ImageMetadata instance."""
        local_path = data.get("local_path", "")
        if isinstance(local_path, str):
            local_path = Path(local_path)

        return ImageMetadata(
            id=str(data.get("id", "")),
            source=str(data.get("source", "")),
            local_path=local_path,
            download_url=str(data.get("download_url", "")),
            page_url=str(data.get("page_url", "")),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            photographer=str(data.get("photographer", "")),
            license_name=str(data.get("license_name", "")),
            query=str(data.get("query", "")),
        )
