"""
Query loader for dataset builder search queries.

Loads category-specific search queries from plain text files,
deduplicates them, and exposes a clean read-only interface.
"""

from __future__ import annotations

from pathlib import Path


class QueryLoader:
    """Load and manage search queries from text files.

    Each category is stored as a ``.txt`` file in the queries
    directory.  One query per line; empty lines and lines starting
    with ``#`` are ignored.  Duplicates within a file are removed.
    """

    def __init__(self, queries_dir: Path) -> None:
        self._queries_dir: Path = queries_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> dict[str, list[str]]:
        """Load queries for every available category.

        Returns
        -------
        dict[str, list[str]]
            Mapping of category name to its list of queries.
        """
        result: dict[str, list[str]] = {}
        for category in self.categories():
            result[category] = self.load_category(category)
        return result

    def load_category(self, category_name: str) -> list[str]:
        """Load queries for a single category.

        Parameters
        ----------
        category_name:
            Name of the category (matches the ``.txt`` filename
            without extension).

        Returns
        -------
        list[str]
            Deduplicated, non-empty queries.

        Raises
        ------
        FileNotFoundError
            When no ``<category_name>.txt`` file exists.
        ValueError
            When the file exists but contains zero valid queries.
        """
        file_path = self._queries_dir / f"{category_name}.txt"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Category file not found: {file_path}"
            )

        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
        queries: list[str] = []
        seen: set[str] = set()

        for line in raw_lines:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                continue

            if stripped not in seen:
                seen.add(stripped)
                queries.append(stripped)

        if not queries:
            raise ValueError(
                f"Category file contains no valid queries: {file_path}"
            )

        return queries

    def categories(self) -> list[str]:
        """Return sorted list of available category names.

        Scans the queries directory for ``.txt`` files and returns
        their stems as category identifiers.

        Returns
        -------
        list[str]
            Sorted category names derived from filenames.
        """
        categories: list[str] = []

        for file_path in self._queries_dir.iterdir():
            if file_path.suffix == ".txt" and file_path.is_file():
                categories.append(file_path.stem)

        return sorted(categories)
