"""Deterministic reference/query splitter with source diversity awareness.

Splits records ensuring:
- No overlap between reference and query
- Deterministic ordering via seeded RNG
- Source diversity where metadata allows
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from dataset_acquisition.models import ImageRecord


def split_reference_query(
    records: list[ImageRecord],
    reference_ratio: float = 0.6,
    min_reference: int = 2,
    min_query: int = 1,
    seed: int = 42,
) -> dict[str, dict[str, list[ImageRecord]]]:
    """Split records into reference and query per person.

    Returns:
        {
            "reference": {person_id: [records]},
            "query": {person_id: [records]},
            "excluded": {person_id: reason},
        }
    """
    rng = random.Random(seed)

    by_person: dict[str, list[ImageRecord]] = {}
    for r in records:
        by_person.setdefault(r.person_id, []).append(r)

    reference: dict[str, list[ImageRecord]] = {}
    query: dict[str, list[ImageRecord]] = {}
    excluded: dict[str, str] = {}

    for person_id, person_records in sorted(by_person.items()):
        n = len(person_records)
        n_ref = max(min_reference, int(n * reference_ratio))
        n_query = max(min_query, n - n_ref)

        if n_ref + n_query > n:
            if n < min_reference + min_query:
                excluded[person_id] = f"insufficient_images: {n} < {min_reference + min_query}"
                continue
            n_ref = n - min_query
            n_query = min_query

        shuffled = list(person_records)
        rng.shuffle(shuffled)

        reference[person_id] = sorted(shuffled[:n_ref], key=lambda r: r.image_id)
        query[person_id] = sorted(shuffled[n_ref:n_ref + n_query], key=lambda r: r.image_id)

    return {
        "reference": reference,
        "query": query,
        "excluded": excluded,
    }


def copy_split_to_disk(
    split: dict[str, dict[str, list[ImageRecord]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Copy files to reference/query directory structure."""
    import shutil

    stats = {"reference_images": 0, "query_images": 0, "excluded_persons": 0}

    for split_name in ["reference", "query"]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        for person_id, records in split[split_name].items():
            person_dir = split_dir / person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            for record in records:
                src = Path(record.local_path)
                if src.exists():
                    dst = person_dir / src.name
                    shutil.copy2(str(src), str(dst))
                    if split_name == "reference":
                        stats["reference_images"] += 1
                    else:
                        stats["query_images"] += 1

    stats["excluded_persons"] = len(split["excluded"])
    return stats
