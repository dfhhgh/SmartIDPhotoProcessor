"""Deterministic positive and negative pair generator for threshold calibration.

Positive pairs: same person, different reference images.
Negative pairs: different persons.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class PairSample:
    person_id_1: str
    image_1: str
    embedding_1: npt.NDArray[np.float32]
    person_id_2: str
    image_2: str
    embedding_2: npt.NDArray[np.float32]
    is_positive: bool


class PairGenerator:
    """Generates deterministic positive and negative pairs from identity groupings."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def generate_pairs(
        self,
        identity_data: dict[str, list[tuple[str, npt.NDArray[np.float32]]]],
        max_positive_pairs: int = 1000,
        max_negative_pairs: int = 1000,
    ) -> tuple[list[PairSample], list[PairSample]]:
        """Generate positive and negative pairs deterministically.

        Parameters
        ----------
        identity_data:
            Dictionary mapping ``person_id`` to a list of ``(image_path, embedding)`` tuples.
        max_positive_pairs:
            Maximum number of positive pairs to sample.
        max_negative_pairs:
            Maximum number of negative pairs to sample.

        Returns
        -------
        tuple[list[PairSample], list[PairSample]]
            A tuple of ``(positive_pairs, negative_pairs)``.
        """
        rng = random.Random(self._seed)
        persons = sorted(identity_data.keys())

        # 1. Generate Positive Pairs (same person, different images)
        positive_pairs: list[PairSample] = []
        for pid in persons:
            items = identity_data[pid]
            if len(items) < 2:
                continue
            # Generate all combinations of different images for this person
            combos = list(itertools.combinations(items, 2))
            for (img1, emb1), (img2, emb2) in combos:
                positive_pairs.append(
                    PairSample(
                        person_id_1=pid,
                        image_1=img1,
                        embedding_1=emb1,
                        person_id_2=pid,
                        image_2=img2,
                        embedding_2=emb2,
                        is_positive=True,
                    )
                )

        if len(positive_pairs) > max_positive_pairs:
            rng.shuffle(positive_pairs)
            positive_pairs = positive_pairs[:max_positive_pairs]

        # 2. Generate Negative Pairs (different persons)
        negative_pairs: list[PairSample] = []
        person_combos = list(itertools.combinations(persons, 2))
        # Shuffle person pairs deterministically
        rng.shuffle(person_combos)

        for pid1, pid2 in person_combos:
            items1 = identity_data[pid1]
            items2 = identity_data[pid2]
            for img1, emb1 in items1:
                for img2, emb2 in items2:
                    negative_pairs.append(
                        PairSample(
                            person_id_1=pid1,
                            image_1=img1,
                            embedding_1=emb1,
                            person_id_2=pid2,
                            image_2=img2,
                            embedding_2=emb2,
                            is_positive=False,
                        )
                    )

        if len(negative_pairs) > max_negative_pairs:
            rng.shuffle(negative_pairs)
            negative_pairs = negative_pairs[:max_negative_pairs]

        return positive_pairs, negative_pairs
