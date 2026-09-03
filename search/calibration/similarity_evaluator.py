"""Similarity evaluator for calibration pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from search.calibration.pair_generator import PairSample


@dataclass(frozen=True, slots=True)
class EvaluatedPair:
    person_id_1: str
    image_1: str
    person_id_2: str
    image_2: str
    similarity: float
    is_positive: bool


class SimilarityEvaluator:
    """Evaluates cosine similarity for calibration pairs."""

    def evaluate_pair(self, pair: PairSample) -> EvaluatedPair:
        """Compute inner product between two L2-normalized embeddings."""
        emb1 = pair.embedding_1.ravel()
        emb2 = pair.embedding_2.ravel()
        sim = float(np.dot(emb1, emb2))
        return EvaluatedPair(
            person_id_1=pair.person_id_1,
            image_1=pair.image_1,
            person_id_2=pair.person_id_2,
            image_2=pair.image_2,
            similarity=sim,
            is_positive=pair.is_positive,
        )

    def evaluate_batch(self, pairs: list[PairSample]) -> list[EvaluatedPair]:
        """Evaluate a batch of pairs."""
        return [self.evaluate_pair(p) for p in pairs]
