"""Optional benchmark for FAISS IndexFlatIP search latency.

These tests are slow and measure raw performance.  They are
intentionally NOT collected by default.  Run explicitly::

    pytest tests/test_search/test_benchmark.py -v -s

Results are EMPIRICALLY VERIFIED only for the machine and
environment where they were executed.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from search.flat_index import FlatIndex


DIM = 512
K = 5


def _make_batch(n: int, dim: int = DIM) -> np.ndarray:
    batch = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(batch, axis=1, keepdims=True)
    return batch / norms


@pytest.mark.slow
class TestBenchmark:
    """Measure search latency for increasing dataset sizes."""

    @pytest.mark.parametrize("n", [1_000, 10_000, 50_000, 100_000])
    def test_search_latency(self, n: int) -> None:
        db = _make_batch(n)
        query = _make_batch(1)[0]

        idx = FlatIndex(dimension=DIM)
        idx.add(db)

        # Warm up
        idx.search(query, k=K)

        # Benchmark: 100 queries
        n_queries = 100
        t0 = time.perf_counter()
        for _ in range(n_queries):
            idx.search(query, k=K)
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / n_queries) * 1000
        theoretical_mb = (n * DIM * 4) / (1024 * 1024)

        print(f"\n  n={n:>7,}  avg_search={avg_ms:.3f}ms  "
              f"theoretical_vectors_MB={theoretical_mb:.1f}")

    @pytest.mark.parametrize("n", [1_000, 10_000, 100_000])
    def test_add_latency(self, n: int) -> None:
        db = _make_batch(n)
        idx = FlatIndex(dimension=DIM)

        t0 = time.perf_counter()
        idx.add(db)
        elapsed = time.perf_counter() - t0

        print(f"\n  add {n:>7,} vectors: {elapsed*1000:.3f}ms")
