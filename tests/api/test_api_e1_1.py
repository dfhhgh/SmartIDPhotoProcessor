"""Phase E.1.1 — Executor hardening, configuration, and lifecycle tests."""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

from smartidphotoprocessor.api import executor


@pytest.fixture(autouse=True)
def clean_executor():
    """Reset executor before and after each test."""
    executor.reset_executor()
    yield
    executor.reset_executor()


# ---------------------------------------------------------------------------
# Issue 1: Initialization Race Condition
# ---------------------------------------------------------------------------

class TestExecutorRaceCondition:
    """Verify that concurrent first calls create exactly one executor."""

    def test_single_instance_under_contention(self):
        """Many threads calling get_executor() simultaneously must produce one instance."""
        barrier = threading.Barrier(20)
        instances = []
        lock = threading.Lock()

        def get_exec():
            barrier.wait()
            e = executor.get_executor()
            with lock:
                instances.append(id(e))

        threads = [threading.Thread(target=get_exec) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(instances) == 20
        assert len(set(instances)) == 1, (
            f"Expected exactly 1 executor instance, got {len(set(instances))}"
        )

    def test_reuse_after_initialization(self):
        """Subsequent calls return the same executor."""
        e1 = executor.get_executor()
        e2 = executor.get_executor()
        assert e1 is e2

    def test_shutdown_and_recreate(self):
        """After shutdown, a fresh executor is created on next call."""
        e1 = executor.get_executor()
        executor.shutdown_executor()
        e2 = executor.get_executor()
        assert e1 is not e2


# ---------------------------------------------------------------------------
# Issue 2 & 3: Configuration Source of Truth & Validation
# ---------------------------------------------------------------------------

class TestConfigurationValidation:
    """Verify ML_MAX_CONCURRENCY validation."""

    def test_default_value_is_4(self):
        """Default ML_MAX_CONCURRENCY should be 4 (Settings default)."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove env var to test default
            os.environ.pop("ML_MAX_CONCURRENCY", None)
            from smartidphotoprocessor.config.settings import Settings
            s = Settings()
            assert s.ML_MAX_CONCURRENCY == 4

    def test_env_var_override(self):
        """Environment variable overrides default in executor."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "4"}):
            executor.reset_executor()
            e = executor.get_executor()
            assert e._max_workers == 4

    def test_valid_value_1(self):
        """ML_MAX_CONCURRENCY=1 is valid."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "1"}):
            executor.reset_executor()
            e = executor.get_executor()
            assert e._max_workers == 1

    def test_valid_value_2(self):
        """ML_MAX_CONCURRENCY=2 is valid."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "2"}):
            executor.reset_executor()
            e = executor.get_executor()
            assert e._max_workers == 2

    def test_valid_value_4(self):
        """ML_MAX_CONCURRENCY=4 is valid."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "4"}):
            executor.reset_executor()
            e = executor.get_executor()
            assert e._max_workers == 4

    def test_invalid_zero_raises(self):
        """ML_MAX_CONCURRENCY=0 must raise ValueError."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "0"}):
            with pytest.raises(ValueError, match="must be >= 1"):
                executor.get_executor()

    def test_invalid_negative_raises(self):
        """ML_MAX_CONCURRENCY=-1 must raise ValueError."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "-1"}):
            with pytest.raises(ValueError, match="must be >= 1"):
                executor.get_executor()

    def test_invalid_non_integer_raises(self):
        """ML_MAX_CONCURRENCY=abc must raise ValueError."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "abc"}):
            with pytest.raises((ValueError, TypeError)):
                executor.get_executor()

    def test_invalid_empty_string_raises(self):
        """ML_MAX_CONCURRENCY='' must raise ValueError or TypeError."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": ""}):
            with pytest.raises((ValueError, TypeError)):
                executor.get_executor()


# ---------------------------------------------------------------------------
# Issue 4 & 5: Executor Lifecycle
# ---------------------------------------------------------------------------

class TestExecutorLifecycle:
    """Verify executor lifecycle behavior."""

    def test_shutdown_waits_for_active_tasks(self):
        """shutdown_executor() waits for submitted tasks to complete."""
        results = []

        def slow_task():
            import time
            time.sleep(0.5)
            results.append("done")

        e = executor.get_executor()
        e.submit(slow_task)
        executor.shutdown_executor()
        assert results == ["done"]

    def test_shutdown_clears_executor(self):
        """After shutdown, _executor is None."""
        executor.get_executor()
        assert executor._executor is not None
        executor.shutdown_executor()
        assert executor._executor is None

    def test_double_shutdown_is_safe(self):
        """Calling shutdown twice does not raise."""
        executor.get_executor()
        executor.shutdown_executor()
        executor.shutdown_executor()  # Should not raise

    def test_shutdown_without_init_is_safe(self):
        """Calling shutdown without prior init does not raise."""
        executor.shutdown_executor()

    def test_post_shutdown_get_creates_new(self):
        """get_executor after shutdown creates a fresh executor."""
        e1 = executor.get_executor()
        executor.shutdown_executor()
        e2 = executor.get_executor()
        assert e1 is not e2
        assert isinstance(e2, ThreadPoolExecutor)


# ---------------------------------------------------------------------------
# Issue 6 & 7: Thread Safety & Request Isolation
# ---------------------------------------------------------------------------

class TestThreadSafetyAndIsolation:
    """Verify executor does not share request-local state."""

    def test_concurrent_requests_get_independent_results(self):
        """Each request executed via executor gets its own result."""
        results = []
        lock = threading.Lock()

        def task(value):
            import time
            time.sleep(0.05)
            result = {"value": value, "thread": threading.current_thread().name}
            with lock:
                results.append(result)

        e = executor.get_executor()
        futures = [e.submit(task, i) for i in range(10)]
        for f in futures:
            f.result()

        assert len(results) == 10
        values = sorted(r["value"] for r in results)
        assert values == list(range(10))

    def test_executor_threads_are_named(self):
        """Executor threads should have the ml_pipeline prefix."""
        names = []
        lock = threading.Lock()

        def task():
            with lock:
                names.append(threading.current_thread().name)

        e = executor.get_executor()
        futures = [e.submit(task) for _ in range(4)]
        for f in futures:
            f.result()

        assert all("ml_pipeline" in n for n in names)


# ---------------------------------------------------------------------------
# Issue 11: Thread Count / Boundary
# ---------------------------------------------------------------------------

class TestThreadPoolBoundary:
    """Verify max_workers matches ML_MAX_CONCURRENCY."""

    def test_executor_max_workers_matches_config(self):
        """ThreadPoolExecutor._max_workers should equal ML_MAX_CONCURRENCY."""
        with patch.dict(os.environ, {"ML_MAX_CONCURRENCY": "3"}):
            executor.reset_executor()
            e = executor.get_executor()
            assert e._max_workers == 3

    def test_no_hidden_executors(self):
        """Only one executor exists in the module."""
        executor.get_executor()
        assert executor._executor is not None
        assert isinstance(executor._executor, ThreadPoolExecutor)
