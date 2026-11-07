"""Bounded executor for offloading synchronous ML work from the event loop.

Provides a lifecycle-managed ThreadPoolExecutor that:
- Initializes once (thread-safe double-checked locking)
- Is reused across all requests
- Shuts down cleanly at application shutdown

Configuration:
    ML_MAX_CONCURRENCY env var controls max executor threads.
    Default: 2 (matches Settings.ML_MAX_CONCURRENCY default).
    Minimum: 1.
    Validated at executor creation time.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)

_executor: ThreadPoolExecutor | None = None
_executor_init_lock = threading.Lock()

_MIN_CONCURRENCY = 1
_DEFAULT_CONCURRENCY = 4


def _resolve_max_workers() -> int:
    """Read ML_MAX_CONCURRENCY from environment and validate.

    Returns a validated integer >= _MIN_CONCURRENCY.
    Raises ValueError for invalid configuration.
    """
    raw = os.environ.get("ML_MAX_CONCURRENCY", str(_DEFAULT_CONCURRENCY))
    try:
        value = int(raw)
    except (ValueError, TypeError):
        raise ValueError(
            f"ML_MAX_CONCURRENCY must be a valid integer, got {raw!r}"
        )
    if value < _MIN_CONCURRENCY:
        raise ValueError(
            f"ML_MAX_CONCURRENCY must be >= {_MIN_CONCURRENCY}, got {value}"
        )
    return value


def get_executor() -> ThreadPoolExecutor:
    """Return the application-scoped executor, creating it exactly once.

    Thread-safe: initialization is serialized via _executor_init_lock
    with double-checked locking.
    """
    global _executor
    if _executor is not None:
        return _executor
    with _executor_init_lock:
        if _executor is not None:
            return _executor
        max_workers = _resolve_max_workers()
        _executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ml_pipeline",
        )
        logger.info("ML executor initialized with max_workers=%d", max_workers)
    return _executor


def shutdown_executor() -> None:
    """Shut down the executor cleanly during application shutdown.

    Policy: wait for all submitted tasks to complete before returning.
    After shutdown, _executor is set to None. If the application
    somehow receives new requests after shutdown, get_executor() will
    create a fresh executor (standard Uvicorn graceful-shutdown behavior
    prevents this in practice).
    """
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        logger.info("ML executor shut down.")
        _executor = None


def reset_executor() -> None:
    """Reset executor state. For testing only — not part of production lifecycle."""
    global _executor
    shutdown_executor()


async def run_in_executor(func: Callable[..., Any], *args: Any) -> Any:
    """Submit a synchronous function to the bounded executor and await the result.

    This offloads CPU-bound ML work from the event-loop thread so that
    FastAPI can continue processing other requests concurrently.
    """
    import asyncio
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_executor(), func, *args)
