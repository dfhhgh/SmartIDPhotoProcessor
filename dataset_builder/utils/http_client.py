"""
Reusable HTTP client for all image source providers.

Encapsulates session management, retry logic, rate limiting, and
default headers so that source implementations never call
``requests`` directly.
"""

from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config.settings import Settings


class HTTPClient:
    """Production-quality HTTP client built on top of ``requests.Session``.

    The client owns a single session, applies default headers and
    timeouts from :class:`Settings`, retries transient failures with
    exponential back-off, and enforces a minimum delay between
    consecutive requests.

    Usage
    -----
    ::

        with HTTPClient(settings) as client:
            response = client.get("https://example.com/api")
    """

    _RETRY_STATUS_CODES: tuple[int, ...] = (429, 500, 502, 503, 504)

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._session: requests.Session = self._build_session()
        self._last_request_time: float = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> HTTPClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        url: str,
        params: dict[str, str | int | float] | None = None,
        headers: dict[str, str] | None = None,
    ) -> requests.Response:
        """Perform an HTTP GET request with retry and rate limiting.

        Parameters
        ----------
        url:
            Target URL.
        params:
            Optional query parameters.
        headers:
            Optional per-request header overrides.

        Returns
        -------
        requests.Response
            The server response.

        Raises
        ------
        requests.ConnectionError
            When a connection cannot be established after all retries.
        requests.Timeout
            When the request times out after all retries.
        requests.HTTPError
            When the response status code indicates a non-retryable error
            (after ``raise_for_status``).
        """
        self._enforce_rate_limit()

        merged_headers = dict(self._session.headers)
        if headers:
            merged_headers.update(headers)

        response = self._session.get(
            url,
            params=params,
            headers=merged_headers,
            timeout=self._settings.DOWNLOAD_TIMEOUT_SECONDS,
        )
        response.raise_for_status()

        self._last_request_time = time.monotonic()

        return response

    def close(self) -> None:
        """Close the underlying session and release resources."""
        self._session.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        """Create and configure a ``requests.Session`` with retries."""
        session = requests.Session()

        session.headers.update(self._settings.DEFAULT_HEADERS)

        retry_strategy = Retry(
            total=self._settings.MAX_RETRIES,
            backoff_factor=self._settings.BACKOFF_FACTOR,
            status_forcelist=list(self._RETRY_STATUS_CODES),
            allowed_methods=["GET"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _enforce_rate_limit(self) -> None:
        """Sleep if needed to respect the minimum request interval."""
        if self._last_request_time <= 0.0:
            return

        elapsed = time.monotonic() - self._last_request_time
        minimum_interval = self._settings.REQUEST_DELAY_SECONDS

        if elapsed < minimum_interval:
            time.sleep(minimum_interval - elapsed)
