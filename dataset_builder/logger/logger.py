"""
Structured logging for the Dataset Builder project.

Provides a thin wrapper around Python's built-in :mod:`logging` package.
All configuration is read from :class:`Settings`.  Duplicate handlers are
prevented via a class-level registry keyed by logger name.

Usage
-----
::

    from dataset_builder.config.settings import Settings
    from dataset_builder.logger import DatasetLogger

    settings = Settings()
    logger = DatasetLogger(name="downloader", settings=settings)

    logger.info("Starting download for category: %s", category)
    logger.error("Failed to download %s: %s", url, exc)
"""

from __future__ import annotations

import logging
from pathlib import Path

from dataset_builder.config.settings import Settings


class DatasetLogger:
    """Structured logger backed by Python's built-in logging module.

    Read all configuration from ``settings``:

    - ``settings.LOG_LEVEL`` — minimum severity for console output.
    - ``settings.LOG_TO_FILE`` — whether to attach a ``FileHandler``.
    - ``settings.LOG_FILENAME`` — filename for the log file.
    - ``settings.LOGS_DIR`` — parent directory for the log file.

    Duplicate handlers are prevented by inspecting the logger's handler list
    before attaching new ones.  Multiple ``DatasetLogger`` instances with the
    same *name* share a single underlying :class:`logging.Logger` from
    the global logging registry, so only one console handler and one file
    handler are ever added per name.

    Parameters
    ----------
    name:
        Identifier for this logger (e.g. ``"downloader"``).
        Used as the :mod:`logging` logger name and to prevent duplicate
        handlers.
    settings:
        Application settings.  All logging parameters are read from this
        object.
    """

    def __init__(self, name: str, settings: Settings) -> None:
        self._name: str = name
        self._settings: Settings = settings
        self._logger: logging.Logger = logging.getLogger(name)

        self._configure()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the logger name.

        Returns
        -------
        str
            The identifier passed at construction time.
        """
        return self._name

    @property
    def logger(self) -> logging.Logger:
        """Return the underlying :class:`logging.Logger`.

        Useful for callers that need to attach custom handlers or
        perform advanced configuration.

        Returns
        -------
        logging.Logger
            The standard-library logger managed by this instance.
        """
        return self._logger

    def debug(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at ``DEBUG`` level.

        Parameters
        ----------
        msg:
            Log format string.
        *args:
            Positional arguments for *msg*.
        **kwargs:
            Keyword arguments forwarded to :meth:`logging.Logger.debug`.
        """
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at ``INFO`` level.

        Parameters
        ----------
        msg:
            Log format string.
        *args:
            Positional arguments for *msg*.
        **kwargs:
            Keyword arguments forwarded to :meth:`logging.Logger.info`.
        """
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at ``WARNING`` level.

        Parameters
        ----------
        msg:
            Log format string.
        *args:
            Positional arguments for *msg*.
        **kwargs:
            Keyword arguments forwarded to :meth:`logging.Logger.warning`.
        """
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at ``ERROR`` level.

        Parameters
        ----------
        msg:
            Log format string.
        *args:
            Positional arguments for *msg*.
        **kwargs:
            Keyword arguments forwarded to :meth:`logging.Logger.error`.
        """
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: str, *args: object, **kwargs: object) -> None:
        """Log at ``ERROR`` level with exception traceback.

        Should be called inside an ``except`` block to automatically
        append the current exception information.

        Parameters
        ----------
        msg:
            Log format string.
        *args:
            Positional arguments for *msg*.
        **kwargs:
            Keyword arguments forwarded to :meth:`logging.Logger.exception`.
        """
        self._logger.exception(msg, *args, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _configure(self) -> None:
        """Attach console and optional file handlers if not already present."""
        level = self._resolve_level()
        self._logger.setLevel(level)
        self._logger.propagate = False

        self._attach_console_handler(level)
        if self._settings.LOG_TO_FILE:
            self._attach_file_handler(level)

    def _resolve_level(self) -> int:
        """Convert the ``LOG_LEVEL`` string to a :mod:`logging` constant."""
        return getattr(logging, self._settings.LOG_LEVEL.upper(), logging.INFO)

    def _attach_console_handler(self, level: int) -> None:
        """Add a :class:`StreamHandler` to ``stderr`` if not already attached."""
        if self._has_handler_type(logging.StreamHandler):
            return

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(self._build_formatter())
        self._logger.addHandler(handler)

    def _attach_file_handler(self, level: int) -> None:
        """Add a :class:`FileHandler` to the configured log file."""
        if self._has_handler_type(logging.FileHandler):
            return

        log_dir = self._settings.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        log_path = log_dir / self._settings.LOG_FILENAME
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(self._build_formatter())
        self._logger.addHandler(handler)

    def _has_handler_type(self, handler_type: type) -> bool:
        """Return ``True`` if the logger already has a handler of *handler_type*.

        Parameters
        ----------
        handler_type:
            The :class:`logging.Handler` subclass to search for.

        Returns
        -------
        bool
            ``True`` when a matching handler is already attached.
        """
        return any(isinstance(h, handler_type) for h in self._logger.handlers)

    @staticmethod
    def _build_formatter() -> logging.Formatter:
        """Return the standard log line format.

        Returns
        -------
        logging.Formatter
            Formatter with timestamp, logger name, level, and message.
        """
        return logging.Formatter(
            fmt="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
