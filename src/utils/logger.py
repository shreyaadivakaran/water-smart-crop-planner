"""
logger.py
---------
Centralised logging setup for the Water-Smart Crop Planner.

All modules should obtain their logger via::

    from src.utils.logger import get_logger
    logger = get_logger(__name__)

Log files are written to the ``logs/`` directory at the project root.
Console output is always enabled; file output is added when the logs
directory is writable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_DEFAULT_LEVEL = logging.DEBUG

# Resolve logs/ relative to project root (three levels up from src/utils/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOGS_DIR = _PROJECT_ROOT / "logs"

# Track which loggers have already been configured to avoid duplicate handlers
_configured_loggers: set[str] = set()


def get_logger(
    name: str,
    log_file: str | None = None,
    level: int = _DEFAULT_LEVEL,
) -> logging.Logger:
    """Return a configured logger that writes to console and optionally a file.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    log_file:
        Filename (not full path) for the log file inside ``logs/``.
        Defaults to ``"pipeline.log"``.
    level:
        Logging level.  Defaults to ``logging.DEBUG``.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if name in _configured_loggers:
        return logging.getLogger(name)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # prevent double-logging to root logger

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    resolved_log_file = log_file or "pipeline.log"
    try:
        _LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(
            _LOGS_DIR / resolved_log_file, encoding="utf-8"
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except OSError as exc:
        logger.warning(
            "Could not create file handler for '%s': %s. "
            "Logging to console only.",
            resolved_log_file,
            exc,
        )

    _configured_loggers.add(name)
    return logger


def get_validation_logger() -> logging.Logger:
    """Return the dedicated data-validation logger.

    Writes to ``logs/data_validation.log`` in addition to the console.

    Returns
    -------
    logging.Logger
    """
    return get_logger(
        "water_smart.validation",
        log_file="data_validation.log",
    )
