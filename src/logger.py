"""
Centralized logging configuration for churn prediction project.

Provides consistent logging setup across all modules with:
- File rotation (daily)
- Console output with colors
- Structured format with timestamp, level, module name
- Environment-based configuration (dev vs prod)
"""

import logging
import logging.handlers
import os
from pathlib import Path

from src.config import PROJECT_ROOT


class LoggerConfig:
    """Centralized logger configuration.

    Attributes:
        LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        LOG_FILE: Path to log file
        LOG_MAX_BYTES: Max bytes per rotated file (not used with daily rotation)
        LOG_FORMAT: Format string for log messages
        DATE_FORMAT: Format for timestamps
    """

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE = os.getenv("LOG_FILE", str(PROJECT_ROOT / "logs" / "app.log"))
    LOG_FORMAT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def get_level(cls) -> int:
        """Get logging level as integer."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(cls.LOG_LEVEL, logging.INFO)


def _setup_log_directory() -> None:
    """Create logs directory if it doesn't exist."""
    log_dir = Path(LoggerConfig.LOG_FILE).parent
    log_dir.mkdir(parents=True, exist_ok=True)


_shared_handlers: list = []


def _get_shared_handlers() -> list:
    """Build the file and console handlers once, then reuse them.

    Every module must share the *same* TimedRotatingFileHandler instance.
    One handler per module would mean several open descriptors on
    logs/app.log, and the midnight rollover would then fail on Windows
    (PermissionError: the file is locked by another handler).
    """
    if _shared_handlers:
        return _shared_handlers

    _setup_log_directory()
    formatter = logging.Formatter(
        LoggerConfig.LOG_FORMAT, datefmt=LoggerConfig.DATE_FORMAT
    )

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=LoggerConfig.LOG_FILE,
        when="midnight",  # Rotate at midnight
        interval=1,  # Every day
        backupCount=30,  # Keep last 30 days
        encoding="utf-8",
    )
    file_handler.setLevel(LoggerConfig.get_level())
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LoggerConfig.get_level())
    console_handler.setFormatter(formatter)

    _shared_handlers.extend([file_handler, console_handler])
    return _shared_handlers


def get_logger(name: str) -> logging.Logger:
    """Get or create a configured logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(LoggerConfig.get_level())

    for handler in _get_shared_handlers():
        if handler not in logger.handlers:
            logger.addHandler(handler)

    return logger


# Module-level logger for this file
logger = get_logger(__name__)
