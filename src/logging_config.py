"""
Centralized logging configuration for IP-GCA Copilot.
Sets up a rich-formatted console handler and an optional rotating file handler.
Import and call setup_logging() once at the entry point of any script or demo.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rich_tracebacks: bool = True
) -> logging.Logger:
    """
    Configures the root logger with a Rich console handler and an optional
    rotating file handler. Safe to call multiple times (idempotent).

    Args:
        level:            Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file:         Optional path for a plain-text rotating log file.
        rich_tracebacks:  Enable rich traceback formatting in the console handler.

    Returns:
        The configured root logger instance.
    """
    root_logger = logging.getLogger()

    # Avoid adding duplicate handlers on repeated calls
    if root_logger.handlers:
        return root_logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    try:
        from rich.logging import RichHandler
        console_handler = RichHandler(
            level=numeric_level,
            rich_tracebacks=rich_tracebacks,
            show_time=True,
            show_path=True,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        root_logger.addHandler(console_handler)
    except ImportError:
        # Fallback: plain StreamHandler if rich is not installed
        plain_handler = logging.StreamHandler(sys.stdout)
        plain_handler.setLevel(numeric_level)
        plain_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")
        )
        root_logger.addHandler(plain_handler)

    if log_file:
        from logging.handlers import RotatingFileHandler
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB per file
            backupCount=3,
            encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Convenience wrapper that returns a named child logger.
    Modules should use this instead of logging.getLogger() directly.

    Usage:
        from src.logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)
