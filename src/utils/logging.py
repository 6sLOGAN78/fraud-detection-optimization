"""Structured logging configuration for the IEEE-CIS Fraud Detection system."""

import json
import logging
import sys
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Custom logging formatter that outputs JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Formats a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A string containing the serialized JSON log entry.
        """
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "file": record.pathname,
            "line": record.lineno,
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logger(
    name: str = "fraud_detection",
    level: str = "INFO",
    console: bool = True,
    json_format: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Configures and retrieves the structured logger.

    Args:
        name: Name of the logger instance.
        level: Severity level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        console: Whether to log to standard outputs.
        json_format: If true, output JSON log format; otherwise, human-readable.
        log_file: Optional path to write log output.

    Returns:
        Structured Python logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    # Raw format strings
    str_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = (
        JSONFormatter()
        if json_format
        else logging.Formatter(str_format, datefmt="%Y-%m-%d %H:%M:%S")
    )

    if console:
        # Standard console logger
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(rich_tracebacks=True, markup=True)
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
        console_handler.setLevel(logger.level)
        logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logger.level)
        logger.addHandler(file_handler)

    return logger
