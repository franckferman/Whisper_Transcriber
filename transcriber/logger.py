#!/usr/bin/env python3
# transcriber/logger.py

"""
Logger Module

Description:
Structured logging setup for the transcription system.
Supports console and optional file output with configurable verbosity.

Created By  : Franck FERMAN
Version     : 2.0.0
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_CONSOLE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    debug: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure root logger with a console handler and an optional file handler.

    Args:
        debug:    If True, set level to DEBUG; otherwise INFO.
        log_file: Optional path to a log file. If None, no file handler is added.
                  If the string 'auto', a timestamped filename is generated.
    """
    root = logging.getLogger()
    level = logging.DEBUG if debug else logging.INFO
    root.setLevel(level)

    # Avoid adding duplicate handlers when called multiple times
    if root.handlers:
        root.handlers.clear()

    formatter = logging.Formatter(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT)

    # Console handler (stderr to keep stdout clean for piping)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional file handler
    if log_file:
        if log_file == "auto":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"transcription_{timestamp}.log"

        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always capture everything in file
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

        logging.getLogger(__name__).info("Logging to file: %s", log_file)

    logging.getLogger(__name__).debug(
        "Logging initialized. Level=%s", "DEBUG" if debug else "INFO"
    )


