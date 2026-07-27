"""Consistent console and persistent run logging."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def configure_run_logging(run_dir: str | Path, level: str = "INFO") -> logging.Logger:
    run_dir = Path(run_dir)
    logger = logging.getLogger("ir_subject_classification")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.addHandler(file_handler)
    warnings_logger.addHandler(console)
    warnings_logger.setLevel(logger.level)
    return logger

