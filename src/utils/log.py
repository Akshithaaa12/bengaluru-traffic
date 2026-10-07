"""Standard logger setup shared by all pipeline steps."""
from __future__ import annotations

import logging
import sys
import time

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger writing UTC-timestamped lines to stderr (configured once per name)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(_FORMAT)
        formatter.converter = time.gmtime
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
