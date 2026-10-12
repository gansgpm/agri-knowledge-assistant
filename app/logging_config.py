"""Logging setup shared by the API process."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once, and quieten noisy third-party loggers."""
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
