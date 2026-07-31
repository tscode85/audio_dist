# file: acoustic_gap/logging_utils.py
"""Central logging configuration for the toolkit."""
from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with a concise, timestamped format."""
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
