"""Logging configuration.

One format, used by every entry point, so a log line from the daemon and one
from the pre-flight can be read side by side. English only — that rule applies
to log messages as much as to code.
"""

from __future__ import annotations

import logging
import warnings

FORMAT = "%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s"

# These are chatty at INFO and say nothing about what MAGI is doing. httpx in
# particular logs one line per request, which during a debate means one per
# turn, drowning the turns themselves.
# `autogen_core.events` dumps every full prompt and response at INFO — around
# 90 KB per debate, which buries the turns it is wrapped around. `autogen_core`
# itself narrates message routing at the same level.
NOISY = (
    "httpx",
    "httpcore",
    "urllib3",
    "asyncio",
    "faster_whisper",
    "autogen_core",
    "autogen_core.events",
)


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging. Safe to call once per process, at the top."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=FORMAT,
    )
    for name in NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    # AutoGen's structured output puts a parsed pydantic model into a field
    # typed `None`, so pydantic warns once per turn. Cosmetic, upstream, and
    # loud enough to bury the debate itself.
    warnings.filterwarnings(
        "ignore",
        message="Pydantic serializer warnings",
        category=UserWarning,
        module="pydantic.main",
    )
