"""Crash-restart supervision for long-running services.

Copied and adapted from latacc-edge's ``__main__.supervise``. Connectivity loss
is the normal case here too: the Spark is a separate machine on WiFi, and a
debate that fails because Ollama blinked must not take the node down with it.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from magi.constants import MAX_BACKOFF_S

logger = logging.getLogger("magi.supervision")


async def supervise(name: str, factory: Callable[[], Awaitable[None]]) -> None:
    """Run *factory* in a crash-restart loop with exponential backoff.

    A service that returns normally is considered finished and is **not**
    restarted: that is how a service opts out of hardware it cannot use (TTS on
    a machine with no speaker). Restarting it would spin the event loop.
    """
    backoff = 1.0
    while True:
        try:
            logger.info("Starting %s", name)
            await factory()
        except asyncio.CancelledError:
            logger.info("Stopped %s (cancelled)", name)
            raise
        except (ConnectionError, OSError) as exc:
            logger.warning("%s: %s — retrying in %.0fs", name, exc, backoff)
        except Exception:
            logger.exception("%s crashed — restarting in %.0fs", name, backoff)
        else:
            logger.info("%s finished — not restarting", name)
            return

        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, MAX_BACKOFF_S)
