"""In-process pub/sub event bus: topic -> set of asyncio queues.

Copied and adapted from latacc-edge. There is no shared package between the two
projects by design (different lifecycles), so this is a deliberate duplicate —
a fix here does not reach latacc-edge, or the reverse.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class Bus:
    """Simple topic-based pub/sub using :class:`asyncio.Queue`.

    Each call to :meth:`subscribe` returns a new queue that will receive all
    messages published to that topic. Call :meth:`unsubscribe` (typically in a
    ``finally`` block) to remove the queue and avoid leaks.

    The bus retains the **last published value** per topic so late subscribers
    (a browser opening mid-debate) receive current state immediately instead of
    an empty screen until the next event.
    """

    def __init__(self) -> None:
        self._subs: dict[str, set[asyncio.Queue[Any]]] = defaultdict(set)
        self._last: dict[str, Any] = {}

    def subscribe(self, topic: str, *, replay_last: bool = False) -> asyncio.Queue[Any]:
        """Subscribe to *topic* and return a new queue.

        If *replay_last* is ``True`` and a value has been published on this
        topic before, it is placed into the queue immediately so the new
        subscriber starts with current state.
        """
        q: asyncio.Queue[Any] = asyncio.Queue()
        self._subs[topic].add(q)
        if replay_last and topic in self._last:
            q.put_nowait(self._last[topic])
        logger.debug("subscribe  topic=%s  subscribers=%d", topic, len(self._subs[topic]))
        return q

    def unsubscribe(self, topic: str, queue: asyncio.Queue[Any]) -> None:
        self._subs[topic].discard(queue)
        if not self._subs[topic]:
            del self._subs[topic]

    async def publish(self, topic: str, data: Any) -> None:
        self._last[topic] = data
        for q in self._subs.get(topic, ()):
            await q.put(data)

    def last(self, topic: str) -> Any | None:
        """Last value published on *topic*, or ``None``."""
        return self._last.get(topic)
