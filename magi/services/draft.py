"""The question under composition: one PTT press per line, until SEND.

**A press does not start a debate.** Speech recognition gets things wrong, and a
debate costs two to four minutes of three models' time — so the operator sees
each transcribed line, deletes the ones Whisper mangled, adds more, and only
then commits. Firing on every press would mean a mis-heard word costs a full
deliberation and the answer to a question nobody asked.

**The draft lives here, not in the browser.** The kiosk and any other client are
two views of the same node: a draft held in JS would give each of them a
different question, and a reload mid-sentence would lose it.

Unlike latacc-edge's report draft this is **not** persisted. That one survives a
reboot because a half-dictated 9-Line is minutes of work in a field where the
node genuinely does reboot. A question is one or two sentences and the cost of
losing it is repeating them, which does not justify a database on the path
between a button and a sentence.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from magi.bus import Bus

logger = logging.getLogger(__name__)

#: A question is a question, not an essay. The cap exists so a stuck PTT button
#: cannot grow the prompt without bound behind the operator's back.
MAX_LINES = 20


@dataclass(frozen=True)
class DraftLine:
    """One transcribed press."""

    id: int
    text: str
    created_at: str

    def as_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "timestamp": self.created_at}


class Draft:
    """Ordered list of transcribed lines, with the ability to drop any of them.

    Ids are monotonic and never reused, so a delete that races the arrival of a
    new line cannot remove the wrong one — with positional indices, deleting
    "line 2" while a press is still transcribing would silently target whatever
    landed there in the meantime.
    """

    def __init__(self) -> None:
        self._lines: list[DraftLine] = []
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()

    async def add(self, text: str) -> DraftLine | None:
        """Append a transcribed line. ``None`` when the draft is full."""
        text = text.strip()
        if not text:
            return None
        async with self._lock:
            if len(self._lines) >= MAX_LINES:
                logger.warning("Draft is full (%d lines) — ignoring", MAX_LINES)
                return None
            line = DraftLine(
                id=next(self._ids),
                text=text,
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
            self._lines.append(line)
            return line

    async def delete(self, line_id: int) -> bool:
        async with self._lock:
            before = len(self._lines)
            self._lines = [line for line in self._lines if line.id != line_id]
            return len(self._lines) != before

    async def clear(self) -> None:
        async with self._lock:
            self._lines = []

    async def lines(self) -> list[DraftLine]:
        async with self._lock:
            return list(self._lines)

    async def question(self) -> str:
        """The composed question: every line, in dictation order.

        Joined with spaces rather than newlines. The advisors receive this as a
        single question, and a multi-line prompt invites a model to answer each
        line separately instead of treating them as one thought.
        """
        async with self._lock:
            return " ".join(line.text for line in self._lines).strip()

    async def as_dict(self) -> dict:
        lines = await self.lines()
        return {
            "lines": [line.as_dict() for line in lines],
            "question": " ".join(line.text for line in lines).strip(),
            "full": len(lines) >= MAX_LINES,
        }


async def publish(bus: Bus, draft: Draft, topic: str) -> None:
    """Push the current draft to every client.

    Called after each mutation rather than on a timer: the operator has just
    spoken and is waiting to read what the machine heard, so the round trip
    should be a push, not a poll.
    """
    await bus.publish(topic, await draft.as_dict())
