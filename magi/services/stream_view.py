"""The terminal view of a running debate.

Lives in the package rather than in ``scripts/ask.py`` because both the CLI and
the daemon need it. ``./launch.sh`` starts the node and the operator watches the
terminal; a token stream that only appeared in a developer script would be a
feature nobody running the actual node could ever see.

Off unless two things are true: at least one advisor has ``stream: true`` in the
persona file, and stdout is a terminal. The second matters on the Pi, where the
daemon runs under systemd and its stdout is the journal — streaming tokens into
a log file is noise, not a view.
"""

from __future__ import annotations

import os
import shutil
import sys

from magi.bus import Bus
from magi.constants import TOPIC_CHUNK

CYAN, DIM, NC = "\033[0;36m", "\033[2m", "\033[0m"

#: Marks the live text as something other than a log line. The screen mixes the
#: two — a streamed `position` forming, then the orchestrator's log line with
#: that turn's `summary` — and without a marker they read as duplicated logging
#: rather than as one thing replacing the other.
MARKER = "  ▸ "

CLEAR_ROW = "\r\033[K"
UP_AND_CLEAR = "\033[A\033[K"


def wanted() -> bool:
    """Whether a human is watching a terminal that can show this."""
    return sys.stdout.isatty()


def _columns() -> int:
    """The real width of the attached terminal.

    Asked of the device rather than of ``shutil.get_terminal_size``, which
    prefers the ``COLUMNS`` environment variable. That variable is frequently
    stale — inherited across an ``ssh`` session or a resize — and a width that
    is wrong by any amount makes the erase below clear the wrong number of
    rows, leaving fragments of a block that was supposed to disappear.
    """
    try:
        return max(os.get_terminal_size(sys.stdout.fileno()).columns, 20)
    except OSError:
        return max(shutil.get_terminal_size(fallback=(80, 24)).columns, 20)


async def render_stream(bus: Bus, quiet: bool = False) -> None:
    """Draw each advisor's `position` as it forms, then take it back down.

    The deltas are raw JSON, because every advisor answers as a ``MagiTurn``, so
    this pulls out the one field worth watching rather than printing braces at
    the operator.

    **The block is transient.** It is erased the moment the position is
    complete, leaving the screen where it was, and a second later the
    orchestrator logs that turn's one-line `summary` in its place. So the live
    text is a thing you watch happen rather than a thing that accumulates, and
    the permanent record stays exactly what it was before any of this existed.
    That is what stops the terminal from carrying two near-identical versions of
    every turn.

    Only ever drawn for a **sole** speaker. Phase A runs the advisors in
    parallel and a terminal has one cursor, so three live streams produce a
    marker flip per delta and a screen nobody can read. Concurrent streams are
    dropped rather than dumped whole when they finish: a block that was never
    watched forming adds nothing the `summary` log line does not already say.
    """
    queue = bus.subscribe(TOPIC_CHUNK)
    readers: dict[str, PositionReader] = {}
    drawn: list[str] = []  # the visible text of the block currently on screen
    owner: str | None = None
    # Advisors whose current position already lost some text to contention.
    # Once that has happened the block can never be honest: picking it up when
    # the screen frees would print a fragment starting mid-sentence under a
    # marker, which reads as a complete thought and is not one. Observed on the
    # Pi as `▸ BALTHASAR  that coverage will be limited and maintenance…`.
    partial: set[str] = set()

    def erase() -> None:
        """Take the current block back off the screen.

        Row arithmetic rather than anything cleverer, because the block is
        plain wrapped text: how many rows it occupies is its visible length over
        the terminal width. ANSI codes are excluded from that length by
        construction, since `drawn` holds only what was printed as characters.

        The one thing that breaks it is something else writing to stdout while
        the block is up, which would push it and leave the erase aimed at the
        wrong rows. Nothing does today: the block only exists while one advisor
        generates, and the orchestrator logs at the turn boundaries either side
        of that.
        """
        nonlocal owner
        if owner is None:
            return
        width = _columns()
        length = len(MARKER) + len(owner) + 2 + sum(len(part) for part in drawn)
        rows = max(1, -(-length // width))  # ceil
        # NC on every exit, not only the tidy one. The block opens with DIM, and
        # that attribute belongs to the terminal rather than to the text: an
        # erase that skipped the reset left every later line grey — the
        # orchestrator's own turn logs included — until some other block
        # happened to close properly and put it back. Observed on the Pi as a
        # whole blind round rendered in a different colour from the rest.
        sys.stdout.write(CLEAR_ROW + UP_AND_CLEAR * (rows - 1) + NC)
        sys.stdout.flush()
        drawn.clear()
        owner = None

    async def pump() -> None:
        nonlocal owner
        while True:
            event = await queue.get()
            if quiet:
                continue
            name = event["advisor"]

            reader = readers.setdefault(name, PositionReader())
            text, closed = reader.feed(event["text"])

            # A second advisor mid-position means the blind round: give up the
            # screen entirely rather than fight over it.
            sole = len([r for r in readers.values() if r.started and not r.done]) <= 1

            if not sole:
                # Nobody may hold the screen while two advisors are mid-position.
                # The block comes down here rather than on its owner's next
                # delta, so contention that starts between two of its chunks
                # does not leave it up for an arbitrary stretch.
                #
                # Its owner is marked too: its text has just left the screen, so
                # resuming it later would draw a continuation with no beginning.
                if owner is not None:
                    partial.add(owner)
                    erase()
                if text:
                    partial.add(name)
            elif text and name in partial:
                # Already lost its opening. Stays undrawable until it closes.
                pass
            elif text:
                if owner != name:
                    erase()
                    sys.stdout.write(f"{MARKER}{CYAN}{name}{NC} {DIM}")
                    owner = name
                sys.stdout.write(text)
                sys.stdout.flush()
                drawn.append(text)

            if closed:
                # Dropped so this advisor's next turn starts clean. There is
                # exactly one `position` per turn, so nothing after the close
                # can match by accident before then.
                readers.pop(name, None)
                partial.discard(name)
                if owner == name:
                    erase()

    try:
        await pump()
    finally:
        # Barge-in cancels this task mid-block. Without this the operator's
        # shell is left dim, and with half a sentence on it, after the debate
        # they interrupted.
        erase()


class PositionReader:
    """Pulls `position` out of a JSON object arriving one delta at a time.

    A state machine rather than a search over an accumulating buffer, because
    the deltas are far smaller than the thing being matched. Measured against
    gemma3:12b, the key alone arrives as four separate chunks:

        '{'  '\\n'  '  '  '"'  'position'  '":'  ' '  '"'  'A'  ' small' …

    so any approach that looks for `"position":` and then assumes the opening
    quote of the value is already in hand prints one stray space and stops. The
    three states exist because each of those boundaries can fall between chunks.
    """

    KEY = '"position":'
    # DONE is not decoration. Without a terminal state the reader falls back
    # into INSIDE on the next delta and starts printing `summary` as though it
    # were still the position, which is invisible in a live terminal because it
    # simply looks like the sentence continuing.
    SEEKING, OPENING, INSIDE, DONE = 0, 1, 2, 3

    def __init__(self) -> None:
        self._state = self.SEEKING
        self._pending = ""
        self._escaped = False

    @property
    def started(self) -> bool:
        """Whether this advisor is inside its `position` value right now.

        The renderer counts these rather than counting readers: a reader exists
        from an advisor's very first delta, which under the blind round is the
        opening brace, so counting readers would call the round contended
        several deltas before anything was actually competing for the screen.
        """
        return self._state is self.INSIDE

    @property
    def done(self) -> bool:
        return self._state is self.DONE

    def feed(self, text: str) -> tuple[str, bool]:
        """Return the printable text in this delta, and whether the value ended."""
        if self._state is self.DONE:
            return "", False

        self._pending += text

        if self._state is self.SEEKING:
            index = self._pending.find(self.KEY)
            if index < 0:
                # Keep only enough tail to still match a key split down the
                # middle by a chunk boundary.
                self._pending = self._pending[-len(self.KEY):]
                return "", False
            self._pending = self._pending[index + len(self.KEY):]
            self._state = self.OPENING

        if self._state is self.OPENING:
            quote = self._pending.find('"')
            if quote < 0:
                self._pending = ""  # only whitespace can legally precede it
                return "", False
            self._pending = self._pending[quote + 1:]
            self._state = self.INSIDE

        out: list[str] = []
        for char in self._pending:
            if self._escaped:
                # The value is one spoken paragraph, so a literal newline in it
                # would break the single-line-per-advisor layout.
                out.append(" " if char in "nrt" else char)
                self._escaped = False
                continue
            if char == "\\":
                self._escaped = True
                continue
            if char == '"':
                self._pending = ""
                self._state = self.DONE
                return "".join(out), True
            out.append(char)

        self._pending = ""
        return "".join(out), False
