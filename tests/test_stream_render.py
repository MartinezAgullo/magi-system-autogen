"""Pulling `position` out of a JSON object that arrives one delta at a time.

The deltas are much smaller than the key being matched. Captured live from
`gemma3:12b` on 2026-08-12, the opening of a turn arrives as:

    '{'  '\\n'  '  '  '"'  'position'  '":'  ' '  '"'  'A'  ' small'  …

Every boundary in `"position": "` can therefore fall between two chunks, which
is what the first version of this got wrong: it searched an accumulating buffer
for the key and then assumed the value's opening quote was already in hand, so
it printed one stray space per advisor and stopped.
"""

from __future__ import annotations

from magi.services.stream_view import PositionReader

#: Exactly as captured from the wire.
REAL_CHUNKS = ["{", "\n", "  ", '"', "position", '":', " ", '"', "A", " small", " team"]


def drain(reader: PositionReader, chunks) -> tuple[str, bool]:
    out, closed = [], False
    for chunk in chunks:
        text, done = reader.feed(chunk)
        out.append(text)
        closed = closed or done
    return "".join(out), closed


def test_a_key_split_across_four_chunks_still_matches():
    """The regression this class exists for."""
    text, closed = drain(PositionReader(), REAL_CHUNKS)

    assert text == "A small team"
    assert not closed


def test_the_whole_object_in_one_chunk_also_works():
    text, closed = drain(PositionReader(), ['{"position": "All at once", "summary": "s"}'])

    assert text == "All at once"
    assert closed


def test_the_value_ends_at_its_closing_quote():
    text, closed = drain(PositionReader(), REAL_CHUNKS + ['."', ', "summary"'])

    assert text == "A small team."
    assert closed


def test_nothing_after_the_close_is_emitted():
    reader = PositionReader()
    drain(reader, ['{"position": "Done."'])

    assert reader.feed(', "summary": "not this"') == ("", False)


def test_escaped_quotes_do_not_end_the_value():
    text, closed = drain(PositionReader(), ['{"position": "He said \\"no\\" firmly."'])

    assert text == 'He said "no" firmly.'
    assert closed


def test_an_escape_split_across_chunks_is_still_one_escape():
    text, _ = drain(PositionReader(), ['{"position": "a\\', 'nb"'])

    # A literal newline inside the value would break the one-line-per-advisor
    # layout, so it renders as a space.
    assert text == "a b"


def test_a_newline_escape_becomes_a_space():
    text, _ = drain(PositionReader(), ['{"position": "one\\ntwo"'])

    assert text == "one two"


def test_fields_before_position_are_skipped():
    """Schema order puts `position` first today, but nothing enforces that at
    run time, and a model that emits another field first must not have it
    printed as if it were the position."""
    text, closed = drain(PositionReader(), ['{"summary": "s", "position": "the real one"'])

    assert text == "the real one"
    assert closed


def test_a_chunk_stream_with_no_position_emits_nothing():
    text, closed = drain(PositionReader(), ['{"summary"', ': "s", "confid', 'ence": 0.8}'])

    assert text == ""
    assert not closed


def test_the_seeking_buffer_does_not_grow_without_bound():
    """A long turn is thousands of deltas. Holding them all to search for a key
    that has already gone past would make the renderer the expensive part."""
    reader = PositionReader()
    for _ in range(500):
        reader.feed("padding that will never match ")

    assert len(reader._pending) <= len(PositionReader.KEY)  # noqa: SLF001


# ── The transient block ──────────────────────────────────────────────────────
#
# The live text is drawn and then taken back off the screen, so that the
# orchestrator's own log line for that turn lands where it was. Without the
# erase the terminal carries two near-identical versions of every turn — the
# streamed `position` and the logged `summary` — which reads as duplicated
# logging rather than as one thing replacing the other.

import asyncio  # noqa: E402
import re  # noqa: E402

import pytest  # noqa: E402

from magi.bus import Bus  # noqa: E402
from magi.constants import TOPIC_CHUNK  # noqa: E402
from magi.services import stream_view  # noqa: E402


async def _render(chunks, monkeypatch, columns=200):
    """Feed deltas through the real renderer and capture what hit stdout."""
    written: list[str] = []
    monkeypatch.setattr(stream_view.sys.stdout, "write", written.append)
    monkeypatch.setattr(stream_view.sys.stdout, "flush", lambda: None)
    monkeypatch.setattr(stream_view, "_columns", lambda: columns)

    bus = Bus()
    task = asyncio.create_task(stream_view.render_stream(bus))
    # The renderer subscribes inside its own coroutine, so it has to be given a
    # slot before anything is published or the first deltas go nowhere.
    await asyncio.sleep(0)
    for advisor, text in chunks:
        await bus.publish(TOPIC_CHUNK, {"advisor": advisor, "text": text})
        await asyncio.sleep(0)
    task.cancel()
    return "".join(written)


async def test_the_marker_distinguishes_live_text_from_a_log_line(monkeypatch):
    out = await _render([("BALTHASAR", '{"position": "A small team')], monkeypatch)

    assert stream_view.MARKER in out
    assert "BALTHASAR" in out
    assert "A small team" in out


async def test_the_block_is_erased_once_the_position_completes(monkeypatch):
    out = await _render(
        [("BALTHASAR", '{"position": "A small team"'), ], monkeypatch
    )

    # One row at 200 columns, so one clear and no cursor-up.
    assert out.endswith(stream_view.CLEAR_ROW + stream_view.NC)
    assert stream_view.UP_AND_CLEAR not in out


async def test_a_wrapped_block_is_erased_row_by_row(monkeypatch):
    """The erase has to know how many rows it drew, or it leaves fragments of
    the block above the log line that replaced it."""
    long = "x" * 90
    out = await _render(
        [("BALTHASAR", f'{{"position": "{long}"')], monkeypatch, columns=40
    )

    # marker(4) + name(9) + 2 + 90 = 105 chars over 40 columns -> 3 rows.
    assert out.count(stream_view.UP_AND_CLEAR) == 2


async def test_concurrent_advisors_are_not_drawn_at_all(monkeypatch):
    """Phase A runs them in parallel and a terminal has one cursor. Dumping the
    loser whole when it finishes would just duplicate its `summary` log line."""
    out = await _render(
        [
            ("BALTHASAR", '{"position": "from balthasar'),
            ("CASPAR", '{"position": "from caspar'),
            ("CASPAR", ' continuing'),
        ],
        monkeypatch,
    )

    assert "from balthasar" in out  # sole speaker until the second one starts
    assert "from caspar" not in out
    assert "continuing" not in out


async def test_a_later_turn_by_the_same_advisor_draws_again(monkeypatch):
    """Deliberation asks each advisor repeatedly; the reader is dropped on close
    so the next turn has to match the key from scratch."""
    out = await _render(
        [
            ("BALTHASAR", '{"position": "round two"'),
            ("BALTHASAR", '{"position": "round three"'),
        ],
        monkeypatch,
    )

    assert "round two" in out
    assert "round three" in out


async def test_a_turn_that_lost_text_to_contention_is_never_picked_up(monkeypatch):
    """Observed on the Pi: `▸ BALTHASAR  that coverage will be limited…`.

    BALTHASAR was suppressed while CASPAR held the screen, then started being
    drawn the moment CASPAR finished — from the middle of its sentence, under a
    marker, reading as a complete thought. A block missing its opening can never
    be honest, so once suppressed it stays suppressed until the turn closes."""
    out = await _render(
        [
            ("CASPAR", '{"position": "caspar has the screen'),  # sole, drawn
            ("BALTHASAR", '{"position": "opening words'),       # contended, dropped
            ("CASPAR", '."'),                                   # caspar closes, erases
            ("BALTHASAR", " and the rest of the sentence"),     # sole again
        ],
        monkeypatch,
    )

    assert "caspar has the screen" in out
    assert "opening words" not in out
    # The point: picking it up here would print a marker followed by a sentence
    # with no beginning.
    assert "and the rest of the sentence" not in out


async def test_the_advisor_is_drawable_again_on_its_next_turn(monkeypatch):
    """The suppression is per turn, not permanent. It clears on close, so the
    advisor is drawn normally the next time it speaks."""
    out = await _render(
        [
            ("CASPAR", '{"position": "caspar has the screen'),
            ("BALTHASAR", '{"position": "suppressed opening'),
            ("CASPAR", '."'),
            ("BALTHASAR", ' suppressed tail"'),   # closes, clearing the suppression
            ("BALTHASAR", '{"position": "a clean new turn'),
        ],
        monkeypatch,
    )

    assert "suppressed" not in out
    assert "a clean new turn" in out


# ── Terminal attributes ──────────────────────────────────────────────────────
#
# The block opens with DIM, which is terminal state rather than a property of
# the text. Observed on the Pi: a block erased because of contention skipped the
# reset, so every line after it — the orchestrator's own turn logs included —
# rendered grey until some later block happened to close tidily and put it back.


def _ends_reset(out: str) -> bool:
    """Whether the terminal is left with no attribute set."""
    codes = re.findall(r"\x1b\[[0-9;]*m", out)
    return bool(codes) and codes[-1] == stream_view.NC


async def test_a_tidy_close_leaves_the_terminal_reset(monkeypatch):
    out = await _render([("BALTHASAR", '{"position": "done."')], monkeypatch)

    assert _ends_reset(out)


async def test_an_erase_forced_by_contention_also_resets(monkeypatch):
    """The regression. DIM leaked out of this path and coloured everything
    printed afterwards, including lines this module never wrote."""
    out = await _render(
        [
            ("BALTHASAR", '{"position": "drawn while sole'),
            ("CASPAR", '{"position": "now contended'),
        ],
        monkeypatch,
    )

    assert _ends_reset(out)


async def test_cancellation_mid_block_resets_too(monkeypatch):
    """Barge-in cancels this task while a block is on screen. Leaving the
    operator's shell dim afterwards would outlive the debate entirely."""
    written: list[str] = []
    monkeypatch.setattr(stream_view.sys.stdout, "write", written.append)
    monkeypatch.setattr(stream_view.sys.stdout, "flush", lambda: None)
    monkeypatch.setattr(stream_view, "_columns", lambda: 200)

    bus = Bus()
    task = asyncio.create_task(stream_view.render_stream(bus))
    await asyncio.sleep(0)
    await bus.publish(TOPIC_CHUNK, {"advisor": "BALTHASAR", "text": '{"position": "mid'})
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _ends_reset("".join(written))
