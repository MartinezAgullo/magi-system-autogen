"""Per-agent activity: who has an LLM call open, right now.

Emitted at the model client rather than at a phase boundary, for the same
reason the call counting lives there: it is the only place that sees every
call, including the ones AutoGen makes on its own behalf. A console fed from
the orchestrator's phases would go dark for the 90 s of deliberation and light
up only when a turn had already landed.
"""

from __future__ import annotations

import pytest

from magi.bus import Bus
from magi.config import Settings
from magi.constants import TOPIC_ACTIVITY
from magi.orchestrator.clients import InstrumentedChatClient, build_client
from magi.personas import PersonaSet


@pytest.fixture
def personas() -> PersonaSet:
    return PersonaSet.model_validate(
        {
            "magi": [
                {"name": "MELCHIOR", "model": "a:1b", "system_prompt": "a"},
                {"name": "BALTHASAR", "model": "b:1b", "system_prompt": "b"},
            ],
            "orchestrator": {"name": "MAGI", "model": "a:1b", "system_prompt": "o"},
        }
    )


class Recorder:
    def __init__(self) -> None:
        self.seen: list[tuple[str, bool]] = []

    async def __call__(self, name: str, busy: bool) -> None:
        self.seen.append((name, busy))


def _client(personas: PersonaSet, hook, counter=None) -> InstrumentedChatClient:
    return build_client(
        Settings(), personas, personas.magi[0], counter, on_activity=hook
    )


async def test_a_call_is_announced_and_then_withdrawn(personas, monkeypatch):
    hook = Recorder()
    client = _client(personas, hook)

    async def ok(*args, **kwargs):
        # Mid-call, the console must already show this advisor as busy —
        # announcing on completion would light the indicator for the moment the
        # work has just stopped, which is exactly backwards.
        assert hook.seen == [("MELCHIOR", True)]
        return _Result()

    monkeypatch.setattr(type(client).__mro__[1], "create", ok)

    await client.create([])

    assert hook.seen == [("MELCHIOR", True), ("MELCHIOR", False)]


async def test_a_failed_call_still_withdraws(personas, monkeypatch):
    """The `finally` this pins is load-bearing. An advisor whose model is
    unreachable would otherwise be drawn as still thinking for the rest of the
    debate, and the console's one rule is that it never asserts more than the
    daemon told it."""
    hook = Recorder()
    client = _client(personas, hook)

    async def boom(*args, **kwargs):
        raise RuntimeError("the Spark is off")

    monkeypatch.setattr(type(client).__mro__[1], "create", boom)

    with pytest.raises(RuntimeError):
        await client.create([])

    assert hook.seen == [("MELCHIOR", True), ("MELCHIOR", False)]


async def test_a_broken_hook_does_not_break_the_debate(personas, monkeypatch):
    """Reporting activity is cosmetic; generating the answer is not. A console
    that cannot be told an advisor is thinking must not cost the answer."""

    async def broken(name: str, busy: bool) -> None:
        raise RuntimeError("the socket went away")

    client = _client(personas, broken)

    async def ok(*args, **kwargs):
        return _Result()

    monkeypatch.setattr(type(client).__mro__[1], "create", ok)

    assert await client.create([]) is not None


async def test_no_hook_is_the_normal_case(personas, monkeypatch):
    """scripts/ask.py runs with no bus at all."""
    client = build_client(Settings(), personas, personas.magi[0])

    async def ok(*args, **kwargs):
        return _Result()

    monkeypatch.setattr(type(client).__mro__[1], "create", ok)

    assert await client.create([]) is not None


# ── Streaming ────────────────────────────────────────────────────────────────
#
# `model_client_stream=True` makes an AssistantAgent call `create_stream` and
# never `create`. Before these, only `create` was instrumented, so switching one
# agent to streaming would have silently zeroed its call count, its tokens, its
# GenAI span and its activity events, with no error anywhere. Verified live
# against gemma3:12b and nemotron3:33b on 2026-08-12: identical counts in both
# modes.


def _stub_stream(chunks, usage_out=42, finish="stop"):
    from autogen_core.models import CreateResult, RequestUsage

    async def stream(self, *args, **kwargs):
        stream.kwargs = kwargs
        for chunk in chunks:
            yield chunk
        yield CreateResult(
            finish_reason=finish,
            content="".join(chunks),
            usage=RequestUsage(prompt_tokens=7, completion_tokens=usage_out),
            cached=False,
        )

    return stream


async def _drain(client, **kwargs):
    return [item async for item in client.create_stream([], **kwargs)]


async def test_a_streamed_call_is_counted_like_any_other(personas, monkeypatch):
    counter_hook = Recorder()
    from magi.services.metrics import CallCounter

    counter = CallCounter()
    client = _client(personas, counter_hook, counter=counter)
    monkeypatch.setattr(type(client).__mro__[1], "create_stream", _stub_stream(["a", "b"]))

    items = await _drain(client)

    assert [i for i in items if isinstance(i, str)] == ["a", "b"]
    # The number the engine comparison is about. Zero here would look like a
    # free call rather than a missing measurement.
    assert counter.total.calls == 1
    assert counter.total.completion_tokens == 42
    assert counter_hook.seen == [("MELCHIOR", True), ("MELCHIOR", False)]


async def test_streaming_asks_for_usage_because_it_is_not_sent_by_default(
    personas, monkeypatch
):
    """OpenAI-compatible streaming returns no token counts unless
    `stream_options.include_usage` is set, and AutoGen only sets it when
    `include_usage` is passed. Left alone, every streamed call would report zero
    tokens."""
    stub = _stub_stream(["a"])
    client = _client(personas, Recorder())
    monkeypatch.setattr(type(client).__mro__[1], "create_stream", stub)

    await _drain(client)

    assert stub.kwargs["include_usage"] is True


async def test_an_explicit_include_usage_is_respected(personas, monkeypatch):
    stub = _stub_stream(["a"])
    client = _client(personas, Recorder())
    monkeypatch.setattr(type(client).__mro__[1], "create_stream", stub)

    await _drain(client, include_usage=False)

    assert stub.kwargs["include_usage"] is False


async def test_an_abandoned_stream_still_withdraws(personas, monkeypatch):
    """A consumer that breaks out mid-stream — barge-in does exactly this —
    must not leave the advisor drawn as still generating."""
    hook = Recorder()
    client = _client(personas, hook)
    monkeypatch.setattr(
        type(client).__mro__[1], "create_stream", _stub_stream(["a", "b", "c"])
    )

    stream = client.create_stream([])
    assert await anext(stream) == "a"
    await stream.aclose()

    assert hook.seen == [("MELCHIOR", True), ("MELCHIOR", False)]


async def test_a_truncated_stream_is_reported_rather_than_retried(
    personas, monkeypatch, caplog
):
    """The length retry cannot exist on this path: streaming does not raise
    LengthFinishReasonError, and by the time `finish_reason` says "length" the
    chunks are already with the consumer. Saying so loudly is the most that can
    be done, and it is the concrete reason not to stream MELCHIOR."""
    import logging

    client = _client(personas, Recorder())
    monkeypatch.setattr(
        type(client).__mro__[1], "create_stream", _stub_stream(["a"], finish="length")
    )

    with caplog.at_level(logging.WARNING):
        await _drain(client)

    assert "no retry is possible" in caplog.text


async def test_the_orchestrator_publishes_activity_to_the_bus(personas):
    """The seam between the client hook and the console."""
    from magi.orchestrator.magi import Magi

    bus = Bus()
    queue = bus.subscribe(TOPIC_ACTIVITY)
    magi = Magi(Settings(), personas, bus=bus)

    await magi._publish_activity("MELCHIOR", True)  # noqa: SLF001

    assert queue.get_nowait() == {"advisor": "MELCHIOR", "busy": True}


class _Result:
    """Enough of a CreateResult for the counting path to read."""

    usage = None
