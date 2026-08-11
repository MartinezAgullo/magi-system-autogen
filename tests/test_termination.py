"""ConsensusTermination.

The behaviour worth pinning: AutoGen hands ``__call__`` only the messages
produced since the last call, so a condition that read just its argument would
never see all three advisors at once and would never fire.
"""

from __future__ import annotations

import pytest
from autogen_agentchat.base import TerminatedException
from autogen_agentchat.messages import StructuredMessage, TextMessage

from magi.models import MagiTurn
from magi.orchestrator.termination import CONSENSUS_SOURCE, ConsensusTermination

ADVISORS = ["MELCHIOR", "BALTHASAR", "CASPAR"]


def msg(source: str, agrees_with: list[str]) -> StructuredMessage[MagiTurn]:
    return StructuredMessage[MagiTurn](
        source=source,
        content=MagiTurn(
            position="A position long enough to be a real one.",
            summary="A summary.",
            agrees_with=agrees_with,
            confidence=0.8,
            critique=[],
        ),
    )


def unanimous_batch() -> list[StructuredMessage[MagiTurn]]:
    return [
        msg("MELCHIOR", ["BALTHASAR", "CASPAR"]),
        msg("BALTHASAR", ["MELCHIOR", "CASPAR"]),
        msg("CASPAR", ["MELCHIOR", "BALTHASAR"]),
    ]


async def test_fires_when_the_last_advisor_completes_the_agreement():
    condition = ConsensusTermination(ADVISORS)

    for message in unanimous_batch()[:2]:
        assert await condition([message]) is None

    stop = await condition([unanimous_batch()[2]])

    assert stop is not None
    assert stop.source == CONSENSUS_SOURCE
    assert condition.terminated


async def test_state_accumulates_across_calls():
    """The reason this class holds state at all: each call sees one turn, and no
    single call ever contains the whole vote."""
    condition = ConsensusTermination(ADVISORS)

    for message in unanimous_batch():
        result = await condition([message])

    assert result is not None
    assert set(condition.latest_turns) == set(ADVISORS)


async def test_does_not_fire_before_every_advisor_has_spoken():
    condition = ConsensusTermination(ADVISORS)

    assert await condition(unanimous_batch()[:2]) is None
    assert not condition.terminated


async def test_does_not_fire_on_a_split_vote():
    condition = ConsensusTermination(ADVISORS)

    result = await condition(
        [
            msg("MELCHIOR", ["BALTHASAR"]),
            msg("BALTHASAR", ["MELCHIOR"]),
            msg("CASPAR", []),
        ]
    )

    assert result is None
    assert not condition.terminated


async def test_a_later_turn_replaces_an_earlier_one():
    """An advisor that changes its mind must not leave its old vote counted."""
    condition = ConsensusTermination(ADVISORS)

    await condition(
        [
            msg("MELCHIOR", ["BALTHASAR", "CASPAR"]),
            msg("BALTHASAR", ["MELCHIOR", "CASPAR"]),
            msg("CASPAR", []),
        ]
    )
    assert not condition.terminated

    stop = await condition([msg("CASPAR", ["MELCHIOR", "BALTHASAR"])])

    assert stop is not None


async def test_non_structured_messages_are_ignored():
    condition = ConsensusTermination(ADVISORS)

    assert await condition([TextMessage(source="MELCHIOR", content="hello")]) is None
    assert condition.latest_turns == {}


async def test_calling_after_termination_raises():
    condition = ConsensusTermination(ADVISORS)
    await condition(unanimous_batch())

    with pytest.raises(TerminatedException):
        await condition([msg("MELCHIOR", [])])


async def test_reset_clears_the_accumulated_votes():
    condition = ConsensusTermination(ADVISORS)
    await condition(unanimous_batch())

    await condition.reset()

    assert not condition.terminated
    assert condition.latest_turns == {}


# ── Budget ───────────────────────────────────────────────────────────────────


def test_budget_leaves_room_for_whole_rounds():
    """MaxMessageTermination counts the seed task as a message. Without the +1
    the last round is cut before its final speaker — and since the order is
    fixed, the same advisor loses its closing turn every single debate."""
    from autogen_agentchat.conditions import ExternalTermination

    from magi.config import Settings
    from magi.orchestrator.termination import build_termination

    settings = Settings(max_rounds=3)
    condition, _ = build_termination(settings, ADVISORS, ExternalTermination())

    # 1 blind round outside the team + 2 deliberation rounds of 3 turns,
    # plus the seed message the team is started with.
    budget = _max_message_budget(condition)
    assert budget == 2 * len(ADVISORS) + 1


def test_a_single_round_budget_still_allows_one_full_round():
    from autogen_agentchat.conditions import ExternalTermination

    from magi.config import Settings
    from magi.orchestrator.termination import build_termination

    condition, _ = build_termination(
        Settings(max_rounds=1), ADVISORS, ExternalTermination()
    )

    assert _max_message_budget(condition) == len(ADVISORS) + 1


def _max_message_budget(condition) -> int:
    """Dig the MaxMessageTermination out of the composed OR condition."""
    from autogen_agentchat.conditions import MaxMessageTermination

    stack = [condition]
    while stack:
        node = stack.pop()
        if isinstance(node, MaxMessageTermination):
            return node._max_messages  # noqa: SLF001 — no public accessor
        stack.extend(getattr(node, "_conditions", []))
    raise AssertionError("no MaxMessageTermination in the composed condition")
