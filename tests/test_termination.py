"""ConsensusTermination.

The behaviour worth pinning: AutoGen hands ``__call__`` only the messages
produced since the last call, so a condition that read just its argument would
never see all three advisors at once and would never fire.
"""

from __future__ import annotations

import asyncio

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
    terminators = build_termination(settings, ADVISORS, ExternalTermination())

    # 1 blind round outside the team + 2 deliberation rounds of 3 turns,
    # plus the seed message the team is started with.
    budget = _max_message_budget(terminators.condition)
    assert budget == 2 * len(ADVISORS) + 1


def test_a_single_round_budget_still_allows_one_full_round():
    from autogen_agentchat.conditions import ExternalTermination

    from magi.config import Settings
    from magi.orchestrator.termination import build_termination

    terminators = build_termination(
        Settings(max_rounds=1), ADVISORS, ExternalTermination()
    )

    assert _max_message_budget(terminators.condition) == len(ADVISORS) + 1


def _max_message_budget(condition) -> int:
    """Dig the MaxMessageTermination out of the composed OR condition."""
    from autogen_agentchat.conditions import MaxMessageTermination

    stack = [condition]
    while stack:
        node = stack.pop()
        if isinstance(node, MaxMessageTermination):
            return node._max_messages  # noqa: SLF001 — no public accessor
        stack.extend(getattr(node, "_conditions", []))
        inner = getattr(node, "_inner", None)  # noqa: SLF001 — Latching wrapper
        if inner is not None:
            stack.append(inner)
    raise AssertionError("no MaxMessageTermination in the composed condition")


# ── Which condition fired ────────────────────────────────────────────────────
#
# The behaviour under test is a framework one, and it is counter-intuitive
# enough to be worth pinning: BaseGroupChatManager resets the whole termination
# stack the instant a child fires, before the caller of run_stream is scheduled
# again. So `condition.terminated` is False everywhere the orchestrator can look,
# and the latch is the only surviving record of what happened.


async def test_a_latch_survives_the_reset_that_follows_firing():
    from magi.constants import TERMINATED_BY_CONSENSUS
    from magi.orchestrator.termination import Latching

    latch = Latching(ConsensusTermination(ADVISORS), TERMINATED_BY_CONSENSUS)

    assert await latch(unanimous_batch()) is not None
    assert latch.fired
    assert latch.terminated

    await latch.reset()

    # The inner condition forgets, exactly as the group chat expects it to.
    assert not latch.terminated
    # The latch does not, which is the whole reason it exists.
    assert latch.fired


async def test_a_latch_that_never_fired_stays_clear():
    from magi.constants import TERMINATED_BY_CONSENSUS
    from magi.orchestrator.termination import Latching

    latch = Latching(ConsensusTermination(ADVISORS), TERMINATED_BY_CONSENSUS)

    assert await latch(unanimous_batch()[:2]) is None

    assert not latch.fired


async def test_timeout_is_reported_as_timeout_and_not_as_budget():
    """The gap this whole mechanism was added to close. Both mean "did not
    converge", but only one of them is an argument for a faster model."""
    from autogen_agentchat.conditions import ExternalTermination

    from magi.config import Settings
    from magi.constants import TERMINATED_BY_TIMEOUT
    from magi.orchestrator.termination import build_termination

    terminators = build_termination(
        Settings(debate_timeout_s=0.01), ADVISORS, ExternalTermination()
    )

    await asyncio.sleep(0.02)
    assert await terminators.condition([msg("MELCHIOR", [])]) is not None
    await terminators.condition.reset()

    assert terminators.fired() == TERMINATED_BY_TIMEOUT


async def test_consensus_outranks_a_budget_that_fired_on_the_same_turn():
    """A debate that converged on its last permitted turn converged. Reporting
    that as `budget` would count a success as a failure."""
    from autogen_agentchat.conditions import ExternalTermination

    from magi.config import Settings
    from magi.constants import TERMINATED_BY_CONSENSUS
    from magi.orchestrator.termination import build_termination

    # max_rounds=1 -> a budget of len(ADVISORS) + 1 messages, exhausted by the
    # same batch that completes the agreement. Note the fourth message keeps
    # MELCHIOR's vote intact: a later turn replaces an earlier one, so an
    # advisor that re-spoke with an empty `agrees_with` would break the tally
    # rather than the budget.
    terminators = build_termination(
        Settings(max_rounds=1), ADVISORS, ExternalTermination()
    )

    stop = await terminators.condition(
        unanimous_batch() + [msg("MELCHIOR", ["BALTHASAR", "CASPAR"])]
    )

    assert stop is not None
    assert terminators.fired() == TERMINATED_BY_CONSENSUS


async def test_barge_in_is_distinguishable_from_every_other_reason():
    from autogen_agentchat.conditions import ExternalTermination

    from magi.config import Settings
    from magi.constants import TERMINATED_BY_BARGE_IN
    from magi.orchestrator.termination import build_termination

    external = ExternalTermination()
    terminators = build_termination(Settings(), ADVISORS, external)

    external.set()
    assert await terminators.condition([msg("MELCHIOR", [])]) is not None

    assert terminators.fired() == TERMINATED_BY_BARGE_IN


def test_nothing_fired_is_not_silently_a_real_reason():
    from autogen_agentchat.conditions import ExternalTermination

    from magi.config import Settings
    from magi.constants import TERMINATED_BY_UNKNOWN
    from magi.orchestrator.termination import build_termination

    terminators = build_termination(Settings(), ADVISORS, ExternalTermination())

    assert terminators.fired() == TERMINATED_BY_UNKNOWN
