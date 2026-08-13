"""Which row of votes the outcome is decided on.

Phase B produces a history, not a snapshot: each advisor speaks several times
and every turn revises the previous one. The tally needs one row, and picking
"whatever each advisor said last" mixes rounds — one advisor's answer to the
group's latest positions counted against another's answer to the previous ones.
A truncated debate reads as a disagreement nobody expressed.

So the rule is the deepest round every advisor reached, with one exception,
which is the second half of what these tests pin down: a debate that stopped
because the votes lined up is reported on the votes that stopped it.

The team is stubbed. What is under test is the orchestrator's bookkeeping around
it, including the one manager behaviour it has to survive — a termination stack
reset the instant a condition fires.
"""

from __future__ import annotations

import pytest
from autogen_agentchat.messages import StructuredMessage

from magi.config import Settings
from magi.constants import TERMINATED_BY_BUDGET, TERMINATED_BY_CONSENSUS
from magi.models import MagiTurn
from magi.orchestrator import magi as magi_module
from magi.orchestrator.magi import Magi
from magi.personas import PersonaSet

ADVISORS = ["MELCHIOR", "BALTHASAR", "CASPAR"]


@pytest.fixture
def personas() -> PersonaSet:
    return PersonaSet.model_validate(
        {
            "magi": [
                {"name": "MELCHIOR", "model": "a:1b", "system_prompt": "a"},
                {"name": "BALTHASAR", "model": "b:1b", "system_prompt": "b"},
                {"name": "CASPAR", "model": "c:1b", "system_prompt": "c"},
            ],
            "orchestrator": {"name": "MAGI", "model": "a:1b", "system_prompt": "o"},
        }
    )


def turn(agrees_with: list[str], position: str = "A real position.") -> MagiTurn:
    return MagiTurn(
        position=position, summary="A summary.", agrees_with=agrees_with, confidence=0.8
    )


def msg(source: str, agrees_with: list[str], position: str):
    return StructuredMessage[MagiTurn](
        source=source, content=turn(agrees_with, position)
    )


class _Advisor:
    def __init__(self, name: str) -> None:
        self.name = name


class _StubTeam:
    """Enough of a group chat to drive the orchestrator's loop.

    It replays a script and plays the one manager behaviour the row selection
    has to survive: the termination stack is consulted after every message and
    reset the instant it fires, which is why the votes cannot be read back off
    the condition once the run is over.
    """

    def __init__(self, condition, script) -> None:
        self._condition = condition
        self._script = script

    async def run_stream(self, task: str):
        for message in self._script:
            yield message
            if await self._condition([message]) is not None:
                await self._condition.reset()
                return


def _stub_phase_b(monkeypatch, script):
    monkeypatch.setattr(
        magi_module,
        "build_advisors",
        lambda *args, **kwargs: [_Advisor(name) for name in ADVISORS],
    )
    monkeypatch.setattr(
        magi_module,
        "build_team",
        lambda settings, personas, present, condition, counter: _StubTeam(
            condition, script
        ),
    )


BLIND = {name: turn([], f"{name} blind") for name in ADVISORS}


async def test_a_round_cut_off_before_its_last_speaker_is_not_tallied(
    personas, monkeypatch
):
    """The failure this rule exists for. MELCHIOR opened round 3 by dropping its
    agreement with BALTHASAR, and the budget ran out before the other two
    answered. Counting that against their round-2 votes turns a MAJORITY into a
    DEADLOCK on the strength of one unanswered turn."""
    _stub_phase_b(
        monkeypatch,
        [
            msg("MELCHIOR", ["BALTHASAR"], "M r2"),
            msg("BALTHASAR", ["MELCHIOR"], "B r2"),
            msg("CASPAR", [], "C r2"),
            msg("MELCHIOR", [], "M r3"),
        ],
    )
    # One deliberation round of three speakers, plus the seed: the budget fires
    # on the fourth message, which is the round-3 turn nobody answers.
    magi = Magi(Settings(max_rounds=2), personas)

    result = await magi._deliberate("q", BLIND)  # noqa: SLF001

    assert result.tallied_round == 2
    assert {name: t.position for name, t in result.turns.items()} == {
        "MELCHIOR": "M r2",
        "BALTHASAR": "B r2",
        "CASPAR": "C r2",
    }
    assert result.terminated_by == TERMINATED_BY_BUDGET


async def test_a_complete_round_is_tallied_whole(personas, monkeypatch):
    _stub_phase_b(
        monkeypatch,
        [
            msg("MELCHIOR", [], "M r2"),
            msg("BALTHASAR", [], "B r2"),
            msg("CASPAR", [], "C r2"),
        ],
    )
    magi = Magi(Settings(max_rounds=3), personas)

    result = await magi._deliberate("q", BLIND)  # noqa: SLF001

    assert result.tallied_round == 2
    assert [t.position for t in result.turns.values()] == ["M r2", "B r2", "C r2"]


async def test_a_debate_stopped_by_consensus_is_reported_on_the_votes_that_stopped_it(
    personas, monkeypatch
):
    """The exception, and it is not optional. ConsensusTermination watches each
    advisor's latest vote; here MELCHIOR's round-3 turn completes the agreement
    and ends the debate. Falling back to the last complete round would report
    DEADLOCK on a run whose stated reason for ending is that it converged."""
    _stub_phase_b(
        monkeypatch,
        [
            msg("MELCHIOR", [], "M r2"),
            msg("BALTHASAR", ["MELCHIOR", "CASPAR"], "B r2"),
            msg("CASPAR", ["MELCHIOR", "BALTHASAR"], "C r2"),
            msg("MELCHIOR", ["BALTHASAR", "CASPAR"], "M r3"),
        ],
    )
    magi = Magi(Settings(max_rounds=4), personas)

    result = await magi._deliberate("q", BLIND)  # noqa: SLF001

    assert result.terminated_by == TERMINATED_BY_CONSENSUS
    assert result.turns["MELCHIOR"].position == "M r3"
    # A floor, not the row: everyone spoke at least twice, MELCHIOR three times.
    assert result.tallied_round == 2


async def test_the_blind_round_stands_alone_when_deliberation_produced_nothing(
    personas, monkeypatch
):
    """Every advisor answered blind and the group chat died before anyone spoke
    again. Round 1 is a complete round like any other."""
    _stub_phase_b(monkeypatch, [])
    magi = Magi(Settings(), personas)

    result = await magi._deliberate("q", BLIND)  # noqa: SLF001

    assert result.tallied_round == 1
    assert [t.position for t in result.turns.values()] == [
        "MELCHIOR blind",
        "BALTHASAR blind",
        "CASPAR blind",
    ]


async def test_an_advisor_missing_from_the_blind_round_does_not_hold_the_row_back(
    personas, monkeypatch
):
    """Two advisors answered, the third never did. The row is theirs, at the
    depth they reached, rather than depth zero because of an absent seat."""
    _stub_phase_b(
        monkeypatch,
        [msg("MELCHIOR", [], "M r2"), msg("BALTHASAR", [], "B r2")],
    )
    magi = Magi(Settings(max_rounds=3), personas)

    result = await magi._deliberate(  # noqa: SLF001
        "q", {name: BLIND[name] for name in ("MELCHIOR", "BALTHASAR")}
    )

    assert result.tallied_round == 2
    assert set(result.turns) == {"MELCHIOR", "BALTHASAR"}


# ── The transcript ───────────────────────────────────────────────────────────
#
# The tally reads one row. The record keeps all of them, and the two are not the
# same thing: the row is the result, the transcript is the evidence, and mode
# collapse — three advisors reproducing each other's sentences — is visible only
# in the second one. The bug these replace kept the last turn per advisor and
# stamped every one of them with the final round, so a three-round debate was
# recorded as three advisors speaking once each, in the last round.


async def test_the_transcript_keeps_every_turn_in_the_order_it_was_spoken(
    personas, monkeypatch
):
    _stub_phase_b(
        monkeypatch,
        [
            msg("MELCHIOR", [], "M r2"),
            msg("BALTHASAR", [], "B r2"),
            msg("CASPAR", [], "C r2"),
            msg("MELCHIOR", [], "M r3"),
            msg("BALTHASAR", [], "B r3"),
            msg("CASPAR", [], "C r3"),
        ],
    )
    magi = Magi(Settings(max_rounds=3), personas)

    result = await magi._deliberate("q", BLIND)  # noqa: SLF001

    assert [(t.advisor, t.round_index) for t in result.transcript] == [
        ("MELCHIOR", 1), ("BALTHASAR", 1), ("CASPAR", 1),
        ("MELCHIOR", 2), ("BALTHASAR", 2), ("CASPAR", 2),
        ("MELCHIOR", 3), ("BALTHASAR", 3), ("CASPAR", 3),
    ]
    assert [t.turn.position for t in result.transcript[:3]] == [
        "MELCHIOR blind", "BALTHASAR blind", "CASPAR blind",
    ]


async def test_the_turns_that_decided_the_outcome_are_the_ones_marked(
    personas, monkeypatch
):
    """A turn that arrived after the row the tally read is still in the record —
    it was generated, paid for and shown on the console. It must not be
    mistakable for a vote that counted, or the transcript would appear to
    contradict the outcome stored beside it."""
    _stub_phase_b(
        monkeypatch,
        [
            msg("MELCHIOR", ["BALTHASAR"], "M r2"),
            msg("BALTHASAR", ["MELCHIOR"], "B r2"),
            msg("CASPAR", [], "C r2"),
            msg("MELCHIOR", [], "M r3"),
        ],
    )
    magi = Magi(Settings(max_rounds=2), personas)

    result = await magi._deliberate("q", BLIND)  # noqa: SLF001

    tallied = [(t.advisor, t.round_index) for t in result.transcript if t.tallied]
    assert tallied == [("MELCHIOR", 2), ("BALTHASAR", 2), ("CASPAR", 2)]
    # The unanswered round-3 turn is kept, and kept out of the arithmetic.
    assert result.transcript[-1].round_index == 3
    assert result.transcript[-1].tallied is False


async def test_a_consensus_row_is_marked_even_though_it_spans_two_rounds(
    personas, monkeypatch
):
    """ConsensusTermination stops on each advisor's latest vote, so the row that
    ended the debate sits at two different depths. Marking by advisor and round
    would mark the wrong turns here; the marking follows the objects the tally
    actually read."""
    _stub_phase_b(
        monkeypatch,
        [
            msg("MELCHIOR", [], "M r2"),
            msg("BALTHASAR", ["MELCHIOR", "CASPAR"], "B r2"),
            msg("CASPAR", ["MELCHIOR", "BALTHASAR"], "C r2"),
            msg("MELCHIOR", ["BALTHASAR", "CASPAR"], "M r3"),
        ],
    )
    magi = Magi(Settings(max_rounds=4), personas)

    result = await magi._deliberate("q", BLIND)  # noqa: SLF001

    assert result.terminated_by == TERMINATED_BY_CONSENSUS
    assert {(t.advisor, t.round_index) for t in result.transcript if t.tallied} == {
        ("MELCHIOR", 3), ("BALTHASAR", 2), ("CASPAR", 2),
    }


async def test_a_debate_that_never_deliberated_still_has_a_transcript(
    personas, monkeypatch
):
    """Fewer than two advisors answered, so the group chat never ran. The blind
    round is the whole debate and the whole record of it."""
    _stub_phase_b(monkeypatch, [])
    magi = Magi(Settings(), personas)

    result = await magi._deliberate(  # noqa: SLF001
        "q", {"MELCHIOR": BLIND["MELCHIOR"]}
    )

    assert [(t.advisor, t.round_index, t.tallied) for t in result.transcript] == [
        ("MELCHIOR", 1, True)
    ]
