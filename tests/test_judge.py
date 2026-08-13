"""The pairwise pass over ambiguous agreement claims.

It exists because the advisors do not vote at the same moment. On one shared
thread the first speaker of a round votes on everyone else's previous positions
and the last speaker votes on their revised ones, so two advisors can hold the
same view and still fail to name each other. The mutual rule discards exactly
those claims, and this asks one question about each before it does.

The LLM is stubbed here. What is being pinned down is the arithmetic around it:
which pairs get asked, what happens to each answer, and what an unreachable
judge must not cause.
"""

from __future__ import annotations

import asyncio

import pytest

from magi.config import Settings
from magi.models import MagiTurn
from magi.orchestrator import judge as judge_module
from magi.orchestrator.judge import JudgeRuling, judge_pairs
from magi.personas import PersonaSet


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


def turn(agrees_with: list[str]) -> MagiTurn:
    return MagiTurn(
        position="A position long enough to be a real one.",
        summary="A summary.",
        agrees_with=agrees_with,
        confidence=0.8,
    )


TURNS = {
    "MELCHIOR": turn(["BALTHASAR"]),
    "BALTHASAR": turn(["CASPAR"]),
    "CASPAR": turn(["MELCHIOR"]),
}


def _stub(answers, seen=None):
    """Replace the single-question judge with a scripted one.

    `answers` maps a frozenset of the two names to the ruling that question
    returns, so a test can make one pair cosmetic and another substantive.
    """

    async def fake(settings, personas, turns, order, counter=None, on_activity=None):
        if seen is not None:
            seen.append(tuple(order))
        return answers[frozenset(turns)]

    return fake


async def test_a_cosmetic_pair_is_repaired_and_a_substantive_one_is_not(
    personas, monkeypatch
):
    monkeypatch.setattr(
        judge_module,
        "judge_disagreement",
        _stub(
            {
                frozenset({"MELCHIOR", "BALTHASAR"}): JudgeRuling(
                    substantive=False, reason="same answer, different words"
                ),
                frozenset({"BALTHASAR", "CASPAR"}): JudgeRuling(
                    substantive=True, reason="one ships, one does not"
                ),
            }
        ),
    )

    rulings = await judge_pairs(
        Settings(),
        personas,
        TURNS,
        [("BALTHASAR", "MELCHIOR"), ("BALTHASAR", "CASPAR")],
    )

    assert rulings.repaired == (("BALTHASAR", "MELCHIOR"),)
    assert rulings.substantive == (("BALTHASAR", "CASPAR"),)


async def test_an_unreachable_judge_leaves_the_votes_exactly_as_cast(
    personas, monkeypatch
):
    """The worst thing this system could do is manufacture consensus out of an
    outage. A pair the judge could not rule on appears in neither tuple, so it
    neither repairs an edge nor cancels the wider judge that follows."""
    monkeypatch.setattr(
        judge_module,
        "judge_disagreement",
        _stub({frozenset({"MELCHIOR", "BALTHASAR"}): None}),
    )

    rulings = await judge_pairs(
        Settings(), personas, TURNS, [("BALTHASAR", "MELCHIOR")]
    )

    assert rulings.repaired == ()
    assert rulings.substantive == ()


async def test_no_pairs_costs_no_calls(personas, monkeypatch):
    """A debate where every claim was reciprocated, or where nobody claimed
    anything, must not pay for a judge that has nothing to rule on."""
    calls = []
    monkeypatch.setattr(judge_module, "judge_disagreement", _stub({}, seen=calls))

    rulings = await judge_pairs(Settings(), personas, TURNS, [])

    assert rulings == judge_module.PairRulings()
    assert calls == []


async def test_only_the_two_positions_in_the_pair_are_shown(personas, monkeypatch):
    """One question per pair, not one question about everything: the answer is
    used per edge, and a judge shown all three positions can only ever rule on
    all three."""
    seen: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        judge_module,
        "judge_disagreement",
        _stub(
            {
                frozenset({"MELCHIOR", "BALTHASAR"}): JudgeRuling(
                    substantive=True, reason="."
                )
            },
            seen=seen,
        ),
    )

    await judge_pairs(Settings(), personas, TURNS, [("BALTHASAR", "MELCHIOR")])

    assert seen == [("BALTHASAR", "MELCHIOR")]


async def test_the_pairs_are_asked_concurrently(personas, monkeypatch):
    """Three pairs on the critical path of a voice interface, after the operator
    has already waited two to four minutes. The token cost is three calls either
    way; the wall clock does not have to be."""
    in_flight = 0
    peak = 0

    async def fake(settings, personas, turns, order, counter=None, on_activity=None):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return JudgeRuling(substantive=True, reason=".")

    monkeypatch.setattr(judge_module, "judge_disagreement", fake)

    await judge_pairs(
        Settings(),
        personas,
        TURNS,
        [("BALTHASAR", "MELCHIOR"), ("BALTHASAR", "CASPAR"), ("CASPAR", "MELCHIOR")],
    )

    assert peak == 3
