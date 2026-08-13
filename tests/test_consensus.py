"""The vote tally — the shared contract with the sibling repo.

If these two implementations tallied differently, the benchmark would be
measuring the tally instead of the framework, so this module is the one that
most needs to be pinned down by tests rather than by reading.
"""

from __future__ import annotations

from magi.models import MagiTurn, Outcome
from magi.orchestrator.consensus import (
    asymmetric_pairs,
    latest_complete_round,
    tally,
)

ADVISORS = ("MELCHIOR", "BALTHASAR", "CASPAR")


def turn(agrees_with: list[str], confidence: float = 0.8) -> MagiTurn:
    return MagiTurn(
        position="A position long enough to be a real one.",
        summary="A summary.",
        agrees_with=agrees_with,
        confidence=confidence,
        critique=[],
    )


def test_all_three_mutually_agreeing_is_unanimous():
    result = tally(
        {
            "MELCHIOR": turn(["BALTHASAR", "CASPAR"]),
            "BALTHASAR": turn(["MELCHIOR", "CASPAR"]),
            "CASPAR": turn(["MELCHIOR", "BALTHASAR"]),
        }
    )

    assert result.outcome is Outcome.UNANIMOUS
    assert result.majority == ADVISORS[::1] or set(result.majority) == set(ADVISORS)
    assert result.dissent == ()


def test_two_mutually_agreeing_is_a_majority_with_named_dissent():
    result = tally(
        {
            "MELCHIOR": turn(["BALTHASAR"]),
            "BALTHASAR": turn(["MELCHIOR"]),
            "CASPAR": turn([]),
        }
    )

    assert result.outcome is Outcome.MAJORITY
    assert set(result.majority) == {"MELCHIOR", "BALTHASAR"}
    assert result.dissent == ("CASPAR",)


def test_one_sided_agreement_is_not_agreement():
    """The load-bearing rule. BALTHASAR being agreeable about MELCHIOR, with
    MELCHIOR saying nothing back, is one advisor conceding — not two
    converging. Counting it would report a consensus nobody reached."""
    result = tally(
        {
            "MELCHIOR": turn([]),
            "BALTHASAR": turn(["MELCHIOR"]),
            "CASPAR": turn(["MELCHIOR"]),
        }
    )

    assert result.outcome is Outcome.DEADLOCK


def test_a_cycle_forms_no_bloc():
    """A→B→C→A has no mutual edge anywhere: nobody actually paired up."""
    result = tally(
        {
            "MELCHIOR": turn(["BALTHASAR"]),
            "BALTHASAR": turn(["CASPAR"]),
            "CASPAR": turn(["MELCHIOR"]),
        }
    )

    assert result.outcome is Outcome.DEADLOCK


def test_three_way_split_is_deadlock():
    result = tally({name: turn([]) for name in ADVISORS})

    assert result.outcome is Outcome.DEADLOCK
    assert len(result.majority) == 1


def test_self_agreement_is_ignored():
    """Models do list themselves in `agrees_with`. Observed on a live run, and
    counting it would make every lone advisor its own majority."""
    result = tally(
        {
            "MELCHIOR": turn(["MELCHIOR"]),
            "BALTHASAR": turn(["BALTHASAR"]),
            "CASPAR": turn(["CASPAR"]),
        }
    )

    assert result.outcome is Outcome.DEADLOCK


def test_unknown_names_are_dropped():
    """A model will occasionally invent a fourth advisor. It must not become a
    phantom vote."""
    result = tally(
        {
            "MELCHIOR": turn(["GASPAR", "BALTHASAR"]),
            "BALTHASAR": turn(["MELCHIOR"]),
            "CASPAR": turn([]),
        }
    )

    assert result.outcome is Outcome.MAJORITY
    assert set(result.majority) == {"MELCHIOR", "BALTHASAR"}


def test_names_are_matched_case_insensitively():
    result = tally(
        {
            "MELCHIOR": turn(["balthasar"]),
            "BALTHASAR": turn(["Melchior"]),
            "CASPAR": turn([]),
        }
    )

    assert result.outcome is Outcome.MAJORITY


def test_two_advisors_agreeing_is_unanimous_not_majority():
    """After a model drops out the debate continues with two. Both agreeing is
    the strongest result available; there is no majority with two voters."""
    result = tally({"MELCHIOR": turn(["CASPAR"]), "CASPAR": turn(["MELCHIOR"])})

    assert result.outcome is Outcome.UNANIMOUS


def test_two_advisors_disagreeing_is_deadlock():
    result = tally({"MELCHIOR": turn([]), "CASPAR": turn([])})

    assert result.outcome is Outcome.DEADLOCK


def test_a_lone_advisor_is_never_unanimous():
    """One model talking to itself is not a deliberation, and must never be
    reported as though three had agreed."""
    result = tally({"MELCHIOR": turn([])})

    assert result.outcome is not Outcome.UNANIMOUS


def test_no_turns_at_all_is_deadlock():
    assert tally({}).outcome is Outcome.DEADLOCK


# ── Reporting a debate that lost an advisor ──────────────────────────────────


def test_verdict_prompt_names_advisors_that_never_answered():
    """Reporting UNANIMOUS from two voices without mentioning the third is
    technically true and practically a lie. Observed on the Pi, where a model
    failed schema validation in the blind round."""
    from magi.orchestrator.prompts import verdict_task

    turns = {"MELCHIOR": turn(["CASPAR"]), "CASPAR": turn(["MELCHIOR"])}

    task = verdict_task(
        "q", turns, ADVISORS, Outcome.UNANIMOUS, ("MELCHIOR", "CASPAR"), (),
        absent=["BALTHASAR"],
    )

    assert "BALTHASAR did not take part" in task
    assert "2 advisors, not 3" in task


def test_a_full_roster_adds_no_such_warning():
    from magi.orchestrator.prompts import verdict_task

    turns = {n: turn([]) for n in ADVISORS}

    task = verdict_task("q", turns, ADVISORS, Outcome.DEADLOCK, (), ADVISORS)

    assert "did not take part" not in task


# ── One round of votes, not a mixture of rounds ──────────────────────────────


def test_the_deepest_row_everyone_reached_is_the_one_tallied():
    history = {
        "MELCHIOR": [turn([]), turn(["CASPAR"])],
        "BALTHASAR": [turn([]), turn([])],
        "CASPAR": [turn([]), turn(["MELCHIOR"])],
    }

    row = latest_complete_round(history)

    assert {name: t.agrees_with for name, t in row.items()} == {
        "MELCHIOR": ["CASPAR"],
        "BALTHASAR": [],
        "CASPAR": ["MELCHIOR"],
    }


def test_a_round_one_advisor_never_finished_is_left_out():
    """The reason this function exists. MELCHIOR spoke in round 3 and the debate
    was cut off before CASPAR answered it, so MELCHIOR's newest vote refers to
    positions the other two have not yet responded to. Counting it against their
    round-2 votes reads as a disagreement that nobody expressed."""
    history = {
        "MELCHIOR": [turn([]), turn(["BALTHASAR"]), turn([])],
        "BALTHASAR": [turn([]), turn(["MELCHIOR"])],
        "CASPAR": [turn([]), turn([])],
    }

    row = latest_complete_round(history)

    assert tally(row).outcome is Outcome.MAJORITY
    assert set(tally(row).majority) == {"MELCHIOR", "BALTHASAR"}


def test_an_empty_history_has_no_round():
    assert latest_complete_round({}) == {}
    assert latest_complete_round({"MELCHIOR": []}) == {}


# ── Asymmetric claims, and repairing them ────────────────────────────────────


def test_a_one_sided_claim_is_an_asymmetric_pair():
    pairs = asymmetric_pairs(
        {
            "MELCHIOR": turn(["BALTHASAR"]),
            "BALTHASAR": turn([]),
            "CASPAR": turn([]),
        }
    )

    assert pairs == (("BALTHASAR", "MELCHIOR"),)


def test_mutual_and_silent_pairs_are_not_ambiguous():
    """Nothing to ask about: the first pair agreed in both directions, and
    nobody said anything about CASPAR at all. Only a claim that exists and was
    not returned is worth an LLM call."""
    pairs = asymmetric_pairs(
        {
            "MELCHIOR": turn(["BALTHASAR"]),
            "BALTHASAR": turn(["MELCHIOR"]),
            "CASPAR": turn([]),
        }
    )

    assert pairs == ()


def test_a_cycle_is_three_asymmetric_pairs():
    """A→B→C→A is the shape that tallies as DEADLOCK while every advisor has
    declared agreement with somebody. Every pair in it is one-sided."""
    pairs = asymmetric_pairs(
        {
            "MELCHIOR": turn(["BALTHASAR"]),
            "BALTHASAR": turn(["CASPAR"]),
            "CASPAR": turn(["MELCHIOR"]),
        }
    )

    assert len(pairs) == 3


def test_a_repaired_edge_turns_a_deadlock_into_a_majority():
    turns = {
        "MELCHIOR": turn(["BALTHASAR"]),
        "BALTHASAR": turn([]),
        "CASPAR": turn([]),
    }

    assert tally(turns).outcome is Outcome.DEADLOCK

    result = tally(turns, extra_edges=[("BALTHASAR", "MELCHIOR")])

    assert result.outcome is Outcome.MAJORITY
    assert set(result.majority) == {"BALTHASAR", "MELCHIOR"}
    assert result.dissent == ("CASPAR",)


def test_repairing_every_edge_of_a_cycle_is_unanimous():
    turns = {
        "MELCHIOR": turn(["BALTHASAR"]),
        "BALTHASAR": turn(["CASPAR"]),
        "CASPAR": turn(["MELCHIOR"]),
    }
    pairs = asymmetric_pairs(turns)

    assert tally(turns).outcome is Outcome.DEADLOCK
    assert tally(turns, extra_edges=pairs).outcome is Outcome.UNANIMOUS


def test_an_extra_edge_for_an_advisor_that_never_spoke_is_ignored():
    """A repair can only ever be about turns that exist. Anything else would let
    an absent advisor into a bloc."""
    turns = {"MELCHIOR": turn([]), "BALTHASAR": turn([])}

    result = tally(turns, extra_edges=[("MELCHIOR", "CASPAR")])

    assert result.outcome is Outcome.DEADLOCK


def test_no_extra_edges_is_the_plain_reading_of_the_votes():
    """The default has to stay the advisors' own words: every caller that does
    not ask for a repair must see exactly what they declared."""
    turns = {
        "MELCHIOR": turn(["BALTHASAR"]),
        "BALTHASAR": turn([]),
        "CASPAR": turn([]),
    }

    assert tally(turns).outcome is Outcome.DEADLOCK
