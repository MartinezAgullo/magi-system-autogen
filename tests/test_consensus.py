"""The vote tally — the shared contract with the sibling repo.

If these two implementations tallied differently, the benchmark would be
measuring the tally instead of the framework, so this module is the one that
most needs to be pinned down by tests rather than by reading.
"""

from __future__ import annotations

from magi.models import MagiTurn, Outcome
from magi.orchestrator.consensus import tally

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
