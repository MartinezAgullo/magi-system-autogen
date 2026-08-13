"""Three answers, or one answer three times.

The tally cannot tell the difference, and that is the point of the module. An
advisor that stops contributing and reproduces the previous speaker produces
three identical positions, which score as the strongest possible agreement — an
UNANIMOUS the judge never had to touch, and which carries exactly as much
information as one model's answer.

The case in `test_a_verbatim_copy_is_the_worst_possible_score` is real, taken
from a debate held against the Spark on 2026-08-13: MELCHIOR and BALTHASAR
returned the same 225 characters, byte for byte, in a UNANIMOUS with
`judged_cosmetic` and `judged_edges` both at zero. Every field the record held
at the time said that debate was the good kind.
"""

from __future__ import annotations

from magi.constants import ECHO_CONTAINMENT
from magi.orchestrator.novelty import containment, measure, shingles, words

MELCHIOR = (
    "A three-person startup should avoid self-hosted Kubernetes and only "
    "evaluate a managed Kubernetes service if the product will soon require "
    "scalable, reliable deployments; otherwise it should start with simpler "
    "infrastructure."
)
CASPAR = (
    "The real risk is not the orchestration layer but the team's ability to "
    "operate it at three in the morning, and nobody has costed that against "
    "the alternative of a single virtual machine."
)


def test_a_verbatim_copy_is_the_worst_possible_score():
    """The measured failure. Two advisors, one text, and an outcome that every
    other field in the record calls an unaided consensus."""
    result = measure({"MELCHIOR": MELCHIOR, "BALTHASAR": MELCHIOR})

    assert result.score == 0.0
    assert result.echoed
    assert result.echoes[0].advisors == ["BALTHASAR", "MELCHIOR"]
    assert result.echoes[0].containment == 1.0


def test_different_answers_to_the_same_question_score_high():
    """The behaviour this must not punish. Two advisors can reach the same
    conclusion by different routes and in different words — that is convergence,
    and it is the thing the system exists to produce."""
    result = measure({"MELCHIOR": MELCHIOR, "CASPAR": CASPAR})

    assert result.score > 0.9
    assert not result.echoed


def test_a_copy_with_a_sentence_added_is_still_a_copy():
    """Why the measure is containment and not similarity. The common shape is
    one advisor's text embedded whole in another's with a line of its own bolted
    on; a symmetric measure would score that pair as half-different and let the
    debate through as two contributions."""
    padded = MELCHIOR + " I would also note that hiring is easier with it."

    result = measure({"MELCHIOR": MELCHIOR, "BALTHASAR": padded})

    assert result.echoed
    assert result.echoes[0].containment == 1.0
    assert result.score == 0.0


def test_typography_does_not_hide_a_copy():
    """The real case arrived with non-breaking hyphens in "three‑person", which
    a byte comparison reads as a different string. An advisor that copied a
    sentence and rendered one dash differently has still copied it."""
    fancy = MELCHIOR.replace("-", "‑").replace(";", " —")

    result = measure({"MELCHIOR": MELCHIOR, "BALTHASAR": fancy})

    assert result.score == 0.0


def test_one_shared_sentence_lowers_the_score_without_crying_echo():
    """The middle ground has to stay expressive. Two advisors sharing a sentence
    inside otherwise distinct positions is worth seeing in a mean over forty
    debates, and is not on its own the collapse the threshold is for."""
    shared = "Operational complexity is the cost nobody prices correctly."
    result = measure({
        "MELCHIOR": f"{MELCHIOR} {shared}",
        "CASPAR": f"{CASPAR} {shared}",
    })

    assert 0.0 < result.score < 1.0
    assert not result.echoed


def test_three_advisors_report_every_offending_pair_worst_first():
    """A debate can collapse in more than one place, and the record should name
    which advisors stopped contributing rather than only that someone did."""
    result = measure({
        "MELCHIOR": MELCHIOR,
        "BALTHASAR": MELCHIOR,
        "CASPAR": CASPAR,
    })

    assert [echo.advisors for echo in result.echoes] == [["BALTHASAR", "MELCHIOR"]]
    assert all(
        first.containment >= second.containment
        for first, second in zip(result.echoes, result.echoes[1:], strict=False)
    )


def test_a_lone_advisor_scores_nothing_rather_than_perfectly():
    """A debate that lost two advisors has nothing to compare. Scoring it 1.0
    would put a perfect novelty score on a monologue, and that average would
    then be read as three advisors doing their job."""
    assert measure({"MELCHIOR": MELCHIOR}).score is None
    assert measure({}).score is None


def test_a_position_shorter_than_one_shingle_still_compares():
    """A model can satisfy the schema with three words. Returning no shingles
    for it would report two identical terse answers as perfectly novel — the
    degenerate case scoring best of all."""
    result = measure({"MELCHIOR": "Avoid it entirely.", "BALTHASAR": "Avoid it entirely."})

    assert result.score == 0.0
    assert result.echoed


def test_the_pieces_behave_as_the_threshold_assumes():
    """The measure is a contract with the sibling repo, so the parts it is
    defined in terms of are worth pinning: tokens are words without
    punctuation, shingles overlap, and containment divides by the smaller
    side."""
    assert words("A three-person startup!") == ["a", "three", "person", "startup"]
    assert len(shingles("one two three four five", size=4)) == 2
    assert containment({1, 2}, {1, 2, 3, 4}) == 1.0
    assert containment({1, 2, 3, 4}, set()) == 0.0
    # The threshold is a fraction of the shorter position, so it is only
    # meaningful in that range.
    assert 0.0 < ECHO_CONTAINMENT <= 1.0
