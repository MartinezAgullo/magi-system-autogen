"""The three prompts the orchestrator writes, as opposed to the persona prompts.

Persona prompts say who an advisor is; these say what it is being asked to do
right now. Keeping them apart means a change to the debate procedure does not
require editing three personalities, and a change to a personality cannot
silently alter the procedure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from magi.models import MagiTurn, Outcome


def blind_task(question: str) -> str:
    """Phase A. No mention of the other advisors at all.

    Naming them would invite the model to imagine their positions and answer
    against a guess, which is the same anchoring the blind round exists to
    prevent — just with an imagined interlocutor instead of a real one.
    """
    return (
        f"{question}\n\n"
        "Give your own answer. Leave `agrees_with` and `critique` empty: you "
        "have not seen anyone else's position yet."
    )


def deliberation_seed(
    question: str, turns: Mapping[str, MagiTurn], order: Sequence[str]
) -> str:
    """Phase B. The three blind positions, then the instruction to engage.

    Positions are given in full rather than as summaries: an advisor asked to
    critique a one-line précis will critique the précis.
    """
    positions = "\n\n".join(
        f"{name} ({turns[name].confidence:.1f} confidence):\n{turns[name].position.strip()}"
        for name in order
        if name in turns
    )
    return (
        f"The question was:\n{question}\n\n"
        f"The advisors answered independently, without seeing each other:\n\n"
        f"{positions}\n\n"
        "Work through this in order.\n\n"
        "1. `critique` — for each OTHER advisor, what does their answer get "
        "wrong, assume without saying, or leave out? Judge them from your own "
        "role, which is the only reason there are three of you. One entry per "
        "advisor. If you genuinely have no objection to one of them, leave that "
        "one out — but having no objection to anybody is rare and usually means "
        "you have stopped reading them critically.\n\n"
        "2. `position` — what you hold now, having read them. It must contain "
        "something none of the others said. Copying another advisor's sentence "
        "is not agreement, it is silence: three identical answers carry exactly "
        "as much information as one, and the whole point of asking three "
        "advisors is lost. Use your own words and your own angle even when you "
        "reach the same conclusion. If they changed your mind, say what changed "
        "it.\n\n"
        "3. `agrees_with` — the OTHER advisors, by exact name, whose position "
        "you could act on. Different wording, different emphasis and a caveat "
        "of your own are not disagreement: list them anyway. Leave one out only "
        "if acting on their position would lead somewhere materially different "
        "from acting on yours. Never list yourself. You can criticise an "
        "advisor in `critique` and still agree with them here, and that "
        "combination is common."
    )


def verdict_task(
    question: str,
    turns: Mapping[str, MagiTurn],
    order: Sequence[str],
    outcome: Outcome,
    majority: Sequence[str],
    dissent: Sequence[str],
    absent: Sequence[str] = (),
) -> str:
    """Phase C. The transcript plus an outcome that is **already decided**.

    The orchestrator is told the outcome rather than asked for it. It writes the
    prose; the arithmetic was done in Python. Asking a 12B model to both count
    votes and phrase the result would put the one part that must be exact at the
    mercy of the one part that cannot be.
    """
    positions = "\n\n".join(
        f"{name}:\n{turns[name].position.strip()}" for name in order if name in turns
    )

    instruction = {
        Outcome.UNANIMOUS: (
            "The advisors reached agreement. State the shared conclusion."
        ),
        Outcome.MAJORITY: (
            f"{', '.join(majority)} agreed; {', '.join(dissent)} did not. "
            "State the majority conclusion, then name the dissenter and their "
            "objection in one clause. Do not bury the dissent."
        ),
        Outcome.DEADLOCK: (
            "The advisors did not converge. Do NOT invent a synthesis or pick a "
            "side. Say that they disagreed, and name the precise point they "
            "split on."
        ),
    }[outcome]

    # A debate that lost an advisor must say so out loud. Reporting UNANIMOUS
    # from two voices without mentioning the third is technically true and
    # practically a lie: the operator hears the system's strongest verdict and
    # has no way to know a third of it never spoke. Observed on the Pi, where a
    # model failed schema validation in the blind round and the remaining two
    # agreed.
    missing = ""
    if absent:
        missing = (
            f"\n\nIMPORTANT: {', '.join(absent)} did not take part — the model "
            f"failed to answer. Say so in your first sentence: this verdict "
            f"comes from {len(turns)} advisors, not {len(turns) + len(absent)}."
        )

    return (
        f"Question:\n{question}\n\n"
        f"Final positions:\n\n{positions}\n\n"
        f"Outcome: {outcome.value}. {instruction}{missing}\n\n"
        f"Set `outcome` to exactly {outcome.value}."
    )


# The threshold is "a materially different outcome", not "a different decision".
# The stricter wording made any difference of degree substantive, which is most
# of why split votes almost never got promoted. Loosening it is deliberately the
# best-instrumented lever available: `judged_cosmetic` is on every DebateRecord,
# so a rise in UNANIMOUS that comes from the judge stays distinguishable from one
# where the advisors actually converged. See docs/agreement-bias.md § (b).
JUDGE_SYSTEM_PROMPT = (
    "You decide one thing and nothing else: whether two or more advisors who "
    "did not formally agree are in fact saying the same thing in different "
    "words.\n\n"
    "Answer SUBSTANTIVE if acting on one position would lead to a materially "
    "different outcome than acting on another. Answer COSMETIC if the "
    "positions would lead to broadly the same course of action and differ in "
    "emphasis, vocabulary, framing or degree.\n\n"
    "Do not judge who is right. Do not add your own opinion on the question."
)


def judge_task(turns: Mapping[str, MagiTurn], order: Sequence[str]) -> str:
    positions = "\n\n".join(
        f"{name}:\n{turns[name].position.strip()}" for name in order if name in turns
    )
    return (
        f"{positions}\n\n"
        "Are these positions substantively different, or the same answer worded "
        "differently?"
    )
