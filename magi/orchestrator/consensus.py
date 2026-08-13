"""Vote tally. Pure functions over ``MagiTurn`` — no AutoGen, no I/O.

**SHARED CONTRACT.** The sibling no-framework repo holds a byte-identical copy
of this module. If the two implementations tallied votes differently, the whole
comparison would be measuring the tally rather than the framework. Keeping it
free of AutoGen imports is what makes that copy possible, and is the reason the
LLM judge lives in ``judge.py`` instead of here.

The counting is deterministic Python and stays that way. A model is not asked to
count to three: it is asked what it thinks, and the arithmetic is ours.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from magi.models import MagiTurn, Outcome


@dataclass(frozen=True)
class Tally:
    """The arithmetic result of one round's votes."""

    outcome: Outcome
    #: The largest bloc of mutually-agreeing advisors, sorted.
    majority: tuple[str, ...]
    #: Everyone outside it. On DEADLOCK this is everyone outside the largest
    #: bloc, which may be several singletons.
    dissent: tuple[str, ...]

    @property
    def unanimous(self) -> bool:
        return self.outcome is Outcome.UNANIMOUS


def _claims(turns: Mapping[str, MagiTurn]) -> dict[str, set[str]]:
    """Each advisor's declared agreements, as a set of known advisor names.

    Names are normalised and unknown ones dropped, because a model will
    occasionally write "Melchior" or invent a fourth advisor, and neither should
    turn into a phantom vote. Self-references are stripped rather than trusted:
    ``agrees_with`` regularly contains the advisor's own name.
    """
    known = set(turns)
    return {
        name: {other.strip().upper() for other in turn.agrees_with} & known - {name}
        for name, turn in turns.items()
    }


def asymmetric_pairs(turns: Mapping[str, MagiTurn]) -> tuple[tuple[str, str], ...]:
    """The pairs where exactly one side claimed agreement, sorted.

    These are precisely the claims the mutual rule throws away, and they are
    worth a second look rather than a shrug, because the advisors do not vote
    simultaneously. On one shared thread the first speaker of a round votes on
    everyone else's *previous* positions while the last speaker votes on their
    revised ones, so two advisors can hold the same view and still fail to name
    each other in the same round. That is a property of the turn order, not a
    disagreement.

    Pure and deliberately decision-free: it says which pairs are ambiguous, not
    what to do about them. Resolving them costs an LLM call, which is why it
    happens in ``judge.py`` and never here.
    """
    claims = _claims(turns)
    names = sorted(turns)
    return tuple(
        (first, second)
        for index, first in enumerate(names)
        for second in names[index + 1 :]
        if (second in claims[first]) != (first in claims[second])
    )


def latest_complete_round(
    history: Mapping[str, Sequence[MagiTurn]],
) -> dict[str, MagiTurn]:
    """The deepest row in which **every** advisor has spoken.

    The alternative, and what this replaces, is each advisor's latest turn
    whenever it happened. That mixes rounds: a debate cut off by the budget, the
    clock or a crash leaves one advisor's round-3 vote next to another's round-2
    vote, and the tally then reads two votes cast against different versions of
    the same positions as a disagreement. Some of the DEADLOCKs this system
    produced were that, rather than any advisor disagreeing with anything.

    Turns deeper than the returned row are not deleted, only left out of the
    arithmetic: they belong in the transcript, they were paid for, and an
    operator watching the console has already read them.

    The cost, stated plainly: under ``SelectorGroupChat`` there are no rounds,
    so an uneven selector that keeps re-asking one advisor holds the whole tally
    at the depth of whichever advisor it asked least. That is a real property of
    what the selector did rather than an artefact, but it means the tallied
    depth has to be recorded next to the outcome, or two rows that read the same
    were not decided on comparable evidence.
    """
    if not history:
        return {}
    depth = min(len(turns) for turns in history.values())
    if depth == 0:
        return {}
    return {name: turns[depth - 1] for name, turns in history.items()}


def _agreement_blocs(
    turns: Mapping[str, MagiTurn],
    extra_edges: Iterable[tuple[str, str]] = (),
) -> list[set[str]]:
    """Connected components of **mutual** agreement.

    Mutual, not one-directional, and that is the load-bearing decision here.
    "A agrees with B" while B says nothing about A is one advisor being
    agreeable, not two advisors converging — and a debate that ended on it would
    report a consensus nobody actually reached. A cycle (A→B→C→A) likewise
    yields no mutual edge and no bloc, which is correct: nobody paired up.

    ``extra_edges`` are treated as if both sides had claimed them. They are how
    an asymmetric claim gets repaired once something outside this module has
    established that the two positions say the same thing (``judge.py``). The
    repair is passed in rather than inferred so that this function stays a pure
    reading of the votes, and so that a debate row can record how many edges
    were repaired: agreement the advisors declared and agreement a judge granted
    must stay distinguishable forever.
    """
    claims = _claims(turns)
    for first, second in extra_edges:
        if first in claims and second in claims:
            claims[first].add(second)
            claims[second].add(first)

    blocs: list[set[str]] = []
    unassigned = set(claims)
    while unassigned:
        seed = unassigned.pop()
        bloc = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            for other in list(unassigned):
                # The edge exists only if both sides claim it.
                if other in claims[current] and current in claims[other]:
                    bloc.add(other)
                    unassigned.discard(other)
                    frontier.append(other)
        blocs.append(bloc)

    blocs.sort(key=len, reverse=True)
    return blocs


def tally(
    turns: Mapping[str, MagiTurn],
    extra_edges: Iterable[tuple[str, str]] = (),
) -> Tally:
    """Decide the outcome from one round of votes.

    ``UNANIMOUS`` when one bloc contains everyone, ``MAJORITY`` when the largest
    bloc is strictly more than half, ``DEADLOCK`` otherwise.

    "More than half" rather than "two or more" so the rule survives a debate
    that lost an advisor: with two voters, a bloc of one is not a majority, and
    the honest answer there is DEADLOCK.

    ``turns`` should be one advisor per name from the *same* round — see
    ``latest_complete_round``. ``extra_edges`` is the judge's repair of
    asymmetric claims, empty by default so that the plain reading of the votes
    is always available.
    """
    if not turns:
        return Tally(Outcome.DEADLOCK, (), ())

    blocs = _agreement_blocs(turns, extra_edges)
    largest = blocs[0]
    rest = tuple(sorted(set(turns) - largest))

    if len(largest) == len(turns) and len(turns) > 1:
        outcome = Outcome.UNANIMOUS
    elif len(largest) * 2 > len(turns):
        outcome = Outcome.MAJORITY
    else:
        outcome = Outcome.DEADLOCK

    return Tally(outcome, tuple(sorted(largest)), rest)


def disagreement_summary(turns: Mapping[str, MagiTurn], names: Sequence[str]) -> str:
    """One line per advisor, for the verdict prompt and the DEADLOCK report."""
    return "\n".join(
        f"{name}: {turns[name].summary.strip()}" for name in names if name in turns
    )
