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

from collections.abc import Mapping, Sequence
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


def _agreement_blocs(turns: Mapping[str, MagiTurn]) -> list[set[str]]:
    """Connected components of **mutual** agreement.

    Mutual, not one-directional, and that is the load-bearing decision here.
    "A agrees with B" while B says nothing about A is one advisor being
    agreeable, not two advisors converging — and a debate that ended on it would
    report a consensus nobody actually reached. A cycle (A→B→C→A) likewise
    yields no mutual edge and no bloc, which is correct: nobody paired up.

    Names are normalised and unknown ones dropped, because a model will
    occasionally write "Melchior" or invent a fourth advisor, and neither should
    turn into a phantom vote.
    """
    known = set(turns)
    claims = {
        name: {other.strip().upper() for other in turn.agrees_with} & known - {name}
        for name, turn in turns.items()
    }

    blocs: list[set[str]] = []
    unassigned = set(known)
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


def tally(turns: Mapping[str, MagiTurn]) -> Tally:
    """Decide the outcome from every advisor's latest turn.

    ``UNANIMOUS`` when one bloc contains everyone, ``MAJORITY`` when the largest
    bloc is strictly more than half, ``DEADLOCK`` otherwise.

    "More than half" rather than "two or more" so the rule survives a debate
    that lost an advisor: with two voters, a bloc of one is not a majority, and
    the honest answer there is DEADLOCK.
    """
    if not turns:
        return Tally(Outcome.DEADLOCK, (), ())

    blocs = _agreement_blocs(turns)
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
