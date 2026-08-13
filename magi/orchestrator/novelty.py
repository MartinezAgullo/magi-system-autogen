"""Is this consensus three answers, or one answer three times?

**SHARED CONTRACT.** Like ``consensus.py``, the sibling no-framework repo holds
a byte-identical copy. Pure functions over text — no AutoGen, no I/O, no model
call. If the two implementations measured novelty differently, a difference in
convergence rates between them would be a difference in arithmetic.

The tally cannot see this failure, and that is the whole reason the module
exists. Agreement has two ways of being cheap:

* **The judge grants it.** Already recorded, as ``judged_cosmetic`` and
  ``judged_edges``. A concession by something outside the debate.
* **An advisor stops contributing.** It echoes the previous speaker, and three
  identical answers score as an unusually strong consensus. Nothing recorded
  this, and it is invisible in every summary: the console prints three
  agreeing advisors, the outcome says UNANIMOUS, and the debate produced one
  model's answer at three models' cost.

Measured on 2026-08-13, in a real debate that reached UNANIMOUS with the judge
never invoked: MELCHIOR's and BALTHASAR's final positions were the same 225
characters, byte for byte. That debate is indistinguishable from a genuine
convergence in every field the record held until now.

**Containment, not similarity.** The measure is how much of the *shorter*
position appears in the longer one, because the shape this catches is one
advisor's text embedded whole in another's with a sentence of its own added.
Jaccard would score that pair as half-different and let it through; containment
calls it what it is.

What this does not do is judge meaning. Three advisors can reach the same
conclusion in three sets of words and that is convergence, which is the thing
the system is for — it will score near 1.0 here and should. This only measures
whether the *text* was reused, which is the one form of collapse that can be
detected without asking a model, and therefore the one that can be computed for
every debate, for free, forever.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass

from magi.constants import ECHO_CONTAINMENT, NOVELTY_SHINGLE
from magi.models import Echo

#: Splits on anything that is not a word character, which is also what makes
#: the comparison survive typography: an advisor that copied a sentence and
#: rendered "three-person" with a non-breaking hyphen has still copied it, and
#: the real measured case did exactly that.
_NOT_WORD = re.compile(r"\W+", re.UNICODE)


@dataclass(frozen=True)
class Novelty:
    """How much of the tallied positions is each advisor's own text."""

    #: 1.0 when no two positions share a phrase, 0.0 when two are identical.
    #: ``None`` when there was nothing to compare — fewer than two positions.
    score: float | None
    #: Pairs at or above :data:`ECHO_CONTAINMENT`, worst first.
    echoes: tuple[Echo, ...]

    @property
    def echoed(self) -> bool:
        return bool(self.echoes)


def words(text: str) -> list[str]:
    """Lowercased word tokens, punctuation and typography removed."""
    normalised = unicodedata.normalize("NFKC", text).lower()
    return [token for token in _NOT_WORD.split(normalised) if token]


def shingles(text: str, size: int = NOVELTY_SHINGLE) -> set[tuple[str, ...]]:
    """Overlapping word n-grams.

    A position shorter than one shingle becomes a single shingle of everything
    it has, rather than an empty set. Two advisors that both answered in three
    words still copied each other if the three words match, and returning
    nothing would have scored that as perfect novelty.
    """
    tokens = words(text)
    if not tokens:
        return set()
    if len(tokens) <= size:
        return {tuple(tokens)}
    return {tuple(tokens[i : i + size]) for i in range(len(tokens) - size + 1)}


def containment(first: set, second: set) -> float:
    """Shared fraction of the smaller set. 1.0 when one contains the other."""
    if not first or not second:
        return 0.0
    return len(first & second) / min(len(first), len(second))


def measure(positions: Mapping[str, str]) -> Novelty:
    """Score one debate's tallied positions.

    Given the row the tally read, not the whole transcript: a repeated position
    matters when it is one of the votes that produced the outcome. An advisor
    that echoed in round 2 and then wrote something of its own in round 3 did
    contribute, and the round-3 turn is what counted.
    """
    if len(positions) < 2:
        return Novelty(None, ())

    grams = {name: shingles(text) for name, text in positions.items()}
    names = sorted(grams)

    found: list[Echo] = []
    worst = 0.0
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = containment(grams[first], grams[second])
            worst = max(worst, overlap)
            if overlap >= ECHO_CONTAINMENT:
                found.append(Echo(advisors=[first, second], containment=overlap))

    found.sort(key=lambda echo: echo.containment, reverse=True)
    return Novelty(1.0 - worst, tuple(found))
