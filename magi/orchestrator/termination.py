"""Stopping the debate, in AutoGen's own idioms.

Consensus is a custom ``TerminationCondition``; the round budget, the wall-clock
budget and PTT barge-in are the framework's own conditions, composed with ``|``.
Writing a bespoke loop that checked all four would have been less code and would
have benchmarked nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from autogen_agentchat.base import TerminatedException, TerminationCondition
from autogen_agentchat.conditions import (
    ExternalTermination,
    MaxMessageTermination,
    TimeoutTermination,
)
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    StopMessage,
    StructuredMessage,
)

from magi.config import Settings
from magi.constants import (
    TERMINATED_BY_BARGE_IN,
    TERMINATED_BY_BUDGET,
    TERMINATED_BY_CONSENSUS,
    TERMINATED_BY_TIMEOUT,
    TERMINATED_BY_UNKNOWN,
)
from magi.models import MagiTurn
from magi.orchestrator.consensus import tally

logger = logging.getLogger(__name__)

#: Set as the StopMessage source so the orchestrator can report which condition
#: fired without inspecting types. Recorded on every debate row: one query over
#: it answers how often debates converge rather than run out of budget.
CONSENSUS_SOURCE = "ConsensusTermination"


class ConsensusTermination(TerminationCondition):
    """Stop as soon as every advisor's latest vote is unanimous.

    **Accumulates state.** ``__call__`` receives only the messages produced
    since the last call, not the whole history, so a condition that inspected
    just its argument would never see all three advisors at once and would never
    fire. The latest turn per advisor is kept here and cleared by ``reset()``.

    Evaluates once every advisor has spoken at least once, and again on each
    turn after that. Waiting for tidy round boundaries would be wrong for
    ``SelectorGroupChat``, which has no rounds — and stopping the moment the
    votes line up is the point of having the condition at all.
    """

    def __init__(self, advisors: Sequence[str]) -> None:
        self._advisors = {name.upper() for name in advisors}
        self._latest: dict[str, MagiTurn] = {}
        self._terminated = False

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def latest_turns(self) -> dict[str, MagiTurn]:
        """The votes as they stood when the debate ended. The orchestrator
        tallies these rather than re-parsing the transcript."""
        return dict(self._latest)

    async def __call__(
        self, messages: Sequence[BaseAgentEvent | BaseChatMessage]
    ) -> StopMessage | None:
        if self._terminated:
            raise TerminatedException("Termination condition has already been reached")

        for message in messages:
            if isinstance(message, StructuredMessage) and isinstance(
                message.content, MagiTurn
            ):
                self._latest[message.source.upper()] = message.content

        if set(self._latest) != self._advisors:
            return None

        result = tally(self._latest)
        if not result.unanimous:
            return None

        self._terminated = True
        logger.info("Consensus reached: %s", ", ".join(result.majority))
        return StopMessage(
            content=f"Unanimous: {', '.join(result.majority)}",
            source=CONSENSUS_SOURCE,
        )

    async def reset(self) -> None:
        self._latest.clear()
        self._terminated = False


class Latching(TerminationCondition):
    """Delegates to one condition and remembers, past ``reset()``, that it fired.

    Necessary because **a fired condition is not observable from outside the
    run.** ``BaseGroupChatManager._apply_termination_condition`` calls
    ``reset()`` on the whole stack the instant a child returns a
    ``StopMessage``, and does so before the caller of ``run_stream`` is
    scheduled. Measured on 0.7.5 with a stub team: every condition reports
    ``terminated is False`` on every message the caller sees, including the
    final ``TaskResult``. Sampling during the stream — the obvious fix, and the
    one this code used to carry — never observes anything either. It is not a
    race that a lucky ordering wins; the reset always gets there first.

    ``TaskResult.stop_reason`` does survive, but only as the human-readable
    *content* of the stop message, comma-joined when several fired at once.
    Classifying a debate by substring-matching AutoGen's English would break on
    a wording change in a patch release.

    So the latch: set in ``__call__``, never cleared by ``reset()``. One
    instance per debate, which is what makes "never cleared" safe.
    """

    def __init__(self, inner: TerminationCondition, label: str) -> None:
        self._inner = inner
        self._label = label
        self._fired = False

    @property
    def label(self) -> str:
        return self._label

    @property
    def fired(self) -> bool:
        """Whether this condition ended the run. Survives the group chat's reset."""
        return self._fired

    @property
    def terminated(self) -> bool:
        return self._inner.terminated

    async def __call__(
        self, messages: Sequence[BaseAgentEvent | BaseChatMessage]
    ) -> StopMessage | None:
        stop = await self._inner(messages)
        if stop is not None:
            self._fired = True
        return stop

    async def reset(self) -> None:
        # Deliberately does not clear `_fired`. That is the entire point.
        await self._inner.reset()


@dataclass
class Terminators:
    """The composed stop condition, plus the parts, for one debate.

    ``consensus`` is exposed unwrapped because the orchestrator reads
    ``latest_turns`` off it. The rest are only ever asked whether they fired.
    """

    condition: TerminationCondition
    consensus: ConsensusTermination
    #: In precedence order. Consensus first, because a debate that converged on
    #: its very last permitted turn converged — reporting that as "budget"
    #: would count a success as a failure in the one statistic the project
    #: exists to produce. Barge-in next, since the operator's intent outranks
    #: any budget the run happened to hit on the same turn.
    latches: tuple[Latching, ...] = field(default_factory=tuple)

    def fired(self) -> str:
        """Why the debate stopped, as a ``TERMINATED_BY_*`` value."""
        for latch in self.latches:
            if latch.fired:
                return latch.label
        return TERMINATED_BY_UNKNOWN


def build_termination(
    settings: Settings,
    advisor_names: Sequence[str],
    external: ExternalTermination,
) -> Terminators:
    """The composed stop condition, and the means to ask afterwards what fired.

    The message budget counts **deliberation** turns only: phase A runs outside
    the team, so the group chat starts at round 2 and has ``max_rounds - 1``
    rounds left.

    The ``+ 1`` is the seed task, which ``MaxMessageTermination`` counts as a
    message like any other. Without it the budget is one short and the last
    round is cut before its final speaker — and because the order is fixed, it
    is always the *same* advisor that loses its closing turn. That is not a
    truncation, it is a bias: one seat systematically never gets to answer the
    others' last word.
    """
    deliberation_rounds = max(settings.max_rounds - 1, 1)
    max_messages = deliberation_rounds * len(advisor_names) + 1

    consensus = ConsensusTermination(advisor_names)
    latches = (
        Latching(consensus, TERMINATED_BY_CONSENSUS),
        Latching(external, TERMINATED_BY_BARGE_IN),
        Latching(TimeoutTermination(settings.debate_timeout_s), TERMINATED_BY_TIMEOUT),
        Latching(MaxMessageTermination(max_messages), TERMINATED_BY_BUDGET),
    )

    condition: TerminationCondition = latches[0]
    for latch in latches[1:]:
        condition = condition | latch

    return Terminators(condition=condition, consensus=consensus, latches=latches)
