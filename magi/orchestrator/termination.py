"""Stopping the debate, in AutoGen's own idioms.

Consensus is a custom ``TerminationCondition``; the round budget, the wall-clock
budget and PTT barge-in are the framework's own conditions, composed with ``|``.
Writing a bespoke loop that checked all four would have been less code and would
have benchmarked nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

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


def build_termination(
    settings: Settings,
    advisor_names: Sequence[str],
    external: ExternalTermination,
) -> tuple[TerminationCondition, ConsensusTermination]:
    """The composed stop condition, and the consensus one for later inspection.

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
    condition = (
        consensus
        | MaxMessageTermination(max_messages)
        | TimeoutTermination(settings.debate_timeout_s)
        | external
    )
    return condition, consensus
