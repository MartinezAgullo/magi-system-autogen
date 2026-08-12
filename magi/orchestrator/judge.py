"""The tie-break, run only when the vote is split.

It answers exactly one question — is the disagreement substantive, or the same
answer worded differently? — and never who is right. Widening its remit would
make it a fourth advisor with a casting vote, which is the opposite of what a
three-way deliberation is for.

Kept out of ``consensus.py`` so that module stays pure and can be copied
verbatim into the sibling repo.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from pydantic import BaseModel, Field

from magi.config import Settings
from magi.models import MagiTurn
from magi.orchestrator.clients import build_client
from magi.orchestrator.prompts import JUDGE_SYSTEM_PROMPT, judge_task
from magi.personas import PersonaSet
from magi.services.metrics import CallCounter

logger = logging.getLogger(__name__)


class JudgeRuling(BaseModel):
    """Structured so the answer cannot arrive as hedged prose."""

    substantive: bool = Field(
        description=(
            "True if acting on one advisor's position would lead to a different "
            "decision than acting on another's. False if they differ only in "
            "wording, emphasis or framing."
        )
    )
    reason: str = Field(description="One sentence. The specific difference, or its absence.")


async def judge_disagreement(
    settings: Settings,
    personas: PersonaSet,
    turns: Mapping[str, MagiTurn],
    order: Sequence[str],
    counter: CallCounter | None = None,
    on_activity=None,
) -> JudgeRuling | None:
    """Return the ruling, or ``None`` if the judge could not be reached.

    ``None`` is not "no disagreement". A judge that failed must leave the split
    vote standing — silently promoting a MAJORITY to UNANIMOUS because an HTTP
    call timed out would manufacture consensus out of an outage, which is the
    single worst thing this system could do.
    """
    judge = AssistantAgent(
        name="JUDGE",
        # The counter matters here: the judge is a real LLM call on the debate's
        # critical path, and leaving it out understates the cost of every split
        # vote. Caught by the traces themselves — 11 `chat` spans against a
        # recorded count of 10.
        # Deliberately left under the orchestrator's label rather than a "JUDGE"
        # one: the label is also what the console lights up, and the judge is
        # MAGI's own work on the critical path. A separate label would leave the
        # core panel dark for the 3-4 s it runs. The span name (SPAN_JUDGE)
        # already separates it for anyone reading traces.
        model_client=build_client(
            settings, personas, personas.orchestrator, counter,
            on_activity=on_activity,
        ),
        system_message=JUDGE_SYSTEM_PROMPT,
        output_content_type=JudgeRuling,
    )

    try:
        result = await judge.run(task=judge_task(turns, order))
    except Exception:
        logger.exception("Judge unreachable — leaving the split vote as it stands")
        return None

    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage) and isinstance(
            message.content, JudgeRuling
        ):
            return message.content

    logger.warning("Judge returned no ruling — leaving the split vote as it stands")
    return None
