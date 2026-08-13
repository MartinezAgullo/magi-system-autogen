"""The tie-break, run only when the vote is split.

It answers exactly one question — is the disagreement substantive, or the same
answer worded differently? — and never who is right. Widening its remit would
make it a fourth advisor with a casting vote, which is the opposite of what a
three-way deliberation is for.

It is asked that question twice, over different scopes, and the difference
matters:

- ``judge_pairs`` runs first, over the pairs where exactly one advisor claimed
  agreement. Those claims exist and were not reciprocated, which on a shared
  thread is as likely to be the turn order as a disagreement, so the judge is
  asked whether the two positions actually differ. A cosmetic ruling repairs the
  edge and the tally is redone.
- ``judge_disagreement`` runs after, over all the positions at once, and can
  promote a split vote outright.

The second is skipped when the first found any real difference. Asking "are all
of these the same answer?" after being told that two specific ones are not would
let a vaguer question overrule a sharper one.

Kept out of ``consensus.py`` so that module stays pure and can be copied
verbatim into the sibling repo.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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

    # Same threshold as JUDGE_SYSTEM_PROMPT, deliberately word for word: this
    # description is also sent to the model, as part of the JSON schema, and two
    # copies of a rule that drift produce a judge arguing with itself.
    substantive: bool = Field(
        description=(
            "True if acting on one advisor's position would lead to a materially "
            "different outcome than acting on another's. False if they would "
            "lead to broadly the same course of action and differ in wording, "
            "emphasis, framing or degree."
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


@dataclass(frozen=True)
class PairRulings:
    """What the judge made of each asymmetric pair.

    A pair the judge could not rule on appears in neither tuple. That is not an
    oversight: an unreachable judge must leave the votes exactly as the advisors
    cast them, in both directions.
    """

    #: Pairs ruled cosmetic. The tally treats these as mutual agreement.
    repaired: tuple[tuple[str, str], ...] = ()
    #: Pairs ruled a real difference. Any of these cancels the wider judge.
    substantive: tuple[tuple[str, str], ...] = ()


async def judge_pairs(
    settings: Settings,
    personas: PersonaSet,
    turns: Mapping[str, MagiTurn],
    pairs: Sequence[tuple[str, str]],
    counter: CallCounter | None = None,
    on_activity=None,
) -> PairRulings:
    """Rule on each asymmetric pair, concurrently.

    One question per pair rather than one question about everything, because the
    answer is used per edge: "MELCHIOR and CASPAR are saying the same thing,
    BALTHASAR is not" is a MAJORITY, and a single global ruling can only ever
    produce all or nothing.

    **Concurrently on purpose.** These sit on the critical path of a voice
    interface, after two to four minutes the operator has already waited. With
    three advisors there are at most three pairs, so the wall-clock cost is one
    judge call rather than three — while the token and call cost is genuinely
    three, which is why the debate row counts calls at the client and not here.
    """
    if not pairs:
        return PairRulings()

    async def rule(pair: tuple[str, str]) -> JudgeRuling | None:
        first, second = pair
        return await judge_disagreement(
            settings,
            personas,
            {first: turns[first], second: turns[second]},
            pair,
            counter,
            on_activity,
        )

    rulings = await asyncio.gather(*(rule(pair) for pair in pairs))

    repaired: list[tuple[str, str]] = []
    substantive: list[tuple[str, str]] = []
    for pair, ruling in zip(pairs, rulings, strict=True):
        if ruling is None:
            continue
        if ruling.substantive:
            substantive.append(pair)
        else:
            repaired.append(pair)
            logger.info(
                "Judge: %s and %s differ only in wording (%s)",
                pair[0], pair[1], ruling.reason,
            )
    return PairRulings(repaired=tuple(repaired), substantive=tuple(substantive))
