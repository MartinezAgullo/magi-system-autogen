"""Building the deliberation team — the one thing the two engines differ in.

Same advisors, same prompts, same termination stack, same verdict agent. Only
turn selection changes, which is the point: the delta between the two engines is
the cost of AutoGen's headline feature and nothing else.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from autogen_agentchat.base import ChatAgent, TerminationCondition
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import BaseGroupChat, RoundRobinGroupChat, SelectorGroupChat

from magi.config import Settings
from magi.constants import SELECTOR_LABEL
from magi.models import MagiTurn
from magi.orchestrator.clients import build_client
from magi.personas import PersonaSet
from magi.services.metrics import CallCounter

logger = logging.getLogger(__name__)

# Replaces AutoGen's default, which opens with "You are in a role play game".
# That framing invites the selector to keep a conversation entertaining; what is
# wanted is coverage — everyone gets heard before anyone speaks twice — and the
# default prompt has no reason to care about that.
#
# Deliberately still a real LLM call. A `selector_func` would short-circuit it
# and produce a benchmark of nothing: the whole reason this engine exists is to
# price the call that a selector function would avoid.
SELECTOR_PROMPT = """You are moderating a deliberation between expert advisors.

Roles:
{roles}

Transcript so far:
{history}

Choose which advisor from {participants} should speak next. Prefer whoever has
been challenged and not yet replied, or whose perspective is missing from the
exchange so far. Return only the name.
"""


def build_team(
    settings: Settings,
    personas: PersonaSet,
    advisors: Sequence[ChatAgent],
    termination: TerminationCondition,
    counter: CallCounter | None = None,
) -> BaseGroupChat:
    """The team for the configured engine.

    ``custom_message_types`` is mandatory for both. A standalone
    ``AssistantAgent`` emits ``StructuredMessage[MagiTurn]`` happily, but a group
    chat routes messages through a serialisation registry and rejects anything
    it was not told about — at run time, from inside a message container.

    Neither passes ``runtime=``. Injecting a ``SingleThreadedAgentRuntime`` to
    attach a tracer provider hangs the run (``run_stream`` only starts the
    runtime it built itself) and silently reverses
    ``ignore_unhandled_exceptions``. AutoGen reads the globally registered
    provider anyway.
    """
    common = {
        "termination_condition": termination,
        "custom_message_types": [StructuredMessage[MagiTurn]],
    }

    if settings.engine == "autogen_roundrobin":
        return RoundRobinGroupChat(list(advisors), **common)

    # The selector's model is the orchestrator's — small and fast, because this
    # call sits on the critical path of every single turn.
    #
    # It is labelled SELECTOR, not MAGI, so the call counter breaks it out from
    # the verdict and the judge. Folding it into the orchestrator's total would
    # hide the exact number this engine exists to produce.
    selector_client = build_client(
        settings, personas, personas.orchestrator, counter, label=SELECTOR_LABEL
    )
    logger.info(
        "SelectorGroupChat: turn selection by %s (one extra LLM call per turn)",
        personas.orchestrator.model,
    )
    return SelectorGroupChat(
        list(advisors),
        model_client=selector_client,
        selector_prompt=SELECTOR_PROMPT,
        # False, so an advisor cannot answer itself into a monologue. With three
        # advisors and a fixed budget, a repeated speaker means another advisor
        # never gets a turn at all — the debate quietly becomes a duologue.
        allow_repeated_speaker=False,
        **common,
    )
