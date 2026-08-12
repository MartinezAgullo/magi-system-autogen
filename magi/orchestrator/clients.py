"""Model clients and advisor agents.

One ``OpenAIChatCompletionClient`` per advisor, pointed at Ollama's
OpenAI-compatible endpoint. The same class serves a real API-key provider by
changing ``base_url`` and ``api_key``, which is what makes this developable on a
machine that runs no local models.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import CreateResult, ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from openai import LengthFinishReasonError

from magi.config import Settings
from magi.constants import (
    ATTR_ADVISOR,
    ATTR_LENGTH_RETRY,
    ATTR_STREAMED,
    GEN_AI_OPERATION,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT,
    GEN_AI_USAGE_OUTPUT,
    LENGTH_RETRY_FACTOR,
    PROBE_MAX_TOKENS,
)
from magi.models import MagiTurn
from magi.personas import Persona, PersonaSet
from magi.services.metrics import CallCounter
from magi.setup.setup_tracing import get_tracer

logger = logging.getLogger(__name__)

#: Called with ``(label, busy)`` when a call opens and again when it closes.
#: Async, because the only consumer publishes to the bus.
ActivityHook = Callable[[str, bool], Awaitable[None]]


class InstrumentedChatClient(OpenAIChatCompletionClient):
    """An Ollama client that counts and traces every call it makes.

    Subclassing rather than wrapping, because a wrapper would have to reimplement
    the whole ``ChatCompletionClient`` surface to stay valid, and every method it
    forgot would be a silently uncounted call. Overriding ``create`` catches all
    of them — including the ones AutoGen makes on its own behalf, which is the
    entire point: ``SelectorGroupChat``'s speaker-selection call never appears as
    a chat message, so message-level accounting would miss the one number the
    engine comparison is about.

    **Both entry points are overridden, and that is not optional.** An
    ``AssistantAgent`` with ``model_client_stream=True`` calls ``create_stream``
    and never touches ``create``. With only ``create`` covered, switching one
    agent to streaming would silently zero its call count, its token count, its
    GenAI span and its activity events, with no error anywhere — every headline
    number this project exists to produce, quietly gone. The two paths share
    :meth:`_record` so they cannot drift.
    """

    def __init__(
        self,
        *,
        magi_label: str,
        counter: CallCounter | None = None,
        on_activity: ActivityHook | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._magi_label = magi_label
        self._magi_counter = counter
        self._magi_activity = on_activity
        self._magi_model = kwargs.get("model", "unknown")
        self._magi_max_tokens = kwargs.get("max_tokens") or PROBE_MAX_TOKENS

    async def _announce(self, busy: bool) -> None:
        """Tell the console this agent started or stopped generating.

        Swallows its own failures. A console that cannot be told an advisor is
        thinking is a cosmetic problem; a debate that dies because the bus
        hiccuped is not, and this sits on the critical path of every call.
        """
        if self._magi_activity is None:
            return
        try:
            await self._magi_activity(self._magi_label, busy)
        except Exception:
            logger.debug("Activity hook failed for %s", self._magi_label, exc_info=True)

    def _open_span(self, span) -> None:
        # Span naming follows the OTel GenAI semantic conventions so these sit
        # in a backend alongside the sibling repo's traces without translation.
        span.set_attribute(GEN_AI_SYSTEM, "ollama")
        span.set_attribute(GEN_AI_OPERATION, "chat")
        span.set_attribute(GEN_AI_REQUEST_MODEL, self._magi_model)
        span.set_attribute(ATTR_ADVISOR, self._magi_label)

    def _record(self, span, result: Any) -> None:
        """Token accounting, shared by the streaming and non-streaming paths.

        One method rather than two copies, because the failure mode of a second
        copy is not a crash: it is a number that is quietly wrong in one mode
        and right in the other.
        """
        usage = getattr(result, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        span.set_attribute(GEN_AI_USAGE_INPUT, prompt_tokens)
        span.set_attribute(GEN_AI_USAGE_OUTPUT, completion_tokens)
        if self._magi_counter is not None:
            self._magi_counter.record(self._magi_label, prompt_tokens, completion_tokens)

    async def create_stream(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """The streaming twin of :meth:`create`, instrumented identically.

        Two things differ from the non-streaming path, and both are traps rather
        than details.

        **Usage must be asked for.** OpenAI-compatible streaming returns no
        token counts unless ``stream_options.include_usage`` is set, and AutoGen
        only sets it when ``include_usage`` is passed explicitly. Left at its
        default, every streamed call would report zero tokens — which reads as a
        free call rather than as a missing measurement. Defaulted to ``True``
        here, and left alone if the caller has an opinion.

        **The length retry does not exist here, and cannot.** Streaming does not
        raise ``LengthFinishReasonError``; a model that spends its whole budget
        on reasoning yields ``finish_reason="length"`` on the final result, by
        which point chunks have already been handed to the consumer and there is
        nothing to transparently retry. The condition is recorded and logged
        loudly instead. This is the concrete reason not to stream MELCHIOR: the
        retry is what stops one advisor's exhausted budget from taking the whole
        group chat down with it.
        """
        kwargs.setdefault("include_usage", True)

        tracer = get_tracer()
        with tracer.start_as_current_span(f"chat {self._magi_model}") as span:
            self._open_span(span)
            span.set_attribute(ATTR_STREAMED, True)

            await self._announce(True)
            try:
                async for item in super().create_stream(*args, **kwargs):
                    # The generator's last item is the CreateResult; everything
                    # before it is a text delta. Recording on that one item is
                    # what keeps a streamed call indistinguishable from a
                    # non-streamed one in the metrics.
                    if isinstance(item, CreateResult):
                        self._record(span, item)
                        if item.finish_reason == "length":
                            span.set_attribute(ATTR_LENGTH_RETRY, False)
                            logger.warning(
                                "%s hit its %d-token budget mid-stream — no retry is "
                                "possible once chunks have been emitted, so this turn "
                                "is truncated",
                                self._magi_label, self._magi_max_tokens,
                            )
                    yield item
            finally:
                await self._announce(False)

    async def create(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        tracer = get_tracer()
        with tracer.start_as_current_span(f"chat {self._magi_model}") as span:
            self._open_span(span)

            # Announced here rather than at the phase boundaries because this is
            # the only place that knows a call is genuinely in flight — including
            # SelectorGroupChat's, which never becomes a chat message. The same
            # seam that counts the calls reports them.
            await self._announce(True)
            try:
                try:
                    result = await super().create(*args, **kwargs)
                except LengthFinishReasonError:
                    # A reasoning model spends its budget thinking before writing
                    # anything, and how much it needs varies with the question —
                    # so no fixed budget is ever reliably enough. Measured:
                    # nemotron3 burned its entire 4000-token allowance on one
                    # question and produced nothing, having finished comfortably
                    # on the previous one.
                    #
                    # This matters more than it looks. The group chat has no
                    # notion of a participant dropping out, so one advisor
                    # hitting the limit takes the whole debate down. One retry
                    # with double the room turns a fatal error into a slow turn.
                    widened = max(
                        self._magi_max_tokens * LENGTH_RETRY_FACTOR, PROBE_MAX_TOKENS
                    )
                    logger.warning(
                        "%s spent its whole %d-token budget on reasoning — retrying at %d",
                        self._magi_label, self._magi_max_tokens, widened,
                    )
                    span.set_attribute(ATTR_LENGTH_RETRY, True)
                    extra = dict(kwargs.pop("extra_create_args", {}) or {})
                    extra["max_tokens"] = widened
                    result = await super().create(*args, extra_create_args=extra, **kwargs)
            finally:
                # In `finally`, so an advisor that raises does not leave the
                # console showing it as still thinking for the rest of the
                # debate. The failure is reported elsewhere; a stuck indicator
                # would be a second and silent lie about it.
                await self._announce(False)

            self._record(span, result)
            return result


def build_client(
    settings: Settings,
    personas: PersonaSet,
    persona: Persona,
    counter: CallCounter | None = None,
    label: str | None = None,
    on_activity: ActivityHook | None = None,
) -> OpenAIChatCompletionClient:
    """A model client configured for one advisor.

    ``model_info`` has to be supplied by hand: AutoGen keeps a table of known
    OpenAI model names and refuses anything it does not recognise, which is
    every Ollama tag. Declaring ``structured_output=True`` is not optimism — it
    is what pre-flight verified per advisor before the node was allowed to boot.

    ``function_calling`` is False deliberately. The advisors have no tools, and
    claiming a capability nobody exercises would only invite a future change to
    rely on it untested.
    """
    create_args: dict = {}

    temperature = personas.temperature_for(persona)
    if temperature is not None:
        create_args["temperature"] = temperature

    max_tokens = personas.max_tokens_for(persona)
    if max_tokens is not None:
        create_args["max_tokens"] = max_tokens

    # Only sent when reasoning is explicitly disabled. Leaving the key out is
    # meaningfully different from sending "none": nemotron3 produces no valid
    # structured output at all with reasoning suppressed, while gemma3 is
    # measurably more reliable with it off.
    if personas.thinking_for(persona) is False:
        create_args["reasoning_effort"] = "none"

    return InstrumentedChatClient(
        magi_label=label or persona.name,
        counter=counter,
        on_activity=on_activity,
        model=persona.model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model_info=ModelInfo(
            vision=False,
            function_calling=False,
            json_output=True,
            structured_output=True,
            family=ModelFamily.UNKNOWN,
            multiple_system_messages=False,
        ),
        **create_args,
    )


def build_advisor(
    settings: Settings,
    personas: PersonaSet,
    persona: Persona,
    counter: CallCounter | None = None,
    on_activity: ActivityHook | None = None,
) -> AssistantAgent:
    """One advisor as an ``AssistantAgent`` that can only speak in ``MagiTurn``.

    ``output_content_type`` is the whole trick: the vote is not a separate phase
    the orchestrator has to ask for, it is the shape of every message. An
    advisor physically cannot answer without stating where it stands relative to
    the others.
    """
    return AssistantAgent(
        name=persona.name,
        model_client=build_client(
            settings, personas, persona, counter, on_activity=on_activity
        ),
        system_message=personas.system_prompt_for(persona),
        description=f"{persona.name}, the {persona.archetype}.",
        output_content_type=MagiTurn,
    )


def build_advisors(
    settings: Settings,
    personas: PersonaSet,
    counter: CallCounter | None = None,
    on_activity: ActivityHook | None = None,
) -> list[AssistantAgent]:
    """A fresh set of advisors.

    Called once per phase rather than once per debate. Phase A leaves its
    question and answer in each agent's model context; reusing those agents in
    the group chat would put every advisor's own blind position into its context
    twice, once as its own memory and once in the seed.
    """
    return [
        build_advisor(settings, personas, p, counter, on_activity)
        for p in personas.magi
    ]
