"""MAGI: drives one debate from a question to a spoken verdict.

Three phases, and the split between them is not cosmetic.

**Phase A — blind.** Each advisor answers alone. This cannot happen inside a
group chat: ``RoundRobinGroupChat`` gives every agent one shared thread, so the
second speaker would read the first one's answer and anchor to it. Three models
that agree because they read each other are not a consensus, they are an echo —
and that failure would be invisible in the output, which is what makes it worth
stepping outside the team abstraction to prevent.

**Phase B — deliberation.** The group chat, seeded with the three blind
positions, until a termination condition fires.

**Phase C — verdict.** The outcome is computed in Python; a final agent writes
the sentence that gets spoken.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination
from autogen_agentchat.messages import StructuredMessage

from magi.bus import Bus
from magi.config import Settings
from magi.constants import (
    ATTR_ADVISOR,
    ATTR_ADVISORS_PRESENT,
    ATTR_DEBATE_ID,
    ATTR_ENGINE,
    ATTR_JUDGED_COSMETIC,
    ATTR_LLM_CALLS,
    ATTR_MODELS,
    ATTR_OUTCOME,
    ATTR_ROUND,
    ATTR_ROUNDS_USED,
    ATTR_SELECTOR_CALLS,
    ATTR_TERMINATED_BY,
    ATTR_TRACING_ENABLED,
    GEN_AI_USAGE_INPUT,
    GEN_AI_USAGE_OUTPUT,
    SELECTOR_LABEL,
    SPAN_DEBATE,
    SPAN_JUDGE,
    SPAN_PHASE_BLIND,
    SPAN_PHASE_DELIBERATION,
    SPAN_PHASE_VERDICT,
    SPAN_TALLY,
    SPAN_TURN,
    TERMINATED_BY_ERROR,
    TERMINATED_BY_INSUFFICIENT,
    TOPIC_QUESTION,
    TOPIC_TURN,
    TOPIC_VERDICT,
)
from magi.models import DebateRecord, MagiTurn, MagiVerdict, Outcome, TurnRecord
from magi.orchestrator import prompts
from magi.orchestrator.clients import build_advisor, build_advisors, build_client
from magi.orchestrator.consensus import Tally, tally
from magi.orchestrator.judge import judge_disagreement
from magi.orchestrator.teams import build_team
from magi.orchestrator.termination import build_termination
from magi.personas import PersonaSet
from magi.services.metrics import CallCounter
from magi.setup.setup_tracing import get_tracer

logger = logging.getLogger(__name__)


class Magi:
    """The orchestrator. One instance per node; one :meth:`debate` per question."""

    def __init__(
        self,
        settings: Settings,
        personas: PersonaSet,
        bus: Bus | None = None,
        tracer_provider=None,
    ) -> None:
        self._settings = settings
        self._personas = personas
        self._bus = bus
        self._external: ExternalTermination | None = None
        # Not passed to AutoGen — it reads the global provider that
        # setup_tracing() registers (see _deliberate). Held only to record on
        # each debate whether the run was instrumented, which is the one fact
        # that makes a traced run comparable, or not, with an untraced one.
        self._tracer_provider = tracer_provider
        self._counter = CallCounter()

    # ── Barge-in ─────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Abort the running debate, if any.

        This is what a second PTT press calls. ``ExternalTermination`` is
        AutoGen's own mechanism for it — one of the places the framework
        genuinely covers a requirement rather than getting in the way of it.
        """
        if self._external is not None and not self._external.terminated:
            logger.info("Barge-in: cancelling the running debate")
            self._external.set()

    # ── Phases ───────────────────────────────────────────────────────────

    async def _blind_round(self, question: str) -> dict[str, MagiTurn]:
        """Every advisor answers in parallel, in isolation."""
        task = prompts.blind_task(question)
        tracer = get_tracer()

        async def ask(persona) -> tuple[str, MagiTurn | None]:
            agent = build_advisor(
                self._settings, self._personas, persona, self._counter
            )
            with tracer.start_as_current_span(SPAN_TURN) as span:
                span.set_attribute(ATTR_ADVISOR, persona.name)
                span.set_attribute(ATTR_ROUND, 1)
                try:
                    result = await agent.run(task=task)
                except Exception as exc:
                    # One unreachable model must not end the debate: the
                    # remaining advisors carry on and the verdict says so. That
                    # is the 2/3 degradation the design calls for, and it is why
                    # this gathers results rather than letting it propagate.
                    logger.exception("%s failed the blind round — continuing without it",
                                     persona.name)
                    span.record_exception(exc)
                    return persona.name, None
                return persona.name, _last_turn(result)

        with tracer.start_as_current_span(SPAN_PHASE_BLIND):
            answers = await asyncio.gather(*(ask(p) for p in self._personas.magi))
            turns = {name: turn for name, turn in answers if turn is not None}

        for name, turn in turns.items():
            await self._publish_turn(name, turn, round_index=1)

        missing = [p.name for p in self._personas.magi if p.name not in turns]
        if missing:
            logger.warning("Blind round produced no answer from: %s", ", ".join(missing))
        return turns

    async def _deliberate(
        self, question: str, blind: Mapping[str, MagiTurn]
    ) -> tuple[dict[str, MagiTurn], str, int]:
        """Run the group chat. Returns the final turns, what stopped it, and rounds used."""
        advisors = build_advisors(self._settings, self._personas, self._counter)
        present = [a for a in advisors if a.name in blind]
        if len(present) < 2:
            logger.warning("Fewer than two advisors answered — skipping deliberation")
            return dict(blind), TERMINATED_BY_INSUFFICIENT, 1

        self._external = ExternalTermination()
        terminators = build_termination(
            self._settings, [a.name for a in present], self._external
        )
        team = build_team(
            self._settings, self._personas, present, terminators.condition, self._counter
        )

        seed = prompts.deliberation_seed(
            question, blind, [p.name for p in self._personas.magi]
        )

        turns = dict(blind)
        deliberation_turns = 0
        # Read from the latches after the run, never from `condition.terminated`
        # and never by sampling during the stream. The group chat manager resets
        # the whole stack the moment a condition fires, before this coroutine is
        # scheduled again, so every condition reads False everywhere out here.
        # `Latching` is what survives that; see termination.py.
        stopped_by: str | None = None

        with get_tracer().start_as_current_span(SPAN_PHASE_DELIBERATION) as phase_span:
            try:
                async for message in team.run_stream(task=seed):
                    if isinstance(message, TaskResult):
                        continue
                    if isinstance(message, StructuredMessage) and isinstance(
                        message.content, MagiTurn
                    ):
                        deliberation_turns += 1
                        name = message.source.upper()
                        turns[name] = message.content
                        # Round 1 was blind, so deliberation starts at round 2.
                        # Under SelectorGroupChat there are no rounds — the
                        # selector may call the same advisor after two others,
                        # or never. This is a turn-count divided by the roster
                        # size, reported as an approximation rather than
                        # pretending the two engines produce the same structure.
                        round_index = 2 + (deliberation_turns - 1) // len(present)
                        # An event rather than a child span: the turn happens
                        # inside AutoGen, which is already opening its own spans
                        # for it. A parallel span of ours would double-count the
                        # same work in every latency breakdown.
                        phase_span.add_event(
                            SPAN_TURN, {ATTR_ADVISOR: name, ATTR_ROUND: round_index}
                        )
                        await self._publish_turn(name, message.content, round_index)
            except Exception as exc:
                # One advisor failing must not lose the debate. This is where
                # the framework fights the 2/3 degradation requirement: in the
                # blind round the failure is contained because we own the
                # gather, but a group chat has no notion of a participant
                # dropping out — any participant raising takes the whole run
                # down with it. So the turns collected so far are kept and the
                # debate is tallied on them, flagged so the record shows it was
                # cut short.
                logger.exception("Deliberation failed — tallying the turns so far")
                phase_span.record_exception(exc)
                stopped_by = TERMINATED_BY_ERROR

            # The exception path wins: a run that raised did not reach whatever
            # the latches would report, and "budget" on a debate that crashed
            # would hide a failure inside a normal-looking statistic.
            if stopped_by is None:
                stopped_by = terminators.fired()

            phase_span.set_attribute(ATTR_TERMINATED_BY, stopped_by)
            phase_span.set_attribute(ATTR_ROUND, deliberation_turns)

        rounds_used = 1 + -(-deliberation_turns // len(present))  # ceil
        return turns, stopped_by, rounds_used

    async def _verdict(
        self, question: str, turns: Mapping[str, MagiTurn], result: Tally
    ) -> MagiVerdict:
        """Write the spoken answer for an outcome that is already decided."""
        order = [p.name for p in self._personas.magi]
        agent = _as_verdict_agent(self._settings, self._personas, self._counter)

        absent = [name for name in order if name not in turns]
        task = prompts.verdict_task(
            question, turns, order, result.outcome, result.majority, result.dissent,
            absent,
        )
        try:
            with get_tracer().start_as_current_span(SPAN_PHASE_VERDICT):
                run = await agent.run(task=task)
        except Exception:
            logger.exception("Verdict agent failed — falling back to the raw tally")
            return _fallback_verdict(turns, order, result)

        verdict = _last_verdict(run)
        if verdict is None:
            logger.warning("Verdict agent returned nothing usable — using the raw tally")
            return _fallback_verdict(turns, order, result)

        # The outcome is ours, not the model's. It is given in the prompt and
        # overwritten here anyway: a model that mislabels a DEADLOCK as
        # UNANIMOUS would report a consensus that never happened.
        return verdict.model_copy(update={"outcome": result.outcome})

    # ── Entry point ──────────────────────────────────────────────────────

    async def debate(self, question: str) -> DebateRecord:
        started = time.monotonic()
        debate_id = uuid.uuid4().hex[:12]
        order = [p.name for p in self._personas.magi]
        self._counter.reset()
        logger.info("[%s] Debate: %s", debate_id, question)

        # Announced before any model is called. The blind round takes 20-30s,
        # and a console that shows nothing until the first turn lands looks
        # broken for exactly as long as the system is working hardest.
        if self._bus is not None:
            await self._bus.publish(
                TOPIC_QUESTION,
                {
                    "debate_id": debate_id,
                    "question": question,
                    "engine": self._settings.engine,
                    "max_rounds": self._settings.max_rounds,
                    "advisors": order,
                },
            )

        tracer = get_tracer()
        with tracer.start_as_current_span(SPAN_DEBATE) as root:
            root.set_attribute(ATTR_DEBATE_ID, debate_id)
            root.set_attribute(ATTR_ENGINE, self._settings.engine)
            root.set_attribute(ATTR_MODELS, sorted(self._personas.distinct_models()))
            # Recorded on the span itself, because a comparison between a traced
            # run and an untraced one is invalid and this is the only place that
            # fact survives.
            root.set_attribute(ATTR_TRACING_ENABLED, self._tracer_provider is not None)

            blind = await self._blind_round(question)
            turns, terminated_by, rounds_used = await self._deliberate(question, blind)

            with tracer.start_as_current_span(SPAN_TALLY):
                result = tally(turns)

            # The judge runs only on a split vote, and only to decide whether
            # the disagreement is real. A ruling of "cosmetic" promotes the
            # outcome; a judge that could not be reached leaves it as it was.
            judged_cosmetic = False
            if not result.unanimous and len(turns) > 1:
                with tracer.start_as_current_span(SPAN_JUDGE) as judge_span:
                    ruling = await judge_disagreement(
                        self._settings, self._personas, turns, order, self._counter
                    )
                    judge_span.set_attribute("magi.judge_ran", ruling is not None)
                    if ruling is not None:
                        judge_span.set_attribute("magi.substantive", ruling.substantive)
                if ruling is not None and not ruling.substantive:
                    logger.info("[%s] Judge: disagreement is cosmetic (%s)",
                                debate_id, ruling.reason)
                    result = Tally(Outcome.UNANIMOUS, tuple(sorted(turns)), ())
                    judged_cosmetic = True

            verdict = await self._verdict(question, turns, result)
            elapsed = time.monotonic() - started

            root.set_attribute(ATTR_OUTCOME, result.outcome.value)
            root.set_attribute(ATTR_ROUNDS_USED, rounds_used)
            root.set_attribute(ATTR_TERMINATED_BY, terminated_by)
            root.set_attribute(ATTR_JUDGED_COSMETIC, judged_cosmetic)
            root.set_attribute(ATTR_ADVISORS_PRESENT, sorted(turns))
            root.set_attribute(ATTR_LLM_CALLS, self._counter.total.calls)
            root.set_attribute(
                ATTR_SELECTOR_CALLS, self._counter.by_label[SELECTOR_LABEL].calls
            )
            root.set_attribute(GEN_AI_USAGE_INPUT, self._counter.total.prompt_tokens)
            root.set_attribute(GEN_AI_USAGE_OUTPUT, self._counter.total.completion_tokens)

            record = DebateRecord(
                debate_id=debate_id,
                question=question,
                engine=self._settings.engine,
                outcome=result.outcome,
                verdict=verdict,
                turns=[
                    TurnRecord(advisor=name, round_index=rounds_used, turn=turn)
                    for name, turn in turns.items()
                ],
                rounds_used=rounds_used,
                terminated_by=terminated_by,
                judged_cosmetic=judged_cosmetic,
                advisors_present=sorted(turns),
                models={p.name: p.model for p in self._personas.magi},
                duration_s=elapsed,
                llm_calls=self._counter.total.calls,
                selector_calls=self._counter.by_label[SELECTOR_LABEL].calls,
                prompt_tokens=self._counter.total.prompt_tokens,
                completion_tokens=self._counter.total.completion_tokens,
                tracing_enabled=self._tracer_provider is not None,
            )

        logger.info(
            "[%s] %s in %.1fs (%d rounds, %d LLM calls incl. %d selector, "
            "stopped by %s)",
            debate_id, result.outcome.value, elapsed, rounds_used,
            self._counter.total.calls,
            self._counter.by_label[SELECTOR_LABEL].calls, terminated_by,
        )
        if self._bus is not None:
            await self._bus.publish(TOPIC_VERDICT, record)
        return record

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _publish_turn(self, name: str, turn: MagiTurn, round_index: int) -> None:
        logger.info("  %-10s r%d  %s", name, round_index, turn.summary.strip())
        if self._bus is not None:
            await self._bus.publish(
                TOPIC_TURN,
                TurnRecord(advisor=name, round_index=round_index, turn=turn),
            )


def _as_verdict_agent(settings: Settings, personas: PersonaSet, counter=None):
    """The orchestrator agent, typed to ``MagiVerdict``.

    Not a ``SocietyOfMindAgent``, which is what the design originally called
    for: in AutoGen 0.7.5 that class takes no ``output_content_type``, so the
    one place its highest-level abstraction fits, it cannot return typed fields.
    A plain ``AssistantAgent`` fed the final positions does the same job and
    keeps `outcome`, `dissent` and `disagreement_point` as data instead of prose
    to be re-parsed.
    """
    from autogen_agentchat.agents import AssistantAgent

    return AssistantAgent(
        name=personas.orchestrator.name,
        model_client=build_client(settings, personas, personas.orchestrator, counter),
        system_message=personas.system_prompt_for(personas.orchestrator),
        output_content_type=MagiVerdict,
    )


def _last_turn(result: TaskResult) -> MagiTurn | None:
    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage) and isinstance(message.content, MagiTurn):
            return message.content
    return None


def _last_verdict(result: TaskResult) -> MagiVerdict | None:
    for message in reversed(result.messages):
        if isinstance(message, StructuredMessage) and isinstance(
            message.content, MagiVerdict
        ):
            return message.content
    return None


def _fallback_verdict(
    turns: Mapping[str, MagiTurn], order, result: Tally
) -> MagiVerdict:
    """What the node says when the orchestrator model is unavailable.

    Blunt, and honest about being blunt. The alternative — staying silent
    because the summariser is down, after three advisors already answered —
    throws away work the operator is waiting on.
    """
    from magi.orchestrator.consensus import disagreement_summary

    lines = disagreement_summary(turns, order)
    absent = [name for name in order if name not in turns]
    prefix = f"Only {len(turns)} of {len(order)} advisors answered. " if absent else ""
    if result.outcome is Outcome.DEADLOCK:
        answer = f"{prefix}The advisors did not converge. {lines}"
    else:
        leader = result.majority[0] if result.majority else next(iter(turns), "")
        answer = prefix + (turns[leader].summary.strip() if leader in turns else lines)
    return MagiVerdict(
        outcome=result.outcome,
        answer=answer,
        dissent=", ".join(result.dissent) or None,
        disagreement_point=lines if result.outcome is Outcome.DEADLOCK else None,
    )
