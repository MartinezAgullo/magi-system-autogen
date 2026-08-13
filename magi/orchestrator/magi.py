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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import ExternalTermination
from autogen_agentchat.messages import (
    ModelClientStreamingChunkEvent,
    StructuredMessage,
)

from magi.bus import Bus
from magi.config import Settings
from magi.constants import (
    ATTR_ADVISOR,
    ATTR_ADVISORS_PRESENT,
    ATTR_DEBATE_ID,
    ATTR_ENGINE,
    ATTR_JUDGED_COSMETIC,
    ATTR_JUDGED_EDGES,
    ATTR_LLM_CALLS,
    ATTR_MODELS,
    ATTR_NOVELTY,
    ATTR_OUTCOME,
    ATTR_ROUND,
    ATTR_ROUNDS_USED,
    ATTR_SELECTOR_CALLS,
    ATTR_TALLIED_ROUND,
    ATTR_TERMINATED_BY,
    ATTR_TRACING_ENABLED,
    BLIND_ROUND,
    GEN_AI_USAGE_INPUT,
    GEN_AI_USAGE_OUTPUT,
    SELECTOR_LABEL,
    SPAN_DEBATE,
    SPAN_JUDGE,
    SPAN_JUDGE_PAIRS,
    SPAN_PHASE_BLIND,
    SPAN_PHASE_DELIBERATION,
    SPAN_PHASE_VERDICT,
    SPAN_TALLY,
    SPAN_TURN,
    TERMINATED_BY_CONSENSUS,
    TERMINATED_BY_ERROR,
    TERMINATED_BY_INSUFFICIENT,
    TOPIC_ACTIVITY,
    TOPIC_CHUNK,
    TOPIC_QUESTION,
    TOPIC_TURN,
    TOPIC_VERDICT,
)
from magi.models import DebateRecord, MagiTurn, MagiVerdict, Outcome, TurnRecord
from magi.orchestrator import prompts
from magi.orchestrator.clients import build_advisor, build_advisors, build_client
from magi.orchestrator.consensus import (
    Tally,
    asymmetric_pairs,
    latest_complete_round,
    tally,
)
from magi.orchestrator.judge import judge_disagreement, judge_pairs
from magi.orchestrator.novelty import measure as measure_novelty
from magi.orchestrator.teams import build_team
from magi.orchestrator.termination import build_termination
from magi.personas import PersonaSet
from magi.services.metrics import CallCounter, ProcessSampler
from magi.setup.setup_tracing import get_tracer
from magi.store.debates import DebateStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Deliberation:
    """What phase B produced, and on how much of it the outcome rests."""

    #: One turn per advisor, all from the same round — the row the tally reads.
    turns: dict[str, MagiTurn]
    terminated_by: str
    #: Rounds in which any turn happened, including the blind one. Work done.
    rounds_used: int
    #: The deepest round every advisor reached. Evidence the outcome rests on.
    tallied_round: int
    #: Every turn of the debate in the order it was spoken, blind round first,
    #: with ``tallied`` set on the ones above. The tally needs a single row; the
    #: record needs all of them, because a transcript that keeps only the last
    #: turn per advisor cannot answer whether the three positions converged or
    #: collapsed into each other.
    transcript: tuple[TurnRecord, ...] = ()


class Magi:
    """The orchestrator. One instance per node; one :meth:`debate` per question."""

    def __init__(
        self,
        settings: Settings,
        personas: PersonaSet,
        bus: Bus | None = None,
        tracer_provider=None,
        store: DebateStore | None = None,
        residency_warning: bool | None = None,
    ) -> None:
        self._settings = settings
        self._personas = personas
        self._bus = bus
        self._store = store
        # What the last pre-flight said about model residency on the inference
        # host, or None when nothing recent looked. Carried on every row so a
        # run whose weights were evicted mid-debate can be excluded from the
        # benchmark afterwards instead of quietly widening every latency figure.
        self._residency_warning = residency_warning
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

    async def _run_agent(self, agent, task: str, name: str) -> TaskResult:
        """Run one agent alone, forwarding any token deltas it emits.

        ``run_stream`` rather than ``run`` so that an advisor configured with
        ``stream: true`` is streamed in phase A as well as phase B. For every
        other advisor this is exactly ``run``: with ``model_client_stream``
        off, no chunk events are ever produced and the loop simply collects the
        final ``TaskResult``.
        """
        result: TaskResult | None = None
        async for message in agent.run_stream(task=task):
            if isinstance(message, TaskResult):
                result = message
            elif isinstance(message, ModelClientStreamingChunkEvent):
                await self._publish_chunk(name, message.content)
        if result is None:
            raise RuntimeError(f"{name} produced no result")
        return result

    async def _blind_round(self, question: str) -> dict[str, MagiTurn]:
        """Every advisor answers in parallel, in isolation."""
        task = prompts.blind_task(question)
        tracer = get_tracer()

        async def ask(persona) -> tuple[str, MagiTurn | None]:
            agent = build_advisor(
                self._settings, self._personas, persona, self._counter,
                self._publish_activity,
            )
            with tracer.start_as_current_span(SPAN_TURN) as span:
                span.set_attribute(ATTR_ADVISOR, persona.name)
                span.set_attribute(ATTR_ROUND, 1)
                try:
                    result = await self._run_agent(agent, task, persona.name)
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
    ) -> _Deliberation:
        """Run the group chat, and pick the row of votes the outcome rests on."""
        advisors = build_advisors(
            self._settings, self._personas, self._counter, self._publish_activity
        )
        # The transcript starts with the blind round: those turns are as much a
        # part of the debate as the deliberated ones, and on a run that never
        # reaches the group chat they are the whole of it.
        transcript: list[TurnRecord] = [
            TurnRecord(advisor=name, round_index=BLIND_ROUND, turn=turn)
            for name, turn in blind.items()
        ]

        present = [a for a in advisors if a.name in blind]
        if len(present) < 2:
            logger.warning("Fewer than two advisors answered — skipping deliberation")
            return _Deliberation(
                dict(blind), TERMINATED_BY_INSUFFICIENT, 1, 1,
                _mark_tallied(transcript, dict(blind)),
            )

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

        # Every turn each advisor produced, in order, blind round first. Kept as
        # a history rather than as "the latest per advisor" because the tally
        # needs one round of votes and not a mixture of rounds: whoever spoke
        # last in a truncated debate would otherwise have its newest position
        # counted against everyone else's previous one.
        history: dict[str, list[MagiTurn]] = {
            name: [turn] for name, turn in blind.items()
        }
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
                    if isinstance(message, ModelClientStreamingChunkEvent):
                        await self._publish_chunk(message.source.upper(), message.content)
                        continue
                    if isinstance(message, StructuredMessage) and isinstance(
                        message.content, MagiTurn
                    ):
                        deliberation_turns += 1
                        name = message.source.upper()
                        history.setdefault(name, []).append(message.content)
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
                        # The round index was already computed here and then
                        # thrown away: the record used to stamp every turn with
                        # the final round, so a three-round debate read as three
                        # advisors speaking once each, in the last round.
                        transcript.append(
                            TurnRecord(
                                advisor=name,
                                round_index=round_index,
                                turn=message.content,
                            )
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
        # "Every advisor has spoken at least this many times." Under a complete
        # round it is the round itself; on the consensus path below the row is
        # deeper than this for whoever spoke last, so it reads as a floor.
        tallied_round = min(len(turns) for turns in history.values())

        if stopped_by == TERMINATED_BY_CONSENSUS:
            # ConsensusTermination watched each advisor's latest vote and
            # stopped the debate on finding them unanimous. Tallying a different
            # row here would let the record say DEADLOCK about a run whose
            # stated reason for ending is that it converged, so the row that
            # stopped it is the row that counts. Reconstructed from `history`
            # rather than read off the condition: the group chat manager resets
            # the whole stack the instant one fires, and `latest_turns` is
            # already empty by the time this line runs.
            decided = {name: turns[-1] for name, turns in history.items()}
        else:
            decided = latest_complete_round(history)

        return _Deliberation(
            decided, stopped_by, rounds_used, tallied_round,
            _mark_tallied(transcript, decided),
        )

    async def _verdict(
        self, question: str, turns: Mapping[str, MagiTurn], result: Tally
    ) -> MagiVerdict:
        """Write the spoken answer for an outcome that is already decided."""
        order = [p.name for p in self._personas.magi]
        agent = _as_verdict_agent(
            self._settings, self._personas, self._counter, self._publish_activity
        )

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
        """One question in, one recorded verdict out.

        The two things wrapped around the debate proper are the two the
        benchmark is made of: a sampler watching what the node spends while it
        runs, and the write that makes the run survive the process.
        """
        async with ProcessSampler(
            self._settings.metrics_sample_interval_s,
            enabled=self._settings.metrics_enabled,
        ) as sampler:
            record = await self._debate(question, sampler)

        await self._persist(record)
        return record

    async def _persist(self, record: DebateRecord) -> None:
        """Write the row, and never lose the answer over a failed write.

        After the verdict has already been published, deliberately. The operator
        is waiting on the spoken answer, not on the benchmark record, and the
        write goes to a thread anyway — but ordering it this way means a full
        disk costs a row and not the debate that produced it.
        """
        if self._store is None:
            return
        try:
            await self._store.save(record)
        except Exception:
            logger.exception("Could not record debate %s — the row is lost, the "
                             "verdict is not", record.debate_id)

    async def _debate(self, question: str, sampler: ProcessSampler) -> DebateRecord:
        started = time.monotonic()
        # Wall clock as well as a monotonic start: the first is what every
        # cross-run query orders by, the second is the only one safe to subtract.
        started_at = datetime.now(UTC)
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
            deliberation = await self._deliberate(question, blind)
            turns = deliberation.turns
            terminated_by = deliberation.terminated_by
            rounds_used = deliberation.rounds_used

            with tracer.start_as_current_span(SPAN_TALLY) as tally_span:
                tally_span.set_attribute(ATTR_TALLIED_ROUND, deliberation.tallied_round)
                result = tally(turns)

            # The judge runs twice over different scopes, and only ever on a
            # split vote. First per ambiguous pair, then — if that found no real
            # difference — over everything at once. See judge.py.
            repaired: tuple[tuple[str, str], ...] = ()
            settled_pair = False
            if not result.unanimous and len(turns) > 1:
                pairs = asymmetric_pairs(turns)
                if pairs:
                    with tracer.start_as_current_span(SPAN_JUDGE_PAIRS) as pair_span:
                        rulings = await judge_pairs(
                            self._settings, self._personas, turns, pairs,
                            self._counter, self._publish_activity,
                        )
                        pair_span.set_attribute("magi.pairs", len(pairs))
                        pair_span.set_attribute("magi.repaired", len(rulings.repaired))
                    repaired = rulings.repaired
                    settled_pair = bool(rulings.substantive)
                    if repaired:
                        result = tally(turns, extra_edges=repaired)
                        logger.info(
                            "[%s] Judge repaired %d one-sided claim(s): %s",
                            debate_id, len(repaired),
                            "; ".join(f"{a}+{b}" for a, b in repaired),
                        )

            # A ruling of "cosmetic" promotes the outcome; a judge that could
            # not be reached leaves it as it was. Skipped when the pairwise pass
            # already found a real difference between two specific positions:
            # a vaguer question must not overrule a sharper one that was asked
            # of the same model minutes earlier.
            judged_cosmetic = False
            if not result.unanimous and len(turns) > 1 and not settled_pair:
                with tracer.start_as_current_span(SPAN_JUDGE) as judge_span:
                    ruling = await judge_disagreement(
                        self._settings, self._personas, turns, order, self._counter,
                        self._publish_activity,
                    )
                    judge_span.set_attribute("magi.judge_ran", ruling is not None)
                    if ruling is not None:
                        judge_span.set_attribute("magi.substantive", ruling.substantive)
                if ruling is not None and not ruling.substantive:
                    logger.info("[%s] Judge: disagreement is cosmetic (%s)",
                                debate_id, ruling.reason)
                    result = Tally(Outcome.UNANIMOUS, tuple(sorted(turns)), ())
                    judged_cosmetic = True

            # The other axis of cheap agreement, and the one the judge fields
            # say nothing about: whether the advisors wrote three answers or
            # copied one. Free — no model call, pure text — so it is measured on
            # every debate rather than only when someone suspects it.
            novelty = measure_novelty(
                {name: turn.position for name, turn in turns.items()}
            )
            if novelty.echoed:
                logger.warning(
                    "[%s] %s reached on repeated text: %s. Three identical "
                    "answers carry as much information as one",
                    debate_id, result.outcome.value,
                    "; ".join(
                        f"{a}≡{b} ({echo.containment:.0%})"
                        for echo in novelty.echoes
                        for a, b in [echo.advisors]
                    ),
                )

            verdict = await self._verdict(question, turns, result)
            elapsed = time.monotonic() - started

            root.set_attribute(ATTR_OUTCOME, result.outcome.value)
            root.set_attribute(ATTR_ROUNDS_USED, rounds_used)
            root.set_attribute(ATTR_TERMINATED_BY, terminated_by)
            root.set_attribute(ATTR_JUDGED_COSMETIC, judged_cosmetic)
            root.set_attribute(ATTR_JUDGED_EDGES, len(repaired))
            if novelty.score is not None:
                # A number, not a transcript: safe on a span, and it makes
                # "show me the debates that agreed by copying" one query.
                root.set_attribute(ATTR_NOVELTY, novelty.score)
            root.set_attribute(ATTR_TALLIED_ROUND, deliberation.tallied_round)
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
                # The whole transcript, not one turn per advisor. Each turn
                # carries the round it was actually spoken in, and the ones the
                # outcome rests on carry `tallied`.
                turns=list(deliberation.transcript),
                rounds_used=rounds_used,
                terminated_by=terminated_by,
                judged_cosmetic=judged_cosmetic,
                judged_edges=len(repaired),
                novelty=novelty.score,
                echoes=list(novelty.echoes),
                tallied_round=deliberation.tallied_round,
                advisors_present=sorted(turns),
                models={p.name: p.model for p in self._personas.magi},
                duration_s=elapsed,
                node_id=self._settings.node_id,
                started_at=started_at,
                max_rounds=self._settings.max_rounds,
                streamed_advisors=self._personas.streaming_names(),
                residency_warning=self._residency_warning,
                llm_calls=self._counter.total.calls,
                selector_calls=self._counter.by_label[SELECTOR_LABEL].calls,
                prompt_tokens=self._counter.total.prompt_tokens,
                completion_tokens=self._counter.total.completion_tokens,
                tracing_enabled=self._tracer_provider is not None,
                # Read here rather than after the sampler stops, so the record is
                # complete the moment it exists. It costs one more sample's worth
                # of resolution and buys a record that is never briefly wrong.
                node=sampler.cost(),
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

    async def _publish_chunk(self, name: str, text: str) -> None:
        """One token delta from an advisor configured to stream.

        Only ever fires for advisors with ``stream: true`` in the persona file,
        so a node that streams nothing publishes nothing and the topic costs
        exactly zero.
        """
        if self._bus is not None:
            await self._bus.publish(TOPIC_CHUNK, {"advisor": name, "text": text})

    async def _publish_activity(self, name: str, busy: bool) -> None:
        """Who has an LLM call open, as it opens and closes.

        Handed to every model client this orchestrator builds. A debate is
        30-90 s of one advisor generating with nothing to show for it until the
        turn lands, and a console that renders only completed turns spends that
        whole time looking like it has crashed.
        """
        if self._bus is not None:
            await self._bus.publish(TOPIC_ACTIVITY, {"advisor": name, "busy": busy})

    async def _publish_turn(self, name: str, turn: MagiTurn, round_index: int) -> None:
        logger.info("  %-10s r%d  %s", name, round_index, turn.summary.strip())
        if self._bus is not None:
            await self._bus.publish(
                TOPIC_TURN,
                TurnRecord(advisor=name, round_index=round_index, turn=turn),
            )


def _as_verdict_agent(
    settings: Settings, personas: PersonaSet, counter=None, on_activity=None
):
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
        model_client=build_client(
            settings, personas, personas.orchestrator, counter, on_activity=on_activity
        ),
        system_message=personas.system_prompt_for(personas.orchestrator),
        output_content_type=MagiVerdict,
    )


def _mark_tallied(
    transcript: Sequence[TurnRecord], decided: Mapping[str, MagiTurn]
) -> tuple[TurnRecord, ...]:
    """Flag the turns the outcome was computed from.

    Matched by object identity rather than by advisor and round, because the two
    paths that pick the decided row do not agree on what a round is: a debate
    stopped by consensus is tallied on each advisor's latest vote, which can sit
    at three different depths. The objects are the same ones the tally read, so
    identity says exactly what happened and no reconstruction can disagree with
    it later.
    """
    for entry in transcript:
        entry.tallied = decided.get(entry.advisor) is entry.turn
    return tuple(transcript)


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
