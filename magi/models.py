"""The typed shapes the debate is made of.

``MagiTurn`` and ``MagiVerdict`` are not merely internal dataclasses: they are
handed to the model as a JSON schema (AutoGen's ``output_content_type``), so
every field name and description is part of the prompt. Renaming a field
changes model behaviour. Treat them as prompt surface, not plumbing.

Schema constraints worth knowing before editing:

* **No arbitrary-key mappings.** ``dict[str, str]`` becomes an
  ``additionalProperties`` schema, which strict JSON-schema modes reject and
  loose ones honour unevenly across models. That is why per-advisor critique is
  a list of objects with an explicit ``advisor`` field rather than a dict keyed
  by name.
* **Keep it shallow.** Every level of nesting is another thing a 12B model can
  get subtly wrong, and a malformed turn costs a whole round.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Outcome(StrEnum):
    """How a debate ended.

    ``DEADLOCK`` covers both a three-way split and running out of round or time
    budget. Both mean the same thing to the operator: the advisors did not
    converge, and no synthesis will be invented for them.
    """

    UNANIMOUS = "UNANIMOUS"
    MAJORITY = "MAJORITY"
    DEADLOCK = "DEADLOCK"


class Critique(BaseModel):
    """One advisor's objection to another's position."""

    advisor: str = Field(description="Name of the advisor being criticised.")
    objection: str = Field(description="The specific objection, in one or two sentences.")


class MagiTurn(BaseModel):
    """One advisor's contribution to one round. Every turn is also a vote.

    The vote is not a separate phase — it is the shape of the message, so an
    advisor cannot answer without stating where it stands relative to the
    others.
    """

    position: str = Field(
        description="Your answer to the question and the reasoning behind it."
    )
    summary: str = Field(
        description=(
            "Your position in a single sentence, phrased to be read aloud "
            "through a speaker. No lists, no markdown."
        )
    )
    # The bar here is deliberately the same one the judge applies to a split
    # vote: would acting on their position lead somewhere materially different?
    # It used to read "would actively defend", which is a third copy of a rule
    # the system prompt and the deliberation seed also state, and the strictest
    # of the three wins in practice. See docs/agreement-bias.md § (d).
    agrees_with: list[str] = Field(
        default_factory=list,
        description=(
            "Names of the other advisors whose position you could act on, even "
            "if you would word it differently or add a caveat. Leave one out "
            "when acting on it would lead somewhere materially different from "
            "your own position. Leave empty in the first round, when you have "
            "not seen them."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in your own position, not in the group converging.",
    )
    critique: list[Critique] = Field(
        default_factory=list,
        description=(
            "Your objections to the other advisors, one entry each. Empty in "
            "the first round."
        ),
    )


class MagiVerdict(BaseModel):
    """The single answer MAGI returns, and the only thing that gets spoken.

    ``outcome`` is decided by the deterministic tally and passed in as a
    constraint. The orchestrator writes prose; it never counts.
    """

    outcome: Outcome
    answer: str = Field(
        description=(
            "Two or three sentences of plain spoken English. No lists, no "
            "headings, no markdown — this is read aloud."
        )
    )
    dissent: str | None = Field(
        default=None,
        description="On MAJORITY: who disagreed and why, in one clause. Otherwise null.",
    )
    disagreement_point: str | None = Field(
        default=None,
        description="On DEADLOCK: the precise point the advisors split on. Otherwise null.",
    )


# ── Records ──────────────────────────────────────────────────────────────────
#
# Below this line the models are ours, not the LLM's: they are never sent as a
# schema, so field names are free to be descriptive rather than promptable.


class TurnRecord(BaseModel):
    """One advisor's turn, with where in the debate it happened."""

    advisor: str
    round_index: int
    turn: MagiTurn
    #: True on the turns the outcome was actually computed from — the row
    #: ``latest_complete_round`` returned, or the votes ``ConsensusTermination``
    #: stopped on. Everything else was generated, paid for, shown on the console
    #: and then left out of the arithmetic, and the difference has to survive
    #: into the record: ``tallied_round`` says how deep the row was, this says
    #: which turns it was. Never true on a turn published live, because at that
    #: point nothing knows yet.
    tallied: bool = False


class Echo(BaseModel):
    """Two advisors whose tallied positions are largely the same text.

    Not an error and not an outcome: a fact about how the consensus was
    produced. Three advisors that agree because two of them stopped writing
    carry exactly as much information as one, and the tally scores it as the
    strongest possible agreement.
    """

    #: The pair, sorted, so the same echo reads the same way in every row.
    advisors: list[str]
    #: How much of the shorter position appears in the longer one, 0.0 to 1.0.
    #: 1.0 is verbatim.
    containment: float


class NodeCost(BaseModel):
    """What one debate cost the node it ran on, as opposed to the inference host.

    This is the number the project exists to produce. The Spark generates the
    tokens either way, so wall-clock barely moves between a MacBook and a Pi;
    what moves is the CPU and the resident memory a framework runtime consumes
    on a 4-core ARM board, and nobody publishes that.

    Sampled over the debate rather than read once at the end: ``getrusage``
    reports a peak for the life of the process, which after an hour of uptime
    answers a different question than "how big did this debate make the node".

    It is the whole **process**, not the debate in isolation. On a node that is
    also serving the console, sampling telemetry and holding a Whisper model,
    all of that is in here. That is the right number for "what does the node
    cost" and the wrong one for "what does the debate cost", so a benchmark run
    should be a node doing nothing else — which is what ``scripts/ask.py`` is.
    """

    #: CPU seconds burned between the start and the end of the debate, from
    #: ``getrusage`` deltas rather than from the samples: an exact figure for the
    #: window costs the same as an approximate one.
    cpu_s: float = 0.0
    #: ``cpu_s`` as a percentage of one core over the debate's wall clock. Above
    #: 100 means more than one core was busy.
    cpu_percent: float = 0.0
    peak_rss_mb: float = 0.0
    mean_rss_mb: float = 0.0
    #: How many RSS samples that mean is over. One sample is a reading, not a
    #: mean, and a row cannot be read without knowing which it got.
    samples: int = 0


class DebateRecord(BaseModel):
    """Everything one debate produced. The unit the SQLite store persists.

    ``terminated_by`` and ``judged_cosmetic`` are here because the interesting
    questions about this system are not about any single answer: how often do
    debates actually converge rather than run out of budget, and how often is an
    apparent disagreement only wording? Neither is recoverable after the fact if
    it was not recorded at the time.

    The same reasoning covers the comparability fields — ``engine``, ``models``,
    ``tracing_enabled``, ``streamed_advisors``, ``residency_warning``,
    ``max_rounds``. A row that cannot say how it was produced cannot be averaged
    with any other row, and finding that out at analysis time means the runs are
    gone.
    """

    debate_id: str
    question: str
    engine: str
    outcome: Outcome
    verdict: MagiVerdict
    #: **Every turn, in the order it was spoken**, blind round first — not one
    #: per advisor. The tally reads a single row (see ``tallied`` above), but the
    #: transcript is what makes "read the final positions for novelty" possible
    #: after the fact, and mode collapse is visible to the eye and invisible to
    #: the arithmetic. See docs/agreement-bias.md § "Measuring whether it worked".
    turns: list[TurnRecord]
    rounds_used: int
    #: A ``TERMINATED_BY_*`` value from ``constants``: ``consensus``,
    #: ``budget``, ``timeout``, ``barge_in``, ``error`` or
    #: ``insufficient_advisors``. ``budget`` and ``timeout`` are separate on
    #: purpose — both are non-convergence, but only one of them is an argument
    #: for a faster model.
    terminated_by: str
    #: True when the judge promoted a split vote by ruling the difference
    #: cosmetic, over all the positions at once.
    judged_cosmetic: bool = False
    #: How many one-sided agreement claims the judge repaired into mutual ones
    #: before the tally. Separate from ``judged_cosmetic`` because the two are
    #: different concessions: this one says two advisors were already agreeing
    #: and the turn order hid it, that one says a whole split vote was wording.
    #: An UNANIMOUS with both at zero is the only one nobody helped.
    judged_edges: int = 0
    #: How much of the tallied positions is each advisor's own text: 1.0 when no
    #: two of them share a phrase, 0.0 when at least two are word for word the
    #: same. ``None`` when fewer than two advisors were tallied and there is
    #: nothing to compare.
    #:
    #: This is the second axis of cheap consensus, and the one nothing else
    #: records. ``judged_cosmetic`` and ``judged_edges`` say the judge granted
    #: the agreement; this says the advisors stopped contributing. A debate can
    #: be unaided on both counts and still be an echo, so an UNANIMOUS is only
    #: earned when the judge stayed out of it *and* this stayed high.
    novelty: float | None = None
    #: The pairs that carried it down, worst first. Empty on a healthy debate.
    echoes: list[Echo] = Field(default_factory=list)
    #: The round the outcome was decided on: the deepest one in which every
    #: advisor spoke. Lower than ``rounds_used`` when the last round was cut off
    #: by the budget, the clock or a crash, and lower again under a selector
    #: that keeps re-asking the same advisor.
    tallied_round: int = 0
    #: Who actually answered. Shorter than the configured roster when a model
    #: was unreachable and the debate went ahead without it.
    advisors_present: list[str] = Field(default_factory=list)
    models: dict[str, str] = Field(default_factory=dict)
    duration_s: float = 0.0
    #: Which node produced the row. Two Pis, or a Pi and a MacBook, write to
    #: databases that may later be concatenated, and CPU figures from different
    #: boards must never be averaged together.
    node_id: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: The round budget this debate ran under. Raising it changes both the
    #: number of chances to converge and the message budget at once, so a
    #: convergence rate is uninterpretable without it.
    max_rounds: int = 0
    #: Advisors configured with ``stream: true``. Streaming moves per-chunk work
    #: onto the node whose CPU is being measured, and removes the length retry,
    #: so a run that mixed streamed and unstreamed advisors without recording
    #: which is not comparable with one that did not.
    streamed_advisors: list[str] = Field(default_factory=list)
    #: Whether pre-flight warned that the inference host was not holding every
    #: model resident. ``None`` when no recent pre-flight observed it. A debate
    #: whose models were evicted mid-run measures storage rather than the
    #: framework, and has to be excludable after the fact rather than silently
    #: skewing the benchmark.
    residency_warning: bool | None = None

    # Counted at the model client, so these include calls AutoGen makes on its
    # own behalf — notably SelectorGroupChat's speaker selection, which never
    # appears as a chat message and is the whole subject of the engine
    # comparison.
    llm_calls: int = 0
    #: Of those, the ones SelectorGroupChat spent choosing who speaks next.
    #: Zero under autogen_roundrobin — the difference between the two engines
    #: is exactly this number plus whatever latency it bought.
    selector_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Tracing is an observer effect on a node whose CPU and memory are being
    #: measured. A row that does not know whether it was traced cannot be
    #: compared with one that was.
    tracing_enabled: bool = False
    #: What the debate cost the node. ``None`` when ``MAGI_METRICS_ENABLED=0``,
    #: which is a real configuration and must not be confused with zero cost.
    node: NodeCost | None = None
