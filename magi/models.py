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


class DebateRecord(BaseModel):
    """Everything one debate produced. The unit the SQLite store persists.

    ``terminated_by`` and ``judged_cosmetic`` are here because the interesting
    questions about this system are not about any single answer: how often do
    debates actually converge rather than run out of budget, and how often is an
    apparent disagreement only wording? Neither is recoverable after the fact if
    it was not recorded at the time.
    """

    debate_id: str
    question: str
    engine: str
    outcome: Outcome
    verdict: MagiVerdict
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
