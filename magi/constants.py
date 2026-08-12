"""Values that are part of the design, not of the deployment.

The split against ``config.py`` is deliberate and worth keeping: **anything an
operator might reasonably want to change per node belongs in ``config.py``**
(hosts, ports, model budgets, feature flags). What lives here is the protocol
itself — bus topic names, span and attribute keys, invariants the code would be
wrong without. Making these configurable would only create ways for two nodes
to disagree about what a debate is.

Span and attribute names matter more than they look: they are the shared
vocabulary with the sibling no-framework repo, so its copy of this file must
match or the two sets of traces cannot be compared.
"""

from __future__ import annotations

# ── Debate invariants ────────────────────────────────────────────────────────

# Two advisors still make a debate (2/2 unanimous, or 1-1 deadlock). One does
# not — a single model talking to itself is not a deliberation.
MIN_ADVISORS = 2

# The first round is blind: advisors answer without seeing each other, so the
# group chat cannot start before round 2. Not a tunable — it is what stops the
# debate from being an echo.
BLIND_ROUND = 1

# ── Bus topics ───────────────────────────────────────────────────────────────

TOPIC_QUESTION = "question"       # a transcribed PTT press, ready to debate
TOPIC_TURN = "turn"               # one advisor's MagiTurn, as it lands
TOPIC_VERDICT = "verdict"         # the MagiVerdict that ends a debate
TOPIC_STATUS = "status"           # connectivity and backend health
TOPIC_SPEAK = "speak"             # text queued for TTS
TOPIC_DRAFT = "draft"             # the question under composition, after every edit
TOPIC_STT = "stt"                 # per-press recognition feedback (busy, empty, error)

# ── OpenTelemetry ────────────────────────────────────────────────────────────

TRACER_NAME = "magi"

SPAN_DEBATE = "magi.debate"
SPAN_PHASE_BLIND = "magi.phase.blind"
SPAN_PHASE_DELIBERATION = "magi.phase.deliberation"
SPAN_PHASE_VERDICT = "magi.phase.verdict"
SPAN_ROUND = "magi.round"
SPAN_TURN = "magi.turn"
SPAN_JUDGE = "magi.judge"
SPAN_TALLY = "magi.tally"
SPAN_STT = "magi.stt"
SPAN_TTS = "magi.tts"

# OTel GenAI semantic conventions. Spelled out rather than imported from the
# semconv package: that package is still unstable and renames keys between
# releases, which would silently split one metric into two in the backend.
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT = "gen_ai.usage.output_tokens"

# Domain attributes, namespaced so they cannot collide with the conventions.
ATTR_ENGINE = "magi.engine"
ATTR_DEBATE_ID = "magi.debate_id"
ATTR_ADVISOR = "magi.advisor"
ATTR_ROUND = "magi.round"
ATTR_ROUNDS_USED = "magi.rounds_used"
ATTR_OUTCOME = "magi.outcome"
ATTR_MODELS = "magi.models"
ATTR_TRACING_ENABLED = "magi.tracing_enabled"
# Which TerminationCondition fired — one of the TERMINATED_BY_* values below.
# One query over this answers how often debates converge rather than run out.
ATTR_TERMINATED_BY = "magi.terminated_by"
ATTR_LLM_CALLS = "magi.llm_calls"
ATTR_ADVISORS_PRESENT = "magi.advisors_present"
ATTR_JUDGED_COSMETIC = "magi.judged_cosmetic"
ATTR_SELECTOR_CALLS = "magi.selector_calls"
#: Set when an advisor exhausted its token budget on reasoning and the call was
#: retried with more room. Worth querying: a model that trips this often is one
#: that does not belong on a latency-bounded interface.
ATTR_LENGTH_RETRY = "magi.length_retry"

# One retry, at double the budget. Bounded on purpose — an unbounded escalation
# would trade a visible failure for an invisible latency cliff, on a node whose
# whole point is that latency is measured.
LENGTH_RETRY_FACTOR = 2

# ── How a debate ended ───────────────────────────────────────────────────────
#
# The vocabulary of ATTR_TERMINATED_BY and DebateRecord.terminated_by. Fixed
# strings rather than a config knob, and enumerated here rather than written
# inline at the point of use, because they are the axis every cross-run query
# groups by — including the sibling repo's, which must produce the same words
# for the same events or the two benchmarks cannot be read side by side.
#
# TERMINATED_BY_BUDGET and TERMINATED_BY_TIMEOUT are deliberately distinct. Both
# mean "did not converge", but one describes a debate with more to say and the
# other one too slow to say it, and only the second is an argument for a faster
# model.
TERMINATED_BY_CONSENSUS = "consensus"
TERMINATED_BY_BUDGET = "budget"
TERMINATED_BY_TIMEOUT = "timeout"
TERMINATED_BY_BARGE_IN = "barge_in"
TERMINATED_BY_ERROR = "error"
#: Fewer than two advisors survived the blind round, so there was nothing to
#: deliberate: the group chat never ran at all.
TERMINATED_BY_INSUFFICIENT = "insufficient_advisors"
#: The stack fired but no latch claimed it — should not happen, and says so
#: rather than silently reporting one of the real reasons.
TERMINATED_BY_UNKNOWN = "unknown"

# The call counter's bucket for SelectorGroupChat's turn-selection calls. Kept
# apart from the orchestrator's own calls even though they share a model: the
# difference between the two engines IS this bucket, so folding it into any
# other total would erase the measurement.
SELECTOR_LABEL = "SELECTOR"

# ── Pre-flight probe ─────────────────────────────────────────────────────────

# A model will happily satisfy the MagiTurn schema with position="Yes." — valid,
# and useless in a debate. Measured real advisors produce 77-249 characters with
# their actual persona prompts, so this threshold rejects the degenerate case
# without being anywhere near the working range.
MIN_POSITION_CHARS = 40

# How many times pre-flight will ask an advisor before declaring it unusable.
#
# One sample is the wrong gate. These models are measurably intermittent — a
# 33B reasoning model answered "Not advisable" (13 chars) on one boot having
# produced 311 characters on the previous one — so a single-shot check blocks a
# node that would have worked, and does it at the worst moment: startup.
#
# Retrying matches how a debate actually behaves. One terse turn among three
# rounds is survivable; the deliberation asks again and the tally copes. A
# model that cannot produce substance in THREE consecutive attempts is a
# different thing, and that is what this check is for.
PROBE_ATTEMPTS = 3

# Thinking models spend their budget on reasoning before emitting anything. At
# 300 tokens they return empty content and the reasoning in a separate field,
# which reads as a decoding failure and is really a budget failure.
PROBE_MAX_TOKENS = 900

# Used only when probing a model that has no persona attached (scripts/). The
# real pre-flight sends the advisor's own system prompt, because a thin prompt
# produces thin answers from every model and makes the probe useless.
GENERIC_ADVISOR_PROMPT = (
    "You are an advisor in a three-way debate. State your position on the "
    "question and the reasoning behind it, then summarise it in one sentence "
    "that could be read aloud. Use the required schema."
)
PROBE_QUESTION = "Should a three-person startup adopt Kubernetes for its first product?"

# ── Timeouts ─────────────────────────────────────────────────────────────────

# Supervision backoff ceiling. Long enough not to hammer a downed Spark, short
# enough that recovery is not noticeable to someone standing at the node.
MAX_BACKOFF_S = 30.0

# Pre-flight. The schema probe may trigger a cold model load, which on a 33B
# model is slow but bounded; the plain API calls should answer instantly or
# something is wrong.
SCHEMA_PROBE_TIMEOUT_S = 120.0
API_TIMEOUT_S = 10.0
COLLECTOR_PROBE_TIMEOUT_S = 3.0
