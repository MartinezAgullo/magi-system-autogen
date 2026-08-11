# magi-system-autogen

A three-model deliberation system running on a Raspberry Pi 5. The operator
asks a question by push-to-talk; three LLMs with different roles — **MELCHIOR**,
**BALTHASAR** and **CASPAR** — debate it until they converge; an orchestrator
called **MAGI** returns one spoken verdict.

The names are an Evangelion reference (Naoko Akagi's three-personality
supercomputer). The canon spelling is MELCHIOR / BALTHASAR / CASPAR — use those
exact uppercase forms everywhere in code, config, logs and UI.

The Pi is the **reasoning client, not the inference host**: STT runs locally
(`faster-whisper`), the debate loop runs locally, and only token generation
travels — HTTP to Ollama on the DGX Spark. No model weights on the node.

## This is the AutoGen half of a pair

There will be a sibling repository, `magi-system` (no framework, hand-rolled
orchestrator), implementing the **same debate protocol** so the two can be
compared on the same hardware. That comparison is a stated goal of the project.

This repo is therefore **AutoGen all the way down**. Where a choice exists
between doing something by hand and doing it through a framework construct,
use the framework construct — that is what is being evaluated. There is no
"native" fallback engine here; the baseline lives in the sibling repo.

> **WARNING — duplicated contract.** `config/magi.yaml` (personas, models) and
> the SQLite metrics schema in `store/debates.py` are **shared by agreement**
> with the sibling repo. If either diverges, the benchmark silently stops being
> a comparison. Any change to them must land in both repos in the same change
> window.

It also reuses the architecture proven in
[`latacc-edge`](/Users/pablo/Desktop/Scripts/latacc-edge) (single asyncio
process, in-process bus, `supervise()`, browser-side PTT, local STT, Piper TTS)
but shares **no code with it as a dependency** — modules are
copied and adapted. See [Relationship to sibling projects](#relationship-to-sibling-projects).

## Decisions already made

Everything in this section is settled. Do not re-litigate it; if a decision
turns out to be wrong, change it here first and then in the code.

| Topic | Decision |
|---|---|
| Multi-agent framework | **AutoGen** (`autogen-agentchat` v0.4+) — chosen deliberately over a hand-rolled loop, see [Choice of framework](#choice-of-framework) |
| Group chat type | **Both** `RoundRobinGroupChat` and `SelectorGroupChat`, selected by config, so the cost of LLM turn-selection can be measured |
| Baseline | The sibling repo `magi-system`. **Not** an engine inside this one |
| Debate protocol | Blind round 1, then up to 2 deliberation rounds with structured voting |
| Consensus | Expressed in AutoGen's own idioms: structured output + a custom `TerminationCondition` + `SocietyOfMindAgent`. See [Consensus, the AutoGen way](#consensus-the-autogen-way) |
| Outcome states | `UNANIMOUS` (3/3) / `MAJORITY` (2/3) / `DEADLOCK` (1-1-1, or no convergence within the round budget) |
| Roles | Three fixed archetypes defined in `config/magi.yaml`, editable without touching code |
| Models | One model per MAGI, rotatable via YAML. A pre-flight check verifies Ollama is actually serving all three |
| Language | **English everywhere** — voice, UI, prompts, code, comments, logs, docs |
| TTS | Final verdict only, condensed to 2-3 sentences. The debate itself is read on screen |
| STT | Local `faster-whisper`, one phrase per PTT press, composed into a draft. **A press never starts a debate** — see [Speech](#speech-ptt-draft-send) |
| Tools | None for now. The tool registry exists and is empty, so search/RAG can be added without a refactor |
| Persistence | SQLite: every debate (question, turns, votes, verdict, metrics). Each debate starts clean — no conversational carry-over |
| Observability | **OpenTelemetry everywhere**, using AutoGen's own `trace_provider` hook, following the OTel GenAI semantic conventions. See [Observability](#observability-opentelemetry) |
| Instrumentation | Full: per-turn latency, token counts, LLM call count, process CPU/RSS sampling |
| Infrastructure | **No MQTT, no broker.** The node talks HTTP to Ollama and nothing else |
| Round budget | Max 3 rounds, global timeout configurable. Cuts early on a unanimous vote |
| Dev backend | Spark's Ollama over WiFi by default; an OpenAI-compatible API-key provider as fallback when the Spark is off |

## Choice of framework

**AutoGen is not the technically optimal choice here, and that is on purpose.**
This must be stated plainly in `README.md` under a "Choice of framework"
heading — it is a stated project goal, not an embarrassment to hide.

Why AutoGen anyway:

- Doing something other than yet another framework-less agent loop.
- Measuring what a framework runtime actually costs on a Raspberry Pi 5:
  CPU, resident memory, and added latency per round.
- `SelectorGroupChat` spends an **extra LLM call per turn** just to choose the
  speaker. Quantifying that on-device is one of the deliverables.

Why a hand-rolled orchestrator would be the correct architectural call for a
production edge node — also to be stated in the README:

- It is an **edge device with voice input**, where every millisecond of
  response time is felt by the user, and the framework adds latency the task
  does not need.
- It needs **2/3 degradation**: if one model is unreachable, the debate should
  continue with the other two. A framework's group chat abstraction fights that.
- It needs **fast cancellation** (barge-in: a new PTT press aborts the running
  debate), which is harder to guarantee through someone else's asyncio runtime.

The sibling repo exists precisely so this claim is measured rather than asserted.

### Corollary: use the framework, do not fight it

Since the whole point is to evaluate AutoGen, the implementation must reach for
AutoGen's constructs first. A design that routes around the framework produces
a benchmark of nothing. Concretely, prefer:

| Need | AutoGen construct — not a hand-rolled equivalent |
|---|---|
| Vote as data | `AssistantAgent(output_content_type=MagiTurn)` -> `StructuredMessage[MagiTurn]` |
| Stop on consensus | Custom `TerminationCondition` subclass |
| Round budget | `MaxMessageTermination` |
| Global time budget | `TimeoutTermination` |
| Barge-in / cancel | `ExternalTermination().set()` + `CancellationToken` |
| Composing stop rules | `cond_a \| cond_b \| cond_c` |
| One verdict from a team | `SocietyOfMindAgent` wrapping the team |
| Deterministic turn order | `RoundRobinGroupChat` |
| Constrained LLM turn choice | `SelectorGroupChat(selector_func=..., candidate_func=...)` |
| Tracing the internals | Register the provider **globally**; do NOT pass `runtime=` (see below) |

The one place where working around the framework is mandatory is the blind
round — see below. It is an exception with a reason, not a precedent.

### What it cost in practice (AutoGen 0.7.5)

Measured while building the first engine. Kept here because a claim about
framework overhead that is never checked against the framework is just an
opinion.

| Expectation | What happened |
|---|---|
| Barge-in is hard through someone else's runtime | **Wrong** — `ExternalTermination().set()` covers it directly |
| `SocietyOfMindAgent` produces the verdict | **Wrong** — it takes no `output_content_type`, so it cannot return typed fields. Replaced by a plain `AssistantAgent` |
| Structured output just works | **Partly** — a group chat rejects `StructuredMessage[MagiTurn]` unless declared in `custom_message_types`, and fails at run time from inside a container |
| The condition that fired can be read afterwards | **Wrong** — `OrTerminationCondition` resets its children on completion. Sample during the stream |
| A message budget is a message budget | **Careful** — `MaxMessageTermination` counts the seed task, so the naive number cuts the last round before its final speaker. Fixed order makes that the *same* advisor every debate: a bias, not a truncation |
| 2/3 degradation fights the abstraction | **Right, and worse than expected** — a group chat has no concept of a participant dropping out; any participant raising takes the whole run down. Best available answer is to catch it and tally what was collected |
| A reasoning advisor needs a big enough token budget | **No fixed budget is enough** — nemotron3 burned its whole 4000-token allowance on one question having finished comfortably on the previous one. Because the group chat cannot lose a participant, that killed the debate. Mitigated with one retry at double the budget in `InstrumentedChatClient` |
| Inject a tracer provider via `runtime=` | **Trap, twice over** — `run_stream` only calls `runtime.start()` for the runtime it built itself, so an injected one hangs the debate forever; and the embedded one is constructed with `ignore_unhandled_exceptions=False` on purpose, which the natural hand-rolled construction silently reverses. Unnecessary anyway: AutoGen resolves `tracer_provider or get_tracer_provider() or NoOp`, so registering globally is enough |

### Mode collapse is the real enemy, and it is not AutoGen's fault

With three agents on one shared thread, the debate collapsed: after a genuinely
varied blind round, the second and third speakers reproduced the first one's
answer **verbatim**. Three identical sentences, which a naive reading scores as
an unusually strong consensus. Telling them not to copy changed nothing.

What fixed it was the **order of the task**: critique the others first, state
your own position second, and require it to contain something nobody else said.
See `prompts.deliberation_seed` — that ordering is load-bearing, not stylistic,
and anyone editing it should re-check that debates still reach MAJORITY and
DEADLOCK on contested questions rather than sliding back to unanimity.

Two structural notes that fall out of the same constraint:

- The group chat gives every participant the **same** seed message; there is no
  way to hand each advisor a personalised task. Anything advisor-specific has to
  live in its system prompt.
- `agrees_with` regularly contains the advisor's own name. `consensus.py` strips
  self-references rather than trusting the model, and requires **mutual**
  agreement for the same reason.

## Architecture

Single asyncio process (`magi`) supervised by systemd, with components
connected through an in-process pub/sub bus. A browser (kiosk on the 7" display,
or any client on the LAN) captures PTT audio and renders the debate.

```
mic (PTT) ──> browser ──> ui/server.py (WS /ui/stream) ──> stt.py (local whisper)
                                                                │
                                                           question
                                                                │
                              orchestrator/magi.py  (SocietyOfMindAgent "MAGI")
                                                                │
   Phase A — blind round: asyncio.gather(agent.run(...)) x3, outside any team
                                                                │
   Phase B — RoundRobinGroupChat | SelectorGroupChat, seeded with the 3 answers
             termination = ConsensusTermination | MaxMessage | Timeout | External
                                                                │
   Phase C — SocietyOfMindAgent emits StructuredMessage[MagiVerdict]
                                                                │
                    MELCHIOR / BALTHASAR / CASPAR ──> Ollama on Spark (:11434)
                                       │
                          verdict (UNANIMOUS|MAJORITY|DEADLOCK)
                                       │
              ┌────────────────────────┼────────────────────────┐
        store/debates.py         tts.py (Piper)          ui/server.py
        (SQLite + metrics)     final verdict only        /ui/stream ──> browser
```

### Debate protocol

Three phases.

**Phase A — blind round.** Each MAGI answers the question independently, with
no visibility of the others. **This is the one thing that cannot be done inside
a group chat**: in `RoundRobinGroupChat` every agent shares one message thread,
so the second speaker would see the first one's answer and anchor to it. Phase A
therefore calls each `AssistantAgent` directly — `asyncio.gather` over
`agent.run(task=question)` — which is still AutoGen's own single-agent API, just
not a team.

Anchoring is the single biggest threat to the value of this system: three models
that converge because they read each other are not a consensus, they are an echo.

Phase B uses **fresh agent instances** (or `on_reset()`), with the three Phase A
positions passed in as the team's task. Reusing the Phase A agents directly
would put each position in its author's context twice.

**Phase B — deliberation.** The group chat runs until the termination stack
fires. Up to `MAGI_MAX_ROUNDS - 1` rounds of mutual critique in which any MAGI
may revise its position.

**Phase C — verdict.** The `SocietyOfMindAgent` named MAGI produces the single
answer, as a `StructuredMessage[MagiVerdict]`.

### Consensus, the AutoGen way

The decision is unchanged — deterministic tally of structured votes, LLM judge
only when the vote is split. What changed is that it is expressed entirely in
framework constructs instead of a bespoke phase.

**Every turn is a vote.** Each MAGI is an `AssistantAgent` with
`output_content_type=MagiTurn`, so it cannot emit free prose:

```python
class Critique(BaseModel):
    advisor: str             # who is being criticised
    objection: str

class MagiTurn(BaseModel):
    position: str            # its stance this round
    summary: str             # one line, what it would say out loud
    agrees_with: list[str]   # subset of the other two MAGI names
    confidence: float        # 0.0 - 1.0
    critique: list[Critique] # empty in the blind round
```

The vote is not a separate phase; it is the shape of the message.

`critique` is a **list of objects, not a `dict[str, str]` keyed by name**. An
arbitrary-key mapping becomes an `additionalProperties` schema, which strict
JSON-schema modes reject outright and loose ones honour unevenly across models.
The same reasoning applies to anything added later: these models are prompt
surface handed to a 12B model as a schema, so keep them shallow and keep every
key fixed.

**Stopping is a `TerminationCondition`.** `ConsensusTermination` subclasses
`autogen_agentchat.base.TerminationCondition`, keeps the latest
`StructuredMessage[MagiTurn]` per source, and returns a `StopMessage` when the
tally is unanimous. `consensus.py` holds the pure tally function it calls, so
the rule stays testable without spinning up a team.

The full stack composes with `|`:

```python
termination = (
    ConsensusTermination(names)                      # unanimous -> stop early
    | MaxMessageTermination(len(names) * max_rounds) # round budget
    | TimeoutTermination(settings.debate_timeout_s)  # wall clock
    | external                                       # ExternalTermination, barge-in
)
```

`external.set()` is what a new PTT press calls. Barge-in is a framework feature
here, not something bolted on.

**The verdict is a `SocietyOfMindAgent`.** Named MAGI, wrapping the team, with
`output_content_type=MagiVerdict`:

```python
class MagiVerdict(BaseModel):
    outcome: Literal["UNANIMOUS", "MAJORITY", "DEADLOCK"]
    answer: str                    # 2-3 sentences — this is what gets spoken
    dissent: str | None            # the minority position, on MAJORITY
    disagreement_point: str | None # the precise fork, on DEADLOCK
```

Division of labour, and it matters: **`outcome` is computed deterministically**
by `consensus.py` from the final votes and passed to the agent as a constraint.
The agent writes the prose. It is not asked to count.

**The judge runs only on a split vote**, and answers one question: is the
disagreement substantive, or the same answer worded differently? Cosmetic
disagreement is promoted to `UNANIMOUS`; substantive disagreement stands. This
is `judge.py`, invoked by `consensus.py` before the outcome is finalised.

### Outcome rules

```
3/3 agree                       -> UNANIMOUS
2/3 agree                       -> MAJORITY   (dissent is reported, not hidden)
1-1-1, or no convergence by the -> DEADLOCK
round budget / timeout
```

On `DEADLOCK`, MAGI reports the three positions and the precise point of
disagreement. **It does not invent a synthesis.** Papering over a real
disagreement would destroy the only thing this system produces that a single
model cannot.

If a model is unreachable, the debate continues with the remaining two and the
verdict says so. Two-agent outcomes are `UNANIMOUS` (2/2) or `DEADLOCK` (1-1);
there is no majority with two voters.

### Engines

Two, selected by `MAGI_ENGINE`. Both are AutoGen; they differ only in the team
class:

| Value | Team |
|---|---|
| `autogen_roundrobin` | `RoundRobinGroupChat` — deterministic turn order, no speaker-selection LLM call |
| `autogen_selector` | `SelectorGroupChat` — an LLM picks the next speaker every turn |

Same agents, same prompts, same termination stack, same verdict agent. The only
variable is turn selection, which is the point: the delta between them is the
cost of AutoGen's headline feature.

`SelectorGroupChat` is configured **without** `selector_func`, deliberately —
a selector function would short-circuit the LLM call being measured.

### LLM client

`autogen-ext[openai]` with `OpenAIChatCompletionClient` pointed at Ollama's
OpenAI-compatible endpoint (`http://{SPARK_HOST}:11434/v1`), **not**
`autogen-ext[ollama]`.

One dependency instead of two, and — more importantly — the exact same client
class works against the Spark and against a real API-key provider by changing
`base_url` and `api_key`. Development happens on a MacBook that runs no local
models, so this is a hard requirement, not a convenience.

Structured output (`output_content_type`) needs the model to honour JSON schema
constraints. Not every Ollama tag does this equally well — if a model produces
malformed turns, that is a finding about the model, and it belongs in the
results, not in a workaround that abandons structured output.

### Observability: OpenTelemetry

**Everything is traced.** OTel is the primary observability layer, and this is
another place the framework earns its keep: `autogen-core`'s
`SingleThreadedAgentRuntime` accepts a `trace_provider`, so AutoGen emits spans
for agent messaging and model calls without the code asking it to. Pass the
provider through when constructing the runtime the teams run on.

Layers, all exporting to the same provider:

| Layer | Source |
|---|---|
| AutoGen internals | `SingleThreadedAgentRuntime(trace_provider=...)` — agent messages, model calls, including the selector's |
| HTTP to Ollama | `opentelemetry-instrumentation-httpx` |
| UI and PTT | `opentelemetry-instrumentation-fastapi` |
| Phases and domain events | Manual spans in `orchestrator/` — one root span per debate, child spans per phase, per round, per turn, plus `stt`, `tts`, `judge`, `consensus_tally` |

Span naming and attributes follow the **OTel GenAI semantic conventions**
(`gen_ai.system`, `gen_ai.request.model`, `gen_ai.operation.name`,
`gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`). Not optional
tidiness: conventions are what let the sibling repo's traces sit in the same
backend and be compared without a translation layer.

Domain attributes on the debate root span: `magi.engine`, `magi.question_id`,
`magi.rounds_used`, `magi.outcome`, `magi.models` (the loaded set),
`magi.terminated_by` (which `TerminationCondition` fired — consensus, max
messages, timeout or barge-in). That last one answers, in one query, how often
debates actually converge versus how often they run out of budget.

Exporters, by config:

- **OTLP** (`MAGI_OTEL_ENDPOINT`) to a collector — Jaeger or Tempo on the Spark
  or the MacBook. The normal path.
- **File** (JSON lines) for a Pi running with no collector reachable, so field
  runs are not lost.
- **Console** for development.

#### Traces and SQLite are not the same job

They coexist deliberately and must not drift into duplicating each other:

- **OTel** is for drill-down: why was *this* debate slow, where did the time go,
  what did the selector cost on turn four.
- **SQLite** (`store/debates.py`) is the durable benchmark record — one row per
  debate with the aggregate numbers, plus the full transcript. It is a **shared
  schema contract** with the sibling repo and must survive with no collector
  present.

Aggregates are computed once in the orchestrator and written to both. Do not
derive the SQLite row by querying a tracing backend.

#### Tracing the thing that measures tracing

OTel instrumentation itself costs CPU and memory, on a node whose CPU and
memory consumption is the object of study. Therefore:

- `MAGI_OTEL_ENABLED` must genuinely disable it — no exporter, no provider,
  no-op spans.
- Every benchmark row records whether tracing was on
  (`magi.tracing_enabled`). A comparison across repos with tracing on in one
  and off in the other is invalid.
- The headline overhead numbers are taken with tracing **off**, in both repos.
  Traces explain the behaviour; they do not produce the published figure.
- Use `BatchSpanProcessor`, never `SimpleSpanProcessor`. Synchronous export on
  a Pi would put network I/O on the critical path of a voice interface.

### Metrics

`services/metrics.py`. Recorded per turn and per debate, into SQLite:

- wall-clock latency per turn and per phase
- prompt / completion tokens
- **number of LLM calls**, which is where `SelectorGroupChat` shows its cost
- process CPU% and RSS, sampled on an interval for the duration of the debate
- the engine, the model set, and whether tracing was enabled — metrics without
  those are uninterpretable

**Count calls at the client, not at the message.** `models_usage` on chat
messages misses the selector's own calls, which are internal to
`SelectorGroupChat`. Wrap `OpenAIChatCompletionClient` in a counting proxy so
every call is counted regardless of who makes it. This is the difference
between measuring the thing and guessing at it. The same wrapper is the natural
place to emit the GenAI spans, so tracing and counting share one seam.

Comparisons this enables:

- `autogen_roundrobin` vs `autogen_selector` — cost of automatic turn-taking
  (within this repo).
- this repo vs the sibling `magi-system` — cost of the runtime itself
  (across repos, which is why the metrics schema is a shared contract).

### Supervision pattern

Copied from `latacc-edge`: every long-running service runs inside
`supervise(name, factory)` — infinite loop, exponential backoff (1s -> 30s) on
exception, clean propagation of `asyncio.CancelledError`.

### Audio capture (browser, not ALSA)

PTT audio is captured **by the browser** with `MediaRecorder` and sent as a
binary frame over the existing `/ui/stream` WebSocket; `ui/server.py` hands it
to `stt.py`. The "microphone" is whatever input the kiosk browser uses (the
Bluetooth headset). There is no ALSA capture path in the daemon.

A new PTT press while a debate is running is **barge-in**: it calls
`external.set()` on the running debate's `ExternalTermination` and starts a new
one. Time-to-abort is worth measuring; it is one of the concrete places a
framework can disappoint on an edge device.

## Speech: PTT, draft, SEND

Speech recognition runs **on the node** (`services/stt.py`, `faster-whisper`).
Audio never leaves the Pi; only the transcript does, and only as far as the
draft. That is the same split the rest of the system follows.

**A PTT press does not start a debate.** It transcribes one phrase and appends
it to a draft (`services/draft.py`) as a numbered line the operator can read
back and delete. Only SEND commits. The reason is arithmetic: recognition gets
things wrong, and a debate costs two to four minutes of three models' time — so
firing per press means a mis-heard word buys a full deliberation on a question
nobody asked. This is the same draft/SEND shape as latacc-edge's report
composition, for a different reason: there it is because one radio report is one
message, here it is because a debate is expensive and speech is unreliable.

**Audio rides the WebSocket that is already open**, as a binary frame, rather
than a POST per press. No handshake per sentence, and a failure surfaces on the
same LINK indicator the operator is already watching. `POST /stt` exists for
scripts and mic-less machines; the console does not use it.

**Every outcome gets its own message**, because the operator's next move differs:

| Result | What the console says | What it means |
|---|---|---|
| text | the line appears | — |
| `""` | "nothing was heard — hold the button while you speak" | released too early |
| hallucination | "no clear speech in that recording" | the recording was bad |
| exception | the error | the node is broken |

Collapsing these into "nothing appeared" would leave someone pressing the button
again into the same wall.

**Whisper narrates silence if you let it.** `hallucination_silence_threshold`,
`no_speech_threshold`, `log_prob_threshold` and `condition_on_previous_text=False`
are set together for that reason, plus a filter for the handful of phrases the
model produces from noise ("thank you", "subscribe"). A hallucinated line in a
question is worse than a missing one — the operator may not reread it.

**Ids, not positions.** A line is deleted by id, because a press still being
transcribed can land between the operator reading a line and tapping its button.
A positional delete would then remove whatever arrived in that slot.

**The draft is not persisted**, unlike latacc-edge's. That one survives a reboot
because a half-dictated 9-Line is minutes of work in a field where the node
genuinely does reboot. A question is one or two sentences, and losing it costs
repeating them — not enough to justify a database between a button and a
sentence.

### The Pi is where speech gets expensive

Measured on the same three-second clips, tracing off:

| | MacBook M3 | Raspberry Pi 5 |
|---|---|---|
| model load | 12.7 s | 8.1 s |
| per phrase | 1.5-2.2 s | **6.6-7.1 s** |
| SoC temperature | — | **76.8 °C** while transcribing |

Three to four times slower, and hot enough to put the console into CONDITION
CAUTION — which is the telemetry earning its place rather than a problem in
itself.

Tolerable, because of the draft: the operator is waiting to *read back* a
phrase, not waiting for an answer, and they can keep speaking while the previous
press transcribes (the semaphore queues them). It would not be tolerable if a
press went straight to a debate.

This is the first hard number for the **AI HAT+ (Hailo-8)**, which the hardware
notes have carried as "an optimisation, not a requirement" since the start.
Seven seconds and 77 °C per phrase on the CPU is what moving Whisper to the
accelerator would buy back. Still not a requirement; no longer a guess.

**The microphone needs a secure context.** Over plain http only `localhost`
qualifies, so a browser opening the node by LAN address gets the console and no
PTT. The button says so rather than silently doing nothing, and `launch.sh`
prints the localhost URL for exactly this reason.

## Personas

Defined in `config/magi.yaml`, loaded by `personas.py`. **Names, models,
parameters and prompts are always externally configurable** — nothing in the
code may hardcode a model tag or a persona name, and `MAGI_PERSONAS_FILE` points
at whichever file a given run should use. The table below is the committed
starting point, not a fixture.

| Name | Archetype | Default model | Focus |
|---|---|---|---|
| **MELCHIOR** | The scientist | `nemotron3:33b` | First principles, evidence, correctness. States uncertainty explicitly rather than guessing |
| **BALTHASAR** | The pragmatist | `gemma3:12b` | Feasibility, cost, tradeoffs, what actually ships and what it takes to maintain it |
| **CASPAR** | The skeptic | `qwen3:14b` | Failure modes, unstated assumptions, second-order effects. Argues the opposing case on purpose |

This maps onto the canon's scientist / mother / woman without being twee: the
mother protects (pragmatism, sustainability), the woman doubts (skepticism).

Persona prompts are **data, not code**. Editing a role must never require a
code change, because comparing role formulations is part of the point — and
because a planned extension replaces the hand-written YAML with three prompts
generated per question by a preliminary LLM call (see "Possible extensions" in
`README.md`). `personas.py` must therefore return personas as a structure a
generator can produce just as easily as the YAML loader does.

```yaml
# config/magi.yaml — shared contract with the sibling repo
magi:
  - name: MELCHIOR
    archetype: scientist
    model: nemotron3:33b
    system_prompt: |
      ...
```

### Model set and Ollama configuration

What the Spark currently serves (2026-08-07):

```
gemma3:12b        qwen3:14b        nemotron3:33b     deepseek-r1:32b
qwen3-coder:30b   qwen3-vl:8b      bge-m3:latest
```

The default three are chosen for **lineage diversity**, which is the whole
premise: three models from the same family agreeing proves nothing. NVIDIA
(`nemotron3`), Google (`gemma3`) and Alibaba (`qwen3`) were trained by
different teams on different data, so their disagreements are informative.
Picking `qwen3:14b` and `qwen3-coder:30b` for two seats would quietly turn the
debate into an echo — the exact failure the blind round exists to prevent.

Notes on the rest:

- **`bge-m3`** is an embedding model, not a chat model. It cannot be a MAGI. It
  would only matter if consensus were ever measured by embedding similarity.
- **`qwen3-vl:8b`** is vision-first and same-family as `qwen3:14b`. No seat.
- **`qwen3-coder:30b`** is the right MELCHIOR *for software questions
  specifically*, which makes it a good first test case for the dynamic-role
  extension rather than a static default.
- **`deepseek-r1:32b`** is tempting for MELCHIOR — reasoning-tuned, fits the
  scientist. Two problems: it emits `<think>` blocks, which fight
  `output_content_type`'s strict JSON, and its latency is high enough to be felt
  on a voice interface. Treat it as a **variant to benchmark**, not a default.
  If it is used, `<think>` stripping must happen before schema validation, and
  its thinking tokens must be counted separately or the token comparison is
  meaningless.
- **`qwen3:14b`** has hybrid thinking. Pin it explicitly (`/no_think`, or the
  equivalent option) rather than leaving it to the model's default, or the same
  config will produce different latencies on different days.

### Model residency on the Spark

With three different models taking turns, Ollama may unload and reload weights
between turns. When it does, tens of seconds per turn disappear into disk I/O
and every latency number becomes a measurement of storage, not of AutoGen.

How real this is depends on the Spark's configuration and is **not assumed** —
the Spark is known to hold at least two models resident concurrently, and its
unified memory comfortably fits the ~60B parameters of the default three. The
constraint is configuration, not capacity:

```
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_KEEP_ALIVE=1h
```

**The Pi cannot read the Spark's environment variables**, so the pre-flight
check must not try to assert on them. It observes instead, via
`GET /api/ps` (the models currently loaded, with their expiry):

- Warn if fewer than the YAML's distinct models are resident, naming which are
  missing and quoting the two env vars above as the fix.
- Warn if any resident model's `expires_at` is soon relative to
  `MAGI_DEBATE_TIMEOUT_S` — it will be evicted mid-debate.

These are **warnings, not aborts**. A missing model tag is a hard error; a
suboptimal residency setup is a caveat on the numbers, and the operator may
have reasons. Whether the warning fired is recorded on the debate row, so a
run with a suspected reload can be excluded from the benchmark rather than
silently skewing it.

At runtime the same problem is detectable after the fact: a turn whose latency
is a large multiple of the same model's median is very likely a reload. Flag
those turns in the metrics rather than trying to prevent them.

### Measured model behaviour (2026-08-11, Ollama 0.23.2)

Structured output was the first thing verified, and it produced three findings
that shaped the config. Re-measure after any Ollama or model upgrade; these are
observations, not properties.

**Reasoning is not a free switch, and it is per model.** Suppressing it
(`reasoning_effort: "none"`, Ollama's OpenAI-compatible `think: false`) made
`gemma3:12b` go from 3/5 to 5/5 valid turns, and made `nemotron3:33b` produce
**0/5** — nothing valid at all. There is therefore deliberately **no global
`thinking` default** in `config/magi.yaml`; it is set per advisor. A single
switch would have silently broken MELCHIOR.

**A reasoning model needs a much larger budget than its answer suggests.**
`nemotron3:33b` emits ~3800 characters of reasoning *before* writing a word of
answer. At the 900-token default it returned empty content with the reasoning in
a separate field — which reads like a schema-decoding failure and is really a
budget failure. MELCHIOR runs at 2500. That makes it by far the slowest and most
expensive advisor; on a voice interface that is a real cost, and it belongs in
the benchmark rather than hidden.

**Schema-valid is not the same as usable, and this is the important one.** Every
model will happily satisfy the schema with `position: "No."` — valid, and unable
to hold a debate. Under a thin probe prompt the measured positions were 7-11
characters; under the real persona prompts, 77-249. Two consequences:

- The pre-flight probe sends the **advisor's own system prompt** and asserts a
  minimum `position` length (`MIN_POSITION_CHARS`), not just schema validity. It
  probes per advisor, not per model tag, because the same model behaves
  differently under a different prompt and thinking setting.
- Any future evaluation of a model must use the real prompts. A thin prompt
  makes every model look incapable and tells you nothing.

The first version of this check validated the schema only, and passed all three
models on a run where two of them were in fact returning nothing usable. A green
check is only worth what it actually asserts.

**And it retries.** The version after that gated startup on a single sample,
which is the opposite mistake: `nemotron3` answered "Not advisable" — 13
characters — on one boot having produced 311 on the previous one, and the node
refused to start over one unlucky draw. Pre-flight now asks up to
`PROBE_ATTEMPTS` times and takes the first usable answer, warning when an
advisor needed more than one. A model that cannot produce substance in three
consecutive tries still fails hard, which is what the check is actually for.

The general rule, learned twice here: **a gate on a non-deterministic system
needs more than one sample.** These models are measurably intermittent, so any
single-shot assertion about them is a coin toss dressed as a check.

## Configuration

`pydantic-settings`, env prefix `MAGI_`, `.env` at the repo root and a committed
`.env.example`.

| Setting | Purpose |
|---|---|
| `MAGI_OLLAMA_HOST` / `MAGI_OLLAMA_PORT` | Spark's Ollama (`192.168.18.52:11434` on the current LAN — DHCP, may change) |
| `MAGI_LLM_BACKEND` | `ollama` (default) or `openai` |
| `MAGI_OPENAI_BASE_URL` / `MAGI_OPENAI_API_KEY` | Fallback provider when the Spark is off |
| `MAGI_ENGINE` | `autogen_roundrobin` \| `autogen_selector` |
| `MAGI_MAX_ROUNDS` | Default 3 (one blind + two deliberation) |
| `MAGI_DEBATE_TIMEOUT_S` | Feeds `TimeoutTermination` |
| `MAGI_PERSONAS_FILE` | Default `config/magi.yaml` |
| `MAGI_STT_MODEL` / `MAGI_STT_PRELOAD` / `MAGI_STT_CPU_THREADS` | `faster-whisper`, `small`, `int8`, preloaded at boot by default |
| `MAGI_TTS_ENABLED` / `MAGI_TTS_VOICE` | Piper, English voice |
| `MAGI_DB_PATH` | Default `data/magi.db` |
| `MAGI_UI_HOST` / `MAGI_UI_PORT` | Default `0.0.0.0:8000` |
| `MAGI_METRICS_ENABLED` / `MAGI_METRICS_SAMPLE_INTERVAL_S` | Instrumentation |
| `MAGI_OTEL_ENABLED` | Master switch. Off means no provider and no-op spans, not a silent exporter |
| `MAGI_OTEL_EXPORTER` | `otlp` \| `file` \| `console` |
| `MAGI_OTEL_ENDPOINT` | OTLP collector (Jaeger/Tempo on the Spark or the MacBook) |
| `MAGI_OTEL_SERVICE_NAME` | Default `magi-autogen` — **must differ from the sibling repo's**, or traces from the two implementations merge into one service |
| `MAGI_FAKE_HW` | Disables Pi-specific services when developing on the MacBook |

Host settings stay independent (Ollama host, TTS, UI) so the Spark, the Pi and
the dev machine can move without touching each other.

## Pre-flight checks

`./launch.sh` runs these before starting the daemon, and `--check-only` runs
them alone. A missing model must **fail loudly at boot**, not surface as a
confusing debate failure three minutes in:

Hard errors abort; residency and observability problems only warn.

1. Ollama reachable at the configured host. **Abort.**
2. `GET /api/tags` lists **every** model named in `config/magi.yaml`. **Abort**
   with the names of the missing ones and the `ollama pull` command to fix it.
3. `GET /api/ps` — model residency. **Warn** (see
   [Model residency](#model-residency-on-the-spark)).
4. Structured output smoke test: one tiny `MagiTurn` request per model.
   **Abort** — a model that cannot honour the schema breaks every debate, and
   finding out at boot is worth the few seconds.
5. Piper voice file present, if TTS is enabled. **Abort.**
6. Whisper model cached, or preload will fetch it. **Warn.**
7. `MAGI_DB_PATH` directory writable. **Abort.**
8. OTLP collector reachable, if `MAGI_OTEL_EXPORTER=otlp`. **Warn** — never let
   a missing collector stop the node from answering questions.

## Layout

**Single package, flat at the repo root.** Not a uv workspace: workspaces exist
for multi-package repos (LATACC has one per agent), and MAGI has exactly one
deliverable — `packages/*/src/` would be three directory levels earning nothing.
The shape follows the house template (package at root, `tests/`, `scripts/`,
`docker/`, a `config.py`/`constants.py` split, a `setup/` subpackage).

The import package is **`magi`**, not `magi_autogen`, deliberately: the sibling
no-framework repo uses the same name, so the files the two share by contract sit
at identical paths and can be diffed directly.

```
magi-system-autogen/
├── magi/
│   ├── __main__.py             # `python -m magi` -> main.main()
│   ├── main.py                 # boot: config, personas, bus, supervised services
│   ├── preflight.py            # `magi-preflight`, exit 0 / 1 / 2
│   ├── config.py               # pydantic-settings — anything an operator may change
│   ├── constants.py            # protocol invariants: bus topics, span names, budgets
│   ├── models.py               # MagiTurn, MagiVerdict, Outcome
│   ├── personas.py             # loads config/magi.yaml
│   ├── bus.py                  # in-process pub/sub (copied from latacc-edge)
│   ├── supervision.py          # supervise() (copied from latacc-edge)
│   ├── setup/
│   │   ├── setup_logs.py
│   │   └── setup_tracing.py    # TODO — lands with the OTel work
│   ├── orchestrator/           # TODO
│   │   ├── magi.py             # drives phases A/B/C, owns SocietyOfMindAgent
│   │   ├── teams.py            # RoundRobinGroupChat | SelectorGroupChat
│   │   ├── termination.py      # ConsensusTermination + the composed stack
│   │   ├── consensus.py        # pure tally -> UNANIMOUS|MAJORITY|DEADLOCK
│   │   ├── judge.py            # LLM tie-break, only on a split vote
│   │   └── prompts/
│   ├── services/
│   │   ├── ollama_check.py     # pre-flight: /api/tags, /api/ps, schema probe
│   │   ├── stt.py              # TODO — faster-whisper (copied, adapted)
│   │   ├── tts.py              # TODO — Piper (copied, adapted)
│   │   └── metrics.py          # TODO — counting client proxy, CPU/RSS sampler
│   ├── store/
│   │   └── debates.py          # TODO — SQLite, SHARED SCHEMA CONTRACT
│   ├── tools/
│   │   └── registry.py         # TODO — empty by design, wired for search/RAG
│   └── ui/                     # TODO
│       ├── server.py           # FastAPI + WS /ui/stream, PTT ingest
│       └── resources/
├── config/magi.yaml            # personas and models — SHARED CONTRACT
├── tests/
├── scripts/probe_models.py     # schema compliance across every model on the host
├── docker/                     # trace backend ONLY — see below
├── docs/
├── pyproject.toml
├── launch.sh                   # pre-flight, then the node
├── setup_env.sh                # fresh clone -> working environment
└── .env.example
```

### Two things this layout deliberately does not have

**No `requirements.txt`.** `pyproject.toml` plus `uv.lock` are the single source
of truth. A second dependency list would drift, and the drift would be
discovered on the Pi.

**No Dockerfile for MAGI itself.** The node runs under systemd on a Pi and needs
the browser's microphone, a Piper binary and the host audio device.
Containerising it would cost the two numbers the project exists to measure:
`psutil` inside a container reports the container's view of CPU and memory, not
the node's. `docker/` holds the OTLP trace backend only — the one piece that
genuinely benefits, and the one pre-flight warns about.

## Conventions

- Python 3.11+, `uv` (no workspace), single flat package `magi/`.
- **`config.py` vs `constants.py`**: anything an operator might reasonably want
  to change per node goes in `config.py` (hosts, ports, budgets, flags).
  Everything else — bus topics, span and attribute names, protocol invariants —
  goes in `constants.py`. Making the latter configurable would only create ways
  for two nodes to disagree about what a debate is.
- PEP 8, full type hints, **English-only code, comments, docstrings, logs and
  UI text**.
- All blocking work (whisper, Piper, any sync SDK call) goes through
  `asyncio.to_thread`. Never block the event loop.
- **Speech is an optional extra, not a core dependency.** `uv sync --extra dev`
  is enough to hold and benchmark debates; `--extra voice` adds
  `faster-whisper` and `piper-tts`. `faster-whisper` pulls `ctranslate2`, which
  on ARM is the longest install in the tree, and nothing imports it until the
  node actually speaks. Anything under `services/stt.py` or `services/tts.py`
  must therefore import lazily, inside the function that needs it — a
  module-level import would make the whole daemon unstartable without the extra.
  The model preloads at startup by default (`MAGI_STT_PRELOAD=1`) when it is
  installed.
- Persona prompts, names and model tags live in YAML, never hardcoded.
- Spans follow the OTel GenAI semantic conventions; domain attributes are
  namespaced `magi.*`. Never put a question or a model answer in a span
  attribute — transcripts belong in SQLite, and spans get shipped to backends
  that were never sized or secured for them.
- Reach for an AutoGen construct before writing a bespoke one — see
  [the corollary](#corollary-use-the-framework-do-not-fight-it).
- `consensus.py` stays a pure function over votes, with no AutoGen imports, so
  the sibling repo can hold a byte-identical copy.
- Any protocol change lands in both engines here **and** in the sibling repo in
  the same change window, otherwise the benchmark silently stops being a
  comparison.

## Relationship to sibling projects

- **`magi-system`** (to be created): same protocol, no framework. The baseline.
  Shares `config/magi.yaml`, the metrics schema and `consensus.py` **by copy and
  agreement**, never by import.
- **`latacc-edge`**: source of the reusable edge infrastructure. Modules are
  **copied and adapted**; no shared package, no import across repos. The two
  have different lifecycles and coupling them would make every change to one a
  risk to the other — the same boundary already drawn between `latacc-edge` and
  the LATACC server.

Worth copying from `latacc-edge`: `bus.py`, `supervise()`, the `config.py`
pattern, `stt.py`, `tts.py`, the `ui/server.py` + `/ui/stream` PTT ingest
pattern, `launch.sh`.

Explicitly **not** carried over: MQTT and everything around it (`mqtt.py`,
`subscribers/`, `publishers/`, `outbox.py`), the clinical downlink, telemetry,
camera. MAGI is a self-contained node whose only external dependency is Ollama.

## Hardware

| Device | Role |
|---|---|
| Raspberry Pi 5 (16 GB) | The MAGI node — daemon, STT, TTS, UI |
| 7" touch display | Kiosk, `chromium-browser --kiosk http://localhost:8000` |
| Bluetooth microphone (AirPods) | PTT input, via the browser |
| NVIDIA DGX Spark | Ollama inference host (`:11434`), three models loaded |
| MacBook Pro M3 | Development (`MAGI_FAKE_HW=1`, points at the Spark or an API key) |
| AI HAT+ (Hailo 8) | Present, unused. Moving STT to it is an optimisation, not a requirement |

Pi access: `ssh agullo@rpi5-01.local`

## Front-end

An Evangelion console: amber on black, hexagonal chrome, the three advisors as
chamfered panels framing MAGI. `magi/ui/server.py` plus three static files in
`resources/` — no build step, no CDN, no external fonts. A Pi that needs npm to
draw its own screen cannot be fixed in the field.

Rules the design holds to:

- **One WebSocket.** Every client gets the same `/ui/stream`. Two sockets would
  mean two reconnect policies and two answers to "are we connected?", in a UI
  whose panels exist to answer that.
- **Colour means state, never decoration.** Amber at rest, cyan once an advisor
  has spoken, green inside the agreeing bloc, red dissenting or in trouble.
  Every state also changes its text, so the console survives daylight and a
  colour-blind operator.
- **The kanji is chrome; the content is English.** 承認 always sits beside IN
  CONSENSUS. The project's one-language rule holds for everything load-bearing.
- **No binary vote.** The anime's nodes answer 承認/否定 to a yes-or-no question.
  Ours hold positions with nuance, so forcing a binary would misreport what the
  system does. What is shown instead is each advisor's declared bloc — which is
  real, and gives the same colour-change drama honestly.
- **The status flags are wired to something.** CONDITION GREEN and the
  temperature come from `services/telemetry.py` reading the Pi's thermal zone
  and throttle bits. A hardcoded green flag would be set dressing.
- **PTT is present and honestly disabled** until speech is installed; `/stt`
  answers 503, not 404, so the SPA can tell "not installed" from "broken".
- **A fullscreen toggle in the status row.** The kiosk is already fullscreen via
  `chromium --kiosk`, so this is for every other way the console gets opened —
  a laptop on the LAN, a tablet, the Pi's own browser started by hand — where
  browser chrome costs about a third of the triad on an 800x480 panel. Its label
  follows `fullscreenchange`, not the last click, because Esc leaves fullscreen
  without going through the button; and it is hidden outright where the API does
  not exist, per the rule that the display never offers a control it cannot
  honour.

Layout at 800x480 is **phase-driven**, because the triad and a full verdict
panel cannot both have the room they want: while the advisors deliberate the
triad owns the screen and shows their forming positions; once the verdict lands
it repeats those positions in full, so the per-node summaries hide and yield
their space to it.

## Status

**`autogen_roundrobin` works end to end.** A question goes in, three advisors
debate it, a tallied outcome and a spoken-length verdict come out. Verified
against the Spark on 2026-08-11: UNANIMOUS, MAJORITY and DEADLOCK all reached on
real questions, 48-155 s per debate.

Built: config, personas, bus, supervision, typed models, pre-flight, and
`orchestrator/` (clients, termination, consensus, judge, prompts, phases A/B/C),
plus `scripts/ask.py` to run one debate without the UI.

**Tracing works.** A debate produces a single trace of ~130 spans: our phase
spans, AutoGen's internal agent-messaging and model-call spans (they join
automatically once the provider is registered globally), the GenAI `chat {model}`
spans from the instrumented client, and httpx spans underneath. Verified against
Jaeger on 2026-08-11.

Measured breakdown of one 132 s DEADLOCK: deliberation 92.9 s, blind round
29.5 s, verdict 6.4 s, judge 3.7 s, tally ~0. Inside the blind round the three
advisors run in parallel and MELCHIOR alone takes 29.4 s against 13-14 s for the
others — the reasoning model gates the whole phase.

**Runs on the Pi.** Debian 13, Python 3.13, `uv sync --extra dev`, both engines
verified, 69 tests passing on-device. Measured there over three runs per engine:
**1-3 % of one core and ~104 MB peak RSS, identical between engines** — the node
is blocked on HTTP to the Spark for the whole debate. See README § "The Pi
measurement, and a premise it demolishes": the framework's CPU/memory overhead,
which this project was partly built to measure, turns out not to exist in any
meaningful amount. Its real cost is extra LLM calls.

**Both engines work.** First measured comparison (same question, same models,
tracing off, MacBook -> Spark): roundrobin 147.4 s / 11 LLM calls, selector
212.5 s / 18 calls of which 7 were turn selection. **+44 % wall clock, +64 %
calls**, same DEADLOCK outcome. The Pi's numbers are the ones that count and are
not taken yet.

**The console works.** `magi/ui/` serves an Evangelion-styled SPA at `:8000` —
the voting triad (BALTHASAR top, CASPAR left, MELCHIOR right, MAGI at the
centre), live turns over one WebSocket, node colour by agreement bloc, and
status flags driven by the Pi's real SoC temperature. No build step, no external
fonts or scripts. Verified on the Pi.

**Speech works.** Hold PTT, speak one phrase, release: `faster-whisper`
transcribes it on the node and it appears as a numbered, deletable line in the
draft. Nothing is debated until SEND. Measured on the Mac: 12.7 s model load,
1.5-2.2 s per phrase, and silence correctly returns nothing rather than an
invented sentence. On the Pi: 8 s load, **6.6-7.1 s per phrase**, and the SoC
reaches 76.8 °C while transcribing.

Still to build: SQLite persistence and TTS (the verdict is read on screen, not
yet spoken).

Build order: ~~skeleton~~ -> ~~personas + pre-flight~~ ->
~~`autogen_roundrobin` end to end~~ -> ~~tracing~~ -> ~~`autogen_selector`~~ ->
**metrics + SQLite** -> STT/TTS wiring -> UI.

Tracing lands before the second engine on purpose: the first thing anyone will
want to know about `SelectorGroupChat` is where its time goes, and retrofitting
spans after both engines exist means instrumenting them twice.

`RoundRobinGroupChat` comes first because it has no LLM in the control path:
when the debate misbehaves, the cause is a prompt or a model, not the selector.
`SelectorGroupChat` is added once there is a working debate to compare it to.
