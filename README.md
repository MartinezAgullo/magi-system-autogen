# MAGI (AutoGen edition)

> Voice-first multi-agent deliberation on a Raspberry Pi 5: three LLMs with
> opposing roles debate a spoken question until they reach consensus. AutoGen
> implementation, fully traced with OpenTelemetry for edge benchmarking.

Three LLMs argue about your question until they agree. Then one of them tells
you the answer, out loud.

**MAGI** is a voice-first deliberation system running on a Raspberry Pi 5. You
hold a push-to-talk button and ask something. Three models with deliberately
different temperaments — **MELCHIOR** the scientist, **BALTHASAR** the
pragmatist, **CASPAR** the skeptic — answer independently, then read each
other's answers, criticise them, and vote. An orchestrator called **MAGI**
returns a single verdict:

```
UNANIMOUS   all three converged
MAJORITY    two agreed, the dissent is reported rather than hidden
DEADLOCK    no convergence — the three positions and the exact point of
            disagreement are returned, with no invented synthesis
```

The names are an Evangelion reference. The three-personality supercomputer is
the whole design idea, not just the branding.

The Pi is the reasoning client, not the inference host. Speech recognition runs
locally (`faster-whisper`), the debate loop runs locally, and only token
generation leaves the node — HTTP to Ollama on a DGX Spark. No model weights
ever touch the Pi.

## This repo is one half of an experiment

This is the **AutoGen** implementation. A sibling repository, `magi-system`,
will implement the identical protocol with a hand-rolled orchestrator and no
framework. Running both on the same Pi, against the same models, with the same
prompts, is how the cost of the framework gets measured instead of debated.

Consequently this repo commits to AutoGen fully: where a choice exists between
doing something by hand and doing it through a framework construct, it uses the
framework construct. An implementation that routes around AutoGen would
benchmark nothing.

## Status

**Debates work, end to end, fully traced.** Ask a question with
`scripts/ask.py` and three advisors argue it out; UNANIMOUS, MAJORITY and
DEADLOCK have all been reached on real questions, 48-160 s per debate. Every
run produces a single ~130-span trace.

**Voice-first, end to end.** Hold PTT, speak, release — the phrase is
transcribed on the node and appears as a deletable line; SEND commits the
question and the three advisors argue it out on the console. Both engines work,
verified on the Pi.

Still to build: the SQLite benchmark store, and TTS so the verdict is spoken as
well as shown. See `CLAUDE.md` § Status.

The load-bearing assumption is **verified**: as of 2026-08-11, on Ollama 0.23.2,
all three default advisors return a `MagiTurn` that is both schema-valid and
substantive over the OpenAI-compatible endpoint. Getting there needed two
per-model settings that are not obvious and are documented in
`config/magi.yaml`: reasoning must stay **on** for `nemotron3:33b` (it produces
nothing valid with it suppressed) and **off** for `gemma3:12b` (3/5 valid with
it on, 5/5 with it off), and a reasoning model needs roughly triple the token
budget because it spends it before writing any answer.

## Choice of framework

**This project uses Microsoft's AutoGen (`autogen-agentchat` v0.4+), and that is
not the technically optimal choice. It was picked on purpose, and it is worth
being honest about why.**

### Why AutoGen

Three reasons, none of them "it is the best tool for the job":

1. **To build something other than another framework-less agent loop.** A
   hand-rolled Think→Act→Reflect loop is a solved problem in this author's
   other projects. There is more to learn from using the thing everyone cites
   than from writing the same 200 lines again.
2. **To measure what a multi-agent runtime actually costs on a Raspberry Pi 5.**
   Framework overhead is usually discussed on laptops and cloud VMs. On a 16 GB
   ARM SBC, the CPU and resident-memory cost of AutoGen's own asyncio runtime
   and message plumbing is a real, measurable number that nobody publishes.
   Getting that number is a deliverable of this project.
3. **To quantify the price of `SelectorGroupChat`.** AutoGen's headline feature
   is automatic turn-taking: an LLM decides who speaks next. That is *an extra
   LLM call per turn*. In a three-model debate over three rounds, that is a
   meaningful fraction of total latency — and on a voice interface, latency is
   the experience. The project implements both `RoundRobinGroupChat`
   (deterministic, no selection call) and `SelectorGroupChat` behind a config
   flag so the difference can be measured rather than argued about.

### Why a hand-rolled orchestrator would be the right call

For a production edge node, the correct architectural decision is to write the
orchestration yourself. This is stated here so the tradeoff is on the record:

- **It is an edge device with voice input.** Every millisecond of response time
  is felt directly by the user. A framework adds latency for coordination the
  task does not actually need — three agents and a fixed protocol is not a
  problem that requires a group-chat abstraction.
- **It needs 2/3 degradation.** If one model becomes unreachable, the right
  behaviour is for the debate to continue with the remaining two and say so.
  Expressing that inside someone else's group chat abstraction is fighting the
  framework instead of using it.
- **It needs fast cancellation.** A new PTT press must abort a running debate
  immediately (barge-in). Guaranteeing prompt, clean cancellation is
  straightforward in your own asyncio loop and considerably less so through a
  third-party runtime.
- **The debate loop is genuinely small.** Three agents, a fixed round
  structure, a vote tally, a stopping condition — roughly 200 lines. The
  framework's dependency footprint is not repaid by 200 lines saved.

### How the claim gets tested

Everything is traced with **OpenTelemetry**. AutoGen instruments itself and
picks up whatever provider is registered globally, so its agent-messaging and
model-call spans join the trace without the node owning its runtime. On top sit
manual spans per debate and phase, one GenAI `chat {model}` span per LLM call
from the instrumented client, and auto-instrumented httpx underneath. Spans
follow the OTel GenAI semantic conventions so both repos' traces land in one
backend and compare directly.

A worked example — one 132 s DEADLOCK, straight off the root span:

```
magi.debate                132.6s
  magi.phase.deliberation   92.9s
  magi.phase.blind          29.5s   MELCHIOR 29.4s | CASPAR 14.2s | BALTHASAR 13.1s
  magi.phase.verdict         6.4s
  magi.judge                 3.7s
  magi.tally                 ~0
```

The blind round runs its three advisors in parallel, so it costs whatever the
slowest one costs: the reasoning model gates it at more than twice the others.
That is the kind of thing the design predicted qualitatively and the traces put
a number on.

Traces and the database do different jobs and do not duplicate each other:
OpenTelemetry answers *why was this debate slow*, SQLite holds the durable
benchmark row — per-turn latency, prompt/completion tokens, **number of LLM
calls**, sampled process CPU/RSS, the engine, and the exact model set.

Tracing is itself an observer effect: OTel costs CPU and memory on the node
whose CPU and memory are the object of study. So `MAGI_OTEL_ENABLED=0` truly
disables it, every row records whether tracing was on, and the headline
overhead figures are taken with it off in both repos. Traces explain the
behaviour; they do not produce the published number.

Two comparisons fall out of that:

```
MAGI_ENGINE=autogen_roundrobin   AutoGen, deterministic turns
MAGI_ENGINE=autogen_selector     AutoGen, LLM-selected turns
```

- `autogen_roundrobin` vs `autogen_selector` — the cost of automatic
  turn-taking, within this repo. **First measurement**, same question, same
  models, tracing off, MacBook driving the Spark:

  | | roundrobin | selector | delta |
  |---|---|---|---|
  | wall clock | 147.4 s | 212.5 s | **+44 %** |
  | LLM calls | 11 | 18 | **+64 %** |
  | of those, turn selection | 0 | 7 | — |

  Seven extra calls to answer "who speaks next", on a question both engines
  resolved to the same DEADLOCK. On a voice interface that is a minute of
  silence bought with nothing.

### The Pi measurement, and a premise it demolishes

Three runs per engine on the Raspberry Pi 5 (Debian 13, Python 3.13, tracing
off, same question, Ollama on the Spark):

| | roundrobin | selector |
|---|---|---|
| node CPU | 1.7-2.2 s | 1.8 s |
| as % of one core | **1-3 %** | **1 %** |
| peak RSS | 103-104 MB | 103-105 MB |
| LLM calls | 11 | 14-19 (5-8 turn selection) |
| wall clock | 54-253 s | 148-216 s |

**Reason 2 for choosing AutoGen was wrong.** The project set out to measure "the
CPU and resident-memory cost of AutoGen's own asyncio runtime on a 16 GB ARM
board". That cost is ~2 seconds of CPU across a four-minute debate and about
104 MB of RSS — and it is *identical* between the two engines. The node spends
the entire debate blocked on HTTP to a remote GPU. There is no meaningful
runtime overhead to find, because the workload is not CPU-bound and never was.

That is worth stating plainly rather than quietly dropping: the premise was
testable, it was tested, and it did not survive. What the framework actually
costs is **latency, in the form of extra LLM calls** — which the selector
comparison does measure, and which is the number that matters on a voice
interface.

Two caveats on the wall clock: it is dominated by the Spark and is noisy enough
(54-253 s for the same engine and question) that no conclusion should be drawn
from it without many more runs. And the ~104 MB is Python plus httpx plus
pydantic plus AutoGen — attributing a share of it to the framework needs the
sibling repo, which is exactly what the sibling repo is for.
- this repo vs the sibling `magi-system` — the cost of the runtime itself,
  across repos.

For that second comparison to mean anything, `config/magi.yaml`, the SQLite
metrics schema and the consensus tally function are a **shared contract**: both
repos hold identical copies, by agreement rather than by import.

LLM calls are counted at the model client, not at the message level, because
`SelectorGroupChat` makes its selection calls internally where per-message
usage data does not see them. Counting them anywhere else would understate
exactly the thing being measured.

## How the debate works

Three phases.

**Phase A — the blind round.** Each MAGI answers independently, with no
visibility of the others. This *cannot* be done inside a group chat: in
`RoundRobinGroupChat` all agents share one message thread, so the second
speaker sees the first one's answer and anchors to it. Phase A therefore calls
each agent directly and in parallel. Anchoring is the single biggest threat to
the value of the whole system — three models that agree because they read each
other are not a consensus, they are an echo. This is the one place the design
deliberately steps outside the framework's team abstraction.

**Phase B — deliberation.** The group chat is seeded with the three positions.
Up to two rounds of mutual critique, in which any MAGI may revise its position.

**Phase C — the verdict.** A `SocietyOfMindAgent` named MAGI turns the debate
into one answer, 2-3 sentences, which is what gets spoken.

Consensus is expressed in AutoGen's own idioms rather than as a bolted-on
phase: every turn is a `StructuredMessage` carrying a typed vote, a custom
`TerminationCondition` stops the debate the moment the vote is unanimous, and
the round budget, wall-clock budget and PTT barge-in are `MaxMessageTermination`,
`TimeoutTermination` and `ExternalTermination` composed with `|`.

The tally itself stays deterministic — the orchestrator counts votes in Python
and hands the outcome to the verdict agent as a constraint. An LLM judge is
consulted only when the vote is split, to decide one thing: whether the
disagreement is substantive or the same answer worded differently.

## What building it on AutoGen actually cost

Findings from getting the first engine working (AutoGen 0.7.5). This is the
evidence the "Choice of framework" argument above was written to be tested
against, so it is recorded whether or not it flatters the framework.

**Where AutoGen paid for itself.** Barge-in was the requirement expected to be
hardest through someone else's runtime, and `ExternalTermination().set()`
covers it directly. Composing four stop rules with `|` is genuinely nicer than
hand-written bookkeeping, and `output_content_type` turning every turn into a
typed vote removed a whole parsing layer.

**Where it charged rent:**

- `SocietyOfMindAgent` — the class the design was going to use for the verdict —
  takes no `output_content_type`. The one place the framework's highest-level
  abstraction fit, it could not return typed fields. Replaced with a plain
  `AssistantAgent`.
- A group chat rejects `StructuredMessage[MagiTurn]` unless it is declared in
  `custom_message_types`. A standalone agent needs no such thing, and the error
  surfaces from inside a message container at run time, not at construction.
- `OrTerminationCondition` resets its children when the run ends, so asking
  which condition fired *after* `run_stream` returns always answers "none". The
  reason has to be sampled during the stream.
- `MaxMessageTermination` counts the seed task as a message. Budgeting without
  the `+1` cuts the final round before its last speaker — and since the order is
  fixed, that is the same advisor every debate. A bias, not a truncation.
- Injecting a tracer provider through `runtime=` hangs the debate — `run_stream`
  only starts the runtime it created itself — and the runtime you would naturally
  construct silently flips `ignore_unhandled_exceptions` back on. None of it is
  needed: registering the provider globally is enough, and AutoGen's internal
  spans then join the trace on their own.
- **2/3 degradation is genuinely hard.** In the blind round, one model failing
  is contained because we own the `asyncio.gather`. Inside the group chat there
  is no notion of a participant dropping out: any participant raising takes the
  whole run down. The best available answer is to catch it and tally the turns
  collected so far. This was the concrete worry in the argument above, and it
  turned out to be real.

**The finding that had nothing to do with AutoGen, and mattered most.** With
three agents on one shared thread, the debate mode-collapsed: after a good blind
round, the second and third speakers reproduced the first one's answer *verbatim*
— three identical sentences that a naive reading scores as strong consensus.
Prompting them not to copy did nothing. What fixed it was reordering the task:
critique each other **first**, state your position **second**, and require it to
contain something nobody else said. Debates now reach real MAJORITY and DEADLOCK
outcomes with named dissent. Any multi-agent system on a shared thread should
expect this failure, and it is invisible in the output unless you look for it.

## Asking it something

Two ways in, and they end in the same place.

**By voice.** Hold the PTT button, say one phrase, release. `faster-whisper`
transcribes it **on the node** — audio never leaves the Pi — and it appears in
the draft as a numbered line. Say another. Delete the one Whisper mangled. When
the question reads right, press SEND.

```
草案 DRAFT                                    2 LINES   CLEAR ALL
01  Should a small team migrate their monolith to microservices?   DEL
02  Consider the operational cost over the next two years.         DEL
```

**A press never starts a debate**, and that is the point. Recognition gets
things wrong and a debate costs two to four minutes of three models' time, so
firing per press would buy a full deliberation on a question nobody asked. The
draft is the gap between what the machine heard and what it acts on.

**By typing.** The text box takes a question directly, for a laptop on the LAN
or a node with no microphone.

Requires `./setup_env.sh --voice`. Without it the PTT button renders present and
honestly disabled rather than hidden — the control is part of what this node is,
and pretending it works would be worse than showing it greyed. Same for a
browser on a LAN address: the microphone needs a secure context, so only
`localhost` gets PTT, and the button says so.

Measured on the same clips: **1.5-2.2 s per phrase on the MacBook, 6.6-7.1 s on
the Pi**, and silence returns nothing rather than an invented sentence.
Transcribing takes the Pi's SoC to 76.8 °C, which puts the console into
CONDITION CAUTION — the telemetry doing its job. That is tolerable only because
the draft means the operator is waiting to read a phrase back, not waiting for
an answer; it is also the first hard number for what moving Whisper to the
AI HAT+ would be worth.

## The console

`./launch.sh` serves it at `http://<node>:8000`. An Evangelion MAGI display:
the three advisors as chamfered panels framing MAGI at the centre, amber on
black, live turns streaming over one WebSocket, and status flags fed by the
Pi's real temperature and throttle bits.

Node colour is the honest analogue of the anime's 承認/否定 vote. The advisors
never cast a binary vote — they hold positions with nuance — but they do declare
who they stand with, so a node turns green inside the agreeing bloc and red
outside it. The drama of the colour change survives; the lie does not.

Kiosk: `chromium-browser --kiosk --noerrdialogs http://localhost:8000`

Opened any other way — a laptop on the LAN, the Pi's own browser by hand — the
`全画面 FULLSCREEN` button in the status row drops the browser chrome, which on
an 800x480 panel is about a third of the triad.

## Models

The Spark currently serves:

```
gemma3:12b        qwen3:14b        nemotron3:33b     deepseek-r1:32b
qwen3-coder:30b   qwen3-vl:8b      bge-m3:latest
```

Defaults, chosen for **lineage diversity** — three models from the same family
agreeing proves nothing:

| MAGI | Model | Lineage |
|---|---|---|
| MELCHIOR | `nemotron3:33b` | NVIDIA |
| BALTHASAR | `gemma3:12b` | Google |
| CASPAR | `qwen3:14b` | Alibaba |

This is a **starting point, not a fixture**. Names, models, parameters and
prompts all live in `config/magi.yaml`; nothing in the code hardcodes a tag,
and `MAGI_PERSONAS_FILE` selects a different set per run. Rotating models is
the intended way to use this thing.

Two practical notes that will otherwise waste an afternoon:

- **Watch for model residency.** The Spark holds multiple models in memory
  fine, but if Ollama is left to evict between turns, every latency number
  becomes a measurement of disk I/O rather than of AutoGen
  (`OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE`). The Pi cannot read the
  Spark's environment, so the pre-flight check observes `GET /api/ps` instead
  and **warns** — it does not abort, and it records that it warned, so a
  suspect run can be excluded from the benchmark rather than skewing it.
- **`deepseek-r1:32b` is a benchmark variant, not a default.** Its `<think>`
  blocks fight strict JSON output and its latency is felt on a voice interface.
  Interesting to measure, wrong to build on.

## Possible extensions

Ideas that are deliberately out of scope for the first version.

### Dynamic role assignment

Today the three roles are fixed archetypes written by hand in `config/magi.yaml`
— scientist, pragmatist, skeptic — and every question is debated by the same
three personalities. The more advanced version is to **generate the roles per
question**: one preliminary LLM call analyses the topic and produces three
tailored system prompts, choosing the three perspectives that would genuinely
disagree most productively about *this* subject.

A question about a database migration might convene a distributed-systems
engineer, an on-call SRE and a cost-conscious CTO. A question about a medical
protocol would convene something else entirely. The static archetypes are
generic by necessity; generated ones can be specific. The same call could pick
the models, too — `qwen3-coder:30b` earns a seat on a software question and
none at all on anything else.

The tradeoffs, for whenever this gets built:

- One extra LLM call before every debate — on a voice interface, that is
  latency the user feels before anything happens.
- Reproducibility drops. Comparing engines and models requires the prompts to
  be held constant, so the benchmark work needs the static path to stay
  available.
- Prompt-generation quality becomes a new failure mode: three generated roles
  that are secretly the same role produce a fake consensus, which is worse than
  no consensus at all.

It fits the existing design cleanly — `personas.py` already loads personas as
data, so a generator only has to produce the same structure the YAML does.

### Others

- **Tools.** The tool registry is wired and empty by design. Web search would
  make debates on factual topics far more useful, at the cost of latency and an
  internet dependency on the node.
- **Embedding-based consensus.** `bge-m3` is already on the Spark. Comparing
  the three positions by vector similarity would be an objective third opinion
  alongside the self-reported vote and the LLM judge.
- **STT on the AI HAT+.** The Hailo-8 is installed and unused; Whisper runs on
  the Pi's CPU. An optimisation, not a requirement.
- **Conversational memory.** Each debate currently starts clean. Follow-up
  questions would feel more natural by voice, at the cost of a growing context
  that contaminates model comparisons.

## Hardware

| Device | Role |
|---|---|
| Raspberry Pi 5 (16 GB) | The MAGI node — daemon, STT, TTS, UI |
| 7" touch display | Kiosk |
| Bluetooth microphone | PTT input, captured by the browser |
| NVIDIA DGX Spark | Ollama inference host, three models resident |
| MacBook Pro M3 | Development |

## Getting started

```bash
./setup_env.sh              # core only — enough to hold and benchmark debates
./setup_env.sh --voice      # adds faster-whisper and Piper (a long install on ARM)
./launch.sh --check-only    # is the backend actually usable?
./launch.sh                 # pre-flight, then the node — prints the console URL
```

`launch.sh` prints the address to open:

```
▸ Starting MAGI
  Console:  http://localhost:8000
  Kiosk:    chromium-browser --kiosk --noerrdialogs http://localhost:8000
```

Pre-flight distinguishes two kinds of problem, and the distinction is the point:

- **Hard errors abort.** Ollama unreachable, a model in `config/magi.yaml` not
  pulled, or a model that cannot return a schema-valid `MagiTurn`. None of those
  leave a working debate, so finding out at boot beats finding out three minutes
  in.
- **Warnings do not.** Models not resident in Ollama's memory, no Piper voice,
  no trace collector. The node still works; the numbers just deserve an
  asterisk, and the fact that it warned is recorded on the debate row.

Two other entry points:

```bash
uv run python scripts/probe_models.py   # schema compliance for every model on the host
./docker/compose.sh up                  # Jaeger, so traces have somewhere to go
```

`probe_models.py` is how you pick advisors with evidence rather than by
reputation: it probes everything Ollama has pulled, not only the three already
configured. `ask.py --no-trace` runs uninstrumented, which is how a benchmark
run must be taken.

With Jaeger up, one debate is a single trace of ~130 spans — phases, rounds,
AutoGen's internals, and one `chat {model}` span per LLM call, following the
OTel GenAI conventions so the sibling repo's traces sit beside them.

### On the inference host

One thing needs doing there, and only for clean benchmark numbers:

```bash
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_KEEP_ALIVE=1h
```

Without it Ollama evicts and reloads weights between turns, and the latency
figures measure storage rather than the framework.

## License

GPL-3.0-or-later
