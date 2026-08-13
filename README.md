# MAGI (AutoGen edition)

> Voice-first multi-agent deliberation on a Raspberry Pi 5: three LLMs with opposing roles debate a spoken question until they reach consensus. AutoGen implementation, fully traced with OpenTelemetry for edge benchmarking.

Three LLMs argue about your question until they agree. Then one of them tells you the answer, out loud.

**MAGI** is a voice-first deliberation system running on a Raspberry Pi 5. You hold a push-to-talk button and ask something. Three models with deliberately different temperaments, **MELCHIOR** the scientist, **BALTHASAR** the pragmatist and **CASPAR** the skeptic, answer independently, then read each other's answers, criticise them, and vote. An orchestrator called **MAGI** returns a single verdict:

```
UNANIMOUS   all three converged
MAJORITY    two agreed, the dissent is reported rather than hidden
DEADLOCK    no convergence: the three positions and the exact point of
            disagreement are returned, with no invented synthesis
```

The names are an Evangelion reference. The three-personality supercomputer is the whole design idea, not just the branding.

The Pi is the reasoning client, not the inference host. Speech recognition runs locally (`faster-whisper`), the debate loop runs locally, and only token generation leaves the node, over HTTP to Ollama on a DGX Spark. No model weights ever touch the Pi.

## This repo is one half of an experiment

This is the **AutoGen** implementation. A sibling repository, `magi-system`, will implement the identical protocol with a hand-rolled orchestrator and no framework. Running both on the same Pi, against the same models, with the same prompts, is how the cost of the framework gets measured instead of debated.

Consequently this repo commits to AutoGen fully: where a choice exists between doing something by hand and doing it through a framework construct, it uses the framework construct. An implementation that routes around AutoGen would benchmark nothing.

## Status

**Debates work, end to end, fully traced.** Ask a question with `scripts/ask.py` and three advisors argue it out; UNANIMOUS, MAJORITY and DEADLOCK have all been reached on real questions, 48-160 s per debate. Every run produces a single ~130-span trace.

**Voice-first, end to end.** Hold PTT, speak, release: the phrase is transcribed on the node and appears as a deletable line, and SEND commits the question so the three advisors argue it out on the console. Both engines work, verified on the Pi.

**Watchable while it thinks.** The console shows which advisor is generating right now, and the terminal can show the sentence forming: set `stream: true` on an advisor in `config/magi.yaml` and its position is drawn live, then erased when complete so the one-line log lands in its place. Off by default, and per advisor rather than global — a streamed call gives up the wider-budget retry, so the reasoning seat should not take it. Pre-flight verifies streaming advisors over the streaming path and warns about that combination.

**Every debate is recorded.** One row per debate in SQLite, with the full transcript and what the run cost the node, written by both the daemon and `scripts/ask.py`. `scripts/report.py` reads it back: the `UNANIMOUS` rate split into what the advisors reached unaided and what the judge granted them, and the roundrobin-vs-selector cost comparison, grouped by engine, with runs that are not comparable with the rest excluded and named rather than quietly averaged in. The schema is a contract shared with the sibling repo, so the two implementations' databases can simply be concatenated.

Still to build: TTS, so the verdict is spoken as well as shown. See `CLAUDE.md` § Status.

The load-bearing assumption is **verified**: as of 2026-08-11, on Ollama 0.23.2, all three default advisors return a `MagiTurn` that is both schema-valid and substantive over the OpenAI-compatible endpoint. Getting there needed two per-model settings that are not obvious and are documented in `config/magi.yaml`: reasoning must stay **on** for `nemotron3:33b` (it produces nothing valid with it suppressed) and **off** for `gemma3:12b` (3/5 valid with it on, 5/5 with it off), and a reasoning model needs roughly triple the token budget because it spends it before writing any answer.

## Choice of framework

This project uses Microsoft's AutoGen (`autogen-agentchat` v0.4+, [docs](https://microsoft.github.io/autogen/stable/), not the deprecated `pyautogen` 0.2.x whose docs still rank first). Using a multi-agent framework at all, rather than orchestrating three agents by hand, is **not the technically optimal choice** for this node. It was picked deliberately, to measure what a framework costs on an edge device instead of asserting it. The baseline is the sibling repo, which has no framework at all.

Two headline results so far:

| Measured | Result |
|---|---|
| `SelectorGroupChat` vs `RoundRobinGroupChat` | **+44 % wall clock, +64 % LLM calls**, same outcome |
| AutoGen's CPU and memory footprint on the Pi | ~2 s of CPU and ~104 MB RSS per debate, identical between engines |

The second one killed a premise the project was partly built on: there is no meaningful runtime overhead to find, because the node spends the whole debate blocked on HTTP to a remote GPU. What the framework actually costs is latency, in extra LLM calls.

**Read the full argument and the numbers in [`docs/choice-of-framework.md`](docs/choice-of-framework.md)**: which AutoGen this is and why the versions confuse everyone, why a hand-rolled orchestrator would be the right call for a production edge node, how the claim is traced and tested, and what building on AutoGen cost in practice.

## How the debate works

Three phases.

**Phase A, the blind round.** Each MAGI answers independently, with no visibility of the others. This *cannot* be done inside a group chat: in `RoundRobinGroupChat` all agents share one message thread, so the second speaker sees the first one's answer and anchors to it. Phase A therefore calls each agent directly and in parallel. Anchoring is the single biggest threat to the value of the whole system, because three models that agree because they read each other are not a consensus, they are an echo. This is the one place the design deliberately steps outside the framework's team abstraction.

**Phase B, deliberation.** The group chat is seeded with the three positions. Up to two rounds of mutual critique, in which any MAGI may revise its position.

**Phase C, the verdict.** A `SocietyOfMindAgent` named MAGI turns the debate into one answer, 2-3 sentences, which is what gets spoken.

Consensus is expressed in AutoGen's own idioms rather than as a bolted-on phase: every turn is a `StructuredMessage` carrying a typed vote, a custom `TerminationCondition` stops the debate the moment the vote is unanimous, and the round budget, wall-clock budget and PTT barge-in are `MaxMessageTermination`, `TimeoutTermination` and `ExternalTermination` composed with `|`.

The tally itself stays deterministic: the orchestrator counts votes in Python and hands the outcome to the verdict agent as a constraint. An LLM judge is consulted only when the vote is split, to decide one thing, whether the disagreement is substantive or the same answer worded differently.

## Asking it something

Two ways in, and they end in the same place.

**By voice.** Hold the PTT button, say one phrase, release. `faster-whisper` transcribes it **on the node**, so audio never leaves the Pi, and it appears in the draft as a numbered line. Say another. Delete the one Whisper mangled. When the question reads right, press SEND.

```
草案 DRAFT                                    2 LINES   CLEAR ALL
01  Should a small team migrate their monolith to microservices?   DEL
02  Consider the operational cost over the next two years.         DEL
```

**A press never starts a debate**, and that is the point. Recognition gets things wrong and a debate costs two to four minutes of three models' time, so firing per press would buy a full deliberation on a question nobody asked. The draft is the gap between what the machine heard and what it acts on.

**By typing.** The text box takes a question directly, for a laptop on the LAN or a node with no microphone.

Requires `./setup_env.sh --voice`. Without it the PTT button renders present and honestly disabled rather than hidden, because the control is part of what this node is and pretending it works would be worse than showing it greyed. Same for a browser on a LAN address: the microphone needs a secure context, so only `localhost` gets PTT, and the button says so.

Measured on the same clips: **1.5-2.2 s per phrase on the MacBook, 6.6-7.1 s on the Pi**, and silence returns nothing rather than an invented sentence. Transcribing takes the Pi's SoC to 76.8 °C, which puts the console into CONDITION CAUTION, the telemetry doing its job. That is tolerable only because the draft means the operator is waiting to read a phrase back, not waiting for an answer. It is also the first hard number for what moving Whisper to the AI HAT+ would be worth.

## The console

`./launch.sh` serves it at `http://<node>:8000`. An Evangelion MAGI display: the three advisors as chamfered panels framing MAGI at the centre, amber on black, live turns streaming over one WebSocket, and status flags fed by the Pi's real temperature and throttle bits.

Node colour is the honest analogue of the anime's 承認/否定 vote. The advisors never cast a binary vote, they hold positions with nuance, but they do declare who they stand with, so a node turns green inside the agreeing bloc and red outside it. The drama of the colour change survives; the lie does not.

Kiosk: `chromium-browser --kiosk --noerrdialogs http://localhost:8000`

Opened any other way, a laptop on the LAN or the Pi's own browser by hand, the `全画面 FULLSCREEN` button in the status row drops the browser chrome, which on an 800x480 panel is about a third of the triad.

## Models

The Spark currently serves:

```
gemma3:12b        qwen3:14b        nemotron3:33b     deepseek-r1:32b
qwen3-coder:30b   qwen3-vl:8b      bge-m3:latest
```

Defaults, chosen for **lineage diversity**, because three models from the same family agreeing proves nothing:

| MAGI | Model | Lineage |
|---|---|---|
| MELCHIOR | `nemotron3:33b` | NVIDIA |
| BALTHASAR | `gemma3:12b` | Google |
| CASPAR | `qwen3:14b` | Alibaba |

This is a **starting point, not a fixture**. Names, models, parameters and prompts all live in `config/magi.yaml`; nothing in the code hardcodes a tag, and `MAGI_PERSONAS_FILE` selects a different set per run. Rotating models is the intended way to use this thing.

Two practical notes that will otherwise waste an afternoon:

- **Watch for model residency.** The Spark holds multiple models in memory fine, but if Ollama is left to evict between turns, every latency number becomes a measurement of disk I/O rather than of AutoGen (`OLLAMA_MAX_LOADED_MODELS`, `OLLAMA_KEEP_ALIVE`). The Pi cannot read the Spark's environment, so the pre-flight check observes `GET /api/ps` instead and **warns**. It does not abort, and it records that it warned, so a suspect run can be excluded from the benchmark rather than skewing it.
- **`deepseek-r1:32b` is a benchmark variant, not a default.** Its `<think>` blocks fight strict JSON output and its latency is felt on a voice interface. Interesting to measure, wrong to build on.

## Possible extensions

Ideas that are deliberately out of scope for the first version.

### Dynamic role assignment

Today the three roles are fixed archetypes written by hand in `config/magi.yaml`, scientist, pragmatist and skeptic, and every question is debated by the same three personalities. The more advanced version is to **generate the roles per question**: one preliminary LLM call analyses the topic and produces three tailored system prompts, choosing the three perspectives that would genuinely disagree most productively about *this* subject.

A question about a database migration might convene a distributed-systems engineer, an on-call SRE and a cost-conscious CTO. A question about a medical protocol would convene something else entirely. The static archetypes are generic by necessity; generated ones can be specific. The same call could pick the models, too, since `qwen3-coder:30b` earns a seat on a software question and none at all on anything else.

The tradeoffs, for whenever this gets built:

- One extra LLM call before every debate, which on a voice interface is latency the user feels before anything happens.
- Reproducibility drops. Comparing engines and models requires the prompts to be held constant, so the benchmark work needs the static path to stay available.
- Prompt-generation quality becomes a new failure mode: three generated roles that are secretly the same role produce a fake consensus, which is worse than no consensus at all.

It fits the existing design cleanly, since `personas.py` already loads personas as data and a generator only has to produce the same structure the YAML does.

### Others

- **Tools.** The tool registry is wired and empty by design. Web search would make debates on factual topics far more useful, at the cost of latency and an internet dependency on the node.
- **Embedding-based consensus.** `bge-m3` is already on the Spark. Comparing the three positions by vector similarity would be an objective third opinion alongside the self-reported vote and the LLM judge.
- **STT on the AI HAT+.** The Hailo-8 is installed and unused; Whisper runs on the Pi's CPU. An optimisation, not a requirement.
- **Conversational memory.** Each debate currently starts clean. Follow-up questions would feel more natural by voice, at the cost of a growing context that contaminates model comparisons.

## Hardware

| Device | Role |
|---|---|
| Raspberry Pi 5 (16 GB) | The MAGI node: daemon, STT, TTS, UI |
| 7" touch display | Kiosk |
| Bluetooth microphone | PTT input, captured by the browser |
| NVIDIA DGX Spark | Ollama inference host, three models resident |
| MacBook Pro M3 | Development |

## Getting started

```bash
./setup_env.sh              # core only, enough to hold and benchmark debates
./setup_env.sh --voice      # adds faster-whisper and Piper (a long install on ARM)
./launch.sh --check-only    # is the backend actually usable?
./launch.sh                 # pre-flight, then the node, prints the console URL
```

`launch.sh` prints the address to open:

```
▸ Starting MAGI
  Console:  http://localhost:8000
  Kiosk:    chromium-browser --kiosk --noerrdialogs http://localhost:8000
```

Pre-flight distinguishes two kinds of problem, and the distinction is the point:

- **Hard errors abort.** Ollama unreachable, a model in `config/magi.yaml` not pulled, or a model that cannot return a schema-valid `MagiTurn`. None of those leave a working debate, so finding out at boot beats finding out three minutes in.
- **Warnings do not.** Models not resident in Ollama's memory, no Piper voice, no trace collector. The node still works; the numbers just deserve an asterisk, and the fact that it warned is recorded on the debate row.

Two other entry points:

```bash
uv run python scripts/probe_models.py   # schema compliance for every model on the host
./docker/compose.sh up                  # Jaeger, so traces have somewhere to go
```

`probe_models.py` is how you pick advisors with evidence rather than by reputation: it probes everything Ollama has pulled, not only the three already configured. `ask.py --no-trace` runs uninstrumented, which is how a benchmark run must be taken.

With Jaeger up, one debate is a single trace of ~130 spans: phases, rounds, AutoGen's internals, and one `chat {model}` span per LLM call, following the OTel GenAI conventions so the sibling repo's traces sit beside them.

### On the inference host

One thing needs doing there, and only for clean benchmark numbers:

```bash
OLLAMA_MAX_LOADED_MODELS=3
OLLAMA_KEEP_ALIVE=1h
```

Without it Ollama evicts and reloads weights between turns, and the latency figures measure storage rather than the framework.

## License

GPL-3.0-or-later
