# Candidates

Things worth considering, none of them decided.

**This is not the build order.** The committed sequence lives in `CLAUDE.md` under "Status", and SQLite persistence, TTS wiring and the Pi's engine comparison come before anything here. What follows is the queue behind that. Each entry states what it would buy, what it would cost, and where the answer is already known, why it has not been done yet.

Entries move out of this file in one of two directions. Either they become a decision in `CLAUDE.md`, or they get struck out here with the reason. An idea that was rejected for a good reason is worth more than one that was never written down, because the second one comes back every six months.

---

## 1. Tools

The registry (`magi/tools/registry.py`) exists and is empty by design, so search or RAG can be added without a refactor. What that sentence hides is that the refactor is not the hard part.

### What it actually takes

Flipping `function_calling=True` in `ModelInfo` (`clients.py`) changes nothing on its own. It is a capability *declaration*, checked before AutoGen will let you attach tools. The real work is four things.

1. **Attach the tools.** `AssistantAgent(tools=[...])` takes plain callables, which AutoGen introspects into a schema from the signature and docstring, or `FunctionTool` / `Workbench` instances. This part is genuinely easy.
2. **Reconcile tools with `output_content_type`.** They compete for the same structured-generation channel. AutoGen's flow is call, tool calls, execute, call again, final answer, and only that last call is shaped by `output_content_type`. `reflect_on_tool_use=True` is needed or the tool result never enters the model's reasoning, and `max_tool_iterations` bounds the loop.
3. **Verify tool calling per model, not per advisor.** Ollama's function-calling support is uneven across tags. This needs its own pre-flight probe next to the existing schema probe, and multi-sampled, per the rule already learned twice here: a gate on a non-deterministic system needs more than one sample.
4. **Decide what a tool result does to the debate protocol.** An advisor that can look things up is no longer arguing from its own weights. That is the point, and it also means two advisors hitting the same source will agree for a reason the tally cannot distinguish from convergence. Worth thinking about before, not after.

### Budget blowup

This is the reason to be careful, and it is not hypothetical.

A tool-using turn is **at minimum two LLM calls instead of one**, plus one more per additional tool round. With `reflect_on_tool_use=True` the second call carries the tool output on top of the whole prompt, so input tokens rise as well.

Against the numbers already measured:

- MELCHIOR needs `max_tokens: 4000` and spends ~3800 characters reasoning before writing a word of answer. It alone takes 29.4 s of the blind round's 29.5 s, while the other two finish in 13-14 s and wait for it.
- It already trips `LengthFinishReasonError` often enough to need the double-budget retry in `InstrumentedChatClient`.
- A debate is 48-155 s today. Tripling MELCHIOR's call count would put a voice interface past three minutes on a routine question.

And the failure mode is worse than slow. The group chat has no notion of a participant dropping out, so an advisor that exhausts its budget mid-tool-loop takes the whole debate down. That is the same trap the length retry exists to paper over, and tools make it more likely, not less.

### Suggestion: one advisor first

Give tools to **CASPAR only**, and leave MELCHIOR and BALTHASAR as they are. Four reasons:

- **It is the skeptic's job.** Checking whether a claim survives contact with a source is what CASPAR is for. MELCHIOR argues correctness from first principles, so handing it a search tool changes its archetype rather than equipping it.
- **It is the cheapest seat to slow down.** CASPAR runs `qwen3:14b` with `thinking: false`, one of the two fast advisors. MELCHIOR is already the critical path, so adding calls to the seat that finishes first spends slack that actually exists.
- **The blind round is `asyncio.gather`.** Phase A costs `max(advisors)`, not the sum, so a slower CASPAR is free until it overtakes MELCHIOR. There is roughly 15 s of headroom before that happens.
- **It isolates the finding.** One advisor with tools against two without is a controlled comparison inside a single debate. Three at once produces one slower number and no explanation for it.

The cost of asymmetry, stated honestly: three advisors no longer have comparable cost profiles, so per-advisor latency stops being a like-for-like column in the benchmark. That is a reporting problem to solve rather than a reason to avoid it, and it is smaller than the problem of not knowing which advisor caused a regression.

### Also worth knowing

`Workbench` and the MCP integration (`autogen_ext.tools.mcp`) let a tool set come from an external server rather than local functions. Relevant if tools ever need to be shared with the sibling repo: a shared MCP server would keep the tool surface identical across implementations without sharing code, which is the same boundary already drawn everywhere else in this project.

---

## 2. GraphFlow as a third engine

`GraphFlow` (AutoGen 0.6+) is a team built from a `DiGraphBuilder`, with explicit nodes, explicit edges, conditional edges, and parallel fan-out with a join.

### The argument for

**It would close the asterisk on the benchmark.** The blind round is currently the one documented place this project steps outside the framework: `asyncio.gather` over `agent.run()`, outside any team, because a group chat gives every participant one shared thread. So today's comparison is really "AutoGen minus the part AutoGen could not do" against the baseline. `GraphFlow` supports parallel nodes with a join, which is exactly the blind round's shape, so the whole protocol could live inside the framework and the caveat goes away.

Beyond that, the protocol genuinely *is* a graph: blind fan-out, join, deliberation loop, conditional edge on the tally, verdict. Consensus would become an edge predicate rather than a `TerminationCondition` subclass, and the phase structure would be declared rather than driven by Python in `orchestrator/magi.py`.

### The argument against

- **It measures a different thing.** `GraphFlow` has no speaker-selection LLM call. The roundrobin/selector delta prices AutoGen's headline feature; a GraphFlow number would price *declared control flow*, which is a separate question and not one the project set out to answer.
- **A third engine is a third protocol to keep in sync** with `magi-system`, and every divergence silently stops the benchmark from being a comparison.
- **It converges with the baseline.** A declared graph with conditional edges is close to what the hand-rolled orchestrator does anyway. The closer the two get, the less the comparison distinguishes.
- **Priority.** It ranks below `Swarm` (§5.2) in interest and below SQLite in necessity.

### Verdict

Worth doing eventually, and specifically for the blind-round argument rather than for its own sake. If the goal is "one more engine", `Swarm` is the better next one.

---

## 3. Small models, running on the Pi

Move one or more advisors off the Spark and onto the node itself.

### Why it is tempting

The Pi is **idle during a debate**. Measured over three runs per engine: 1-3 % of one core and ~104 MB peak RSS, identical between engines, because the node is blocked on HTTP to the Spark for the entire debate. There is a 16 GB Raspberry Pi 5 doing essentially nothing for two minutes at a time.

It would also change what the project can claim. Right now the node is a reasoning client with a hard dependency on an expensive box on the same LAN. An advisor that runs locally is the difference between "an edge orchestrator" and "an edge system", and it makes the Spark's absence a degradation rather than an outage.

### Why it is harder than the idle CPU suggests

**The STT number is the warning.** `faster-whisper small` on a three-second clip costs 6.6-7.1 s and drives the SoC to 76.8 °C, enough to put the console into CONDITION CAUTION. That is a ~240 M parameter encoder-decoder doing one short transcription. An LLM generating several hundred tokens of structured output is a much larger job, and the Pi has no headroom the thermal envelope has not already found.

**Structured output is where small models fail first.** The pre-flight probe already asserts `MIN_POSITION_CHARS` rather than mere schema validity, precisely because "schema-valid" and "usable in a debate" turned out to be different things: measured positions were 7-11 characters under a thin prompt and 77-249 under the real persona prompts. A 1-3 B model is the most likely thing in this project to satisfy the schema with `position: "Yes."`. It would pass a naive check and hold no debate at all.

**Lineage diversity has to survive.** The premise is that three models trained by different teams disagree informatively. A local seat has to keep that. A small Qwen next to `qwen3:14b` would quietly turn a three-way debate into a two-and-a-half-way one, which is the exact failure the blind round exists to prevent.

**The AI HAT+ is not the answer, at least not yet.** The Hailo-8 is present and unused, and moving Whisper to it is already noted as an optimisation. Its tooling is vision-first, so treat LLM inference on it as something to verify before planning around rather than as available capacity.

### How to try it without wrecking the benchmark

- **One seat, and not MELCHIOR.** BALTHASAR is the candidate: it already runs with `thinking: false`, it is the cheapest advisor, and pragmatism is the archetype least dependent on long reasoning chains.
- **Run `ollama serve` on the Pi and point that persona at it.** This needs per-persona backend configuration, which does not exist today (§5.1). Everything else is already data: the model tag is YAML, and pre-flight already checks whatever the YAML names.
- **Expect the phase-A structure to invert.** Phase A costs `max(advisors)`. If the local seat becomes the slowest, it rather than MELCHIOR gates the blind round, and the whole latency profile of the system changes shape.
- **Record it on the row, or the numbers are uninterpretable.** A debate with one local advisor is not comparable to one with three remote advisors, in the same way and for the same reason a traced run is not comparable to an untraced one. `DebateRecord` already carries `models`; it would need to carry where each one ran.
- **Watch CPU and RSS, which are now real.** The whole "framework overhead does not exist in any meaningful amount" finding rests on the node being blocked on I/O. Put inference on the node and that premise stops holding, which is interesting in itself and needs saying out loud in the README rather than quietly invalidating a headline number.

### The honest framing

This is a **capability** change, not a performance one. Nothing about it will be faster. It buys autonomy from the Spark and a genuinely edge-native story, at the cost of latency, heat, and a seat that argues less well. Whether that is a good trade depends on what the project is for, and it is worth being explicit that a small local advisor is a worse advisor rather than a cheaper equivalent one.

---

## 4. Token streaming

Researched against AutoGen 0.7.5's installed source and **verified live** against the Spark on 2026-08-12, with `gemma3:12b` and `nemotron3:33b` on Ollama 0.23.2.

### What is already done

**Streaming itself is built, opt-in per advisor, and off by default.** `stream: true` on a persona in `config/magi.yaml` makes that advisor's `AssistantAgent` emit token deltas; both `scripts/ask.py` and the daemon render them to the terminal (`magi/services/stream_view.py`), so `./launch.sh` shows them too. Nothing in Python knows which seats or which model tags can stream: it asks the persona file, and pre-flight verifies the answer by probing those advisors through the streaming path rather than the plain one. Verified live with `gemma3:12b` and `qwen3:14b` streaming while `nemotron3:33b` did not.

The one dangerous combination is checked as a property, not a name: an advisor with both `stream` and `thinking` on has lost the wider-budget retry, and is exactly the advisor that needs it. Pre-flight warns and lets the operator proceed.

The **per-advisor activity indicator** is built and shipped, and it solves most of what streaming was wanted for. `TOPIC_ACTIVITY` carries `{advisor, busy}` from `InstrumentedChatClient` to the console, so each node shows GENERATING and pulses while it has a call open. Blind round: all three light at once and the one still lit after the others go quiet is the model gating the phase. Deliberation: one at a time under roundrobin, which also shows the turn order as it happens.

That was the cheap fix for a console that previously showed nothing for 13-30 s of blind round and ~90 s of deliberation, which is exactly when an operator concludes the node has hung. Everything below is the expensive version.

### Finding 1: streaming would have silently broken the instrumentation. Fixed

`model_client_stream=True` routes calls through `ChatCompletionClient.create_stream`, not `create`, and `InstrumentedChatClient` overrode **only `create`**.

So enabling streaming would have silently stopped counting LLM calls, stopped counting tokens, stopped emitting the `chat {model}` GenAI spans, and stopped publishing activity events. Every headline number this project exists to produce, quietly gone, with no error anywhere.

**Fixed.** `create_stream` is now overridden alongside `create`, and the two share `_record` so they cannot drift. Verified live: one call and non-zero tokens counted in both modes, for both `gemma3:12b` and `nemotron3:33b`. Calls made through the streaming path carry `magi.streamed` on the span, because the two paths differ in ways a benchmark cares about and a run that mixed them without recording which was which would be uninterpretable.

Two things that came out of doing it:

- **Usage has to be asked for.** OpenAI-compatible streaming sends no token counts unless `stream_options.include_usage` is set, and AutoGen only sets it when `include_usage` is passed explicitly. Left at its default, every streamed call reports zero tokens, which reads as a free call rather than as a missing measurement. Now defaulted to `True`. Ollama honours it.
- **The length retry cannot exist on the streaming path.** Streaming never raises `LengthFinishReasonError`; a model that spends its whole budget on reasoning produces `finish_reason="length"` on the final result, by which point chunks are already with the consumer and there is nothing to transparently retry. The condition is logged loudly instead. This is the concrete reason not to stream MELCHIOR: that retry is what stops one advisor's exhausted budget from taking the whole group chat down.

### Finding 2: reasoning does not reach AutoGen from Ollama, and it is a one-word mismatch

This one reversed twice, so the evidence matters.

AutoGen's `create_stream` reads `reasoning_content` off each delta's `model_extra`, wraps the run in literal `<think>` / `</think>` markers, and yields it as chunks like ordinary content. So a reasoning model *can* stream its reasoning, and the earlier note in this project saying otherwise was wrong.

Against Ollama it does not, because **Ollama calls the field `reasoning`**:

```
# /v1/chat/completions, nemotron3:33b, stream=true
delta: {"role":"assistant","content":"","reasoning":"We"}
delta: {"role":"assistant","content":"","reasoning":" need"}

# native /api/chat calls it something else again
message: {"role":"assistant","content":"","thinking":"We"}
```

AutoGen keys on `reasoning_content` exclusively, so every one of those deltas is dropped. Confirmed by the live probe: `nemotron3:33b` under `model_client_stream=True` produced 85 chunks, all of them JSON, and no `<think>` at all.

Three consequences:

- The `<think>` idea in the CLI plan below is **blocked**, not merely unverified. It needs either a patch to AutoGen's stream parsing or `OllamaChatCompletionClient` (§5.3), which speaks the native API where the field is `thinking`.
- It is dropped rather than misrouted, so it does not corrupt the JSON. Nothing is broken by it today.
- It would work as designed against a provider that uses `reasoning_content`, which is the field DeepSeek's API and several OpenAI-compatible servers use. So this is an Ollama-specific gap, not an AutoGen defect.

### Finding 3: structured output streams through OpenAI's *beta* client, and Ollama copes

When `response_format` is set, which is what `output_content_type` compiles to, `create_stream` dispatches to `_create_stream_chunks_beta_client`, which uses `client.beta.chat.completions.stream(...)`. That helper is stricter about the event protocol than plain chat completions.

**Verified working** against Ollama 0.23.2 for both models tested, with schema-valid `MagiTurn` output in both modes. The compatibility risk was real and did not materialise.

AutoGen does throw away the useful part, though. The beta helper emits `content.parsed` events carrying an incrementally parsed partial object, which is exactly what a nice renderer would want, and AutoGen's code comments that it "does not handle other event types" and yields only raw `chunk` events. Partial parsing is available in principle and not reachable through AutoGen without patching.

### Finding 4: the JSON problem is real and smaller than it looks. Confirmed

Content deltas are raw JSON text, so rendering them directly shows accumulating syntax. First chunks, live:

```
gemma3:12b     '{\n  "position": "No. Kubernetes introduces significant'
nemotron3:33b  '\n{\n  "position": "Adopt Kubernetes only when the startup'
```

Which confirms the mitigating detail: **`position` arrives first**, whole, before any other field starts, in both models. `MagiTurn` declares it first and schema-constrained decoding honours the order. Extracting it incrementally is a small state machine over the accumulated buffer rather than a general partial-JSON parser: wait for `"position":`, then stream string content until the closing quote. Roughly 40 lines.

### Should the CLI do this? Yes, and before the web console

The renderer was first built into `scripts/ask.py` alone, which was a mistake worth recording: the node is started with `./launch.sh`, and a view that only existed in a developer script was invisible to anyone running the actual thing. It now lives in the package and both use it, gated on stdout being a terminal so it never streams into the journal under systemd.

What is worth building, in order:

1. ~~**Incremental `position` extraction.**~~ **Done.** `PositionReader` in `magi/services/stream_view.py`. It is a three-state machine rather than a search over an accumulating buffer, because the deltas turn out to be far smaller than the key: `gemma3:12b` sends `'{'`, `'\n'`, `'  '`, `'"'`, `'position'`, `'":'`, so every boundary in `"position": "` can fall between two chunks. The first version searched for the key and assumed the value's opening quote was already in hand, which printed one stray space per advisor and stopped.
2. **A per-advisor elapsed counter**, driven by the activity events. The renderer currently holds concurrent streams and prints them whole, because phase A runs three advisors at once and a plain terminal has one cursor; a header flip per token is unreadable. So the blind round is still 13-30 s of nothing, and an elapsed line per advisor would cover it without cursor gymnastics.
3. **The `<think>` stream for the reasoning advisor** is **blocked** by Finding 2. It remains the highest information-per-line output available anywhere in the system, because it shows *why* the slowest advisor is slow, and it is unreachable until either AutoGen learns Ollama's field name or §5.3 lands. Worth revisiting if `OllamaChatCompletionClient` is ever adopted, since that is most of the case for adopting it.

The one hard rule: **benchmark runs must not stream.** Streaming changes the request, changes how the client handles the response, adds per-chunk work on the node whose CPU is the object of study, and loses the length retry. If streaming ever ships it needs its own flag, defaulting off, recorded on the debate row next to `tracing_enabled` for the same reason that one is there. The `magi.streamed` span attribute is already in place.

### Should the web console do this? Probably not

The activity indicator already covers the "is it working?" question, which was the actual complaint. Beyond that, the console's job at 800x480 is the triad and the verdict, and there is no room to render a token stream without taking space from one of them. A dictated question is answered by three advisors in two minutes; watching MELCHIOR reason in real time is a developer's interest, not an operator's.

Reconsider if and only if the CLI version proves the rendering is genuinely legible.

---

## 5. Shorter items

### 5.1 Per-persona backends

`settings.llm_base_url` and `llm_api_key` derive from one global `MAGI_LLM_BACKEND`, so every agent in a run talks to the same place. Ollama *or* an API provider, never both.

Optional `backend` / `base_url` / `api_key` fields on `Persona`, preferred over the global in `build_client`, would fix it. This is a **prerequisite** for §3 (an advisor on the Pi while the others are on the Spark) and for any run mixing local models with a hosted one. Small change, blocks two larger ones.

### 5.2 `Swarm` as the next engine

Turn selection where the current speaker names its successor via `HandoffMessage`. It sits between the two existing engines on the one axis they differ on: dynamic speaker choice, like `SelectorGroupChat`, but at **zero extra LLM calls**, because the choice is folded into the speaker's own generation.

That makes it the most informative third data point available, since it separates "dynamic turn-taking" from "an extra call per turn", which the current two engines conflate. Costs one more field on `MagiTurn`, which is prompt surface, so it is not free.

### 5.3 `OllamaChatCompletionClient`

`autogen_ext.models.ollama`, experimental, not currently installed. Would give first-class access to `num_ctx`, `keep_alive` and `think`.

Three of those matter here. **`keep_alive`** is the model-residency problem that pre-flight currently only *observes* via `/api/ps`, so the client could ask for what the warning is about instead of reporting its absence. **`num_ctx`** matters because MELCHIOR at 4000 output tokens plus a growing thread is the advisor most likely to exceed Ollama's default context, which truncates **silently**. And **`think`** is the native API's own reasoning switch, which is where the streamed reasoning lives that §4's Finding 2 cannot reach: the native protocol calls the field `thinking` and this client reads it.

Against: it is a second client class where there is currently one, it does not work against an API-key provider, and `InstrumentedChatClient` would need a parallel subclass or the counting and tracing seam splits in two. Best done together with §5.1, which makes client choice per-seat anyway.

### 5.4 Bounded `model_context`

Every advisor uses the default `UnboundedChatCompletionContext`, so prompts grow monotonically through a debate. Harmless at three rounds, superlinear against a fixed output budget if `MAGI_MAX_ROUNDS` is ever raised. `BufferedChatCompletionContext` or `TokenLimitedChatCompletionContext` is the fix, and it is also the right lever if an advisor should ever see only the last round rather than the whole thread.

### 5.5 `confidence` is collected and never used

It is on every `MagiTurn`, it is shown in the deliberation seed, and `tally()` ignores it completely. "A low-confidence dissent does not block consensus" is one of the more interesting rules available and needs no prompt change at all. Note the constraint: `tally()` is a shared contract with the sibling repo. See [agreement-bias.md](agreement-bias.md).

### 5.6 `confidence` has a range the model is not told about

Observed live on 2026-08-12, and it cost a debate an advisor.

`nemotron3:33b` returned `confidence: 70`, meaning 70 %. The field is declared `ge=0.0, le=1.0`, so pydantic rejected the turn, MELCHIOR produced nothing in the blind round, and the debate ran 2/3. The degradation path worked exactly as designed and the verdict said so, which is the good news; the trigger is avoidable, which is the bad.

The bounds are in the JSON schema as `minimum` and `maximum`, but constrained decoding does not reliably enforce numeric ranges, and the field's `description` never mentions one. So the only thing actually telling the model the scale is the field name, and "confidence" reads as a percentage to at least one model.

The fix is a few words in the description: state 0.0 to 1.0 explicitly, with an example. Cheap, and it addresses the general rule already learned here twice, that a schema constraint a model cannot see is not a constraint. Note that `MagiTurn` is protocol surface shared with the sibling repo, so it lands in both or in neither.

Worth pairing with §5.5, since anything that starts *using* `confidence` in the tally makes a silent 70-versus-0.7 confusion much more expensive than a rejected turn.

### 5.7 The MAGI / `Magi` name collision

`Magi` (the class in `orchestrator/magi.py`) orchestrates. `MAGI` (the `AssistantAgent`) writes two sentences for an outcome it is handed as a constraint. Same name, opposite amounts of authority.

The boot banner has been fixed. It read `MAGI orchestrator gemma3:12b`, which told the operator a 12B model decided the verdict. Still open: the persona's own system prompt opens "You are MAGI, the orchestrator of a three-advisor deliberation", which tells a writer it is in charge and is then immediately contradicted by the next line. That one is sent to the model, so it is the one that might actually be costing something.

The YAML key `orchestrator:` is a shared contract and should probably stay, name collision and all. A renamed key that lands in only one repo is a worse problem than a confusing one that lands in both.
