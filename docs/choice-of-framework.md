# Choice of framework

The full argument, and the measurements that test it. The README carries only the summary.

The short version: using a multi-agent framework at all, rather than orchestrating three agents by hand, is not the technically optimal choice for this node. It was picked on purpose, to measure what the framework costs instead of asserting it. The sibling repo `magi-system` implements the identical protocol with no framework, and is the baseline.

## Which AutoGen: `autogen-agentchat` 0.4+, not `pyautogen` 0.2

Two incompatible projects share the name, and the version numbers run backwards. **This repo uses the post-rewrite one.**

| | Pre-rewrite | **This repo** |
|---|---|---|
| Package | `pyautogen` 0.2.x | `autogen-agentchat` / `autogen-core` / `autogen-ext` (0.7.5 pinned) |
| API | `initiate_chat`, `GroupChatManager`, `UserProxyAgent` | `RoundRobinGroupChat`, `SelectorGroupChat`, `TerminationCondition`, `output_content_type` |
| Docs | [`autogen/0.2/`](https://microsoft.github.io/autogen/0.2/) | **[`autogen/stable/`](https://microsoft.github.io/autogen/stable/)** |

The 2024 rewrite reset the version to 0.4 and split one package into three, so the higher-looking `0.2` is the older API. Nothing in this codebase exists there: `output_content_type`, the composable termination stack and the group-chat classes are all post-rewrite. A snippet that calls `initiate_chat` is from the wrong AutoGen.

Reference documentation for what is used here:

- [AgentChat user guide](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html), agents, teams and termination
- [`autogen_agentchat.teams`](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.teams.html), `RoundRobinGroupChat` and `SelectorGroupChat`
- [`autogen_agentchat.conditions`](https://microsoft.github.io/autogen/stable/reference/python/autogen_agentchat.conditions.html), `MaxMessageTermination`, `TimeoutTermination`, `ExternalTermination`
- [Structured output](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html#structured-output), how every turn becomes a typed vote
- [Source](https://github.com/microsoft/autogen)

## Why AutoGen

Three reasons, none of them "it is the best tool for the job":

1. **To build something other than another framework-less agent loop.** A hand-rolled Think/Act/Reflect loop is a solved problem in this author's other projects. There is more to learn from using the thing everyone cites than from writing the same 200 lines again.
2. **To measure what a multi-agent runtime costs on a Raspberry Pi 5.** Framework overhead is usually discussed on laptops and cloud VMs. On a 16 GB ARM board, the CPU and resident-memory cost of AutoGen's own asyncio runtime and message plumbing is a real number that nobody publishes.
3. **To quantify the price of `SelectorGroupChat`.** AutoGen's headline feature is automatic turn-taking: an LLM decides who speaks next, which is one extra LLM call per turn. Both engines sit behind a config flag so the difference can be measured rather than argued about.

## Why a hand-rolled orchestrator would be the right call

For a production edge node, writing the orchestration yourself is the correct architectural decision. Stated here so the tradeoff is on the record:

- **It is an edge device with voice input.** Every millisecond is felt by the user, and three agents on a fixed protocol is not a problem that needs a group-chat abstraction.
- **It needs 2/3 degradation.** If one model becomes unreachable the debate should continue with the remaining two and say so. Expressing that inside someone else's group chat is fighting the framework.
- **It needs fast cancellation.** A new PTT press must abort a running debate (barge-in). Clean cancellation is straightforward in your own asyncio loop and less so through a third-party runtime.
- **The debate loop is genuinely small.** Three agents, a fixed round structure, a vote tally and a stopping condition: roughly 200 lines. A dependency footprint is not repaid by 200 lines saved.

## How the claim gets tested

Everything is traced with OpenTelemetry. AutoGen instruments itself and picks up whatever provider is registered globally, so its agent-messaging and model-call spans join the trace without the node owning its runtime. On top sit manual spans per debate and phase, one GenAI `chat {model}` span per LLM call from the instrumented client, and auto-instrumented httpx underneath. Spans follow the OTel GenAI semantic conventions so both repos' traces land in one backend and compare directly.

A worked example, one 132 s DEADLOCK straight off the root span:

```
magi.debate                132.6s
  magi.phase.deliberation   92.9s
  magi.phase.blind          29.5s   MELCHIOR 29.4s | CASPAR 14.2s | BALTHASAR 13.1s
  magi.phase.verdict         6.4s
  magi.judge                 3.7s
  magi.tally                 ~0
```

The blind round runs its three advisors in parallel, so it costs whatever the slowest one costs: the reasoning model gates it at more than twice the others. The design predicted that qualitatively; the traces put a number on it.

Traces and the database do different jobs and must not duplicate each other. OpenTelemetry answers *why was this debate slow*. SQLite holds the durable benchmark row: per-turn latency, prompt and completion tokens, number of LLM calls, sampled process CPU and RSS, the engine, and the exact model set.

Tracing is itself an observer effect, since OTel costs CPU and memory on the node whose CPU and memory are the object of study. So `MAGI_OTEL_ENABLED=0` truly disables it, every row records whether tracing was on, and the headline overhead figures are taken with it off in both repos. Traces explain the behaviour; they do not produce the published number.

LLM calls are counted at the model client, not at the message level, because `SelectorGroupChat` makes its selection calls internally where per-message usage data cannot see them. Counting them anywhere else would understate exactly the thing being measured.

Two comparisons fall out of that.

### Within this repo: roundrobin vs selector

```
MAGI_ENGINE=autogen_roundrobin   AutoGen, deterministic turns
MAGI_ENGINE=autogen_selector     AutoGen, LLM-selected turns
```

First measurement, same question, same models, tracing off, MacBook driving the Spark:

| | roundrobin | selector | delta |
|---|---|---|---|
| wall clock | 147.4 s | 212.5 s | **+44 %** |
| LLM calls | 11 | 18 | **+64 %** |
| of those, turn selection | 0 | 7 | |

Seven extra calls to answer "who speaks next", on a question both engines resolved to the same DEADLOCK. On a voice interface that is a minute of silence bought with nothing.

### Across repos: this vs the sibling `magi-system`

The cost of the runtime itself. For it to mean anything, `config/magi.yaml`, the SQLite metrics schema and the consensus tally function are a shared contract: both repos hold identical copies, by agreement rather than by import.

## The Pi measurement, and a premise it demolishes

Three runs per engine on the Raspberry Pi 5 (Debian 13, Python 3.13, tracing off, same question, Ollama on the Spark):

| | roundrobin | selector |
|---|---|---|
| node CPU | 1.7-2.2 s | 1.8 s |
| as % of one core | **1-3 %** | **1 %** |
| peak RSS | 103-104 MB | 103-105 MB |
| LLM calls | 11 | 14-19 (5-8 turn selection) |
| wall clock | 54-253 s | 148-216 s |

**Reason 2 for choosing AutoGen was wrong.** The project set out to measure the CPU and resident-memory cost of AutoGen's asyncio runtime on a 16 GB ARM board. That cost is about 2 seconds of CPU across a four-minute debate and about 104 MB of RSS, and it is *identical* between the two engines. The node spends the entire debate blocked on HTTP to a remote GPU. There is no meaningful runtime overhead to find, because the workload is not CPU-bound and never was.

Worth stating plainly rather than quietly dropping: the premise was testable, it was tested, and it did not survive. What the framework actually costs is latency, in the form of extra LLM calls, which the selector comparison does measure and which is the number that matters on a voice interface.

Two caveats on the wall clock. It is dominated by the Spark and is noisy enough (54-253 s for the same engine and question) that no conclusion should be drawn from it without many more runs. And the ~104 MB is Python plus httpx plus pydantic plus AutoGen; attributing a share of it to the framework needs the sibling repo, which is exactly what the sibling repo is for.

## What building it on AutoGen actually cost

Findings from getting the first engine working (AutoGen 0.7.5). This is the evidence the argument above was written to be tested against, so it is recorded whether or not it flatters the framework.

**Where AutoGen paid for itself.** Barge-in was the requirement expected to be hardest through someone else's runtime, and `ExternalTermination().set()` covers it directly. Composing four stop rules with `|` is genuinely nicer than hand-written bookkeeping, and `output_content_type` turning every turn into a typed vote removed a whole parsing layer.

**Where it charged rent:**

- `SocietyOfMindAgent`, the class the design was going to use for the verdict, takes no `output_content_type`. The one place the framework's highest-level abstraction fit, it could not return typed fields. Replaced with a plain `AssistantAgent`.
- A group chat rejects `StructuredMessage[MagiTurn]` unless it is declared in `custom_message_types`. A standalone agent needs no such thing, and the error surfaces from inside a message container at run time, not at construction.
- `OrTerminationCondition` resets its children when the run ends, so asking which condition fired *after* `run_stream` returns always answers "none". The reason has to be sampled during the stream.
- `MaxMessageTermination` counts the seed task as a message. Budgeting without the `+1` cuts the final round before its last speaker, and since the order is fixed that is the same advisor every debate. A bias, not a truncation.
- Injecting a tracer provider through `runtime=` hangs the debate, because `run_stream` only starts the runtime it created itself, and the runtime you would naturally construct silently flips `ignore_unhandled_exceptions` back on. None of it is needed: registering the provider globally is enough, and AutoGen's internal spans then join the trace on their own.
- **2/3 degradation is genuinely hard.** In the blind round, one model failing is contained because we own the `asyncio.gather`. Inside the group chat there is no notion of a participant dropping out: any participant raising takes the whole run down. The best available answer is to catch it and tally the turns collected so far. This was the concrete worry in the argument above, and it turned out to be real.

**The finding that had nothing to do with AutoGen, and mattered most.** With three agents on one shared thread the debate mode-collapsed: after a good blind round, the second and third speakers reproduced the first one's answer *verbatim*, three identical sentences that a naive reading scores as strong consensus. Prompting them not to copy did nothing. What fixed it was reordering the task: critique each other **first**, state your position **second**, and require it to contain something nobody else said. Debates now reach real MAJORITY and DEADLOCK outcomes with named dissent. Any multi-agent system on a shared thread should expect this failure, and it is invisible in the output unless you look for it.
