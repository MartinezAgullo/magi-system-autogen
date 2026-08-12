# Making the advisors eager to agree

How to tune MAGI's bias towards consensus, and — more importantly — which of the
available knobs produce agreement you can defend and which only produce
agreement you can report.

Nothing here is currently applied. The committed configuration is deliberately
biased the *other* way: towards earned disagreement. This document exists
because "make them agree more" is a reasonable thing to want from a
deliberation system, and because the obvious way to do it is the wrong one.

> **A note on the sibling repo.** Two of the levers below — the tally rule and
> the judge — sit in files held by contract with `magi-system`. Changing them
> here alone silently stops the benchmark from being a comparison. See
> [Relationship to sibling projects](../CLAUDE.md#relationship-to-sibling-projects).

## The distinction that governs everything else

There are two different things that look identical in the output:

**Convergence.** An advisor reads the others, finds an argument better than its
own, and changes position. Three advisors then hold the same view for three
sets of reasons. This is what the system is for.

**Agreement inflation.** An advisor stops contributing — it echoes, hedges, or
ticks `agrees_with` without engaging — and the tally scores that as consensus.
Three identical answers carry exactly as much information as one.

Both produce `UNANIMOUS`. Only one is worth the two to four minutes of three
models' time that produced it. Every lever below is classified by which it
tends to produce, and that classification is the actual content of this
document.

## The levers, by effect size

### (a) The tally rule — largest effect, zero LLM cost

`magi/orchestrator/consensus.py`, `_agreement_blocs()`.

Agreement is currently an edge in an undirected graph, and the edge exists only
if **both** advisors claim it. "A agrees with B" while B says nothing about A is
one advisor being agreeable, not two converging.

Three progressively looser rules, in order of how much consensus they buy:

| Rule | Effect |
|---|---|
| Mutual edges (current) | A cycle A→B→C→A yields no bloc at all. Strictest |
| One-directional edges | A single agreeable advisor can pull a bloc together on its own |
| Transitive closure over one-directional claims | Almost any partial agreement becomes unanimity |

This lever is honest in a way the prompt levers are not: it changes what the
system *counts* as consensus without changing what the models say, so the
transcript still shows exactly what happened and a reader can disagree with the
arithmetic. It is also the one to reach for first if the complaint is
"UNANIMOUS almost never fires", because the current rule is strict enough that
genuinely converged debates can still tally as MAJORITY.

**Verdict: produces convergence, or at least reports it faithfully.** The risk
is over-reporting, not fabrication.

> Shared contract with `magi-system`. Land it in both repos in the same change
> window, or the benchmark is measuring the tally rather than the framework.

### (b) The judge's threshold — large effect, one LLM call

`magi/orchestrator/prompts.py`, `JUDGE_SYSTEM_PROMPT`.

The judge runs only on a split vote and answers one question: is the
disagreement substantive, or the same answer worded differently? A COSMETIC
ruling promotes the outcome to `UNANIMOUS`
(`magi/orchestrator/magi.py`, in `debate()`).

The current criterion is strict and action-based:

> Answer SUBSTANTIVE if acting on one position would lead to a different
> decision than acting on another.

Loosening it is a one-sentence edit with a large, measurable effect — for
example from "a different decision" to "a materially different outcome", which
lets differences of degree pass as cosmetic.

This is the best-instrumented lever in the system: `judged_cosmetic` is
recorded on every `DebateRecord`, so the change can be quantified after the
fact rather than argued about. Before touching anything else, it is worth
querying how often the judge already runs and how often it already promotes.

**Verdict: produces reportable agreement, and admits it.** The flag on the row
is what keeps it honest — a debate promoted by the judge is distinguishable
from one that converged on its own, forever.

### (c) The seed prompt's ordering — large effect, and a trap

`magi/orchestrator/prompts.py`, `deliberation_seed()`.

The seed currently forces a specific order: critique the others first, state
your own position second, and require that position to contain something nobody
else said.

That ordering is load-bearing anti-collapse machinery, not style. It was added
after a measured failure: with three agents on one shared thread and no such
constraint, the second and third speakers reproduced the first one's answer
**verbatim**, which the tally scored as an unusually strong consensus. Telling
them not to copy changed nothing; changing the order of the task did.

Inverting it — position first, critique optional — is the most direct route to
a higher `UNANIMOUS` rate, and it is the one route this document recommends
against. It does not make the advisors agree. It makes them stop reading each
other.

**Verdict: pure agreement inflation.** Avoid.

### (d) The prompts — moderate effect, cheap to try

`config/magi.yaml`.

Three separate surfaces, in descending order of leverage:

1. **`common_prompt`** currently says "agreement that is not earned is worth
   nothing here", and gates the vote on positions "you would defend, not merely
   tolerate". Changing *defend* to *tolerate* is the single smallest edit in
   this document with a real effect, because it lowers the bar for the one
   field the tally reads.
2. **CASPAR's persona** contains an explicit manufactured-dissent instruction:
   "Argue the opposing case even when you privately lean the other way." That
   is a per-seat agreement suppressor, and CASPAR is by construction the seat
   most likely to be the lone dissenter in a MAJORITY.
3. **`common_prompt`'s mind-changing rule** — "Changing your mind for a good
   reason is the point of this process. Changing it to end the debate is a
   failure." — is the sentence that most directly discourages capitulation.

Note that (1) and (2) differ in kind. Relaxing the `agrees_with` bar changes
how an advisor *reports* a position it already holds. Removing CASPAR's
contrarian instruction changes what position it holds at all.

**Verdict: (1) is reportable agreement; (2) is genuine, and costs you the
skeptic.** Removing the case against is not a tuning change, it is a decision
to run a two-perspective system with three models.

### (e) Sampling and budget — small effect, no prose changes

`config/magi.yaml`, and `MAGI_MAX_ROUNDS`.

- **Temperature.** CASPAR runs at 0.8, the highest of the three, which
  mechanically raises how many novel objections it generates. Lowering it
  toward 0.3 narrows the search and reduces dissent — but also reduces the
  quality of the dissent that remains, which is most of what CASPAR is for.
- **Rounds.** More deliberation rounds means more opportunities to converge.
  The confound: `MAGI_MAX_ROUNDS` feeds `MaxMessageTermination` *and* the
  number of chances, so raising it changes two variables at once and makes the
  resulting convergence rate uninterpretable. If you raise it to measure
  convergence, hold the message budget fixed separately.

**Verdict: genuine but weak, and the rounds knob confounds its own
measurement.**

## What is not available today

Two things worth knowing before designing an experiment around this.

**There is no agreement gradient.** `agrees_with` is hard set membership. An
advisor cannot say "mostly, except on the cost argument". Every lever above
therefore operates on a binary, and "agree when the difference is small" is not
expressible in the current schema.

**`confidence` is collected and never used.** It is on every `MagiTurn`, it is
shown in the deliberation seed, and `tally()` ignores it entirely. A rule like
"a low-confidence dissent does not block consensus" is one of the more
interesting things to try here, and it needs no prompt changes at all — only a
change to `tally()`.

Both are `MagiTurn` questions, and `MagiTurn` is prompt surface, not plumbing:
its field names and descriptions are handed to a 12B model as a JSON schema, so
adding a field changes behaviour beyond the arithmetic that reads it. See the
module docstring in `magi/models.py`.

## If you only do one thing

Change the judge's threshold (b), and read `judged_cosmetic` afterwards. It is
one sentence, it is already instrumented, and it separates the two failure
modes for you: a rise in `UNANIMOUS` that shows up in `judged_cosmetic` is the
judge being more permissive, while a rise that does not is the advisors
actually converging.

## Measuring whether it worked

Whatever you change, the question is not "did `UNANIMOUS` go up". It is
whether the debates still produce information. Three checks:

1. **Contested questions must still reach MAJORITY and DEADLOCK.** A
   configuration that reaches `UNANIMOUS` on a genuinely contested question has
   broken, not improved. Keep a fixed set of such questions and re-run them.
2. **Read the positions for novelty.** Mode collapse is visible to the eye and
   invisible to the tally: if the three `position` fields in the final round
   share sentences, the consensus is an echo. This is what the `position`
   novelty requirement in `deliberation_seed()` exists to prevent.
3. **Split `judged_cosmetic` out of the `UNANIMOUS` rate.** They answer
   different questions and averaging them hides which lever moved.

`DebateRecord` carries `terminated_by`, `judged_cosmetic`, `rounds_used` and
`outcome`, which is enough for all three without adding instrumentation.
