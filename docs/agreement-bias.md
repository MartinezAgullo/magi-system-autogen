# Making the advisors eager to agree

How to tune MAGI's bias towards consensus and, more importantly, which of the available knobs produce agreement you can defend and which only produce agreement you can report.

**Status, 2026-08-13: (b) and (d) are applied. Everything else is not**, and each lever below says which it is. The original configuration was deliberately biased the *other* way, towards earned disagreement, and every debate run against it ended in DEADLOCK. That is the reason for the change: not that disagreement is bad, but that a system which cannot express the other two outcomes is not reporting a result, it is reporting its own prompt. This document exists because "make them agree more" is a reasonable thing to want from a deliberation system, and because the obvious way to do it is the wrong one.

> **A note on the sibling repo.** Two of the levers below (the tally rule and the judge) sit in files held by contract with `magi-system`. Changing them here alone silently stops the benchmark from being a comparison. See [Relationship to sibling projects](../CLAUDE.md#relationship-to-sibling-projects).

## The distinction that governs everything else

There are two different things that look identical in the output:

**Convergence.** An advisor reads the others, finds an argument better than its own, and changes position. Three advisors then hold the same view for three sets of reasons. This is what the system is for.

**Agreement inflation.** An advisor stops contributing: it echoes, hedges, or ticks `agrees_with` without engaging, and the tally scores that as consensus. Three identical answers carry exactly as much information as one.

Both produce `UNANIMOUS`. Only one is worth the two to four minutes of three models' time that produced it. Every lever below is classified by which it tends to produce, and that classification is the actual content of this document.

## The levers, by effect size

### (a) The tally rule: largest effect, zero LLM cost — NOT APPLIED

`magi/orchestrator/consensus.py`, `_agreement_blocs()`.

Agreement is currently an edge in an undirected graph, and the edge exists only if **both** advisors claim it. "A agrees with B" while B says nothing about A is one advisor being agreeable, not two converging.

Three progressively looser rules, in order of how much consensus they buy:

| Rule | Effect |
|---|---|
| Mutual edges (current) | A cycle A→B→C→A yields no bloc at all. Strictest |
| One-directional edges | A single agreeable advisor can pull a bloc together on its own |
| Transitive closure over one-directional claims | Almost any partial agreement becomes unanimity |

#### What that means concretely

Take one real shape of final round. MELCHIOR lists `agrees_with: [BALTHASAR]`, BALTHASAR lists `agrees_with: [CASPAR]`, CASPAR lists `agrees_with: [MELCHIOR]`. Every advisor named someone. Nobody was named back by the advisor they named.

Under the three rules the same votes tally differently:

| Rule | Blocs | Outcome |
|---|---|---|
| Mutual (current) | `{M}`, `{B}`, `{C}` | DEADLOCK |
| One-directional | `{M, B, C}` | UNANIMOUS |
| Transitive closure | `{M, B, C}` | UNANIMOUS |

Nothing about the models changed between those three rows. The advisors said exactly the same words; only the arithmetic reading them changed. That is the whole lever, and it is why it costs nothing and moves the most.

The second rule differs from the third only when the claims do not form a cycle. With `M → B` and `B → C` and nothing else, one-directional edges give `{M, B, C}` too, since it is still one connected component of the undirected graph; transitive closure only starts to matter on directed rules that ask "does A reach C". In practice with three advisors the useful distinction is the first row against the other two, and the honest middle option is **one-directional edges**.

The argument for changing it: the current rule requires two advisors to name each other *in the same round*, which is a coordination problem the advisors cannot see. An advisor writing its turn does not know whether the others will name it back, so a bloc forms only when both happen to reciprocate simultaneously. Three rounds is very few attempts at that.

The argument against: a one-directional edge really can be one agreeable advisor pulling a bloc together on its own, and with the vote bar now lowered by (d) that is more likely than it was. If both are applied at once and `UNANIMOUS` starts firing everywhere, this is the one to revert first, because it is the one that changed the meaning of the word rather than what the models were asked.

This lever is honest in a way the prompt levers are not: it changes what the system *counts* as consensus without changing what the models say, so the transcript still shows exactly what happened and a reader can disagree with the arithmetic. It is also the one to reach for first if the complaint is "UNANIMOUS almost never fires", because the current rule is strict enough that genuinely converged debates can still tally as MAJORITY.

**Verdict: produces convergence, or at least reports it faithfully.** The risk is over-reporting, not fabrication.

> Shared contract with `magi-system`. Land it in both repos in the same change window, or the benchmark is measuring the tally rather than the framework.

### (b) The judge's threshold: large effect, one LLM call — APPLIED 2026-08-13

`magi/orchestrator/prompts.py`, `JUDGE_SYSTEM_PROMPT`.

The judge runs only on a split vote and answers one question: is the disagreement substantive, or the same answer worded differently? A COSMETIC ruling promotes the outcome to `UNANIMOUS` (`magi/orchestrator/magi.py`, in `debate()`).

The criterion was strict and action-based:

> Answer SUBSTANTIVE if acting on one position would lead to a different decision than acting on another. Answer COSMETIC only if the positions would lead to the same action and differ in emphasis, vocabulary or framing.

It now reads:

> Answer SUBSTANTIVE if acting on one position would lead to a materially different outcome than acting on another. Answer COSMETIC if the positions would lead to broadly the same course of action and differ in emphasis, vocabulary, framing or degree.

Three words carry the change. "Materially different outcome" replaces "different decision", so a difference that exists but does not matter no longer counts; "broadly the same course of action" replaces "the same action", so the positions need not be identical in method; and "degree" joins the list of differences that are cosmetic, which is the one that most often applied and had nowhere to go. The dropped "only" matters too: it was reading as an instruction to look for a reason to rule SUBSTANTIVE.

This is the best-instrumented lever in the system: `judged_cosmetic` is recorded on every `DebateRecord`, so the change can be quantified after the fact rather than argued about. Before touching anything else, it is worth querying how often the judge already runs and how often it already promotes.

**Verdict: produces reportable agreement, and admits it.** The flag on the row is what keeps it honest: a debate promoted by the judge stays distinguishable from one that converged on its own, forever.

### (c) The seed prompt's ordering: large effect, and a trap — NOT APPLIED, AND SHOULD NOT BE

`magi/orchestrator/prompts.py`, `deliberation_seed()`.

The seed currently forces a specific order: critique the others first, state your own position second, and require that position to contain something nobody else said.

That ordering is load-bearing anti-collapse machinery, not style. It was added after a measured failure: with three agents on one shared thread and no such constraint, the second and third speakers reproduced the first one's answer **verbatim**, which the tally scored as an unusually strong consensus. Telling them not to copy changed nothing; changing the order of the task did.

Inverting it (position first, critique optional) is the most direct route to a higher `UNANIMOUS` rate, and it is the one route this document recommends against. It does not make the advisors agree. It makes them stop reading each other.

**Verdict: pure agreement inflation.** Avoid.

The seed's *ordering* is untouched and stays untouched. Its `agrees_with` instruction is not: it repeated the strict bar ("advisors whose position you would defend as your own"), which would have overridden the relaxed `common_prompt` by virtue of arriving last. See (d).

### (d) The prompts: moderate effect, cheap to try — APPLIED 2026-08-13

`config/magi.yaml`, and the `agrees_with` clause of `deliberation_seed()` in `magi/orchestrator/prompts.py`.

Three separate surfaces, in descending order of leverage. What each one said, and what was done with it:

**1. The `common_prompt` premise and the vote bar. Changed.** It opened with "you are expected to disagree when you actually disagree, agreement that is not earned is worth nothing here", and gated the vote on positions "you would defend, not merely tolerate". Two advisors reaching the same conclusion by different routes would each withhold the vote because the other's phrasing was not theirs.

The premise now says the goal is a conclusion all three can stand behind, and asks them to disagree when the difference would change what someone should do. The bar for `agrees_with` is now the same test the judge applies in (b): would acting on their position lead somewhere materially different from acting on mine? Different wording, different emphasis and an added caveat are explicitly not disagreement. A sentence was added saying that agreeing does not mean giving up your own angle, because the seed still requires each position to contain something nobody else said and an advisor should not read those two as being in tension.

**The identical clause in `deliberation_seed()` was changed to match**, and this is the part that would have silently undone the rest. The seed arrives after the system prompt and immediately before the model writes, so a relaxed `common_prompt` under an unchanged seed is a relaxed rule the model reads first and a strict one it reads last. It now also says out loud that criticising an advisor in `critique` and agreeing with them in `agrees_with` is a normal combination, which under the old wording read as a contradiction.

**2. CASPAR's manufactured-dissent instruction. Softened, not removed.** It read "Argue the opposing case even when you privately lean the other way", one sentence before "do not manufacture objections", which is close to a contradiction and resolves in favour of dissent. CASPAR is by construction the seat most likely to be the lone dissenter in a MAJORITY.

The case against survives intact: CASPAR still hunts unstated assumptions, failure modes and second-order effects, which is the entire reason the seat exists. What was removed is the standing instruction to *hold* a position it does not believe, replaced with the explicit converse: a risk it has named and judged acceptable is a reason to agree, not a reason to abstain. That is the difference between a skeptic and a contrarian, and only the second one was costing outcomes.

**3. The mind-changing rule. Kept, reframed.** "Changing your mind for a good reason is the point of this process. Changing it to end the debate is a failure." The second half stays, because it is the sentence that keeps convergence honest. The first half now says being persuaded by a better argument is not a concession, which is the same rule with the shame taken out of it.

Note that (1) and (2) differ in kind. Relaxing the `agrees_with` bar changes how an advisor *reports* a position it already holds. Changing CASPAR's instruction changes what position it holds at all, which is why it got the more conservative edit of the two.

**Verdict: (1) is reportable agreement; (2) is genuine, and a full removal would have cost you the skeptic.** Removing the case against is not a tuning change, it is a decision to run a two-perspective system with three models, and that is not what was done here.

**If this needs reverting**, revert (2) first and (1) second. (2) changes what the debate contains; (1) only changes how it is scored, and (1) is the one the DEADLOCK problem actually pointed at.

### (e) Sampling and budget: small effect, no prose changes — NOT APPLIED

`config/magi.yaml`, and `MAGI_MAX_ROUNDS`.

- **Temperature.** CASPAR runs at 0.8, the highest of the three, which mechanically raises how many novel objections it generates. Lowering it toward 0.3 narrows the search and reduces dissent, but it also reduces the quality of the dissent that remains, which is most of what CASPAR is for.
- **Rounds.** More deliberation rounds means more opportunities to converge. The confound: `MAGI_MAX_ROUNDS` feeds `MaxMessageTermination` *and* the number of chances, so raising it changes two variables at once and makes the resulting convergence rate uninterpretable. If you raise it to measure convergence, hold the message budget fixed separately.

**Verdict: genuine but weak, and the rounds knob confounds its own measurement.**

## What is not available today

Two things worth knowing before designing an experiment around this.

**There is no agreement gradient.** `agrees_with` is hard set membership. An advisor cannot say "mostly, except on the cost argument". Every lever above therefore operates on a binary, and "agree when the difference is small" is not expressible in the current schema.

**`confidence` is collected and never used.** It is on every `MagiTurn`, it is shown in the deliberation seed, and `tally()` ignores it entirely. A rule like "a low-confidence dissent does not block consensus" is one of the more interesting things to try here, and it needs no prompt changes at all, only a change to `tally()`.

Both are `MagiTurn` questions, and `MagiTurn` is prompt surface, not plumbing: its field names and descriptions are handed to a 12B model as a JSON schema, so adding a field changes behaviour beyond the arithmetic that reads it. See the module docstring in `magi/models.py`.

Both are now queued together as [candidates.md § 6](candidates.md#6-an-agreement-gradient-and-actually-using-confidence), which is the structural answer to the same problem the applied levers address by rewording. It replaces a threshold stated three times in prose with one number that can be swept and recorded.

## What to do next

The original advice here was "if you only do one thing, change the judge's threshold (b), because it is one sentence and it is already instrumented". That is done, along with (d), and the two were applied together deliberately: (b) only fires on a split vote, so on a roster that deadlocked every time it is the lever with the fewest opportunities to act, and (d) is what produces the split votes for it to act on.

The next step is measurement, not another lever. Re-run the fixed set of contested questions and read `judged_cosmetic` against the `UNANIMOUS` rate: a rise that shows up in `judged_cosmetic` is the judge being more permissive, a rise that does not is the advisors actually converging, and those are two different results that should never be averaged.

Only if debates still deadlock uniformly after that, move to (a), which is the largest remaining effect and the only one that costs nothing per debate.

## Measuring whether it worked

Whatever you change, the question is not "did `UNANIMOUS` go up". It is whether the debates still produce information. Three checks:

1. **Contested questions must still reach MAJORITY and DEADLOCK.** A configuration that reaches `UNANIMOUS` on a genuinely contested question has broken, not improved. Keep a fixed set of such questions and re-run them.
2. **Read the positions for novelty.** Mode collapse is visible to the eye and invisible to the tally: if the three `position` fields in the final round share sentences, the consensus is an echo. This is what the `position` novelty requirement in `deliberation_seed()` exists to prevent.
3. **Split `judged_cosmetic` out of the `UNANIMOUS` rate.** They answer different questions and averaging them hides which lever moved.

`DebateRecord` carries `terminated_by`, `judged_cosmetic`, `rounds_used` and `outcome`, which is enough for all three without adding instrumentation.
