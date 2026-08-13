#!/usr/bin/env python
"""Run one debate from the command line.

The whole system minus the microphone and the speaker, which is what you want
while the debate itself is the thing under development.

    uv run python scripts/ask.py "Should a three-person startup adopt Kubernetes?"
    uv run python scripts/ask.py --engine autogen_selector "..."
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from magi import personas as personas_mod
from magi.bus import Bus
from magi.config import Settings
from magi.constants import PREFLIGHT_FRESH_S, TERMINATED_BY_CONSENSUS
from magi.models import Outcome
from magi.orchestrator import Magi
from magi.services.stream_view import render_stream
from magi.setup.setup_logs import setup_logging
from magi.setup.setup_tracing import setup_tracing, shutdown_tracing
from magi.store.debates import DebateStore

GREEN, YELLOW, RED, CYAN, DIM, NC = (
    "\033[0;32m", "\033[1;33m", "\033[0;31m", "\033[0;36m", "\033[2m", "\033[0m"
)
COLOURS = {Outcome.UNANIMOUS: GREEN, Outcome.MAJORITY: YELLOW, Outcome.DEADLOCK: RED}


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("-q", "--quiet", action="store_true", help="verdict only")
    parser.add_argument("--no-trace", action="store_true",
                        help="run uninstrumented, as a benchmark run must")
    parser.add_argument("--no-store", action="store_true",
                        help="do not write the debate to the benchmark database")
    args = parser.parse_args()

    overrides = {}
    if args.engine:
        overrides["engine"] = args.engine
    if args.rounds:
        overrides["max_rounds"] = args.rounds
    if args.no_trace:
        overrides["otel_enabled"] = False
    settings = Settings(**overrides)

    setup_logging("WARNING" if args.quiet else settings.log_level)
    personas = personas_mod.load(settings.personas_file)
    provider = setup_tracing(settings)

    # A bus only so the token stream has somewhere to arrive. Advisors with
    # `stream: false` — all of them, by default — publish nothing here, so this
    # costs one queue and no output unless the persona file asked for it.
    bus = Bus()
    streamer = asyncio.create_task(render_stream(bus, quiet=args.quiet))

    # Runs from here land in the same database as the daemon's, which is the
    # point: a benchmark set is normally a loop over this script, and a run that
    # was not recorded is a run that has to be repeated. `--no-store` is for
    # trying a prompt out without polluting the record.
    store = None if args.no_store else DebateStore(settings.db_path)
    if store is not None:
        await store.open()
        # Nothing has recorded pre-flight for a bare `ask.py` unless the operator
        # ran it, so residency stays whatever the last recent check found.
        run = await store.latest_preflight(settings.node_id)
        residency = (
            run.residency_warning
            if run is not None and run.age_s <= PREFLIGHT_FRESH_S
            else None
        )
    else:
        residency = None

    try:
        magi = Magi(
            settings, personas, bus=bus, tracer_provider=provider,
            store=store, residency_warning=residency,
        )
        record = await magi.debate(" ".join(args.question))
    finally:
        streamer.cancel()
        # Spans are exported in batches, so without this the run you are
        # waiting to look at is exactly the one that never reaches the backend.
        shutdown_tracing()

    colour = COLOURS[record.outcome]
    # Only worth the words when turns were generated, paid for and then left out
    # of the arithmetic because nobody answered them. Not on the consensus path:
    # there the outcome rests on each advisor's latest vote, and `tallied_round`
    # is a floor on how often everyone spoke rather than the row that was read.
    cut = ("" if record.terminated_by == TERMINATED_BY_CONSENSUS
           or record.tallied_round >= record.rounds_used
           else f", decided on round {record.tallied_round}")
    print(f"\n{colour}▸ {record.outcome.value}{NC}  "
          f"{DIM}{record.rounds_used} rounds{cut}, {record.duration_s:.1f}s, "
          f"stopped by {record.terminated_by}{NC}")
    selector = (f", {record.selector_calls} of them turn-selection"
                if record.selector_calls else "")
    print(f"  {DIM}{record.engine}: {record.llm_calls} LLM calls{selector}, "
          f"{record.prompt_tokens}+{record.completion_tokens} tokens, "
          f"tracing {'on' if record.tracing_enabled else 'off'}{NC}")
    # The node's own cost, which is the number this project exists to produce:
    # the Spark generates the tokens either way, so only this moves between a
    # laptop and a Pi. Measured over the debate rather than read off the process
    # at the end, so a long-lived daemon reports this debate and not its uptime.
    if record.node is not None:
        print(f"  {DIM}node: {record.node.cpu_s:.1f}s CPU "
              f"({record.node.cpu_percent:.0f}% of one core), "
              f"{record.node.peak_rss_mb:.0f} MB peak RSS "
              f"({record.node.samples} samples){NC}")
    # Both say the outcome had help, and they are not the same help: one
    # repaired agreement two advisors had already declared one-sidedly, the
    # other promoted a whole split vote. An outcome with neither is the only one
    # the advisors reached on their own.
    if record.judged_edges:
        print(f"  {DIM}(the judge repaired {record.judged_edges} one-sided "
              f"agreement claim(s)){NC}")
    if record.judged_cosmetic:
        print(f"  {DIM}(the judge ruled the disagreement cosmetic){NC}")
    print(f"\n{record.verdict.answer}\n")
    if record.verdict.dissent:
        print(f"{YELLOW}Dissent:{NC} {record.verdict.dissent}")
    if record.verdict.disagreement_point:
        print(f"{RED}Split on:{NC} {record.verdict.disagreement_point}")

    if not args.quiet:
        # The tallied row, not the whole transcript: `record.turns` now carries
        # every turn of every round, and the ones that decided the outcome are
        # the ones worth reading next to it.
        print(f"\n{CYAN}Final positions{NC}  {DIM}(the row the outcome was "
              f"decided on){NC}")
        for turn in (t for t in record.turns if t.tallied):
            print(f"  {turn.advisor:<10} {DIM}r{turn.round_index} agrees_with="
                  f"{turn.turn.agrees_with or '[]'} conf={turn.turn.confidence:.1f}{NC}")
            print(f"    {turn.turn.summary.strip()}")
        if store is not None:
            print(f"\n{DIM}Recorded in {store.path} as {record.debate_id}{NC}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
