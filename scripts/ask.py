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
from magi.models import Outcome
from magi.orchestrator import Magi
from magi.services.metrics import process_stats
from magi.services.stream_view import render_stream
from magi.setup.setup_logs import setup_logging
from magi.setup.setup_tracing import setup_tracing, shutdown_tracing

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

    try:
        magi = Magi(settings, personas, bus=bus, tracer_provider=provider)
        record = await magi.debate(" ".join(args.question))
    finally:
        streamer.cancel()
        # Spans are exported in batches, so without this the run you are
        # waiting to look at is exactly the one that never reaches the backend.
        shutdown_tracing()

    colour = COLOURS[record.outcome]
    print(f"\n{colour}▸ {record.outcome.value}{NC}  "
          f"{DIM}{record.rounds_used} rounds, {record.duration_s:.1f}s, "
          f"stopped by {record.terminated_by}{NC}")
    stats = process_stats()
    selector = (f", {record.selector_calls} of them turn-selection"
                if record.selector_calls else "")
    print(f"  {DIM}{record.engine}: {record.llm_calls} LLM calls{selector}, "
          f"{record.prompt_tokens}+{record.completion_tokens} tokens, "
          f"tracing {'on' if record.tracing_enabled else 'off'}{NC}")
    # The node's own cost, which is the number this project exists to produce:
    # the Spark generates the tokens either way, so only this moves between a
    # laptop and a Pi.
    print(f"  {DIM}node: {stats.cpu_s:.1f}s CPU "
          f"({100 * stats.cpu_s / record.duration_s:.0f}% of one core), "
          f"{stats.peak_rss_mb:.0f} MB peak RSS{NC}")
    if record.judged_cosmetic:
        print(f"  {DIM}(the judge ruled the disagreement cosmetic){NC}")
    print(f"\n{record.verdict.answer}\n")
    if record.verdict.dissent:
        print(f"{YELLOW}Dissent:{NC} {record.verdict.dissent}")
    if record.verdict.disagreement_point:
        print(f"{RED}Split on:{NC} {record.verdict.disagreement_point}")

    if not args.quiet:
        print(f"\n{CYAN}Final positions{NC}")
        for turn in record.turns:
            print(f"  {turn.advisor:<10} {DIM}agrees_with="
                  f"{turn.turn.agrees_with or '[]'} conf={turn.turn.confidence:.1f}{NC}")
            print(f"    {turn.turn.summary.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
