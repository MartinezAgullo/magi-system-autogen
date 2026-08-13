"""``python -m magi.preflight`` — run the checks and report.

Exit codes: ``0`` all clear (warnings included), ``1`` at least one hard error,
``2`` the configuration itself is unusable (bad or missing persona file).

Warnings deliberately do not fail the run. The distinction is the whole point:
a missing model tag is a fact that stops the node working, while a model that
is not resident only means the latency numbers deserve an asterisk.
"""

from __future__ import annotations

import asyncio
import sys

from magi import personas as personas_mod
from magi.config import Settings
from magi.services.ollama_check import PreflightReport, run_preflight
from magi.store.debates import DebateStore

GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
CYAN = "\033[0;36m"
DIM = "\033[2m"
NC = "\033[0m"

MARKS = {"ok": (GREEN, "✓"), "warn": (YELLOW, "⚠"), "error": (RED, "✗")}


def _print(report: PreflightReport) -> None:
    for result in report.results:
        colour, mark = MARKS[result.level]
        print(f"  {colour}{mark}{NC}  {result.message}")
        for hint in result.hints:
            print(f"      {DIM}{hint}{NC}")


async def _record(settings: Settings, report: PreflightReport) -> None:
    """Leave the verdict where the daemon can find it.

    This runs as its own process, seconds before the node starts, so the
    database is the only channel between the two — and the fact worth passing
    along is model residency on the inference host, which decides whether the
    latency figures a run produces are about the framework or about the Spark's
    disk. Every debate row then carries it, and a suspect run can be excluded
    from the benchmark afterwards instead of silently widening the averages.

    Best effort, deliberately. Pre-flight's job is to say whether the node can
    work; failing it over a bookkeeping write would be the check causing the
    outage it exists to prevent.
    """
    try:
        store = DebateStore(settings.db_path)
        await asyncio.to_thread(
            store.record_preflight_sync,
            settings.node_id,
            failed=report.failed,
            warned=report.warned,
            residency_warning=report.residency_warning,
            detail=report.residency_detail(),
        )
    except Exception as exc:  # noqa: BLE001 — nothing here is worth failing over
        print(f"      {DIM}(could not record this run in {settings.db_path}: {exc}){NC}")


async def _run() -> int:
    settings = Settings()

    print(f"\n{CYAN}▸ MAGI pre-flight{NC}")
    print(f"  {DIM}engine={settings.engine}  backend={settings.llm_backend}  "
          f"personas={settings.personas_file.name}{NC}\n")

    try:
        personas = personas_mod.load(settings.personas_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"  {RED}✗{NC}  {settings.personas_file}: {exc}")
        print(f"      {DIM}Nothing else can be checked until the advisors load.{NC}\n")
        return 2

    report = await run_preflight(settings, personas)
    _print(report)
    await _record(settings, report)
    print()

    if report.failed:
        print(f"  {RED}Pre-flight failed.{NC} Fix the errors above, "
              f"or run ./launch.sh --skip-checks to start anyway.\n")
        return 1
    if report.warned:
        print(f"  {YELLOW}Ready, with warnings.{NC}\n")
    else:
        print(f"  {GREEN}Ready.{NC}\n")
    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
