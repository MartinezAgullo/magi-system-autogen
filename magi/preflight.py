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
