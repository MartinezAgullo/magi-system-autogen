#!/usr/bin/env python
"""Probe every model on the inference host for structured-output compliance.

Pre-flight checks the three models you have already chosen. This checks *all* of
them, so you can pick with evidence rather than by reputation — the whole design
rests on every advisor honouring a JSON schema, and per-model compliance is the
one thing you cannot infer from a model card.

    uv run python scripts/probe_models.py
    uv run python scripts/probe_models.py --host 192.168.68.121

Reports pass/fail and the round-trip time per model. The timing is indicative
only: a cold model load dominates it, so a slow first result usually means the
weights were not resident, not that the model is slow.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

import httpx

from magi.config import Settings
from magi.constants import API_TIMEOUT_S
from magi.services.ollama_check import list_tags, probe_structured_output

GREEN, YELLOW, RED, DIM, NC = "\033[0;32m", "\033[1;33m", "\033[0;31m", "\033[2m", "\033[0m"

# Embedding models cannot chat at all, so failing them says nothing useful.
# Substring match rather than an exact list: the point is to skip the obvious,
# not to maintain a registry of every embedder in existence.
NOT_CHAT_MODELS = ("bge-", "embed", "nomic-")


async def main() -> int:
    settings = Settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings.ollama_host)
    parser.add_argument("--port", type=int, default=settings.ollama_port)
    args = parser.parse_args()

    base = f"http://{args.host}:{args.port}"
    print(f"\n▸ Structured-output probe against {base}\n")

    async with httpx.AsyncClient() as client:
        try:
            tags = sorted(await list_tags(client, base))
        except httpx.HTTPError as exc:
            print(f"  {RED}✗{NC}  Cannot reach Ollama at {base}: {exc}")
            return 1

        try:
            running = {
                m["name"]
                for m in (
                    await client.get(f"{base}/api/ps", timeout=API_TIMEOUT_S)
                ).json().get("models", [])
            }
        except httpx.HTTPError:
            running = set()

        for tag in tags:
            if any(marker in tag for marker in NOT_CHAT_MODELS):
                print(f"  {DIM}—  {tag:<22} skipped (not a chat model){NC}")
                continue

            resident = " " if tag in running else "*"
            started = time.monotonic()
            ok, detail = await probe_structured_output(
                client, f"{base}/v1", "ollama", tag
            )
            elapsed = time.monotonic() - started

            mark = f"{GREEN}✓{NC}" if ok else f"{RED}✗{NC}"
            print(f"  {mark}{resident} {tag:<22} {elapsed:6.1f}s  {detail}")

    print(f"\n  {DIM}* not resident when probed — the time includes a cold load{NC}")
    print(f"  {YELLOW}Pick advisors from different lineages:{NC} three models from one "
          f"family agreeing proves nothing.\n")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
