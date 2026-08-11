"""Counting LLM calls and tokens.

**Counted at the model client, not at the message.** ``models_usage`` on chat
messages only covers turns that produced a message, which misses exactly the
calls that matter most: ``SelectorGroupChat`` makes its speaker-selection call
internally, and that call is the single thing the roundrobin-vs-selector
comparison exists to measure. Counting anywhere else would understate the cost
of the feature under test.

The counter is deliberately dumb — it accumulates, and something else decides
what a debate's numbers mean. The SQLite record and the OTel spans both read
from it rather than each computing their own version.
"""

from __future__ import annotations

import resource
import sys
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ModelUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class CallCounter:
    """Per-debate totals, plus a breakdown per advisor.

    The breakdown is what shows a reasoning advisor costing several times its
    neighbours — a fact that is invisible in a single total and that decides
    whether a model belongs on a voice interface.
    """

    total: ModelUsage = field(default_factory=ModelUsage)
    by_label: dict[str, ModelUsage] = field(default_factory=lambda: defaultdict(ModelUsage))

    def record(self, label: str, prompt_tokens: int, completion_tokens: int) -> None:
        for bucket in (self.total, self.by_label[label]):
            bucket.calls += 1
            bucket.prompt_tokens += prompt_tokens
            bucket.completion_tokens += completion_tokens

    def reset(self) -> None:
        self.total = ModelUsage()
        self.by_label.clear()

    def snapshot(self) -> dict[str, ModelUsage]:
        """A plain dict copy, safe to store after the counter moves on."""
        return {label: ModelUsage(u.calls, u.prompt_tokens, u.completion_tokens)
                for label, u in self.by_label.items()}


# ── Process cost ─────────────────────────────────────────────────────────────


@dataclass
class ProcessStats:
    """What this process cost the node, as opposed to what it cost the Spark.

    The whole premise of the project is that the interesting number is here and
    not in the token count: the Spark generates the tokens either way, so the
    engines' wall-clock barely moves between a MacBook and a Pi. What does move
    is the CPU and memory the framework's runtime consumes on a 4-core ARM
    board, and that is what nobody publishes.
    """

    user_s: float
    system_s: float
    peak_rss_mb: float

    @property
    def cpu_s(self) -> float:
        return self.user_s + self.system_s


def process_stats() -> ProcessStats:
    """Peak RSS and CPU time for this process, via ``getrusage``.

    Deliberately not ``/usr/bin/time -v``: GNU time is not installed on a stock
    Debian 13, and a measurement that depends on an optional package is one the
    Pi cannot be trusted to reproduce. ``ru_maxrss`` is kilobytes on Linux and
    bytes on macOS — a units trap that would quietly report a 1000x difference
    between the dev machine and the node being measured.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return ProcessStats(
        user_s=usage.ru_utime,
        system_s=usage.ru_stime,
        peak_rss_mb=usage.ru_maxrss / divisor,
    )
