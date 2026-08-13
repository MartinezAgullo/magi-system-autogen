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

import asyncio
import contextlib
import logging
import resource
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import psutil

from magi.models import NodeCost

logger = logging.getLogger(__name__)


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

#: ``ru_maxrss`` is kilobytes on Linux and bytes on macOS. Getting this wrong
#: reports a 1000x difference between the dev machine and the node under test,
#: in the direction that flatters whichever one you were not watching.
_MAXRSS_DIVISOR = 1024 * 1024 if sys.platform == "darwin" else 1024


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
    return ProcessStats(
        user_s=usage.ru_utime,
        system_s=usage.ru_stime,
        peak_rss_mb=usage.ru_maxrss / _MAXRSS_DIVISOR,
    )


def _cpu_seconds() -> float:
    """User + system CPU consumed by this process so far, all threads included."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


class ProcessSampler:
    """Watches what one debate costs the node, for the duration of the debate.

    Two measurements, taken two different ways on purpose:

    * **CPU** is the delta of ``getrusage`` across the window. Exact, and free —
      sampling it would only add error to a number the kernel is already
      accumulating.
    * **RSS** has to be sampled, because there is no per-window peak to read.
      ``ru_maxrss`` is the high-water mark for the life of the process, so after
      an hour of uptime it answers "how big has this node ever been", which is
      not the question. ``psutil`` gives the current value; the peak and the mean
      over the debate come from the series.

    Runs as one asyncio task doing a couple of ``/proc`` reads per second. That
    is deliberately not put on a thread: at this rate a thread hand-off costs
    more than the read, on the very CPU budget being measured.

    ``enabled=False`` is a genuine no-op — no task, no samples, and
    :meth:`cost` returns ``None`` rather than a row of zeros. Instrumentation
    that cannot be switched off is not measurable itself, and the headline
    numbers are taken with as little of it running as possible.
    """

    def __init__(self, interval_s: float = 1.0, *, enabled: bool = True) -> None:
        self._interval_s = max(interval_s, 0.05)
        self._enabled = enabled
        self._task: asyncio.Task[None] | None = None
        self._process: psutil.Process | None = None
        self._rss_mb: list[float] = []
        self._started_cpu_s = 0.0
        self._started_at = 0.0

    async def __aenter__(self) -> ProcessSampler:
        if not self._enabled:
            return self
        self._started_cpu_s = _cpu_seconds()
        self._started_at = time.monotonic()
        self._process = psutil.Process()
        # One reading before the loop, so a debate shorter than the interval —
        # a failed one, or a barge-in — still reports a measurement rather than
        # an empty series that reads as "no memory used".
        self._sample()
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self._sample()

    def _sample(self) -> None:
        if self._process is None:
            return
        try:
            self._rss_mb.append(self._process.memory_info().rss / (1024 * 1024))
        except psutil.Error:
            # The process cannot read its own memory info on some hardened
            # kernels. Losing the series is a gap in the benchmark, not a reason
            # to lose the debate that was being measured.
            logger.debug("RSS sample failed", exc_info=True)

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            self._sample()

    def cost(self) -> NodeCost | None:
        """The reading so far. Valid during the debate as well as after it.

        Callable mid-run because the record is assembled inside the debate's own
        span, before the sampler's context manager has exited. The alternative —
        exiting first and patching the record afterwards — is a record that is
        briefly wrong, and a record that is briefly wrong eventually gets read.
        """
        if not self._enabled:
            return None

        elapsed = max(time.monotonic() - self._started_at, 1e-9)
        cpu_s = _cpu_seconds() - self._started_cpu_s
        return NodeCost(
            cpu_s=cpu_s,
            cpu_percent=100.0 * cpu_s / elapsed,
            peak_rss_mb=max(self._rss_mb, default=0.0),
            mean_rss_mb=sum(self._rss_mb) / len(self._rss_mb) if self._rss_mb else 0.0,
            samples=len(self._rss_mb),
        )
