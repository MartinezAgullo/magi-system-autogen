"""What a debate costs the node it runs on.

This is the number the project exists to produce. The Spark generates the tokens
either way, so wall-clock barely moves between a MacBook and a Pi; what moves is
the CPU and the resident memory the framework's runtime consumes on a 4-core ARM
board, and a claim about framework overhead that is never measured is an
opinion.

Two properties are worth pinning down, and they are the two that would fail
silently. A measurement window has to be a window — reading ``getrusage`` at the
end reports the life of the process, which after an hour of uptime is a
different question. And switching the instrumentation off has to mean nothing
runs, on a node whose CPU consumption is the object of study.
"""

from __future__ import annotations

import asyncio

from magi.services.metrics import ProcessSampler


async def test_a_disabled_sampler_measures_nothing_and_says_so():
    """MAGI_METRICS_ENABLED=0 has to be a genuine no-op: no task, no samples,
    and None rather than a row of zeros. A zero CPU reading averages into a
    benchmark as a node that used no CPU; None does not average at all, which is
    the correct behaviour for a measurement nobody took."""
    async with ProcessSampler(0.05, enabled=False) as sampler:
        await asyncio.sleep(0.15)

    assert sampler.cost() is None


async def test_a_debate_shorter_than_the_sample_interval_still_gets_a_reading():
    """A barge-in, or a debate that died in the blind round. One reading is
    taken before the loop starts precisely so these do not report an empty
    series, which reads as "used no memory" rather than as "ended quickly"."""
    async with ProcessSampler(60.0) as sampler:
        pass

    cost = sampler.cost()
    assert cost is not None
    assert cost.samples >= 1
    assert cost.peak_rss_mb > 0


async def test_the_series_grows_while_the_debate_runs():
    """RSS has to be sampled rather than read once: ru_maxrss is the high-water
    mark for the life of the process, so there is no per-debate peak to look
    up."""
    async with ProcessSampler(0.05) as sampler:
        await asyncio.sleep(0.3)

    cost = sampler.cost()
    assert cost is not None
    assert cost.samples >= 3
    assert cost.peak_rss_mb >= cost.mean_rss_mb > 0


async def test_the_cost_is_readable_while_the_debate_is_still_running():
    """The record is assembled inside the debate's own span, before the sampler
    has exited. The alternative — exit first, then patch the record — leaves a
    record that is briefly wrong, and a record that is briefly wrong eventually
    gets read."""
    async with ProcessSampler(0.05) as sampler:
        await asyncio.sleep(0.15)
        during = sampler.cost()
        assert during is not None
        assert during.samples >= 1
    after = sampler.cost()

    assert after is not None
    assert after.samples >= during.samples


async def test_cpu_is_measured_over_the_window_and_not_since_boot():
    """The trap this exists for: `getrusage` is cumulative for the process. A
    node that has been up for an hour would report an hour of CPU against a
    two-minute debate, and the resulting percentage would be nonsense in the
    direction that makes the framework look expensive."""
    # Burn a little CPU before the window opens. None of it belongs to the
    # debate, and all of it is in getrusage by the time the sampler starts.
    sum(index * index for index in range(400_000))

    async with ProcessSampler(0.05) as sampler:
        await asyncio.sleep(0.1)

    cost = sampler.cost()
    assert cost is not None
    # An idle debate spends almost no CPU on the node — which is the actual
    # finding on the Pi, where the node is blocked on HTTP to the Spark for the
    # whole debate. Anything near a full core here means the pre-window work
    # leaked into the measurement.
    assert cost.cpu_s < 0.05
    assert cost.cpu_percent < 50


async def test_the_sampler_stops_when_the_debate_does():
    """It runs for the duration of a debate, not for the life of the node. A
    task left behind would keep sampling between questions and quietly spend the
    CPU budget it exists to measure."""
    sampler = ProcessSampler(0.05)
    async with sampler:
        await asyncio.sleep(0.1)
    settled = sampler.cost()
    assert settled is not None

    await asyncio.sleep(0.2)

    assert sampler.cost().samples == settled.samples
