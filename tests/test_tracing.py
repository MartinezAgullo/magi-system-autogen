"""Tracing setup and call counting.

The property that matters most is the negative one: with tracing disabled there
must be no provider and no recording spans. This node's CPU and memory are the
object of study, so a benchmark run has to be genuinely uninstrumented rather
than instrumented-and-discarding.
"""

from __future__ import annotations

import json

import pytest

from magi.config import Settings
from magi.services.metrics import CallCounter
from magi.setup import setup_tracing as tracing_mod
from magi.setup.setup_tracing import (
    JsonLinesSpanExporter,
    get_tracer,
    setup_tracing,
    shutdown_tracing,
)


@pytest.fixture(autouse=True)
def _clean_provider():
    """Tracing is process-global, so leaking a provider between tests would make
    later ones assert against an earlier one's state."""
    tracing_mod._provider = None
    yield
    tracing_mod._provider = None


def test_disabled_returns_no_provider(tmp_path):
    settings = Settings(otel_enabled=False, db_path=tmp_path / "magi.db")

    assert setup_tracing(settings) is None


def test_spans_do_not_record_when_tracing_was_never_set_up():
    """Callers never branch on whether tracing is on — they open spans either
    way — so a no-op span has to be safe and cheap."""
    with get_tracer().start_as_current_span("magi.debate") as span:
        span.set_attribute("magi.outcome", "UNANIMOUS")

        assert not span.is_recording()


def test_file_exporter_writes_one_json_object_per_span(tmp_path):
    settings = Settings(
        otel_enabled=True,
        otel_exporter="file",
        otel_file_path=tmp_path / "traces.jsonl",
        db_path=tmp_path / "magi.db",
    )
    provider = setup_tracing(settings)
    assert provider is not None

    with get_tracer().start_as_current_span("magi.debate") as span:
        span.set_attribute("magi.outcome", "DEADLOCK")

    shutdown_tracing()

    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["name"] == "magi.debate"


def test_file_exporter_survives_an_unwritable_path(tmp_path):
    """A trace sink that cannot be written must not take the debate down with
    it — losing spans is acceptable, losing the answer is not."""
    exporter = JsonLinesSpanExporter(tmp_path / "traces.jsonl")
    exporter._path = tmp_path / "no" / "such" / "dir" / "traces.jsonl"

    from opentelemetry.sdk.trace.export import SpanExportResult

    assert exporter.export([]) is SpanExportResult.FAILURE


def test_setup_is_idempotent(tmp_path):
    settings = Settings(
        otel_enabled=True,
        otel_exporter="console",
        db_path=tmp_path / "magi.db",
    )

    first = setup_tracing(settings)
    second = setup_tracing(settings)

    assert first is second


# ── Call counting ────────────────────────────────────────────────────────────


def test_counter_accumulates_totals_and_a_per_advisor_breakdown():
    counter = CallCounter()

    counter.record("MELCHIOR", 1000, 200)
    counter.record("MELCHIOR", 1200, 250)
    counter.record("BALTHASAR", 800, 100)

    assert counter.total.calls == 3
    assert counter.total.prompt_tokens == 3000
    assert counter.total.completion_tokens == 550
    # The breakdown is what shows a reasoning advisor costing several times its
    # neighbours — invisible in a single total.
    assert counter.by_label["MELCHIOR"].calls == 2
    assert counter.by_label["BALTHASAR"].total_tokens == 900


def test_reset_clears_between_debates():
    """Each debate starts clean, so a per-debate cost must not inherit the
    previous one's."""
    counter = CallCounter()
    counter.record("MELCHIOR", 100, 10)

    counter.reset()

    assert counter.total.calls == 0
    assert counter.by_label == {}


def test_snapshot_is_detached_from_the_counter():
    counter = CallCounter()
    counter.record("MELCHIOR", 100, 10)

    snapshot = counter.snapshot()
    counter.record("MELCHIOR", 100, 10)

    assert snapshot["MELCHIOR"].calls == 1
