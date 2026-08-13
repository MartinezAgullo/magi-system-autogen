"""The report that answers the question the agreement levers were changed for.

`docs/agreement-bias.md` § (b) and (d) lowered the bar for agreement, and the
document is explicit that the question afterwards is not "did UNANIMOUS go up".
It is whether the rise is the advisors converging or the judge becoming more
permissive, and those must never be averaged into one number.

So what these tests pin down is the split, and the honesty of the filtering:
a row left out of an average has to be counted and named, because a filter
nobody can see is how a systematically excluded run goes unnoticed.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from magi.models import Outcome
from magi.store.debates import DebateStore, connect
from tests.test_store import ADVISORS, a_record

REPORT = Path(__file__).resolve().parents[1] / "scripts" / "report.py"


def _load_report():
    """scripts/ is not a package — it holds entry points, not importable code."""
    spec = importlib.util.spec_from_file_location("magi_report", REPORT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load_report()


@pytest.fixture
async def rows(tmp_path):
    """Four debates: one converged unaided, one repaired, one promoted, one not."""
    store = DebateStore(tmp_path / "magi.db")
    start = datetime(2026, 8, 13, 9, 0, tzinfo=UTC)
    cases = [
        ("unaided00000", Outcome.UNANIMOUS, False, 0),
        ("repaired0000", Outcome.UNANIMOUS, False, 2),
        ("promoted0000", Outcome.UNANIMOUS, True, 1),
        ("deadlock0000", Outcome.DEADLOCK, False, 0),
    ]
    for index, (debate_id, outcome, cosmetic, edges) in enumerate(cases):
        await store.save(
            a_record(
                debate_id=debate_id,
                outcome=outcome,
                judged_cosmetic=cosmetic,
                judged_edges=edges,
                started_at=start + timedelta(minutes=index),
            )
        )
    with connect(store.path) as conn:
        return conn.execute("SELECT * FROM debates ORDER BY started_at").fetchall()


def test_the_unanimous_rate_is_split_into_earned_and_granted(rows):
    """Three of four debates ended UNANIMOUS, and only one of them did so with
    nobody's help. A report that printed 75% would be true and useless: it is
    exactly the number that cannot tell a better prompt from a looser judge."""
    unaided, repaired, promoted = report.unanimous_split(rows)

    assert (unaided, repaired, promoted) == (1, 1, 1)
    printed = report.convergence(rows, keep_all=False)
    assert "3 (75%)" in printed   # UNANIMOUS
    assert "1 (25%)" in printed   # of which unaided


def test_a_debate_that_was_both_repaired_and_promoted_counts_as_promoted(rows):
    """The buckets have to be disjoint or they do not sum to the UNANIMOUS rate,
    and the larger concession is the one that describes the debate: an edge
    repair says two advisors were already agreeing, a cosmetic ruling says a
    whole split vote was wording."""
    both = [row for row in rows if row["debate_id"] == "promoted0000"][0]
    assert both["judged_cosmetic"] and both["judged_edges"]

    unaided, repaired, promoted = report.unanimous_split([both])

    assert (unaided, repaired, promoted) == (0, 0, 1)


async def test_tracing_spoils_the_cost_table_and_not_the_convergence_one(tmp_path):
    """The two blocks filter differently on purpose. Tracing costs the node CPU
    on a node whose CPU is the object of study, so a traced row cannot be
    averaged into the cost table; it says nothing at all about whether three
    advisors agreed, so excluding it from the convergence table would throw away
    a perfectly good debate."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record(debate_id="traced000000", tracing_enabled=True,
                              residency_warning=False))
    await store.save(a_record(debate_id="clean0000000", tracing_enabled=False,
                              residency_warning=False))
    with connect(store.path) as conn:
        rows = conn.execute("SELECT * FROM debates").fetchall()

    assert sum(report.convergence_excuse(row) is None for row in rows) == 2
    assert [report.cost_excuse(row) for row in rows].count("traced") == 1


async def test_a_residency_warning_takes_a_row_out_of_the_cost_table(tmp_path):
    """The whole reason the flag is on the row. A debate whose weights were
    evicted mid-run measures the Spark's disk, and its latency would widen every
    average it landed in — silently, and in the direction that makes the
    framework look slow."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record(debate_id="suspect00000", tracing_enabled=False,
                              residency_warning=True))
    with connect(store.path) as conn:
        row = conn.execute("SELECT * FROM debates").fetchone()

    assert report.cost_excuse(row) == "residency warning"
    assert report.convergence_excuse(row) is None


async def test_excluded_rows_are_counted_and_named(tmp_path):
    """A filter nobody can see is worse than no filter. Whatever is dropped is
    reported under the table it was dropped from, with the reason."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record(debate_id="traced000000", tracing_enabled=True,
                              residency_warning=False))
    await store.save(a_record(debate_id="tworunners00", tracing_enabled=False,
                              residency_warning=False,
                              advisors_present=["BALTHASAR", "MELCHIOR"]))
    with connect(store.path) as conn:
        rows = conn.execute("SELECT * FROM debates").fetchall()

    printed = report.cost(rows, keep_all=False)

    assert "2 row(s) excluded" in printed
    assert "1 traced" in printed
    assert "1 lost an advisor" in printed
    # And --all keeps them.
    assert "excluded" not in report.cost(rows, keep_all=True)


async def test_a_mixed_roster_is_reported_rather_than_filtered(tmp_path):
    """Rotating the models is the point of the project, so a report over two
    rosters is a legitimate thing to ask for. What it must not do is present
    them as one configuration."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record(debate_id="nemotron0000"))
    await store.save(
        a_record(
            debate_id="deepseek0000",
            models={"MELCHIOR": "deepseek-r1:32b", "BALTHASAR": "gemma3:12b",
                    "CASPAR": "qwen3:14b"},
        )
    )
    with connect(store.path) as conn:
        rows = conn.execute("SELECT * FROM debates").fetchall()

    assert "2 different rosters" in report.caveats(rows)


async def test_the_engine_delta_is_taken_against_roundrobin(tmp_path):
    """The comparison the repo exists for, and the direction matters: the
    baseline is the engine with no LLM in the control path, so the delta reads
    as what the selector cost rather than as what it saved."""
    store = DebateStore(tmp_path / "magi.db")
    for index in range(2):
        await store.save(a_record(
            debate_id=f"rr{index:010d}", engine="autogen_roundrobin",
            duration_s=100.0, llm_calls=10, selector_calls=0,
            tracing_enabled=False, residency_warning=False,
        ))
        await store.save(a_record(
            debate_id=f"sel{index:09d}", engine="autogen_selector",
            duration_s=150.0, llm_calls=17, selector_calls=7,
            tracing_enabled=False, residency_warning=False,
        ))
    with connect(store.path) as conn:
        rows = conn.execute("SELECT * FROM debates").fetchall()

    printed = report.cost(rows, keep_all=False)

    assert "autogen_selector vs autogen_roundrobin" in printed
    assert "+50% wall clock" in printed
    assert "+70% LLM calls" in printed


async def test_the_csv_says_what_it_would_have_excluded_rather_than_hiding_it(
    tmp_path, capsys
):
    """A hidden filter inside an export is how a spreadsheet ends up disagreeing
    with the terminal for reasons nobody can reconstruct. Every selected row is
    dumped, with this script's verdict as two columns."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record(debate_id="traced000000", tracing_enabled=True,
                              residency_warning=False))
    with connect(store.path) as conn:
        rows = conn.execute("SELECT * FROM debates").fetchall()

    report.dump_csv(rows)

    out = capsys.readouterr().out
    assert "convergence_comparable,cost_comparable" in out.splitlines()[0]
    assert out.strip().endswith("1,0")
    assert str(ADVISORS[0]) in out
