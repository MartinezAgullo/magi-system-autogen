"""The benchmark record, which is the thing the project is for.

A debate costs two to four minutes of three models' time. Everything else in
this system can be run again; a row that was written wrong, or not written at
all, is a run that has to be repeated — and by then the configuration that
produced it has usually moved.

So these tests are about durability rather than about SQL: a fresh clone has to
create the file, a full record has to survive the round trip with the parts that
make it comparable intact, a debate that lost an advisor has to be readable as
such rather than as a normal debate, and two writes landing at once must not
leave a file that no later run can append to.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from magi.config import Settings
from magi.constants import TERMINATED_BY_BUDGET, TERMINATED_BY_CONSENSUS
from magi.models import (
    Critique,
    DebateRecord,
    MagiTurn,
    MagiVerdict,
    NodeCost,
    Outcome,
    TurnRecord,
)
from magi.orchestrator.magi import Magi, _Deliberation
from magi.personas import PersonaSet
from magi.store.debates import (
    SCHEMA_VERSION,
    DebateStore,
    IncompatibleSchema,
    connect,
)

ADVISORS = ("MELCHIOR", "BALTHASAR", "CASPAR")


def a_turn(advisor: str, round_index: int, *, agrees=(), tallied=False) -> TurnRecord:
    return TurnRecord(
        advisor=advisor,
        round_index=round_index,
        tallied=tallied,
        turn=MagiTurn(
            position=f"{advisor} in round {round_index}, at some length.",
            summary=f"{advisor} says something.",
            agrees_with=list(agrees),
            confidence=0.7,
            critique=[Critique(advisor="CASPAR", objection="Too optimistic.")],
        ),
    )


def a_record(**overrides) -> DebateRecord:
    """A complete debate: two rounds, three advisors, the second round tallied."""
    turns = [a_turn(name, 1) for name in ADVISORS]
    turns += [a_turn(name, 2, agrees=["MELCHIOR"], tallied=True) for name in ADVISORS]
    defaults = dict(
        debate_id="abc123def456",
        question="Should a three-person startup adopt Kubernetes?",
        engine="autogen_roundrobin",
        outcome=Outcome.MAJORITY,
        verdict=MagiVerdict(
            outcome=Outcome.MAJORITY,
            answer="Two of the three would not.",
            dissent="CASPAR would, for the hiring argument.",
        ),
        turns=turns,
        rounds_used=2,
        terminated_by=TERMINATED_BY_BUDGET,
        judged_cosmetic=False,
        judged_edges=1,
        tallied_round=2,
        advisors_present=sorted(ADVISORS),
        models={"MELCHIOR": "nemotron3:33b", "BALTHASAR": "gemma3:12b",
                "CASPAR": "qwen3:14b"},
        duration_s=132.4,
        node_id="magi-01",
        started_at=datetime(2026, 8, 13, 9, 30, tzinfo=UTC),
        max_rounds=3,
        streamed_advisors=["BALTHASAR"],
        residency_warning=True,
        llm_calls=11,
        selector_calls=0,
        prompt_tokens=8123,
        completion_tokens=2044,
        tracing_enabled=False,
        node=NodeCost(cpu_s=2.8, cpu_percent=2.1, peak_rss_mb=104.5,
                      mean_rss_mb=98.2, samples=132),
    )
    return DebateRecord(**{**defaults, **overrides})


# ── Creating the file ────────────────────────────────────────────────────────


async def test_a_fresh_clone_creates_the_database_and_its_directory(tmp_path):
    """`data/` is not in the repository, so the first run on a new machine
    arrives here with nothing. Pre-flight has already asserted the parent is
    writable; creating the rest is this module's job, not an operator's."""
    store = DebateStore(tmp_path / "data" / "magi.db")
    assert not (tmp_path / "data").exists()

    await store.open()

    with connect(store.path) as conn:
        tables = {
            row["name"]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"debates", "turns", "preflight_runs", "meta"} <= tables
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


async def test_opening_an_existing_database_twice_changes_nothing(tmp_path):
    """The daemon opens it at every boot, and `ask.py` opens the same file from
    another process minutes later. Neither may wipe what is there."""
    store = DebateStore(tmp_path / "magi.db")
    await store.open()
    await store.save(a_record())

    await DebateStore(tmp_path / "magi.db").open()

    with connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM debates").fetchone()[0] == 1


async def test_a_database_from_a_future_schema_is_refused_rather_than_half_read(
    tmp_path,
):
    """The schema is a contract with a sibling repo that does not exist yet. The
    failure this guards against is not a crash — it is the sibling adding a
    column, this code ignoring it, and both benchmarks continuing to produce
    numbers that are quietly about different things."""
    path = tmp_path / "magi.db"
    await DebateStore(path).open()
    with connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(IncompatibleSchema, match="shared contract"):
        DebateStore(path).initialise()


# ── The round trip ───────────────────────────────────────────────────────────


async def test_a_full_record_survives_the_round_trip(tmp_path):
    """Every field that makes a row comparable with another row: the engine, the
    models, whether it was traced, whether anyone streamed, whether pre-flight
    warned about residency. A row missing any of them cannot be averaged with
    anything, and there is no way to recover it after the run."""
    store = DebateStore(tmp_path / "magi.db")
    record = a_record()
    await store.save(record)

    with connect(store.path) as conn:
        row = conn.execute("SELECT * FROM debates").fetchone()

    assert row["debate_id"] == record.debate_id
    assert row["question"] == record.question
    assert row["engine"] == "autogen_roundrobin"
    assert row["outcome"] == "MAJORITY"
    assert row["terminated_by"] == TERMINATED_BY_BUDGET
    assert row["node_id"] == "magi-01"
    assert row["started_at"].startswith("2026-08-13T09:30:00")
    assert json.loads(row["models"])["MELCHIOR"] == "nemotron3:33b"
    assert json.loads(row["streamed_advisors"]) == ["BALTHASAR"]
    assert row["tracing_enabled"] == 0
    assert row["residency_warning"] == 1
    assert row["judged_edges"] == 1
    assert row["tallied_round"] == 2 and row["max_rounds"] == 3
    assert row["selector_calls"] == 0 and row["llm_calls"] == 11
    assert row["peak_rss_mb"] == pytest.approx(104.5)
    assert row["verdict_dissent"].startswith("CASPAR")
    # DEADLOCK-only, and the round trip must not invent one.
    assert row["verdict_split_on"] is None


async def test_the_whole_transcript_is_kept_and_the_tallied_row_is_marked(tmp_path):
    """The bug this replaces kept one turn per advisor, all stamped with the
    final round, so a three-round debate read as three advisors speaking once.
    Both halves matter: the transcript is the only place mode collapse is
    visible, and the tally is only defensible if the row it read is identifiable
    among the turns that were generated and then not counted."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record())

    with connect(store.path) as conn:
        rows = conn.execute("SELECT * FROM turns ORDER BY seq").fetchall()

    assert len(rows) == 6
    assert [r["round_index"] for r in rows] == [1, 1, 1, 2, 2, 2]
    assert [r["advisor"] for r in rows[:3]] == list(ADVISORS)
    assert [bool(r["tallied"]) for r in rows] == [False] * 3 + [True] * 3
    assert json.loads(rows[3]["agrees_with"]) == ["MELCHIOR"]
    assert json.loads(rows[0]["critique"])[0]["objection"] == "Too optimistic."
    assert rows[0]["position"].startswith("MELCHIOR in round 1")


async def test_a_consensus_row_may_sit_at_several_depths(tmp_path):
    """A debate stopped by ConsensusTermination is tallied on each advisor's
    latest vote, which is not one round: the advisor whose turn completed the
    agreement is a round ahead of the others. The store records what happened
    rather than a tidier version of it."""
    turns = [a_turn(name, 1) for name in ADVISORS]
    turns += [a_turn(name, 2, tallied=(name != "MELCHIOR")) for name in ADVISORS]
    turns += [a_turn("MELCHIOR", 3, tallied=True)]
    store = DebateStore(tmp_path / "magi.db")
    await store.save(
        a_record(turns=turns, terminated_by=TERMINATED_BY_CONSENSUS,
                 outcome=Outcome.UNANIMOUS, rounds_used=3, tallied_round=2)
    )

    with connect(store.path) as conn:
        tallied = conn.execute(
            "SELECT advisor, round_index FROM turns WHERE tallied ORDER BY seq"
        ).fetchall()

    assert {(r["advisor"], r["round_index"]) for r in tallied} == {
        ("BALTHASAR", 2), ("CASPAR", 2), ("MELCHIOR", 3),
    }


async def test_a_debate_that_lost_an_advisor_reads_as_one(tmp_path):
    """2/3 degradation is a designed outcome, not a failure, and its rows go
    into the same benchmark as everything else. `advisors_present` shorter than
    `models` is how a later query excludes them — a two-voter debate cannot
    reach MAJORITY at all, so averaging its outcome with three-voter debates
    would understate the convergence rate for a reason that has nothing to do
    with the advisors."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(
        a_record(
            advisors_present=["BALTHASAR", "MELCHIOR"],
            turns=[a_turn("MELCHIOR", 1, tallied=True),
                   a_turn("BALTHASAR", 1, tallied=True)],
            outcome=Outcome.DEADLOCK,
            verdict=MagiVerdict(
                outcome=Outcome.DEADLOCK,
                answer="Two advisors answered and did not converge.",
                disagreement_point="Whether the operational cost is recoverable.",
            ),
        )
    )

    with connect(store.path) as conn:
        # json_array_length over the column, because "how many advisors
        # answered" is the filter every cross-run query starts with and it has
        # to be answerable in SQL rather than by loading the row.
        row = conn.execute(
            "SELECT *, json_array_length(advisors_present) AS present FROM debates"
        ).fetchone()

    assert row["present"] == 2
    assert len(json.loads(row["models"])) == 3
    assert row["outcome"] == "DEADLOCK"
    assert row["verdict_split_on"].startswith("Whether")
    assert "CASPAR" not in json.loads(row["advisors_present"])


async def test_metrics_switched_off_records_null_rather_than_zero(tmp_path):
    """MAGI_METRICS_ENABLED=0 is a real configuration, and a row of zeros would
    average into every CPU figure as a node that used no CPU. NULL is the only
    honest answer to a measurement that was never taken."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record(node=None, residency_warning=None))

    with connect(store.path) as conn:
        row = conn.execute("SELECT * FROM debates").fetchone()

    assert row["cpu_s"] is None
    assert row["peak_rss_mb"] is None
    assert row["cost_samples"] is None
    # Same reasoning, different fact: nobody looked at residency.
    assert row["residency_warning"] is None


async def test_saving_the_same_debate_twice_replaces_it(tmp_path):
    """Re-importing a run has to be idempotent, and a debate re-saved must not
    leave the turns of both attempts interleaved in its transcript."""
    store = DebateStore(tmp_path / "magi.db")
    await store.save(a_record())
    await store.save(a_record(rounds_used=3, turns=[a_turn("CASPAR", 1, tallied=True)]))

    with connect(store.path) as conn:
        assert conn.execute("SELECT count(*) FROM debates").fetchone()[0] == 1
        assert conn.execute("SELECT rounds_used FROM debates").fetchone()[0] == 3
        assert conn.execute("SELECT count(*) FROM turns").fetchone()[0] == 1


# ── Two things writing at once ───────────────────────────────────────────────


async def test_concurrent_writes_do_not_corrupt_the_file(tmp_path):
    """A node running debates while `ask.py` writes a benchmark run to the same
    database, or simply a save landing on a thread while another is mid-commit.
    Every write goes through to_thread, so this is genuine thread concurrency
    and not an interleaving the event loop controls."""
    store = DebateStore(tmp_path / "magi.db")
    await store.open()

    await asyncio.gather(
        *(store.save(a_record(debate_id=f"debate{index:04d}")) for index in range(24))
    )

    with connect(store.path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert conn.execute("SELECT count(*) FROM debates").fetchone()[0] == 24
        assert conn.execute("SELECT count(*) FROM turns").fetchone()[0] == 24 * 6
        # Nothing half-written: every debate kept all six of its turns.
        counts = conn.execute(
            "SELECT count(*) AS n FROM turns GROUP BY debate_id"
        ).fetchall()
        assert {row["n"] for row in counts} == {6}


async def test_a_second_store_object_on_the_same_file_appends(tmp_path):
    """Two processes, which is the normal case: `magi-preflight` writes its run
    and exits, then the daemon opens the same file. Neither owns it."""
    path = tmp_path / "magi.db"
    await DebateStore(path).save(a_record(debate_id="first0000000"))
    await DebateStore(path).save(a_record(debate_id="second000000"))

    with connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM debates").fetchone()[0] == 2


# ── Pre-flight's verdict, carried between processes ──────────────────────────


async def test_the_daemon_reads_back_what_preflight_saw(tmp_path):
    """`magi-preflight` is a separate process that launch.sh runs seconds before
    the node. The database is the only channel between them, and the fact worth
    carrying is model residency: a debate whose weights were evicted mid-run
    measures the Spark's disk rather than the framework."""
    store = DebateStore(tmp_path / "magi.db")
    store.record_preflight_sync(
        "magi-01", failed=False, warned=True, residency_warning=True,
        detail="2 of 3 models are not resident",
    )

    run = await store.latest_preflight("magi-01")

    assert run is not None
    assert run.residency_warning is True
    assert run.detail.startswith("2 of 3")
    assert run.age_s < 60


async def test_the_most_recent_preflight_wins(tmp_path):
    """Nodes are restarted, and a run that warned about residency an hour ago
    must not overrule the clean one from thirty seconds ago."""
    store = DebateStore(tmp_path / "magi.db")
    store.record_preflight_sync("magi-01", failed=False, warned=True,
                                residency_warning=True, detail="stale")
    store.record_preflight_sync("magi-01", failed=False, warned=False,
                                residency_warning=False, detail="")

    run = await store.latest_preflight("magi-01")

    assert run is not None and run.residency_warning is False


async def test_preflight_from_another_node_is_not_this_node_s(tmp_path):
    """Two Pis benchmarking against one Spark can share a database file. The
    residency verdict is about the pair, not about the fleet."""
    store = DebateStore(tmp_path / "magi.db")
    store.record_preflight_sync("magi-02", failed=False, warned=True,
                                residency_warning=True, detail="not resident")

    assert await store.latest_preflight("magi-01") is None


async def test_an_unobservable_residency_is_recorded_as_unknown(tmp_path):
    """An API-key backend has no Ollama to ask, and an older build has no
    /api/ps. Recording False there would claim a check that never ran, which is
    the one thing a benchmark caveat must never do."""
    store = DebateStore(tmp_path / "magi.db")
    store.record_preflight_sync("magi-01", failed=False, warned=False,
                                residency_warning=None, detail="")

    run = await store.latest_preflight("magi-01")

    assert run is not None and run.residency_warning is None


async def test_no_recorded_preflight_reads_as_unknown_not_as_clean(tmp_path):
    """A node started with --skip-checks, or a database that predates the
    check. The absence of a warning is not the absence of a problem."""
    assert await DebateStore(tmp_path / "magi.db").latest_preflight("magi-01") is None


async def test_a_stale_preflight_is_still_returned_with_its_age(tmp_path):
    """The store reports what happened; how old is too old is a question about
    how the node was launched, and main.py is where that is decided."""
    store = DebateStore(tmp_path / "magi.db")
    store.record_preflight_sync("magi-01", failed=False, warned=True,
                                residency_warning=True, detail="old news")
    old = (datetime.now(UTC) - timedelta(hours=5)).isoformat(timespec="milliseconds")
    with connect(store.path) as conn:
        conn.execute("UPDATE preflight_runs SET ran_at = ?", (old,))

    run = await store.latest_preflight("magi-01")

    assert run is not None
    assert run.age_s > 4 * 3600


# ── The wiring, end to end ───────────────────────────────────────────────────


async def test_a_completed_debate_reaches_the_database(tmp_path, monkeypatch):
    """The whole point of the module, with the three phases stubbed and
    everything around them real: the record `debate()` assembles is the record
    that lands, including the fields nothing else would notice were missing —
    the node it ran on, the round budget, who was streaming, and what pre-flight
    said about residency."""
    personas = PersonaSet.model_validate(
        {
            "magi": [
                {"name": "MELCHIOR", "model": "a:1b", "system_prompt": "a"},
                {"name": "BALTHASAR", "model": "b:1b", "system_prompt": "b",
                 "stream": True},
                {"name": "CASPAR", "model": "c:1b", "system_prompt": "c"},
            ],
            "orchestrator": {"name": "MAGI", "model": "a:1b", "system_prompt": "o"},
        }
    )
    settings = Settings(
        db_path=tmp_path / "magi.db", node_id="magi-test", max_rounds=3,
        otel_enabled=False, metrics_sample_interval_s=0.05,
    )
    store = DebateStore(settings.db_path)
    magi = Magi(settings, personas, store=store, residency_warning=True)

    # Unanimous on purpose: a split vote would send the judge to a model, and
    # what is under test here is the bookkeeping, not the debate.
    blind = {name: a_turn(name, 1).turn for name in ADVISORS}
    decided = {
        name: MagiTurn(
            position=f"{name} finally.", summary=f"{name} agrees.",
            agrees_with=[other for other in ADVISORS if other != name],
            confidence=0.9,
        )
        for name in ADVISORS
    }
    transcript = [TurnRecord(advisor=n, round_index=1, turn=t) for n, t in blind.items()]
    transcript += [
        TurnRecord(advisor=n, round_index=2, turn=t, tallied=True)
        for n, t in decided.items()
    ]

    async def fake_blind(question):
        return blind

    async def fake_deliberate(question, blind_turns):
        return _Deliberation(decided, TERMINATED_BY_CONSENSUS, 2, 2, tuple(transcript))

    async def fake_verdict(question, turns, result):
        return MagiVerdict(outcome=Outcome.UNANIMOUS, answer="All three agree.")

    monkeypatch.setattr(magi, "_blind_round", fake_blind)
    monkeypatch.setattr(magi, "_deliberate", fake_deliberate)
    monkeypatch.setattr(magi, "_verdict", fake_verdict)

    record = await magi.debate("Should the node write this down?")

    with connect(store.path) as conn:
        row = conn.execute("SELECT * FROM debates").fetchone()
        turns = conn.execute("SELECT * FROM turns ORDER BY seq").fetchall()

    assert row["debate_id"] == record.debate_id
    assert row["outcome"] == "UNANIMOUS"
    assert row["terminated_by"] == TERMINATED_BY_CONSENSUS
    assert row["node_id"] == "magi-test"
    assert row["max_rounds"] == 3
    assert json.loads(row["streamed_advisors"]) == ["BALTHASAR"]
    assert row["residency_warning"] == 1
    assert row["tracing_enabled"] == 0
    # The sampler ran for the length of the debate, so the node's cost is a
    # measurement rather than a default.
    assert row["cost_samples"] >= 1
    assert row["peak_rss_mb"] > 0
    assert len(turns) == 6
    assert [bool(t["tallied"]) for t in turns] == [False] * 3 + [True] * 3


async def test_a_failed_write_does_not_cost_the_verdict(tmp_path, monkeypatch, caplog):
    """The record is worth having; it is not worth more than the answer three
    models just spent two minutes producing. A full disk loses a row and says
    so."""
    personas = PersonaSet.model_validate(
        {
            "magi": [
                {"name": "MELCHIOR", "model": "a:1b", "system_prompt": "a"},
                {"name": "BALTHASAR", "model": "b:1b", "system_prompt": "b"},
            ],
            "orchestrator": {"name": "MAGI", "model": "a:1b", "system_prompt": "o"},
        }
    )
    store = DebateStore(tmp_path / "magi.db")

    async def explode(record):
        raise OSError("no space left on device")

    monkeypatch.setattr(store, "save", explode)
    magi = Magi(Settings(otel_enabled=False), personas, store=store)

    await magi._persist(a_record())  # noqa: SLF001

    assert "no space left" in caplog.text


async def test_a_file_that_is_not_a_database_does_not_stop_the_node(tmp_path):
    """Not knowing whether pre-flight warned is a caveat on the numbers. Failing
    to boot over it is not proportionate — the node's job is to answer
    questions."""
    path = tmp_path / "magi.db"
    path.write_text("this is not a database")

    with pytest.raises(sqlite3.DatabaseError):
        DebateStore(path).initialise()
    assert DebateStore(path).latest_preflight_sync("magi-01") is None
