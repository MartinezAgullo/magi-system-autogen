"""The durable benchmark record: one row per debate, plus its full transcript.

> **SHARED SCHEMA CONTRACT with the sibling repo ``magi-system``.** This file
> *defines* the contract — the sibling does not exist yet, so nothing here can
> be negotiated later without a migration in two places. Any change to the
> schema, including adding a column, bumps :data:`SCHEMA_VERSION` and lands in
> both repos in the same change window. If the two diverge, the benchmark
> silently stops being a comparison, and it stops in the way that is hardest to
> notice: both databases still load, and the numbers are simply about different
> things.

**Nothing in the schema assumes AutoGen.** A hand-rolled orchestrator with no
framework writes exactly these rows: ``engine`` is a free string, ``selector_calls``
is zero where there is no selector, and every column is a fact about a debate
rather than about the runtime that held it. That is the point — the two
implementations are compared by concatenating their databases, not by
translating between them.

**This is not the tracing backend, and the two must not drift into duplicating
each other.** OTel answers "why was *this* debate slow, where did the time go,
what did the selector cost on turn four". This answers "over forty debates, what
happened" — and it has to answer it on a Pi in a field with no collector
reachable, with ``MAGI_OTEL_ENABLED=0``, which is exactly how the headline
numbers are taken. A row is never derived by querying a tracing backend.
Aggregates are computed once in the orchestrator and written to both.

**The transcript lives here, and only here.** Span attributes get shipped to
backends that were never sized or secured for questions and model answers; a
SQLite file on the node is where those belong.

Concurrency: one connection per write, WAL, and a busy timeout. A debate takes
two to four minutes, so a connection setup per row costs nothing measurable, and
in exchange the reporting script, a second node writing to a synced file, or an
interrupted daemon cannot corrupt what is already there. All of it runs through
``asyncio.to_thread`` — a synchronous write on the critical path of a voice
interface is the same mistake ``BatchSpanProcessor`` exists to avoid.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from magi.models import DebateRecord

logger = logging.getLogger(__name__)

#: Bumped by any change to the DDL below. Both repos must agree on it: a
#: database written at a version this code does not know is refused rather than
#: half-read, because a silently ignored column is a benchmark that reports the
#: wrong thing rather than failing.
#:
#: v2 added `novelty` and `echoes`: whether the advisors wrote three answers or
#: copied one. See `orchestrator/novelty.py`.
SCHEMA_VERSION = 2

#: How long a write waits for another writer before giving up. Generous, because
#: the alternative to waiting is losing a debate that took three minutes to
#: produce, and the only contention here is a reporting script reading.
BUSY_TIMEOUT_MS = 10_000


class IncompatibleSchema(RuntimeError):
    """The database on disk was written by a different version of the contract."""


# ── Schema ───────────────────────────────────────────────────────────────────
#
# Times are ISO 8601 with an explicit +00:00 offset, always UTC. Text rather
# than a Unix epoch so the file is readable with the sqlite3 CLI, and always the
# same offset so lexicographic ordering is chronological ordering.
#
# Lists and mappings are JSON in a TEXT column rather than side tables. They are
# read whole, never joined on and never aggregated over, and a normalised
# `debate_models(debate_id, advisor, model)` table would earn nothing but three
# more joins in every query.

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS debates (
        debate_id          TEXT    PRIMARY KEY,
        schema_version     INTEGER NOT NULL,
        node_id            TEXT    NOT NULL,
        started_at         TEXT    NOT NULL,
        question           TEXT    NOT NULL,
        engine             TEXT    NOT NULL,
        outcome            TEXT    NOT NULL,
        terminated_by      TEXT    NOT NULL,
        rounds_used        INTEGER NOT NULL,
        tallied_round      INTEGER NOT NULL,
        max_rounds         INTEGER NOT NULL,
        judged_cosmetic    INTEGER NOT NULL,
        judged_edges       INTEGER NOT NULL,
        novelty            REAL,
        echoes             TEXT,
        advisors_present   TEXT    NOT NULL,
        models             TEXT    NOT NULL,
        streamed_advisors  TEXT    NOT NULL,
        tracing_enabled    INTEGER NOT NULL,
        residency_warning  INTEGER,
        duration_s         REAL    NOT NULL,
        llm_calls          INTEGER NOT NULL,
        selector_calls     INTEGER NOT NULL,
        prompt_tokens      INTEGER NOT NULL,
        completion_tokens  INTEGER NOT NULL,
        cpu_s              REAL,
        cpu_percent        REAL,
        peak_rss_mb        REAL,
        mean_rss_mb        REAL,
        cost_samples       INTEGER,
        verdict_answer     TEXT    NOT NULL,
        verdict_dissent    TEXT,
        verdict_split_on   TEXT
    )
    """,
    # Every turn of every debate, in the order it was spoken. `tallied` marks
    # the row the outcome was computed from; the rest were generated, paid for,
    # shown on the console and left out of the arithmetic. Both are needed:
    # the tally is the result, the transcript is the evidence, and mode collapse
    # is only visible in the second one.
    """
    CREATE TABLE IF NOT EXISTS turns (
        debate_id    TEXT    NOT NULL REFERENCES debates(debate_id) ON DELETE CASCADE,
        seq          INTEGER NOT NULL,
        advisor      TEXT    NOT NULL,
        round_index  INTEGER NOT NULL,
        tallied      INTEGER NOT NULL,
        position     TEXT    NOT NULL,
        summary      TEXT    NOT NULL,
        agrees_with  TEXT    NOT NULL,
        confidence   REAL    NOT NULL,
        critique     TEXT    NOT NULL,
        PRIMARY KEY (debate_id, seq)
    )
    """,
    # Pre-flight's verdict on the inference host, written by `magi-preflight`
    # and read by the daemon it is about to start. It is a separate process, so
    # the database is the only channel between the two — and the fact worth
    # carrying across is model residency, which decides whether a run's latency
    # figures measure the framework or the Spark's disk.
    #
    # Optional by contract: an implementation with no pre-flight leaves this
    # empty and its debates carry residency_warning = NULL, which reads as
    # "nobody looked" rather than as "all clear".
    """
    CREATE TABLE IF NOT EXISTS preflight_runs (
        id                 INTEGER PRIMARY KEY AUTOINCREMENT,
        node_id            TEXT    NOT NULL,
        ran_at             TEXT    NOT NULL,
        failed             INTEGER NOT NULL,
        warned             INTEGER NOT NULL,
        residency_warning  INTEGER,
        detail             TEXT    NOT NULL
    )
    """,
    # The engine comparison groups by engine and reads the most recent N rows;
    # the convergence report does the same and adds outcome. One composite index
    # keeps both off a full scan for as long as this file is worth reading.
    "CREATE INDEX IF NOT EXISTS idx_debates_engine_started ON debates(engine, started_at)",
    "CREATE INDEX IF NOT EXISTS idx_debates_started ON debates(started_at)",
    "CREATE INDEX IF NOT EXISTS idx_preflight_node ON preflight_runs(node_id, ran_at)",
)

# Columns named rather than positional. A bare `VALUES (...)` is one reordered
# column away from writing the engine into the outcome, and the whole file is a
# contract two repositories edit independently.
_DEBATE_COLUMNS = (
    "debate_id", "schema_version", "node_id", "started_at", "question", "engine",
    "outcome", "terminated_by", "rounds_used", "tallied_round", "max_rounds",
    "judged_cosmetic", "judged_edges", "novelty", "echoes", "advisors_present",
    "models", "streamed_advisors", "tracing_enabled", "residency_warning",
    "duration_s", "llm_calls", "selector_calls", "prompt_tokens",
    "completion_tokens", "cpu_s", "cpu_percent", "peak_rss_mb", "mean_rss_mb",
    "cost_samples", "verdict_answer", "verdict_dissent", "verdict_split_on",
)

_TURN_COLUMNS = (
    "debate_id", "seq", "advisor", "round_index", "tallied", "position",
    "summary", "agrees_with", "confidence", "critique",
)


def _insert(table: str, columns: tuple[str, ...]) -> str:
    names = ", ".join(columns)
    placeholders = ", ".join(f":{column}" for column in columns)
    return f"INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})"


_INSERT_DEBATE = _insert("debates", _DEBATE_COLUMNS)
_INSERT_TURN = _insert("turns", _TURN_COLUMNS)


@dataclass(frozen=True)
class PreflightRun:
    """One recorded pre-flight, as the daemon reads it back."""

    node_id: str
    ran_at: datetime
    failed: bool
    warned: bool
    #: ``None`` when residency was not observable — no Ollama backend, or an
    #: older build with no ``/api/ps``.
    residency_warning: bool | None
    detail: str

    @property
    def age_s(self) -> float:
        return (datetime.now(UTC) - self.ran_at).total_seconds()


def connect(path: Path | str) -> sqlite3.Connection:
    """A connection configured the one way every caller wants it.

    WAL so a reader (the reporting script, or a second look at a debate that is
    still running) never blocks the node from finishing one; a busy timeout so a
    writer waits instead of raising; foreign keys on, because they are off by
    default in SQLite and a schema that declares them without enabling them is
    documentation rather than a constraint.
    """
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class DebateStore:
    """The SQLite benchmark record at ``settings.db_path``.

    Cheap to construct and safe to hold for the life of the node: it owns no
    connection between calls, so there is nothing to reopen after a disk goes
    away and comes back, and nothing to serialise across the debate loop.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._ready = False

    @property
    def path(self) -> Path:
        return self._path

    # ── Schema ───────────────────────────────────────────────────────────

    def initialise(self) -> None:
        """Create the database if it is not there, and refuse a foreign one.

        A fresh clone reaches this with no ``data/`` directory at all, which is
        the normal first run rather than an error: pre-flight has already
        asserted the parent is writable, and this creates it.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._write() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version == 0:
                for statement in _SCHEMA:
                    conn.execute(statement)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                conn.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('created_at', ?)",
                    (_iso(datetime.now(UTC)),),
                )
                logger.info("Created benchmark database at %s (v%d)",
                            self._path, SCHEMA_VERSION)
            elif version < SCHEMA_VERSION:
                _migrate(conn, version)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            elif version > SCHEMA_VERSION:
                # Forwards only. A file from a newer version may contain columns
                # this code would silently ignore, and a benchmark that quietly
                # reports the wrong thing is worse than one that refuses to run.
                raise IncompatibleSchema(
                    f"{self._path} was written at schema v{version}, this code "
                    f"speaks v{SCHEMA_VERSION}. The schema is a shared contract "
                    f"with the sibling repo — check both are on the same version."
                )
            conn.execute(
                "INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        self._ready = True

    async def open(self) -> None:
        """:meth:`initialise`, off the event loop. Call once at boot."""
        await asyncio.to_thread(self.initialise)

    # ── Writing ──────────────────────────────────────────────────────────

    async def save(self, record: DebateRecord) -> None:
        """Persist one debate. Never called on the path the operator is waiting on."""
        await asyncio.to_thread(self.save_sync, record)

    def save_sync(self, record: DebateRecord) -> None:
        """The write itself, in one transaction.

        ``INSERT OR REPLACE`` on the debate and a delete-then-insert on its
        turns, so saving the same ``debate_id`` twice replaces it rather than
        raising or duplicating it. That makes re-importing a run idempotent,
        which matters more than catching a UUID collision that will not happen.
        """
        if not self._ready:
            self.initialise()

        with self._write() as conn:
            conn.execute(_INSERT_DEBATE, _debate_row(record))
            conn.execute("DELETE FROM turns WHERE debate_id = ?", (record.debate_id,))
            conn.executemany(
                _INSERT_TURN,
                [
                    {
                        "debate_id": record.debate_id,
                        "seq": seq,
                        "advisor": turn.advisor,
                        "round_index": turn.round_index,
                        "tallied": int(turn.tallied),
                        "position": turn.turn.position,
                        "summary": turn.turn.summary,
                        "agrees_with": json.dumps(turn.turn.agrees_with),
                        "confidence": turn.turn.confidence,
                        "critique": json.dumps(
                            [c.model_dump() for c in turn.turn.critique]
                        ),
                    }
                    for seq, turn in enumerate(record.turns)
                ],
            )

    # ── Pre-flight ───────────────────────────────────────────────────────

    def record_preflight_sync(
        self,
        node_id: str,
        *,
        failed: bool,
        warned: bool,
        residency_warning: bool | None,
        detail: str,
    ) -> None:
        if not self._ready:
            self.initialise()
        with self._write() as conn:
            conn.execute(
                "INSERT INTO preflight_runs "
                "(node_id, ran_at, failed, warned, residency_warning, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    node_id,
                    _iso(datetime.now(UTC)),
                    int(failed),
                    int(warned),
                    None if residency_warning is None else int(residency_warning),
                    detail,
                ),
            )

    def latest_preflight_sync(self, node_id: str) -> PreflightRun | None:
        """The most recent pre-flight recorded for *node_id*, if any.

        Age is left to the caller rather than filtered here: how stale is too
        stale is a question about how the node was launched, and the store's job
        is to say what happened, not to decide what still counts.
        """
        if not self._path.exists():
            return None

        conn: sqlite3.Connection | None = None
        try:
            # Inside the try, not before it: opening is where a file that is not
            # a database fails, and that is precisely the case this has to
            # survive. Not knowing whether pre-flight warned is a caveat on the
            # numbers; refusing to boot over it is not proportionate.
            conn = connect(self._path)
            row = conn.execute(
                "SELECT * FROM preflight_runs WHERE node_id = ? "
                "ORDER BY ran_at DESC, id DESC LIMIT 1",
                (node_id,),
            ).fetchone()
        except sqlite3.DatabaseError:
            logger.warning("Could not read pre-flight history from %s", self._path,
                           exc_info=True)
            return None
        finally:
            if conn is not None:
                conn.close()

        if row is None:
            return None
        return PreflightRun(
            node_id=row["node_id"],
            ran_at=datetime.fromisoformat(row["ran_at"]),
            failed=bool(row["failed"]),
            warned=bool(row["warned"]),
            residency_warning=(
                None if row["residency_warning"] is None
                else bool(row["residency_warning"])
            ),
            detail=row["detail"],
        )

    async def latest_preflight(self, node_id: str) -> PreflightRun | None:
        return await asyncio.to_thread(self.latest_preflight_sync, node_id)

    # ── Internals ────────────────────────────────────────────────────────

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One connection, one transaction, always closed.

        ``with conn`` commits or rolls back but does not close, which is the
        SQLite API's oldest trap: a store that only used it would leak a handle
        per debate and hit the open-file limit some days into a benchmark run.
        """
        conn = connect(self._path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()


def _migrate(conn: sqlite3.Connection, version: int) -> None:
    """Bring an older database up to :data:`SCHEMA_VERSION`, in place.

    Deleting the file and starting again would be simpler and is the wrong
    trade: the rows in it are debates that took minutes of three models' time
    under a configuration that has probably moved since. That is the same
    reasoning that made this module exist.

    **v1 -> v2** adds the novelty columns, and backfills them from the stored
    transcripts. That backfill is the argument for keeping the transcript in the
    first place, demonstrated: the measure did not exist when those debates ran,
    the evidence was kept anyway, and the old runs can answer the new question
    without being repeated.
    """
    if version < 2:  # noqa: PLR2004 — the version this step produces
        from magi.orchestrator.novelty import measure

        conn.execute("ALTER TABLE debates ADD COLUMN novelty REAL")
        conn.execute("ALTER TABLE debates ADD COLUMN echoes TEXT")
        for row in conn.execute("SELECT debate_id FROM debates").fetchall():
            positions = {
                turn["advisor"]: turn["position"]
                for turn in conn.execute(
                    "SELECT advisor, position FROM turns "
                    "WHERE debate_id = ? AND tallied ORDER BY seq",
                    (row["debate_id"],),
                )
            }
            result = measure(positions)
            conn.execute(
                "UPDATE debates SET novelty = ?, echoes = ? WHERE debate_id = ?",
                (
                    result.score,
                    json.dumps([echo.model_dump() for echo in result.echoes]),
                    row["debate_id"],
                ),
            )
        logger.info("Migrated %s to schema v2 (novelty backfilled)", conn)


def _iso(moment: datetime) -> str:
    """UTC, with the offset spelled out, to the millisecond.

    Normalised rather than stored as given: two rows written in different
    timezones would still sort correctly by instant, but not by string, and
    every query in the reporting script orders by this column.
    """
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat(timespec="milliseconds")


def _debate_row(record: DebateRecord) -> dict:
    cost = record.node
    return {
        "debate_id": record.debate_id,
        "schema_version": SCHEMA_VERSION,
        "node_id": record.node_id,
        "started_at": _iso(record.started_at),
        "question": record.question,
        "engine": record.engine,
        "outcome": record.outcome.value,
        "terminated_by": record.terminated_by,
        "rounds_used": record.rounds_used,
        "tallied_round": record.tallied_round,
        "max_rounds": record.max_rounds,
        "judged_cosmetic": int(record.judged_cosmetic),
        "judged_edges": record.judged_edges,
        # NULL when fewer than two advisors were tallied: there was nothing to
        # compare, which is not the same as nothing being repeated.
        "novelty": record.novelty,
        "echoes": json.dumps([echo.model_dump() for echo in record.echoes]),
        "advisors_present": json.dumps(record.advisors_present),
        "models": json.dumps(record.models),
        "streamed_advisors": json.dumps(record.streamed_advisors),
        "tracing_enabled": int(record.tracing_enabled),
        "residency_warning": (
            None if record.residency_warning is None else int(record.residency_warning)
        ),
        "duration_s": record.duration_s,
        "llm_calls": record.llm_calls,
        "selector_calls": record.selector_calls,
        "prompt_tokens": record.prompt_tokens,
        "completion_tokens": record.completion_tokens,
        # NULL rather than zero when metrics are off. Zero CPU is a measurement
        # that never happens, so a row of zeros would be a lie that averages.
        "cpu_s": None if cost is None else cost.cpu_s,
        "cpu_percent": None if cost is None else cost.cpu_percent,
        "peak_rss_mb": None if cost is None else cost.peak_rss_mb,
        "mean_rss_mb": None if cost is None else cost.mean_rss_mb,
        "cost_samples": None if cost is None else cost.samples,
        "verdict_answer": record.verdict.answer,
        "verdict_dissent": record.verdict.dissent,
        "verdict_split_on": record.verdict.disagreement_point,
    }
