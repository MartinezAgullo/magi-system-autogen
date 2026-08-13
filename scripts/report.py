#!/usr/bin/env python
"""Read the benchmark database: did the debates converge, and what did they cost.

Two questions, and they are deliberately kept apart on screen because averaging
them together is how a tuning change gets mistaken for a result.

**Convergence.** The `UNANIMOUS` rate is split three ways: the outcomes the
advisors reached unaided, the ones where the judge repaired a one-sided
agreement claim before the tally (`judged_edges`), and the ones where the judge
promoted a whole split vote by ruling the difference cosmetic (`judged_cosmetic`).
A rise in the first is convergence; a rise in the other two is a more permissive
judge. See docs/agreement-bias.md § "Measuring whether it worked" — this script
exists because that document ends with "it is not yet written anywhere".

**Cost.** The roundrobin-vs-selector comparison: wall clock, LLM calls, and how
many of those calls went on choosing who speaks next. That number is the whole
subject of the engine comparison.

    uv run python scripts/report.py
    uv run python scripts/report.py --last 40 --since 2026-08-11
    uv run python scripts/report.py --engine autogen_selector
    uv run python scripts/report.py --csv > runs.csv

Rows that cannot honestly be averaged with the others are left out by default
and the footer says how many and why. `--all` keeps them. The two blocks do not
exclude the same rows, because they are not spoiled by the same things: tracing
costs the node CPU and changes nothing about whether three advisors agreed.

The SQL selects; the arithmetic is done here in Python. Both because the
selection is what an index can help with and the arithmetic is not, and because
a convergence rate computed in a CASE expression is a rate nobody re-reads.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from magi.config import Settings
from magi.store.debates import connect

BOLD, DIM, YELLOW, NC = "\033[1m", "\033[2m", "\033[1;33m", "\033[0m"


# ── Selection ────────────────────────────────────────────────────────────────


def select(conn: sqlite3.Connection, args) -> list[sqlite3.Row]:
    """The rows the report is about, oldest first.

    Ordered and limited in SQL so `idx_debates_engine_started` does the work,
    then reversed here: `--last 40` means the most recent forty, and a table
    reads forwards.
    """
    where, params = ["1 = 1"], []
    if args.since:
        where.append("started_at >= ?")
        params.append(f"{args.since}T00:00:00.000+00:00")
    if args.engine:
        where.append("engine = ?")
        params.append(args.engine)
    if args.node:
        where.append("node_id = ?")
        params.append(args.node)

    sql = (
        f"SELECT * FROM debates WHERE {' AND '.join(where)} "  # noqa: S608 — fixed clauses
        f"ORDER BY started_at DESC"
    )
    if args.last:
        sql += " LIMIT ?"
        params.append(args.last)

    return list(reversed(conn.execute(sql, params).fetchall()))


def lost_an_advisor(row: sqlite3.Row) -> bool:
    """Fewer advisors answered than were configured.

    A two-voter debate cannot reach MAJORITY at all — the outcome rules leave it
    UNANIMOUS or DEADLOCK — so averaging one into a convergence rate moves the
    number for a reason that has nothing to do with the advisors.
    """
    return len(json.loads(row["advisors_present"])) < len(json.loads(row["models"]))


def convergence_excuse(row: sqlite3.Row) -> str | None:
    """Why this row does not belong in the convergence table, if it does not."""
    if lost_an_advisor(row):
        return "lost an advisor"
    return None


def cost_excuse(row: sqlite3.Row) -> str | None:
    """Why this row does not belong in the cost table, if it does not.

    Stricter than the convergence rule, and the extra two are the reason the two
    blocks filter separately. Tracing is an observer effect on the node whose
    CPU is the object of study; a residency warning means the latency may be the
    inference host reloading weights rather than anything this code did.
    """
    if row["tracing_enabled"]:
        return "traced"
    if row["residency_warning"]:
        return "residency warning"
    if lost_an_advisor(row):
        return "lost an advisor"
    return None


# ── Formatting ───────────────────────────────────────────────────────────────


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Right-aligned columns, except the first. No dependency, no box drawing."""
    widths = [
        max(len(str(headers[i])), *(len(str(row[i])) for row in rows)) if rows
        else len(str(headers[i]))
        for i in range(len(headers))
    ]
    def line(cells: Sequence[str], colour: str = "") -> str:
        out = [str(cells[0]).ljust(widths[0])]
        out += [str(cell).rjust(widths[i + 1]) for i, cell in enumerate(cells[1:])]
        return f"{colour}{'  '.join(out)}{NC if colour else ''}"

    return "\n".join([line(headers, DIM), *(line(row) for row in rows)])


def share(part: int, whole: int) -> str:
    return f"{part} ({100 * part / whole:.0f}%)" if whole else "0"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def by_engine(rows: Sequence[sqlite3.Row]) -> dict[str, list[sqlite3.Row]]:
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(row["engine"], []).append(row)
    # roundrobin first where present: it is the baseline the deltas are against,
    # being the engine with no LLM in the control path.
    return dict(sorted(grouped.items()))


def excuses(rows: Sequence[sqlite3.Row], reason) -> str:
    """A footer naming what was dropped and why. Silence here would let a
    systematically excluded run go unnoticed, which is the failure mode of
    filtering at all."""
    counts: dict[str, int] = {}
    for row in rows:
        why = reason(row)
        if why is not None:
            counts[why] = counts.get(why, 0) + 1
    if not counts:
        return ""
    detail = ", ".join(f"{count} {why}" for why, count in sorted(counts.items()))
    total = sum(counts.values())
    return (f"\n  {YELLOW}{total} row(s) excluded as not comparable{NC} "
            f"{DIM}({detail}) — pass --all to keep them{NC}")


# ── The two blocks ───────────────────────────────────────────────────────────


def unanimous_split(rows: Sequence[sqlite3.Row]) -> tuple[int, int, int]:
    """UNANIMOUS debates as (unaided, edge-repaired, promoted).

    The whole content of this report, and the reason it is a function rather
    than three expressions inside a table: `UNANIMOUS` went up is the number
    that cannot distinguish a better prompt from a looser judge, and these three
    can.

    Disjoint, and ranked by how much help was given. A debate that had an edge
    repaired *and* was then promoted counts as promoted: the larger concession
    is the one that describes it, and buckets that overlap do not add up to the
    rate they are splitting.
    """
    unanimous = [row for row in rows if row["outcome"] == "UNANIMOUS"]
    promoted = sum(1 for row in unanimous if row["judged_cosmetic"])
    repaired = sum(
        1 for row in unanimous if not row["judged_cosmetic"] and row["judged_edges"]
    )
    return len(unanimous) - promoted - repaired, repaired, promoted


def convergence(rows: Sequence[sqlite3.Row], keep_all: bool) -> str:
    """Did they agree, and did they need help agreeing."""
    usable = rows if keep_all else [r for r in rows if convergence_excuse(r) is None]
    out = [f"{BOLD}CONVERGENCE{NC}  {DIM}{_span(rows)}{NC}\n"]

    body = []
    for engine, group in by_engine(usable).items():
        total = len(group)
        unaided, repaired, promoted = unanimous_split(group)
        body.append([
            engine, total,
            share(unaided + repaired + promoted, total),
            share(unaided, total),
            repaired,
            promoted,
            share(sum(r["outcome"] == "MAJORITY" for r in group), total),
            share(sum(r["outcome"] == "DEADLOCK" for r in group), total),
        ])

    out.append(table(
        ["engine", "N", "UNANIMOUS", "unaided", "+edges", "+cosmetic",
         "MAJORITY", "DEADLOCK"],
        body,
    ))

    stopped: dict[str, int] = {}
    for row in usable:
        stopped[row["terminated_by"]] = stopped.get(row["terminated_by"], 0) + 1
    out.append(
        f"\n  {DIM}stopped by{NC}   "
        + " · ".join(f"{why} {count}" for why, count in sorted(stopped.items()))
    )
    # A row decided a round short of the budget was not decided on the same
    # evidence as one that used the whole of it, whatever its outcome says.
    full = sum(row["tallied_round"] >= row["rounds_used"] for row in usable)
    out.append(f"  {DIM}tallied at the deepest round on {full} of {len(usable)}{NC}")
    if not keep_all:
        out.append(excuses(rows, convergence_excuse))
    return "\n".join(out)


def cost(rows: Sequence[sqlite3.Row], keep_all: bool) -> str:
    """What each engine spent, and the delta that is the point of the project."""
    usable = rows if keep_all else [r for r in rows if cost_excuse(r) is None]
    grouped = by_engine(usable)
    out = [f"\n{BOLD}COST PER DEBATE{NC}\n"]

    body, means = [], {}
    for engine, group in grouped.items():
        wall = mean([r["duration_s"] for r in group])
        calls = mean([r["llm_calls"] for r in group])
        means[engine] = (wall, calls)
        cpu = [r["cpu_s"] for r in group if r["cpu_s"] is not None]
        rss = [r["peak_rss_mb"] for r in group if r["peak_rss_mb"] is not None]
        body.append([
            engine, len(group), f"{wall:.1f}s", f"{calls:.1f}",
            f"{mean([r['selector_calls'] for r in group]):.1f}",
            f"{mean([r['prompt_tokens'] for r in group]) / 1000:.1f}k / "
            f"{mean([r['completion_tokens'] for r in group]) / 1000:.1f}k",
            f"{mean(cpu):.1f}" if cpu else "—",
            f"{mean(rss):.0f}" if rss else "—",
        ])

    out.append(table(
        ["engine", "N", "wall", "calls", "selector", "tokens in/out",
         "CPU s", "RSS MB"],
        body,
    ))

    # The delta against roundrobin, which is the baseline because it is the
    # engine with no LLM in the control path. Two numbers: the extra wall clock
    # an operator waits, and the extra calls that bought it.
    baseline = means.get("autogen_roundrobin")
    if baseline and len(means) > 1:
        for engine, (wall, calls) in means.items():
            if engine == "autogen_roundrobin":
                continue
            out.append(
                f"\n  {DIM}{engine} vs autogen_roundrobin:{NC} "
                f"{_delta(wall, baseline[0])} wall clock, "
                f"{_delta(calls, baseline[1])} LLM calls"
            )
    if not keep_all:
        out.append(excuses(rows, cost_excuse))
    return "\n".join(out)


def caveats(rows: Sequence[sqlite3.Row]) -> str:
    """Things that make the whole report a comparison of two different setups.

    Reported rather than filtered on: a mixed roster or a changed round budget
    may be exactly what was being tried, and the script has no way to know. What
    it can do is refuse to let it pass unmentioned.
    """
    notes = []
    for column, label in (
        ("models", "roster"),
        ("max_rounds", "round budget"),
        ("streamed_advisors", "streaming set"),
        ("node_id", "node"),
    ):
        distinct = {row[column] for row in rows}
        if len(distinct) > 1:
            notes.append(f"{len(distinct)} different {label}s")
    if not notes:
        return ""
    return (f"\n  {YELLOW}These rows do not describe one configuration:{NC} "
            f"{DIM}{', '.join(notes)}{NC}")


def _span(rows: Sequence[sqlite3.Row]) -> str:
    if not rows:
        return "no debates"
    first, last = rows[0]["started_at"][:10], rows[-1]["started_at"][:10]
    period = first if first == last else f"{first} → {last}"
    return f"{len(rows)} debates, {period}"


def _delta(value: float, baseline: float) -> str:
    if not baseline:
        return "—"
    return f"{100 * (value - baseline) / baseline:+.0f}%"


# ── CSV ──────────────────────────────────────────────────────────────────────


def dump_csv(rows: Sequence[sqlite3.Row]) -> None:
    """Every selected row, unfiltered, with the exclusion verdicts as columns.

    Unfiltered on purpose: a hidden filter inside an export is how a spreadsheet
    ends up disagreeing with the terminal for reasons nobody can reconstruct.
    The two `*_comparable` columns say what this script would have done, and the
    spreadsheet can decide for itself.
    """
    if not rows:
        return
    fields = list(rows[0].keys()) + ["convergence_comparable", "cost_comparable"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        writer.writerow({
            **dict(row),
            "convergence_comparable": int(convergence_excuse(row) is None),
            "cost_comparable": int(cost_excuse(row) is None),
        })


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None, help="database (default: MAGI_DB_PATH)")
    parser.add_argument("--last", type=int, default=0,
                        help="only the most recent N debates")
    parser.add_argument("--since", default=None, help="from this date, YYYY-MM-DD")
    parser.add_argument("--engine", default=None, help="one engine only")
    parser.add_argument("--node", default=None, help="one node only")
    parser.add_argument("--all", action="store_true",
                        help="include rows that are not comparable with the rest")
    parser.add_argument("--csv", action="store_true",
                        help="write the selected rows as CSV on stdout instead")
    args = parser.parse_args()

    path = args.db or Settings().db_path
    conn = connect(path)
    try:
        rows = select(conn, args)
    except sqlite3.DatabaseError as exc:
        print(f"Cannot read {path}: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if not rows:
        print(f"No debates in {path} match. Hold one:  "
              f"uv run python scripts/ask.py \"...\"", file=sys.stderr)
        return 1

    if args.csv:
        dump_csv(rows)
        return 0

    print()
    print(convergence(rows, args.all))
    print(cost(rows, args.all))
    if (note := caveats(rows)):
        print(note)
    print(f"\n{DIM}{path}, read {datetime.now(UTC):%Y-%m-%d %H:%M} UTC{NC}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
