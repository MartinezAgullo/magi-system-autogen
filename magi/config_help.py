"""``magi-config`` — print every setting, its effective value and where it came from.

Generated from :class:`magi.config.Settings` itself, never from a hand-written
list beside it. The question this answers at a terminal is not "what options
exist" — ``.env.example`` covers the common ones — but "what is this node
*actually* configured to do right now, and which file said so". Two of the
project's recurring questions are of exactly that shape: is it talking to the
right Spark, and was tracing really off for that benchmark run.

Precedence, which the ``source`` column reports:

    environment variable  >  .env  >  default in config.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import ValidationError
from pydantic_settings import BaseSettings

from magi.config import REPO_ROOT, SECTIONS, Settings

# Anything whose name contains one of these is printed masked. `magi-config`
# output ends up pasted into issues and chat windows.
_SECRET_HINTS = ("api_key", "secret", "token", "password")
_MASK = "••••••••"

_ENV_PREFIX = "MAGI_"


class Row(NamedTuple):
    """One setting, resolved."""

    field: str
    env_name: str
    section: str
    description: str
    value: Any
    default: Any
    source: str  # "environment" | ".env" | "default"
    secret: bool

    @property
    def overridden(self) -> bool:
        return self.source != "default"


# ── Resolution ───────────────────────────────────────────────────────────────


def _env_file_path(model: type[BaseSettings] = Settings) -> Path | None:
    raw = model.model_config.get("env_file")
    return Path(raw) if raw else None


def parse_env_file(path: Path | None) -> dict[str, str]:
    """Read a ``.env`` into a dict of upper-cased keys.

    Deliberately not a full dotenv implementation: this only needs to know
    *which* keys the file sets, so the source column can name it.
    """
    if path is None or not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        values[key.strip().upper()] = value.strip().strip("\"'")
    return values


def _in_environ(env_name: str) -> bool:
    """Case-insensitive, because pydantic-settings matches env vars that way."""
    target = env_name.upper()
    return any(k.upper() == target for k in os.environ)


def collect(settings: Settings | None = None) -> list[Row]:
    """Resolve every field to a :class:`Row`, in declaration order."""
    settings = settings or Settings()
    dotenv = parse_env_file(_env_file_path())

    rows: list[Row] = []
    for name, field in Settings.model_fields.items():
        env_name = f"{_ENV_PREFIX}{name.upper()}"
        extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}

        if _in_environ(env_name):
            source = "environment"
        elif env_name in dotenv:
            source = ".env"
        else:
            source = "default"

        rows.append(
            Row(
                field=name,
                env_name=env_name,
                section=str(extra.get("section", "Misc")),
                description=field.description or "",
                value=getattr(settings, name),
                default=field.default,
                source=source,
                secret=any(hint in name for hint in _SECRET_HINTS),
            )
        )
    return rows


def _grouped(rows: Sequence[Row]) -> list[tuple[str, list[Row]]]:
    """Rows by section, in ``SECTIONS`` order. Unknown sections come last."""
    order = {name: i for i, name in enumerate(SECTIONS)}
    seen: dict[str, list[Row]] = {}
    for row in rows:
        seen.setdefault(row.section, []).append(row)
    return sorted(seen.items(), key=lambda kv: order.get(kv[0], len(order)))


# ── Formatting ───────────────────────────────────────────────────────────────


def _shorten(value: Any) -> str:
    """Paths inside the repo print relative to it.

    The absolute form is three times as wide and says nothing the operator
    does not already know about where the checkout is.
    """
    if isinstance(value, Path):
        try:
            return str(value.relative_to(REPO_ROOT))
        except ValueError:
            return str(value)
    return str(value)


def display_value(row: Row) -> str:
    """The value as shown. Secrets never appear, set or unset."""
    if row.secret:
        return _MASK if row.value else "(unset)"
    if isinstance(row.value, bool):
        return "true" if row.value else "false"
    if row.value == "":
        return '""'
    return _shorten(row.value)


def env_value(row: Row) -> str:
    """The value as an env-file literal. ``1``/``0`` for bools, as pydantic reads.

    Paths stay absolute here, unlike in the table. A relative path in a ``.env``
    resolves against the working directory, so the shortened form would only
    mean the same thing when the file is loaded from the repo root — which is
    true of ``launch.sh`` and of nothing else.
    """
    if row.secret:
        return ""
    if isinstance(row.value, bool):
        return "1" if row.value else "0"
    return str(row.value)


class _Ink:
    """ANSI colours, or nothing at all. Same palette as launch.sh."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def __call__(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def cyan(self, t: str) -> str:
        return self("0;36", t)

    def green(self, t: str) -> str:
        return self("0;32", t)

    def dim(self, t: str) -> str:
        return self("2", t)


def _wrap(text: str, width: int, indent: str) -> list[str]:
    """Greedy wrap. Avoids textwrap only to keep the indent handling obvious."""
    if not text:
        return []
    lines: list[str] = []
    current = indent
    for word in text.split():
        candidate = f"{current} {word}" if current != indent else f"{indent}{word}"
        if len(candidate) > width and current != indent:
            lines.append(current)
            current = f"{indent}{word}"
        else:
            current = candidate
    lines.append(current)
    return lines


def render_table(rows: Sequence[Row], *, color: bool = False, width: int | None = None) -> str:
    """The human view: grouped, with the effective value and its source."""
    ink = _Ink(color)
    width = width or min(shutil.get_terminal_size((100, 24)).columns, 110)

    name_w = max((len(r.env_name) for r in rows), default=0)
    value_w = max((len(display_value(r)) for r in rows), default=0)
    value_w = min(value_w, 34)

    env_file = _env_file_path()
    env_note = (
        f"{env_file} (present)" if env_file and env_file.is_file() else f"{env_file} (not present)"
    )

    out: list[str] = [
        ink.cyan("MAGI configuration"),
        ink.dim(f"  {len(rows)} settings, all overridable with the {_ENV_PREFIX} prefix."),
        ink.dim(f"  env file: {env_note}"),
        ink.dim("  advisors, models and prompts are not here — see config/magi.yaml"),
        "",
    ]

    for section, section_rows in _grouped(rows):
        out.append(ink.cyan(f"{section}"))
        for row in section_rows:
            value = display_value(row)
            painted = ink.green(value) if row.overridden else value
            pad = " " * max(0, value_w - len(value))
            out.append(
                f"  {row.env_name:<{name_w}}  {painted}{pad}  {ink.dim('(' + row.source + ')')}"
            )
            for line in _wrap(row.description, width, " " * (name_w + 4)):
                out.append(ink.dim(line))
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def render_env(rows: Sequence[Row]) -> str:
    """A complete ``.env`` reflecting this node's effective configuration.

    Not what ``.env.example`` is: that file is curated down to what an operator
    normally touches, and its prose is worth more than completeness. This is
    the exhaustive dump, for capturing a working node or scaffolding a new one.
    """
    out: list[str] = [
        "# Generated by `magi-config --format env`.",
        "# Effective configuration of this node. Secrets are blank; fill them in.",
        "",
    ]
    for section, section_rows in _grouped(rows):
        out.append(f"# ── {section} " + "─" * max(0, 60 - len(section)))
        for row in section_rows:
            if row.description:
                out.append(f"# {row.description}")
            out.append(f"{row.env_name}={env_value(row)}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ── Entry point ──────────────────────────────────────────────────────────────


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="magi-config",
        description="Print every MAGI setting, its effective value and where it came from.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "env"),
        default="table",
        help="table: the human view (default). env: an exhaustive .env dump.",
    )
    parser.add_argument("--no-color", action="store_true", help="Never emit ANSI colour.")
    args = parser.parse_args(argv)

    # This is a diagnostic tool, so it has to survive the configuration being
    # the thing that is broken. Failing with a raw traceback would hide the
    # answer the operator came for.
    try:
        rows = collect()
    except ValidationError as exc:
        print("Configuration is invalid:\n", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    if args.format == "env":
        sys.stdout.write(render_env(rows))
        return 0

    color = not args.no_color and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
    sys.stdout.write(render_table(rows, color=color))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
