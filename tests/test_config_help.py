"""`magi-config` exists so the operator-facing list of settings is generated
rather than copied. These tests defend that: they fail if a new setting is
added without a section or a description, if `.env.example` names a variable
that no longer exists, or if the API key ever reaches stdout."""

from __future__ import annotations

import os

import pytest

from magi.config import REPO_ROOT, SECTIONS, Settings
from magi.config_help import (
    _MASK,
    collect,
    display_value,
    env_value,
    main,
    parse_env_file,
    render_env,
    render_table,
)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch, tmp_path):
    """Resolve against an empty env file unless a test says otherwise.

    Without this the developer's own `.env` decides what "default" means, and
    the source column is exactly what is under test.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", tmp_path / "absent.env")
    for key in [k for k in list(os.environ) if k.upper().startswith("MAGI_")]:
        monkeypatch.delenv(key, raising=False)


# ── The generator's contract with config.py ──────────────────────────────────


def test_every_field_is_documented():
    for row in collect():
        assert row.description, f"{row.field} has no description"
        assert row.section in SECTIONS, f"{row.field} has an unknown section {row.section!r}"


def test_every_field_appears_in_the_table():
    rows = collect()
    table = render_table(rows)
    for row in rows:
        assert row.env_name in table


def test_sections_print_in_declared_order():
    table = render_table(collect())
    positions = [table.find(s) for s in SECTIONS if s in table]
    assert positions == sorted(positions)


# ── Source resolution ────────────────────────────────────────────────────────


def test_default_is_reported_as_default():
    row = next(r for r in collect() if r.field == "ui_port")
    assert row.source == "default"
    assert row.value == 8000
    assert not row.overridden


def test_environment_wins_and_is_named(monkeypatch):
    monkeypatch.setenv("MAGI_UI_PORT", "9999")
    row = next(r for r in collect() if r.field == "ui_port")
    assert row.source == "environment"
    assert row.value == 9999


def test_env_file_is_named_when_it_is_the_source(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("MAGI_ENGINE=autogen_selector\n", encoding="utf-8")
    monkeypatch.setitem(Settings.model_config, "env_file", env)

    row = next(r for r in collect(Settings(_env_file=env)) if r.field == "engine")
    assert row.source == ".env"
    assert row.value == "autogen_selector"


def test_env_file_value_equal_to_default_still_counts_as_set(monkeypatch, tmp_path):
    """Explicitly pinning a value is not the same as never having chosen one."""
    env = tmp_path / ".env"
    env.write_text("MAGI_UI_PORT=8000\n", encoding="utf-8")
    monkeypatch.setitem(Settings.model_config, "env_file", env)

    row = next(r for r in collect(Settings(_env_file=env)) if r.field == "ui_port")
    assert row.source == ".env"
    assert row.overridden


def test_parse_env_file_handles_comments_quotes_and_export(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        '# a comment\n\nexport MAGI_ENGINE="autogen_selector"\nMAGI_UI_PORT=8000\nbroken line\n',
        encoding="utf-8",
    )
    assert parse_env_file(env) == {
        "MAGI_ENGINE": "autogen_selector",
        "MAGI_UI_PORT": "8000",
    }


def test_parse_env_file_tolerates_a_missing_file(tmp_path):
    assert parse_env_file(tmp_path / "nope.env") == {}


# ── Secrets ──────────────────────────────────────────────────────────────────


def test_api_key_never_reaches_stdout(monkeypatch, capsys):
    monkeypatch.setenv("MAGI_OPENAI_API_KEY", "sk-do-not-print-me")

    assert main([]) == 0
    assert main(["--format", "env"]) == 0

    out = capsys.readouterr().out
    assert "sk-do-not-print-me" not in out
    assert _MASK in out
    # The name still shows, so the operator can see that a key is configured.
    assert "MAGI_OPENAI_API_KEY" in out


def test_unset_secret_says_so():
    row = next(r for r in collect() if r.field == "openai_api_key")
    assert row.secret
    assert display_value(row) == "(unset)"


# ── The env dump ─────────────────────────────────────────────────────────────


def test_env_dump_covers_every_setting_and_parses_back():
    rows = collect()
    parsed = parse_env_file_text(render_env(rows))
    assert set(parsed) == {r.env_name for r in rows}


def parse_env_file_text(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def test_env_dump_round_trips_from_any_directory(tmp_path, monkeypatch):
    """Loading the dump back must reproduce the node, not a CWD-dependent
    approximation of it. Paths are the trap: relative ones resolve against the
    working directory."""
    dump = tmp_path / "dump.env"
    dump.write_text(render_env(collect()), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    restored = Settings(_env_file=dump)
    live = Settings(_env_file=None)
    differing = {
        name
        for name in Settings.model_fields
        if getattr(restored, name) != getattr(live, name) and name != "openai_api_key"
    }
    assert not differing


def test_env_dump_writes_bools_as_pydantic_reads_them():
    row = next(r for r in collect() if r.field == "otel_enabled")
    assert env_value(row) == "1"
    assert display_value(row) == "true"


# ── The committed .env.example ───────────────────────────────────────────────


def test_env_example_names_only_real_settings():
    """The one drift direction that actually bites: a variable in .env.example
    that the code no longer reads sets nothing, silently."""
    known = {f"MAGI_{name.upper()}" for name in Settings.model_fields}
    example = REPO_ROOT / ".env.example"

    named = set()
    for line in example.read_text(encoding="utf-8").splitlines():
        line = line.strip().lstrip("#").strip()
        if line.startswith("MAGI_") and "=" in line:
            named.add(line.partition("=")[0].strip())

    assert named, "nothing was parsed out of .env.example"
    assert named <= known, f"unknown settings in .env.example: {sorted(named - known)}"
