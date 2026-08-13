"""Durable storage. One module, one job: the benchmark record."""

from magi.store.debates import SCHEMA_VERSION, DebateStore, PreflightRun, connect

__all__ = ["SCHEMA_VERSION", "DebateStore", "PreflightRun", "connect"]
