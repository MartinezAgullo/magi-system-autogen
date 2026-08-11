"""Settings via pydantic-settings (env prefix ``MAGI_``, reads ``.env``).

Every field can be overridden by an environment variable with the ``MAGI_``
prefix: ``MAGI_OLLAMA_HOST=10.0.0.5`` overrides :pyattr:`ollama_host`. Values in
a ``.env`` at the repo root are loaded automatically, but real environment
variables always win.

Nothing here names a model or an advisor. Those live in ``config/magi.yaml``
(see ``personas.py``) because rotating them is the point of the project, not an
edge case.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: .../magi-system-autogen/magi/config.py
REPO_ROOT = Path(__file__).resolve().parents[1]

Engine = Literal["autogen_roundrobin", "autogen_selector"]


class Settings(BaseSettings):
    """MAGI daemon configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MAGI_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Identity ─────────────────────────────────────────────────────────
    node_id: str = "magi-01"

    # ── Inference backend ────────────────────────────────────────────────
    # The Spark's Ollama, reached over WiFi. The node holds no weights.
    # DHCP: this moves whenever the network does. Current LAN as of 2026-08-11.
    ollama_host: str = "192.168.68.121"
    ollama_port: int = 11434

    # `ollama` talks to the Spark; `openai` talks to a real API-key provider,
    # which is how this is developed on a MacBook with the Spark powered off.
    # Both go through the same AutoGen client class — only base_url and api_key
    # change, so there is no second code path to keep working.
    llm_backend: Literal["ollama", "openai"] = "ollama"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""

    # ── Debate ───────────────────────────────────────────────────────────
    engine: Engine = "autogen_roundrobin"
    personas_file: Path = REPO_ROOT / "config" / "magi.yaml"

    # One blind round plus (max_rounds - 1) deliberation rounds.
    max_rounds: int = Field(default=3, ge=1)
    # Feeds AutoGen's TimeoutTermination. A voice interface that thinks for
    # five minutes is broken regardless of how good the answer is.
    debate_timeout_s: float = Field(default=180.0, gt=0)

    # ── Speech ───────────────────────────────────────────────────────────
    stt_model: str = "small"
    stt_compute_type: str = "int8"
    # The Pi 5 has four cores and the daemon is otherwise idle while
    # transcribing, so it may have all of them.
    stt_cpu_threads: int = 4
    stt_preload: bool = True
    # Pinned rather than auto-detected. Whisper's language detection is another
    # thing that can be wrong about a two-second clip, and this project is
    # English throughout.
    stt_language: str = "en"
    # 1 is greedy. 5 is measurably better on short noisy clips and costs
    # perhaps a second on a Pi — worth it when the alternative is the operator
    # deleting the line and saying it again.
    stt_beam_size: int = 5

    tts_enabled: bool = True
    # Path to a Piper .onnx voice. Empty means the node stays silent; the
    # screen still works, which is a legitimate way to run it.
    tts_voice: str = ""
    tts_binary: str = "piper"

    # ── Storage ──────────────────────────────────────────────────────────
    # The durable benchmark record. Shared schema with the sibling repo.
    db_path: Path = REPO_ROOT / "data" / "magi.db"

    # ── UI ───────────────────────────────────────────────────────────────
    ui_host: str = "0.0.0.0"
    ui_port: int = 8000

    # ── Observability ────────────────────────────────────────────────────
    # OTel costs CPU and memory on the node whose CPU and memory are the object
    # of study. This flag must genuinely disable it — no provider, no exporter,
    # no-op spans — and headline overhead figures are taken with it off.
    otel_enabled: bool = True
    otel_exporter: Literal["otlp", "file", "console"] = "otlp"
    # The collector runs wherever it is convenient — docker/ brings one up on
    # the dev machine, which is why this defaults to the MacBook rather than
    # the Spark. The Spark's job is inference, not observability.
    otel_endpoint: str = "http://192.168.68.117:4318"
    # Must differ from the sibling repo's, or traces from the two
    # implementations merge into one service and stop being comparable.
    otel_service_name: str = "magi-autogen"
    otel_file_path: Path = REPO_ROOT / "data" / "traces.jsonl"

    metrics_enabled: bool = True
    metrics_sample_interval_s: float = 1.0

    # How often the node reports its own temperature and load to the console.
    # 5s is slow enough to be free and fast enough that a thermal problem shows
    # up while the operator is still looking at the screen. 0 disables it.
    telemetry_interval_s: float = 5.0

    # ── Misc ─────────────────────────────────────────────────────────────
    # Disables Pi-specific services so the daemon runs on the MacBook.
    fake_hw: bool = False
    log_level: str = "INFO"

    # ── Derived ──────────────────────────────────────────────────────────

    @property
    def ollama_base_url(self) -> str:
        """Ollama's native API root (``/api/tags``, ``/api/ps``, ``/api/chat``)."""
        return f"http://{self.ollama_host}:{self.ollama_port}"

    @property
    def llm_base_url(self) -> str:
        """OpenAI-compatible root, which is what AutoGen's client is given.

        Ollama exposes one at ``/v1``. Pointing the same client class at either
        this or a real provider is what makes MacBook development possible.
        """
        if self.llm_backend == "ollama":
            return f"{self.ollama_base_url}/v1"
        return self.openai_base_url

    @property
    def llm_api_key(self) -> str:
        """Ollama ignores the key but the OpenAI client requires a non-empty one."""
        if self.llm_backend == "ollama":
            return "ollama"
        return self.openai_api_key
