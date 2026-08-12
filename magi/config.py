"""Settings via pydantic-settings (env prefix ``MAGI_``, reads ``.env``).

Every field can be overridden by an environment variable with the ``MAGI_``
prefix: ``MAGI_OLLAMA_HOST=10.0.0.5`` overrides :pyattr:`ollama_host`. Values in
a ``.env`` at the repo root are loaded automatically, but real environment
variables always win.

Nothing here names a model or an advisor. Those live in ``config/magi.yaml``
(see ``personas.py``) because rotating them is the point of the project, not an
edge case.

Every field also carries a section and a one-line description, via
:func:`_setting`. That is what ``magi-config`` (and ``./launch.sh --help``)
prints, so the operator-facing list of options is generated from this file
rather than hand-copied beside it. A second copy would drift, and the drift
would be discovered on the Pi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: .../magi-system-autogen/magi/config.py
REPO_ROOT = Path(__file__).resolve().parents[1]

Engine = Literal["autogen_roundrobin", "autogen_selector"]

# Print order for `magi-config`. Every field's section must be one of these;
# tests/test_config_help.py asserts it, so adding a section here is the only
# thing needed to group new settings.
SECTIONS: tuple[str, ...] = (
    "Identity",
    "Inference",
    "Debate",
    "Speech",
    "Storage",
    "UI",
    "Observability",
    "Misc",
)


def _setting(default: Any, section: str, description: str, **kwargs: Any) -> Any:
    """A settings field that ``magi-config`` can document.

    The section and the one-line description live on the field itself. The
    longer "why" belongs in the comments around it, which no generator reads
    and no operator needs at a terminal.
    """
    return Field(
        default=default,
        description=description,
        json_schema_extra={"section": section},
        **kwargs,
    )


class Settings(BaseSettings):
    """MAGI daemon configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MAGI_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Identity ─────────────────────────────────────────────────────────
    node_id: str = _setting(
        "magi-01", "Identity", "Name of this node, recorded alongside every debate."
    )

    # ── Inference backend ────────────────────────────────────────────────
    # The Spark's Ollama, reached over WiFi. The node holds no weights.
    # DHCP: this moves whenever the network does. Current LAN as of 2026-08-11.
    ollama_host: str = _setting(
        "192.168.68.121", "Inference", "Host serving Ollama. Only tokens travel; no weights."
    )
    ollama_port: int = _setting(11434, "Inference", "Port Ollama listens on.")

    # `ollama` talks to the Spark; `openai` talks to a real API-key provider,
    # which is how this is developed on a MacBook with the Spark powered off.
    # Both go through the same AutoGen client class — only base_url and api_key
    # change, so there is no second code path to keep working.
    llm_backend: Literal["ollama", "openai"] = _setting(
        "ollama", "Inference", "`ollama` for the Spark, `openai` for an API-key provider."
    )
    openai_base_url: str = _setting(
        "https://api.openai.com/v1",
        "Inference",
        "OpenAI-compatible root, used when llm_backend=openai.",
    )
    openai_api_key: str = _setting(
        "", "Inference", "API key for the openai backend. Ollama ignores it."
    )

    # ── Debate ───────────────────────────────────────────────────────────
    engine: Engine = _setting(
        "autogen_roundrobin",
        "Debate",
        "Which AutoGen team drives the debate. selector costs an LLM call per turn.",
    )
    personas_file: Path = _setting(
        REPO_ROOT / "config" / "magi.yaml",
        "Debate",
        "YAML defining the three advisors: names, models, prompts.",
    )

    # One blind round plus (max_rounds - 1) deliberation rounds.
    max_rounds: int = _setting(
        3, "Debate", "One blind round plus (max_rounds - 1) deliberation rounds.", ge=1
    )
    # Feeds AutoGen's TimeoutTermination. A voice interface that thinks for
    # five minutes is broken regardless of how good the answer is.
    debate_timeout_s: float = _setting(
        180.0, "Debate", "Wall-clock budget for one debate; feeds TimeoutTermination.", gt=0
    )

    # ── Speech ───────────────────────────────────────────────────────────
    stt_model: str = _setting("small", "Speech", "faster-whisper model size.")
    stt_compute_type: str = _setting(
        "int8", "Speech", "faster-whisper quantisation. int8 on the Pi."
    )
    # The Pi 5 has four cores and the daemon is otherwise idle while
    # transcribing, so it may have all of them.
    stt_cpu_threads: int = _setting(4, "Speech", "Cores faster-whisper may use.")
    stt_preload: bool = _setting(
        True, "Speech", "Load the Whisper model at boot rather than on the first press."
    )
    # Pinned rather than auto-detected. Whisper's language detection is another
    # thing that can be wrong about a two-second clip, and this project is
    # English throughout.
    stt_language: str = _setting(
        "en",
        "Speech",
        "Pinned transcription language. Detection is another thing that can be wrong.",
    )
    # 1 is greedy. 5 is measurably better on short noisy clips and costs
    # perhaps a second on a Pi — worth it when the alternative is the operator
    # deleting the line and saying it again.
    stt_beam_size: int = _setting(
        5, "Speech", "Whisper beam width. 1 is greedy; 5 is better on short noisy clips."
    )

    tts_enabled: bool = _setting(True, "Speech", "Speak the final verdict aloud.")
    # Path to a Piper .onnx voice. Empty means the node stays silent; the
    # screen still works, which is a legitimate way to run it.
    tts_voice: str = _setting(
        "", "Speech", "Path to a Piper .onnx voice. Empty means the node stays silent."
    )
    tts_binary: str = _setting("piper", "Speech", "Piper executable to invoke.")

    # ── Storage ──────────────────────────────────────────────────────────
    # The durable benchmark record. Shared schema with the sibling repo.
    db_path: Path = _setting(
        REPO_ROOT / "data" / "magi.db",
        "Storage",
        "SQLite benchmark record. Shared schema with the sibling repo.",
    )

    # ── UI ───────────────────────────────────────────────────────────────
    ui_host: str = _setting("0.0.0.0", "UI", "Address the console binds to.")
    ui_port: int = _setting(8000, "UI", "Port the console is served on.")

    # ── Observability ────────────────────────────────────────────────────
    # OTel costs CPU and memory on the node whose CPU and memory are the object
    # of study. This flag must genuinely disable it — no provider, no exporter,
    # no-op spans — and headline overhead figures are taken with it off.
    otel_enabled: bool = _setting(
        True, "Observability", "Master tracing switch. Off means no provider and no-op spans."
    )
    otel_exporter: Literal["otlp", "file", "console"] = _setting(
        "otlp", "Observability", "Where spans go."
    )
    # The collector runs wherever it is convenient — docker/ brings one up on
    # the dev machine, which is why this defaults to the MacBook rather than
    # the Spark. The Spark's job is inference, not observability.
    otel_endpoint: str = _setting(
        "http://192.168.68.117:4318", "Observability", "OTLP collector, when exporter=otlp."
    )
    # Must differ from the sibling repo's, or traces from the two
    # implementations merge into one service and stop being comparable.
    otel_service_name: str = _setting(
        "magi-autogen",
        "Observability",
        "Service name in the backend. Must differ from the sibling repo's.",
    )
    otel_file_path: Path = _setting(
        REPO_ROOT / "data" / "traces.jsonl",
        "Observability",
        "Where the `file` exporter writes JSON-lines spans.",
    )

    metrics_enabled: bool = _setting(
        True, "Observability", "Record per-turn and per-debate metrics."
    )
    metrics_sample_interval_s: float = _setting(
        1.0, "Observability", "How often CPU and RSS are sampled during a debate."
    )

    # How often the node reports its own temperature and load to the console.
    # 5s is slow enough to be free and fast enough that a thermal problem shows
    # up while the operator is still looking at the screen. 0 disables it.
    telemetry_interval_s: float = _setting(
        5.0, "Observability", "How often the console gets SoC temperature and load. 0 disables."
    )

    # ── Misc ─────────────────────────────────────────────────────────────
    # Disables Pi-specific services so the daemon runs on the MacBook.
    fake_hw: bool = _setting(
        False, "Misc", "Disable Pi-specific services so the daemon runs on a laptop."
    )
    log_level: str = _setting("INFO", "Misc", "Root log level.")

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
