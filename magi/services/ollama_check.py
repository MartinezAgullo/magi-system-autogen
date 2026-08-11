"""Pre-flight checks against the inference backend.

Two of these carry most of the value.

**The structured-output smoke test.** The entire design rests on every model
honouring a JSON schema (AutoGen's ``output_content_type``). Ollama supports
schema-constrained decoding over its OpenAI-compatible endpoint, but per-model
compliance varies, and a model that cannot produce a valid ``MagiTurn`` breaks
every debate it takes part in. Finding that out at boot costs a few seconds;
finding it out mid-debate costs the session. So it is a hard error.

**Model residency.** With several models taking turns, Ollama may evict and
reload weights between them — and then every latency number measures storage
rather than the framework. The Pi cannot read the Spark's environment, so this
cannot be asserted; ``/api/ps`` is observed instead and the result is a warning
that gets recorded, so a suspect run can be excluded from the benchmark rather
than quietly skewing it.

Errors abort the launch, warnings do not. A missing model is a fact; a
suboptimal residency setup is a caveat on the numbers, and the operator may
have their reasons.
"""

from __future__ import annotations

import json
import logging
import socket
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx

from magi.config import Settings
from magi.constants import (
    API_TIMEOUT_S,
    COLLECTOR_PROBE_TIMEOUT_S,
    GENERIC_ADVISOR_PROMPT,
    MIN_POSITION_CHARS,
    PROBE_ATTEMPTS,
    PROBE_MAX_TOKENS,
    PROBE_QUESTION,
    SCHEMA_PROBE_TIMEOUT_S,
)
from magi.models import MagiTurn
from magi.personas import PersonaSet

logger = logging.getLogger(__name__)

Level = Literal["ok", "warn", "error"]


@dataclass
class CheckResult:
    level: Level
    message: str
    hints: list[str] = field(default_factory=list)


@dataclass
class PreflightReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(self, level: Level, message: str, *hints: str) -> None:
        self.results.append(CheckResult(level, message, list(hints)))

    @property
    def failed(self) -> bool:
        return any(r.level == "error" for r in self.results)

    @property
    def warned(self) -> bool:
        return any(r.level == "warn" for r in self.results)


# ── Ollama API ───────────────────────────────────────────────────────────────


async def list_tags(client: httpx.AsyncClient, base_url: str) -> set[str]:
    """Model tags Ollama has pulled (``GET /api/tags``)."""
    resp = await client.get(f"{base_url}/api/tags", timeout=API_TIMEOUT_S)
    resp.raise_for_status()
    return {m["name"] for m in resp.json().get("models", [])}


async def list_running(client: httpx.AsyncClient, base_url: str) -> list[dict]:
    """Models currently resident in memory (``GET /api/ps``).

    Older Ollama builds do not have this endpoint. Absence is not a failure —
    it only means residency cannot be observed, which is reported as such.
    """
    resp = await client.get(f"{base_url}/api/ps", timeout=API_TIMEOUT_S)
    resp.raise_for_status()
    return resp.json().get("models", [])


async def probe_structured_output(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    *,
    system_prompt: str = GENERIC_ADVISOR_PROMPT,
    question: str = PROBE_QUESTION,
    max_tokens: int = PROBE_MAX_TOKENS,
    temperature: float = 0.3,
    thinking: bool | None = None,
) -> tuple[bool, str]:
    """Ask *model* for one ``MagiTurn`` and judge whether it is usable.

    Goes through the OpenAI-compatible endpoint rather than Ollama's native
    ``/api/chat``: that is the path AutoGen will use, and the two do not fail in
    the same ways. Testing the path you will not use proves nothing.

    Three things this checks, learned the hard way — the first version checked
    only the middle one and passed models that could not hold a debate:

    1. **Non-empty content.** A thinking model given a small token budget spends
       it all on reasoning and returns an empty ``content`` with the reasoning in
       a separate field. It looks like a decoding failure and is really a budget
       failure, so the probe uses a realistic budget and says so when it fails.
    2. **Schema validity.**
    3. **Substance.** A model will happily satisfy the schema with
       ``position: "Yes."`` — valid, and useless in a debate. Measured against
       :data:`MIN_POSITION_CHARS`.

    The caller passes the **real persona prompt**, not a generic one. A thin
    prompt produces thin answers from every model, which had this probe
    condemning models that were fine and would have had it bless models that
    were not.

    Returns ``(ok, detail)``.
    """
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "magi_turn",
                "schema": MagiTurn.model_json_schema(),
            },
        },
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    # Ollama's OpenAI-compatible equivalent of `think: false`. Only sent when
    # thinking is explicitly disabled: some models (nemotron3) fail to produce
    # valid structured output at all with reasoning suppressed, so "leave the
    # model's default alone" has to stay expressible.
    if thinking is False:
        payload["reasoning_effort"] = "none"

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        resp = await client.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=SCHEMA_PROBE_TIMEOUT_S,
        )
    except httpx.TimeoutException:
        return False, f"timed out after {SCHEMA_PROBE_TIMEOUT_S:.0f}s (cold model load?)"
    except httpx.HTTPError as exc:
        return False, f"request failed: {exc}"

    if resp.status_code != httpx.codes.OK:
        body = resp.text[:200].replace("\n", " ")
        return False, f"HTTP {resp.status_code}: {body}"

    try:
        message = resp.json()["choices"][0]["message"]
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return False, f"unexpected response envelope: {exc}"

    content = (message.get("content") or "").strip()
    if not content:
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        if reasoning:
            return False, (
                f"spent the whole {max_tokens}-token budget on reasoning "
                f"({len(reasoning)} chars) and emitted no answer"
            )
        return False, "returned an empty message (schema decoding may have failed)"

    try:
        turn = MagiTurn.model_validate_json(content)
    except ValueError as exc:
        excerpt = content[:160].replace("\n", " ")
        return False, f"output did not match MagiTurn: {excerpt!r} ({type(exc).__name__})"

    if len(turn.position.strip()) < MIN_POSITION_CHARS:
        return False, (
            f"schema-valid but empty of content: position was "
            f"{len(turn.position.strip())} chars ({turn.position.strip()!r})"
        )

    return True, f"valid MagiTurn, {len(turn.position)} chars of position"


# ── The checks ───────────────────────────────────────────────────────────────


def _check_keep_alive(
    report: PreflightReport,
    running: list[dict],
    wanted: set[str],
    debate_timeout_s: float,
) -> None:
    """Warn about models due to be evicted before a debate could finish.

    Residency at boot is not the same as residency for the next five minutes.
    Ollama's default keep-alive is five minutes, so a model can be loaded when
    pre-flight looks and gone by MELCHIOR's second turn — the resulting reload
    lands in the middle of the run and shows up as one inexplicably slow turn.

    Compared against the debate timeout rather than a fixed number because that
    is the longest a single debate can take by construction: anything expiring
    sooner can bite during one.
    """
    now = datetime.now(UTC)
    soon: list[str] = []

    for model in running:
        if model.get("name") not in wanted:
            continue
        raw = model.get("expires_at")
        if not raw:
            continue
        try:
            expires = datetime.fromisoformat(raw)
        except ValueError:
            continue
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        remaining = (expires - now).total_seconds()
        if remaining < debate_timeout_s:
            soon.append(f"{model['name']} ({remaining / 60:.0f} min)")

    if soon:
        report.add(
            "warn",
            "Models expire sooner than one debate can take: " + ", ".join(soon),
            "They will be evicted and reloaded mid-debate, which shows up as one",
            "inexplicably slow turn. On the inference host:  OLLAMA_KEEP_ALIVE=1h",
        )


async def _check_ollama(
    report: PreflightReport, settings: Settings, personas: PersonaSet
) -> None:
    base = settings.ollama_base_url
    wanted = personas.distinct_models()

    async with httpx.AsyncClient() as client:
        try:
            tags = await list_tags(client, base)
        except httpx.HTTPError as exc:
            report.add(
                "error",
                f"Ollama not reachable at {base} ({type(exc).__name__})",
                f"Is the host up?  ping -c1 {settings.ollama_host}",
                "If it answers, Ollama is bound to 127.0.0.1 — on that host run:",
                "  OLLAMA_HOST=0.0.0.0:11434 ollama serve",
            )
            return

        report.add("ok", f"Ollama reachable at {base} ({len(tags)} models pulled)")

        missing = sorted(wanted - tags)
        if missing:
            report.add(
                "error",
                f"Models named in {settings.personas_file.name} are not pulled: "
                + ", ".join(missing),
                "On the inference host:",
                *[f"  ollama pull {m}" for m in missing],
            )
            return

        report.add("ok", "All configured models are pulled: " + ", ".join(sorted(wanted)))

        # ── Structured output (hard) ────────────────────────────────────
        # Probed per ADVISOR rather than per model: the same model behaves
        # differently under a different system prompt and a different thinking
        # setting, and it is the advisor that has to work, not the tag.
        for persona in personas.magi:
            # Up to PROBE_ATTEMPTS, first success wins. These models are
            # intermittent enough that one sample is not evidence: gating boot
            # on a single draw blocks nodes that would have worked fine.
            failures: list[str] = []
            for _ in range(PROBE_ATTEMPTS):
                ok, detail = await probe_structured_output(
                    client,
                    settings.llm_base_url,
                    settings.llm_api_key,
                    persona.model,
                    system_prompt=personas.system_prompt_for(persona),
                    max_tokens=personas.max_tokens_for(persona) or PROBE_MAX_TOKENS,
                    temperature=personas.temperature_for(persona) or 0.3,
                    thinking=personas.thinking_for(persona),
                )
                if ok:
                    break
                failures.append(detail)

            attempts = len(failures) + (1 if ok else 0)
            if ok:
                # The attempts that failed still get said out loud. An advisor
                # that needed three tries at boot will need them mid-debate
                # too, and that is worth knowing before the numbers look odd.
                suffix = f" (after {attempts} attempts)" if attempts > 1 else ""
                report.add(
                    "ok",
                    f"Structured output OK: {persona.name} ({persona.model}) — "
                    f"{detail}{suffix}",
                )
                if failures:
                    report.add(
                        "warn",
                        f"{persona.name} needed {attempts} attempts — it is "
                        f"unreliable under this configuration",
                        *[f"  attempt {i}: {f}" for i, f in enumerate(failures, 1)],
                    )
            else:
                report.add(
                    "error",
                    f"Structured output FAILED: {persona.name} ({persona.model}) — "
                    f"{PROBE_ATTEMPTS} attempts, last: {detail}",
                    "Every debate turn is a schema-constrained MagiTurn, so this",
                    "advisor cannot take part. In config/magi.yaml: swap the model,",
                    "raise max_tokens, or change `thinking` — some models produce",
                    "nothing valid with reasoning suppressed, others need it off.",
                )

        # ── Residency (warn only) ───────────────────────────────────────
        # Deliberately AFTER the probe. The probe itself loads every advisor's
        # model, so asking before it runs reports a cold state that pre-flight
        # is about to invalidate — it warned "3 of 3 not resident" and then
        # loaded all three, two lines later. What matters is the state the node
        # will actually start a debate in, which is the state after the probe.
        try:
            running = await list_running(client, base)
        except httpx.HTTPError:
            report.add(
                "warn",
                "Could not read /api/ps — model residency is unknown",
                "Latency figures from this run may include model reloads.",
            )
        else:
            resident = {m["name"] for m in running}
            not_resident = sorted(wanted - resident)
            if not_resident:
                report.add(
                    "warn",
                    f"{len(not_resident)} of {len(wanted)} models are not resident "
                    "even after loading them: " + ", ".join(not_resident),
                    "Ollama is evicting them, so it will reload weights between turns",
                    "and latency will measure disk I/O rather than the framework.",
                    "On the inference host:",
                    f"  OLLAMA_MAX_LOADED_MODELS={len(wanted)}",
                    "  OLLAMA_KEEP_ALIVE=1h",
                )
            else:
                total_gb = sum(m.get("size", 0) for m in running) / 1e9
                report.add(
                    "ok",
                    f"All {len(wanted)} models resident ({total_gb:.0f} GB)",
                )

            _check_keep_alive(report, running, wanted, settings.debate_timeout_s)


def _check_api_backend(report: PreflightReport, settings: Settings) -> None:
    if settings.openai_api_key:
        report.add("ok", f"API backend configured: {settings.openai_base_url}")
    else:
        report.add(
            "error",
            "MAGI_LLM_BACKEND=openai but MAGI_OPENAI_API_KEY is empty",
            "Set the key, or switch to MAGI_LLM_BACKEND=ollama.",
        )


def _check_port(report: PreflightReport, settings: Settings) -> None:
    """Is the console's port free?

    Checked here rather than discovered by uvicorn because the failure is an
    ordinary operator mistake — a second instance, or one left running in
    another terminal — and uvicorn answers it with forty lines of asyncio
    traceback ending in SystemExit. That reads like a crash. It is a busy port.
    """
    host = "127.0.0.1" if settings.ui_host in ("0.0.0.0", "::") else settings.ui_host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2.0)
        free = probe.connect_ex((host, settings.ui_port)) != 0

    if free:
        report.add("ok", f"Console port {settings.ui_port} is free")
    else:
        report.add(
            "error",
            f"Port {settings.ui_port} is already in use — MAGI is probably "
            "already running",
            "Find it and stop it:",
            "  pgrep -af 'python -m magi'",
            "  pkill -f 'python -m magi'",
            "Or serve on another port:  MAGI_UI_PORT=8001 ./launch.sh",
        )


def _check_local(report: PreflightReport, settings: Settings) -> None:
    # Storage — the benchmark record has nowhere to go without it.
    db_dir = settings.db_path.parent
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        probe = db_dir / ".magi_write_probe"
        probe.touch()
        probe.unlink()
    except OSError as exc:
        report.add("error", f"Database directory not writable: {db_dir} ({exc})")
    else:
        report.add("ok", f"Database directory writable: {db_dir}")

    # Voice output. Silence is a legitimate configuration; a voice that is
    # configured but broken is not.
    if not settings.tts_enabled:
        report.add("ok", "TTS disabled (MAGI_TTS_ENABLED=0) — the node stays silent")
    elif not settings.tts_voice:
        report.add(
            "warn",
            "No Piper voice configured — verdicts will be shown but not spoken",
            "Set MAGI_TTS_VOICE to a .onnx path, or MAGI_TTS_ENABLED=0 to be explicit.",
        )
    else:
        voice = Path(settings.tts_voice)
        if not voice.exists():
            report.add("error", f"Piper voice not found: {voice}")
        elif not voice.with_suffix(voice.suffix + ".json").exists():
            report.add(
                "error",
                f"Piper voice config missing: {voice}.json must sit next to the .onnx",
            )
        else:
            report.add("ok", f"TTS ready: {voice.name}")


async def _check_otel(report: PreflightReport, settings: Settings) -> None:
    if not settings.otel_enabled:
        report.add("ok", "Tracing disabled (MAGI_OTEL_ENABLED=0) — benchmark-clean run")
        return

    if settings.otel_exporter != "otlp":
        report.add("ok", f"Tracing on, exporter={settings.otel_exporter}")
        return

    try:
        async with httpx.AsyncClient() as client:
            await client.get(settings.otel_endpoint, timeout=COLLECTOR_PROBE_TIMEOUT_S)
    except httpx.HTTPError:
        report.add(
            "warn",
            f"OTLP collector not reachable at {settings.otel_endpoint}",
            "Spans will be dropped. The node still answers questions — a missing",
            "collector must never stop a debate.",
        )
    else:
        report.add("ok", f"OTLP collector reachable at {settings.otel_endpoint}")


async def run_preflight(settings: Settings, personas: PersonaSet) -> PreflightReport:
    """Run every check and return the collected results.

    Never raises for a check failure — the caller decides what an error means,
    which is how ``--check-only`` can report everything instead of stopping at
    the first problem.
    """
    report = PreflightReport()

    report.add(
        "ok",
        f"{len(personas.magi)} advisors: "
        + ", ".join(f"{p.name}={p.model}" for p in personas.magi),
    )

    if settings.llm_backend == "ollama":
        await _check_ollama(report, settings, personas)
    else:
        _check_api_backend(report, settings)

    _check_port(report, settings)
    _check_local(report, settings)
    await _check_otel(report, settings)
    return report
