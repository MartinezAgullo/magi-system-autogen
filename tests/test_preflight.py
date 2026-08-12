"""Pre-flight checks, tested against a fake Ollama.

The distinction these tests defend is the one the whole check system rests on:
**a missing or non-compliant model is an error that stops the launch, while
model residency is only a warning.** Getting that backwards either bricks the
node over a tuning detail or lets it run debates that cannot possibly work.
"""

from __future__ import annotations

import json
import socket as sock_module
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from magi.config import Settings
from magi.models import MagiTurn
from magi.personas import PersonaSet
from magi.services import ollama_check

VALID_TURN = MagiTurn(
    position=(
        "Kubernetes is premature here: three people cannot absorb the operational "
        "surface it adds before they have a product to run on it."
    ),
    summary="No — the operational cost outweighs the benefit at this size.",
    agrees_with=[],
    confidence=0.9,
    critique=[],
).model_dump_json()

# Schema-valid and useless. A model that answers like this passes a validity
# check and cannot hold a debate, which is exactly what the substance check is
# there to catch.
HOLLOW_TURN = MagiTurn(
    position="No.",
    summary="No.",
    agrees_with=[],
    confidence=0.9,
    critique=[],
).model_dump_json()


@pytest.fixture
def personas() -> PersonaSet:
    return PersonaSet.model_validate(
        {
            "common_prompt": "rules",
            "magi": [
                {"name": "MELCHIOR", "model": "model-a:1b", "system_prompt": "a"},
                {"name": "CASPAR", "model": "model-b:1b", "system_prompt": "b"},
            ],
            "orchestrator": {"name": "MAGI", "model": "model-a:1b", "system_prompt": "o"},
        }
    )


@pytest.fixture
def free_port() -> int:
    """A port nothing is listening on.

    Pre-flight checks that the console's port is free, so a fixture left on the
    default 8000 would make the whole suite depend on whether a MAGI daemon
    happens to be running on the developer's machine. Tests must not care what
    else is on the box.
    """
    with sock_module.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def settings(tmp_path, free_port) -> Settings:
    return Settings(
        llm_backend="ollama",
        ollama_host="fake-spark",
        ui_port=free_port,
        db_path=tmp_path / "data" / "magi.db",
        tts_enabled=False,
        otel_enabled=False,
    )


def _fake_ollama(
    *,
    tags: list[str],
    running: list[str],
    schema_ok: bool = True,
    schema_status: int = 200,
    schema_body: str | None = None,
    keep_alive_s: float = 3600.0,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": t} for t in tags]})
        if path == "/api/ps":
            expiry = (datetime.now(UTC) + timedelta(seconds=keep_alive_s)).isoformat()
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": t, "expires_at": expiry, "size": 8_000_000_000}
                        for t in running
                    ]
                },
            )
        if path.endswith("/chat/completions"):
            if schema_status != 200:
                return httpx.Response(schema_status, text="unsupported response_format")
            content = schema_body or (VALID_TURN if schema_ok else "Sure, here goes.")
            return httpx.Response(
                200, json={"choices": [{"message": {"content": content}}]}
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


@pytest.fixture
def patch_client(monkeypatch):
    def _install(transport: httpx.MockTransport) -> None:
        original = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return _install


def _messages(report, level: str) -> str:
    return " | ".join(r.message for r in report.results if r.level == level)


async def test_all_green(settings, personas, patch_client):
    patch_client(_fake_ollama(tags=["model-a:1b", "model-b:1b"],
                              running=["model-a:1b", "model-b:1b"]))

    report = await ollama_check.run_preflight(settings, personas)

    assert not report.failed
    assert not report.warned, _messages(report, "warn")


async def test_missing_model_is_a_hard_error(settings, personas, patch_client):
    patch_client(_fake_ollama(tags=["model-a:1b"], running=["model-a:1b"]))

    report = await ollama_check.run_preflight(settings, personas)

    assert report.failed
    assert "model-b:1b" in _messages(report, "error")


async def test_missing_model_short_circuits_the_schema_probe(
    settings, personas, patch_client
):
    """Probing a model that is not pulled would trigger a multi-GB download on
    the inference host as a side effect of a health check."""
    patch_client(_fake_ollama(tags=["model-a:1b"], running=[]))

    report = await ollama_check.run_preflight(settings, personas)

    assert "Structured output" not in _messages(report, "ok")


async def test_non_resident_models_only_warn(settings, personas, patch_client):
    patch_client(_fake_ollama(tags=["model-a:1b", "model-b:1b"], running=["model-a:1b"]))

    report = await ollama_check.run_preflight(settings, personas)

    assert not report.failed
    assert report.warned
    assert "model-b:1b" in _messages(report, "warn")


async def test_residency_is_checked_after_the_probe_has_loaded_the_models(
    settings, personas, patch_client
):
    """The probe loads every advisor's model, so asking /api/ps first reports a
    cold state that pre-flight is about to invalidate itself."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "model-a:1b"}, {"name": "model-b:1b"}]}
            )
        if path == "/api/ps":
            expiry = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
            return httpx.Response(
                200,
                json={
                    "models": [
                        {"name": n, "expires_at": expiry, "size": 8_000_000_000}
                        for n in ("model-a:1b", "model-b:1b")
                    ]
                },
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_TURN}}]})

    patch_client(httpx.MockTransport(handler))

    await ollama_check.run_preflight(settings, personas)

    assert calls.index("/api/ps") > calls.index("/v1/chat/completions")


async def test_prose_instead_of_schema_is_a_hard_error(settings, personas, patch_client):
    """A model that answers in prose cannot take a turn: every turn is a
    schema-constrained MagiTurn."""
    patch_client(
        _fake_ollama(
            tags=["model-a:1b", "model-b:1b"],
            running=["model-a:1b", "model-b:1b"],
            schema_ok=False,
        )
    )

    report = await ollama_check.run_preflight(settings, personas)

    assert report.failed
    assert "Structured output FAILED" in _messages(report, "error")


async def test_backend_rejecting_json_schema_is_a_hard_error(
    settings, personas, patch_client
):
    """An older Ollama that does not know `response_format: json_schema` fails
    every model at once — the hint has to point at the server, not the model."""
    patch_client(
        _fake_ollama(
            tags=["model-a:1b", "model-b:1b"],
            running=["model-a:1b", "model-b:1b"],
            schema_status=400,
        )
    )

    report = await ollama_check.run_preflight(settings, personas)

    assert report.failed
    assert "HTTP 400" in _messages(report, "error")


async def test_unreachable_ollama_is_a_hard_error(settings, personas, patch_client):
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    patch_client(httpx.MockTransport(refuse))

    report = await ollama_check.run_preflight(settings, personas)

    assert report.failed
    assert "not reachable" in _messages(report, "error")


async def test_missing_ps_endpoint_warns_but_does_not_fail(
    settings, personas, patch_client
):
    """Older Ollama builds have no /api/ps. Residency becomes unknown, which is
    a caveat on the numbers, not a reason to refuse to run."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(404)
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "model-a:1b"}, {"name": "model-b:1b"}]}
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_TURN}}]})

    patch_client(httpx.MockTransport(handler))

    report = await ollama_check.run_preflight(settings, personas)

    assert not report.failed
    assert "residency is unknown" in _messages(report, "warn")


async def test_api_backend_without_a_key_is_a_hard_error(tmp_path, personas, free_port):
    settings = Settings(
        llm_backend="openai",
        openai_api_key="",
        ui_port=free_port,
        db_path=tmp_path / "data" / "magi.db",
        tts_enabled=False,
        otel_enabled=False,
    )

    report = await ollama_check.run_preflight(settings, personas)

    assert report.failed


async def test_schema_probe_sends_the_real_magiturn_schema(settings, personas):
    """The probe must exercise the schema the debate will actually use — a
    simplified one would pass on models that cannot handle nested critique."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_TURN}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ok, detail = await ollama_check.probe_structured_output(
            client, "http://fake/v1", "key", "model-a:1b"
        )

    assert ok, detail
    schema = seen["response_format"]["json_schema"]["schema"]
    assert set(MagiTurn.model_json_schema()["properties"]) == set(schema["properties"])


async def test_schema_valid_but_hollow_answers_are_rejected(
    settings, personas, patch_client
):
    """The check that the first version of this probe was missing. A model will
    satisfy the schema with position="No." — valid, and unable to debate."""
    patch_client(
        _fake_ollama(
            tags=["model-a:1b", "model-b:1b"],
            running=["model-a:1b", "model-b:1b"],
            schema_body=HOLLOW_TURN,
        )
    )

    report = await ollama_check.run_preflight(settings, personas)

    assert report.failed
    assert "empty of content" in _messages(report, "error")


async def test_reasoning_that_eats_the_budget_is_reported_as_such(settings, personas):
    """A thinking model with too small a budget returns empty content and its
    reasoning in a separate field. That is a budget failure, and saying
    "decoding failed" would send you chasing the wrong thing."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "", "reasoning": "x" * 1200}}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ok, detail = await ollama_check.probe_structured_output(
            client, "http://fake/v1", "key", "model-a:1b"
        )

    assert not ok
    assert "budget on reasoning" in detail


async def test_thinking_false_suppresses_reasoning_thinking_none_does_not():
    """`thinking: None` must stay distinguishable from `thinking: False`:
    nemotron3 produces no valid output at all with reasoning suppressed, so
    "leave the model's default alone" has to be expressible."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_TURN}}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        for thinking in (False, None, True):
            await ollama_check.probe_structured_output(
                client, "http://fake/v1", "key", "m:1b", thinking=thinking
            )

    assert seen[0]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in seen[1]
    assert "reasoning_effort" not in seen[2]


async def test_probe_uses_the_advisors_own_prompt_not_a_generic_one(
    settings, personas, patch_client
):
    """A thin probe prompt produces thin answers from every model, which had an
    earlier version of this check condemning models that were fine."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "model-a:1b"}, {"name": "model-b:1b"}]}
            )
        if request.url.path == "/api/ps":
            return httpx.Response(
                200, json={"models": [{"name": "model-a:1b"}, {"name": "model-b:1b"}]}
            )
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": VALID_TURN}}]})

    patch_client(httpx.MockTransport(handler))

    await ollama_check.run_preflight(settings, personas)

    # One probe per advisor, not per model tag: the same model behaves
    # differently under a different prompt and thinking setting.
    assert len(seen) == len(personas.magi)
    prompts = [p["messages"][0]["content"] for p in seen]
    assert prompts[0] == personas.system_prompt_for(personas.magi[0])


async def test_models_expiring_before_a_debate_ends_are_flagged(
    settings, personas, patch_client
):
    """Resident now is not resident in five minutes. Ollama's default keep-alive
    is 5 min, so a model can be loaded when pre-flight looks and evicted by the
    second round — surfacing as one inexplicably slow turn."""
    patch_client(
        _fake_ollama(
            tags=["model-a:1b", "model-b:1b"],
            running=["model-a:1b", "model-b:1b"],
            keep_alive_s=60.0,  # shorter than the 180s debate timeout
        )
    )

    report = await ollama_check.run_preflight(settings, personas)

    assert not report.failed
    assert "expire sooner than one debate" in _messages(report, "warn")


async def test_a_long_keep_alive_does_not_warn(settings, personas, patch_client):
    patch_client(
        _fake_ollama(
            tags=["model-a:1b", "model-b:1b"],
            running=["model-a:1b", "model-b:1b"],
            keep_alive_s=3600.0,
        )
    )

    report = await ollama_check.run_preflight(settings, personas)

    assert "expire sooner" not in _messages(report, "warn")


# ── Intermittent advisors ────────────────────────────────────────────────────


async def test_a_flaky_advisor_is_retried_rather_than_blocking_the_boot(
    settings, personas, patch_client
):
    """Gating startup on a single sample was wrong. Measured: nemotron3 answered
    "Not advisable" (13 chars) on one boot having produced 311 characters on the
    previous one — so a one-shot check blocks a node that would have worked, at
    the worst possible moment."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": "model-a:1b"}, {"name": "model-b:1b"}]}
            )
        if path == "/api/ps":
            expiry = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
            return httpx.Response(200, json={"models": [
                {"name": n, "expires_at": expiry, "size": 8_000_000_000}
                for n in ("model-a:1b", "model-b:1b")]})
        calls["n"] += 1
        # Terse on the first attempt for each advisor, substantive after.
        body = HOLLOW_TURN if calls["n"] % 2 == 1 else VALID_TURN
        return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})

    patch_client(httpx.MockTransport(handler))

    report = await ollama_check.run_preflight(settings, personas)

    assert not report.failed
    assert "needed 2 attempts" in _messages(report, "warn")


async def test_an_advisor_that_is_never_substantive_still_fails(
    settings, personas, patch_client
):
    """The check keeps its teeth: a model that cannot produce a real position in
    three consecutive tries breaks every debate it joins."""
    patch_client(
        _fake_ollama(
            tags=["model-a:1b", "model-b:1b"],
            running=["model-a:1b", "model-b:1b"],
            schema_body=HOLLOW_TURN,
        )
    )

    report = await ollama_check.run_preflight(settings, personas)

    assert report.failed
    assert "3 attempts" in _messages(report, "error")


# ── The console port ─────────────────────────────────────────────────────────


async def test_a_busy_console_port_fails_preflight(personas, patch_client, tmp_path):
    """Running the node twice is an ordinary mistake, and uvicorn answers it
    with forty lines of asyncio traceback ending in SystemExit — which reads
    like a crash. Caught at boot with one sentence instead."""
    with sock_module.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]

        busy = Settings(
            llm_backend="ollama", ollama_host="fake-spark", ui_port=port,
            db_path=tmp_path / "data" / "magi.db", tts_enabled=False, otel_enabled=False,
        )
        patch_client(_fake_ollama(tags=["model-a:1b", "model-b:1b"],
                                  running=["model-a:1b", "model-b:1b"]))
        report = await ollama_check.run_preflight(busy, personas)

    assert report.failed
    assert "already in use" in _messages(report, "error")
    assert "pkill" in " ".join(
        h for r in report.results for h in r.hints
    ), "the fix has to be in the hints, not left as an exercise"


async def test_a_free_console_port_passes(settings, personas, patch_client):
    patch_client(_fake_ollama(tags=["model-a:1b", "model-b:1b"],
                              running=["model-a:1b", "model-b:1b"]))

    report = await ollama_check.run_preflight(settings, personas)

    assert "port" in _messages(report, "ok").lower()


# ── Streaming ────────────────────────────────────────────────────────────────
#
# Whether an advisor streams is configuration, never a rule about a seat or a
# model tag. What the code owes in return is verifying that configuration
# against the model that is actually serving, because the streaming and
# non-streaming paths do not fail in the same ways: structured output goes
# through a stricter helper when streamed, and token usage is only reported if
# it is asked for.


def _streaming_personas(**overrides) -> PersonaSet:
    advisor = {"name": "CASPAR", "model": "model-b:1b", "system_prompt": "b"}
    advisor.update(overrides)
    return PersonaSet.model_validate(
        {
            "magi": [
                {"name": "MELCHIOR", "model": "model-a:1b", "system_prompt": "a"},
                advisor,
            ],
            "orchestrator": {"name": "MAGI", "model": "model-a:1b", "system_prompt": "o"},
        }
    )


def test_streaming_is_off_unless_asked_for():
    """A default that quietly turned it on would remove the length retry from
    every advisor at once, and the symptom is one truncated debate a week
    later."""
    personas = _streaming_personas()

    assert personas.streaming_names() == []
    assert personas.stream_for(personas.magi[1]) is False


def test_a_persona_overrides_the_default_in_both_directions():
    on = PersonaSet.model_validate(
        {
            "defaults": {"stream": True},
            "magi": [
                {"name": "MELCHIOR", "model": "a:1b", "system_prompt": "a",
                 "stream": False},
                {"name": "CASPAR", "model": "b:1b", "system_prompt": "b"},
            ],
            "orchestrator": {"name": "MAGI", "model": "a:1b", "system_prompt": "o"},
        }
    )

    assert on.streaming_names() == ["CASPAR"]


async def test_an_advisor_that_streams_is_probed_through_the_streaming_path():
    """Verifying a streaming advisor without streaming has not verified it."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if not body.get("stream"):
            return httpx.Response(
                200, json={"choices": [{"message": {"content": VALID_TURN}}]}
            )
        chunks = "".join(
            f'data: {{"choices":[{{"delta":{{"content":{json.dumps(c)}}}}}]}}\n\n'
            for c in [VALID_TURN[:20], VALID_TURN[20:]]
        )
        return httpx.Response(200, text=chunks + "data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ok_plain, _ = await ollama_check.probe_structured_output(
            client, "http://fake/v1", "key", "m:1b", stream=False
        )
        ok_stream, detail = await ollama_check.probe_structured_output(
            client, "http://fake/v1", "key", "m:1b", stream=True
        )

    assert ok_plain and ok_stream
    assert "streamed" in detail
    assert "stream" not in seen[0]
    assert seen[1]["stream"] is True
    # Without this a streamed probe reports no usage at all, so it would bless a
    # configuration that then silently measures nothing.
    assert seen[1]["stream_options"] == {"include_usage": True}


async def test_a_stream_that_produces_no_chunks_fails_the_probe():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="data: [DONE]\n\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        ok, detail = await ollama_check.probe_structured_output(
            client, "http://fake/v1", "key", "m:1b", stream=True
        )

    assert not ok
    assert "no chunks" in detail


def test_streaming_while_reasoning_warns_because_the_retry_is_gone():
    """The one combination streaming makes dangerous, expressed over the
    persona's settings rather than its name: a reasoning advisor is the one that
    needs the wider-budget retry, and streaming is the one thing that removes
    it."""
    report = ollama_check.PreflightReport()

    ollama_check._check_streaming_config(
        report, _streaming_personas(stream=True, thinking=True)
    )

    warnings = [r for r in report.results if r.level == "warn"]
    assert len(warnings) == 1
    assert "CASPAR" in warnings[0].message
    assert "cannot be retried" in warnings[0].message


def test_streaming_without_reasoning_does_not_warn():
    report = ollama_check.PreflightReport()

    ollama_check._check_streaming_config(
        report, _streaming_personas(stream=True, thinking=False)
    )

    assert [r for r in report.results if r.level == "warn"] == []
