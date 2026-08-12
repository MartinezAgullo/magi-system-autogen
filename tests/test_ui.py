"""The UI server's contract with the SPA.

The browser is not tested here; what is tested is that the daemon offers the
shape the SPA reads, and that the one-socket rule holds — every topic the
console renders must arrive on `/ui/stream`, because a panel fed by a second
connection would need a second reconnect policy in a UI whose job is to answer
"are we connected?" unambiguously.
"""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
from fastapi.testclient import TestClient

from magi.bus import Bus
from magi.config import Settings
from magi.constants import (
    TOPIC_ACTIVITY,
    TOPIC_STATUS,
    TOPIC_STT,
    TOPIC_TURN,
    TOPIC_VERDICT,
)
from magi.models import MagiTurn, TurnRecord
from magi.personas import PersonaSet
from magi.services.draft import Draft
from magi.ui.server import FORWARDED, create_app


@pytest.fixture
def unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def personas() -> PersonaSet:
    return PersonaSet.model_validate(
        {
            "common_prompt": "rules",
            "magi": [
                {"name": "MELCHIOR", "archetype": "scientist",
                 "model": "model-a:1b", "system_prompt": "a"},
                {"name": "CASPAR", "archetype": "skeptic",
                 "model": "model-b:1b", "system_prompt": "b"},
            ],
            "orchestrator": {"name": "MAGI", "model": "model-a:1b", "system_prompt": "o"},
        }
    )


class FakeMagi:
    def __init__(self):
        self.asked: list[str] = []
        self.cancelled = 0

    async def debate(self, question: str):
        self.asked.append(question)

    def cancel(self) -> None:
        self.cancelled += 1


@pytest.fixture
def client(personas):
    bus = Bus()
    magi = FakeMagi()
    app = create_app(bus, magi, personas, Settings())
    with TestClient(app) as c:
        c.bus = bus
        c.magi = magi
        yield c


def test_config_exposes_the_roster_from_the_yaml(client):
    """The SPA draws one node per advisor from this, rather than hardcoding
    three names — a fourth advisor in the YAML must appear without a front-end
    change."""
    config = client.get("/api/config").json()

    assert [a["name"] for a in config["advisors"]] == ["MELCHIOR", "CASPAR"]
    assert config["advisors"][0]["archetype"] == "scientist"
    # The model is rendered under each name: which one is behind a seat is the
    # thing about this system that changes most often, so the screen says it
    # rather than leaving it to whoever remembers what the YAML held.
    assert config["advisors"][0]["model"] == "model-a:1b"
    assert config["orchestrator"] == "MAGI"


def test_config_reports_whether_speech_is_installed(client):
    """The PTT button renders disabled-but-present off this flag. Hiding it
    would misrepresent what the node is for; enabling it would be worse."""
    assert client.get("/api/config").json()["speech_available"] in (True, False)


def test_index_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert "magi.css" in client.get("/").text
    assert client.get("/static/magi.css").status_code == 200
    assert client.get("/static/magi.js").status_code == 200


def test_stt_answers_503_not_404_while_speech_is_unwired(client):
    """404 would be indistinguishable from a broken route, and the SPA would
    show a network error for a capability that is simply not installed."""
    response = client.post("/stt")

    assert response.status_code == 503
    assert "voice" in response.json()["hint"]


def test_an_empty_question_is_rejected(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400


async def test_a_late_client_is_replayed_the_current_state(personas, unused_port):
    """A browser that opens after a debate must render it, not a blank screen
    until someone asks again. That is what `replay_last` buys, and it is the
    difference between a kiosk that survives a reload mid-mission and one that
    does not.

    Against a real uvicorn rather than `TestClient`: TestClient bridges the
    socket across a thread portal and cancels the server task when the block
    exits, which surfaces as a CancelledError that has nothing to do with the
    code under test. Two genuine bugs hid behind that noise — a TaskGroup
    wrapping WebSocketDisconnect in an ExceptionGroup so the handler never
    fired, and `task.exception()` raising on a cancelled task — so the noise
    was worth removing rather than tolerating.
    """
    import uvicorn
    import websockets

    bus = Bus()
    await bus.publish(TOPIC_STATUS, {"condition": "GREEN", "temp_c": 56.0})
    app = create_app(bus, FakeMagi(), personas, Settings())

    config = uvicorn.Config(app, host="127.0.0.1", port=unused_port, log_level="error")
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)

        async with websockets.connect(f"ws://127.0.0.1:{unused_port}/ui/stream") as ws:
            frame = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))

        assert frame["topic"] == TOPIC_STATUS
        assert frame["data"]["temp_c"] == 56.0
    finally:
        server.should_exit = True
        await serving


async def test_a_disconnecting_client_is_not_an_error(personas, unused_port, caplog):
    """Every browser close ends this way. If it logged a traceback, the journal
    would fill with stack traces for the most ordinary event the node sees."""
    import logging

    import uvicorn
    import websockets

    bus = Bus()
    app = create_app(bus, FakeMagi(), personas, Settings())
    config = uvicorn.Config(app, host="127.0.0.1", port=unused_port, log_level="error")
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        while not server.started:
            await asyncio.sleep(0.02)
        with caplog.at_level(logging.ERROR):
            async with websockets.connect(f"ws://127.0.0.1:{unused_port}/ui/stream"):
                pass
            await asyncio.sleep(0.2)
    finally:
        server.should_exit = True
        await serving

    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_pydantic_records_are_serialised_for_the_browser():
    """TurnRecord reaches the socket as a pydantic model; sending it raw would
    fail json.dumps and take the whole stream down with it."""
    from magi.ui.server import _plain

    record = TurnRecord(
        advisor="MELCHIOR",
        round_index=2,
        turn=MagiTurn(position="p" * 50, summary="s", confidence=0.5),
    )

    plain = _plain(record)

    assert plain["advisor"] == "MELCHIOR"
    assert json.dumps(plain)


def test_forwarded_covers_what_the_orchestrator_publishes():
    """The bus topics Magi writes to and the ones the UI forwards must not
    drift apart: a turn published to a topic nobody forwards is a turn the
    console never draws."""
    assert TOPIC_TURN in FORWARDED
    assert TOPIC_VERDICT in FORWARDED
    assert TOPIC_STATUS in FORWARDED
    # Without this the console shows nothing at all for the 13-30 s of a blind
    # round and the ~90 s of deliberation, which is precisely when an operator
    # concludes the node has hung.
    assert TOPIC_ACTIVITY in FORWARDED


# ── Speech and the draft ─────────────────────────────────────────────────────


class FakeSTT:
    """Stands in for faster-whisper, which is an optional extra."""

    HALLUCINATION = "__hallucination__"

    def __init__(self, result="Should we adopt Kubernetes?"):
        self.result = result
        self.calls = 0

    async def transcribe(self, audio: bytes) -> str:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def _audio_outcome(personas, result) -> tuple[list, Draft]:
    """Push one clip through the audio handler and collect what was published."""
    from magi.services.stt import STTService
    from magi.ui.server import _on_audio

    bus = Bus()
    draft = Draft()
    published: list[dict] = []

    queue = bus.subscribe(TOPIC_STT)
    fake = FakeSTT(result)
    fake.HALLUCINATION = STTService.HALLUCINATION
    await _on_audio(b"audio", fake, draft, bus)
    while not queue.empty():
        published.append(queue.get_nowait())
    return published, draft


async def test_a_good_press_becomes_a_draft_line(personas):
    published, draft = await _audio_outcome(personas, "Should we adopt Kubernetes?")

    assert [p["state"] for p in published] == ["transcribing", "ok"]
    assert await draft.question() == "Should we adopt Kubernetes?"


async def test_an_empty_press_says_so_instead_of_failing_silently(personas):
    """Released-too-early is the commonest PTT mistake. "Nothing appeared" would
    leave the operator pressing the button again into the same wall."""
    published, draft = await _audio_outcome(personas, "")

    assert published[-1]["state"] == "empty"
    assert await draft.question() == ""


async def test_a_hallucination_is_reported_differently_from_silence(personas):
    """They need different messages: one is a bad recording, the other is a
    button released too early — and the operator's next move differs."""
    from magi.services.stt import STTService

    published, draft = await _audio_outcome(personas, STTService.HALLUCINATION)

    assert published[-1]["state"] == "discarded"
    assert await draft.question() == ""


async def test_a_transcription_failure_reaches_the_console(personas):
    published, draft = await _audio_outcome(personas, RuntimeError("model exploded"))

    assert published[-1]["state"] == "error"
    assert "model exploded" in published[-1]["detail"]


async def test_send_starts_a_debate_with_the_composed_question(personas):
    """The draft is a question in pieces; SEND is the only thing that spends
    three models' time on it."""
    bus = Bus()
    magi = FakeMagi()
    draft = Draft()
    await draft.add("Should we adopt Kubernetes")
    await draft.add("for our first product?")

    app = create_app(bus, magi, personas, Settings(), None, draft)
    with TestClient(app) as client:
        response = client.post("/api/send")

    assert response.status_code == 200
    assert magi.asked == ["Should we adopt Kubernetes for our first product?"]
    # Cleared on send: the next press starts a new question, not an addendum.
    assert await draft.question() == ""


async def test_send_on_an_empty_draft_is_refused(personas):
    app = create_app(Bus(), FakeMagi(), personas, Settings(), None, Draft())
    with TestClient(app) as client:
        assert client.post("/api/send").status_code == 400


async def test_a_line_can_be_deleted_over_rest(personas):
    draft = Draft()
    await draft.add("Should we adopt Kubernetes")
    bad = await draft.add("for our fur spot duck")
    await draft.add("for our first product?")

    app = create_app(Bus(), FakeMagi(), personas, Settings(), None, draft)
    with TestClient(app) as client:
        assert client.delete(f"/api/draft/{bad.id}").json()["removed"]

    assert await draft.question() == "Should we adopt Kubernetes for our first product?"


def test_stt_over_http_is_503_when_speech_is_not_installed(personas):
    """503, not 404: the SPA must tell "not installed" from "broken route"."""
    app = create_app(Bus(), FakeMagi(), personas, Settings(), None, Draft())
    with TestClient(app) as client:
        response = client.post("/stt", content=b"audio")

    assert response.status_code == 503
    assert "voice" in response.json()["hint"]


def test_the_console_ships_a_fullscreen_control(client):
    """The kiosk is already fullscreen via `chromium --kiosk`, but every other
    way in — a laptop on the LAN, the Pi's own browser opened by hand — shows
    the console inside browser chrome, which on an 800x480 panel costs a third
    of the triad.

    Asserted here rather than in a JS suite because there is no front-end test
    harness: this catches the markup being dropped, and the behaviour is
    verified against a real browser.
    """
    page = client.get("/").text
    script = client.get("/static/magi.js").text

    assert 'id="fullscreen"' in page
    assert 'id="fs-label"' in page
    # Both spellings: Safari still needs the prefix, and the Pi's Chromium does
    # not — a control that works on only one of them is worse than none.
    assert "requestFullscreen" in script
    assert "webkitRequestFullscreen" in script
    # The label follows the real state, not the last click: Esc leaves
    # fullscreen without going through the button.
    assert "fullscreenchange" in script


def test_assets_are_never_served_stale(client):
    """The console is deployed by rsyncing files onto the node, and the kiosk
    browser may not be reloaded for days. Without revalidation it serves
    yesterday's JS from memory cache and the operator sees a screen that no
    longer matches the daemon behind it — worse than an obviously broken one,
    because everything looks fine.

    `no-cache`, not `no-store`: the browser may keep the file, it just has to
    ask. ETag then turns the usual reload into a 304 with no body.
    """
    for path in ("/", "/static/magi.js", "/static/magi.css"):
        response = client.get(path)
        assert response.headers["cache-control"] == "no-cache, must-revalidate", path
        assert response.headers.get("etag"), f"{path} has no etag to revalidate against"


def test_an_unchanged_asset_costs_a_round_trip_not_a_download(client):
    """Always-fresh must not mean always-downloaded, on a node whose browser
    reloads over WiFi."""
    etag = client.get("/static/magi.js").headers["etag"]

    response = client.get("/static/magi.js", headers={"If-None-Match": etag})

    assert response.status_code == 304
    assert not response.content
