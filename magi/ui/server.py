"""FastAPI: the static SPA, and one WebSocket carrying everything.

**One socket, not several.** Every client — the 7" kiosk, a laptop on the LAN —
opens ``/ui/stream`` and receives the same envelope stream. Two sockets would
mean two reconnect policies and two answers to "are we connected?", in a UI
whose panels exist to answer exactly that.

**No build step.** Plain HTML, CSS and JS served from ``resources/``. A Pi that
needs npm to render its own screen is a Pi that cannot be fixed in the field.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from magi.bus import Bus
from magi.config import Settings
from magi.constants import (
    SPAN_STT,
    TOPIC_ACTIVITY,
    TOPIC_DRAFT,
    TOPIC_QUESTION,
    TOPIC_STATUS,
    TOPIC_STT,
    TOPIC_TURN,
    TOPIC_VERDICT,
)
from magi.orchestrator import Magi
from magi.personas import PersonaSet
from magi.services import draft as draft_mod
from magi.services.draft import Draft
from magi.services.stt import STTService, STTUnavailable, is_available
from magi.setup.setup_tracing import get_tracer

logger = logging.getLogger(__name__)

RESOURCES = Path(__file__).parent / "resources"

#: The console is deployed by rsyncing files onto the node and the browser is a
#: long-lived kiosk that may not be reloaded for days. Without this it happily
#: serves yesterday's JS from memory cache and the operator sees a screen that
#: no longer matches the daemon behind it — which is worse than an obviously
#: broken one, because everything looks fine.
#:
#: `no-cache` rather than `no-store`: the browser may keep the file, it just has
#: to revalidate. ETag then makes the usual reload a 304 with no body, so being
#: always-fresh costs a round trip and not a download.
NO_CACHE = {"Cache-Control": "no-cache, must-revalidate"}


class _FreshStatics(StaticFiles):
    """StaticFiles that always revalidates. See :data:`NO_CACHE`."""

    def file_response(self, *args: Any, **kwargs: Any) -> Any:
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response

#: Every topic the SPA renders. Adding one here is all it takes to reach the
#: browser — the forwarder is generic on purpose, so a new panel needs no
#: change to the transport.
FORWARDED = (
    TOPIC_QUESTION, TOPIC_TURN, TOPIC_VERDICT, TOPIC_STATUS, TOPIC_DRAFT, TOPIC_STT,
    TOPIC_ACTIVITY,
)


class Ask(BaseModel):
    question: str


def create_app(
    bus: Bus,
    magi: Magi,
    personas: PersonaSet,
    settings: Settings,
    stt: STTService | None = None,
    draft: Draft | None = None,
) -> FastAPI:
    app = FastAPI(title="MAGI", docs_url=None, redoc_url=None)
    app.mount("/static", _FreshStatics(directory=RESOURCES), name="static")

    # One debate at a time. A second question does not queue behind the first —
    # it cancels it, which is the same barge-in a second PTT press performs.
    # Queueing would leave the operator watching an answer to a question they
    # have already moved on from.
    running: dict[str, asyncio.Task] = {}
    draft = draft if draft is not None else Draft()

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(RESOURCES / "index.html", headers=NO_CACHE)

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        """What the SPA needs to draw itself before any debate happens.

        The advisor roster is config, not code: the UI reads it from here rather
        than hardcoding three names, so a fourth advisor in the YAML appears on
        screen without a front-end change.
        """
        return {
            "node_id": settings.node_id,
            "engine": settings.engine,
            "max_rounds": settings.max_rounds,
            "advisors": [
                {"name": p.name, "archetype": p.archetype, "model": p.model}
                for p in personas.magi
            ],
            "orchestrator": personas.orchestrator.name,
            "speech_available": is_available(),
            "max_draft_lines": draft_mod.MAX_LINES,
        }

    @app.post("/api/ask")
    async def ask(body: Ask) -> JSONResponse:
        question = body.question.strip()
        if not question:
            return JSONResponse({"error": "empty question"}, status_code=400)

        await _cancel_running(running, magi)
        running["debate"] = asyncio.create_task(_run_debate(magi, question))
        return JSONResponse({"accepted": True})

    @app.get("/api/draft")
    async def get_draft() -> dict:
        return await draft.as_dict()

    @app.delete("/api/draft/{line_id}")
    async def delete_line(line_id: int) -> JSONResponse:
        """Drop one mis-heard line.

        By id, not by position: a delete that races an in-flight transcription
        would otherwise remove whichever line happened to land in that slot.
        """
        removed = await draft.delete(line_id)
        await draft_mod.publish(bus, draft, TOPIC_DRAFT)
        return JSONResponse({"removed": removed})

    @app.delete("/api/draft")
    async def clear_draft() -> JSONResponse:
        await draft.clear()
        await draft_mod.publish(bus, draft, TOPIC_DRAFT)
        return JSONResponse({"cleared": True})

    @app.post("/api/send")
    async def send() -> JSONResponse:
        """Commit the draft and start the debate.

        This is the only thing that spends three models' time, which is why a
        PTT press does not do it. Recognition gets things wrong and a debate
        costs minutes; the operator reads what was heard, deletes what was
        mangled, and only then commits.
        """
        question = await draft.question()
        if not question:
            return JSONResponse({"error": "the draft is empty"}, status_code=400)
        await draft.clear()
        await draft_mod.publish(bus, draft, TOPIC_DRAFT)
        await _cancel_running(running, magi)
        running["debate"] = asyncio.create_task(_run_debate(magi, question))
        return JSONResponse({"accepted": True, "question": question})

    @app.post("/api/cancel")
    async def cancel() -> JSONResponse:
        await _cancel_running(running, magi)
        return JSONResponse({"cancelled": True})

    @app.post("/stt")
    async def stt_http(request: Request) -> JSONResponse:
        """Transcribe one clip over HTTP.

        The console does not use this — PTT audio rides the WebSocket that is
        already open, saving a connection per press. It exists so a clip can be
        pushed from a script or a mic-less machine without driving a browser.
        """
        if stt is None or not is_available():
            return JSONResponse(
                {"error": "speech recognition is not installed on this node",
                 "hint": "./setup_env.sh --voice"},
                status_code=503,
            )
        audio = await request.body()
        if not audio:
            return JSONResponse({"error": "empty body"}, status_code=400)
        text = await stt.transcribe(audio)
        return JSONResponse({"text": "" if text == STTService.HALLUCINATION else text})

    @app.websocket("/ui/stream")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        # replay_last on every topic: a browser that connects mid-debate must
        # render the current state, not a blank screen until the next event.
        queues = {t: bus.subscribe(t, replay_last=True) for t in FORWARDED}
        logger.info("UI client connected")
        try:
            await _pump(ws, queues, stt, draft, bus)
        except (WebSocketDisconnect, RuntimeError):
            # RuntimeError is what Starlette raises when sending on a socket the
            # client already closed. Same event as WebSocketDisconnect, just
            # noticed from the other side; neither deserves a stack trace.
            logger.info("UI client disconnected")
        finally:
            for topic, queue in queues.items():
                bus.unsubscribe(topic, queue)

    return app


async def _pump(
    ws: WebSocket,
    queues: dict[str, asyncio.Queue],
    stt: STTService | None,
    draft: Draft,
    bus: Bus,
) -> None:
    """Forward every topic to one socket, and receive PTT audio on the same one.

    Audio comes back as **binary frames on this socket** rather than a POST per
    press: the connection is already open and already the thing the console
    trusts for liveness, so a press costs no handshake and a failure shows up
    on the one indicator the operator is already watching.
    """

    async def forward(topic: str, queue: asyncio.Queue) -> None:
        while True:
            payload = await queue.get()
            await ws.send_text(json.dumps({"topic": topic, "data": _plain(payload)}))

    async def reader() -> None:
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))
            if message.get("bytes"):
                await _on_audio(message["bytes"], stt, draft, bus)

    # asyncio.wait(FIRST_COMPLETED) rather than a TaskGroup. A TaskGroup wraps
    # whatever its children raise in an ExceptionGroup, which means a plain
    # `except WebSocketDisconnect` around this never fires and every ordinary
    # browser close logs an unhandled traceback. Here the first task to finish
    # is the disconnect, the rest are cancelled, and the original exception is
    # re-raised unwrapped for the caller to recognise.
    tasks = [asyncio.create_task(forward(t, q)) for t, q in queues.items()]
    tasks.append(asyncio.create_task(reader()))
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    for task in done:
        # `task.cancelled()` first: calling `.exception()` on a cancelled task
        # raises CancelledError rather than returning it, which would turn an
        # ordinary shutdown into a spurious error on the way out.
        if task.cancelled():
            continue
        if (error := task.exception()) is not None:
            raise error


def _plain(payload: Any) -> Any:
    """Pydantic records to JSON-safe dicts, everything else untouched."""
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    return payload


async def _run_debate(magi: Magi, question: str) -> None:
    try:
        await magi.debate(question)
    except asyncio.CancelledError:
        logger.info("Debate cancelled")
        raise
    except Exception:
        logger.exception("Debate failed")


async def _cancel_running(running: dict[str, asyncio.Task], magi: Magi) -> None:
    task = running.pop("debate", None)
    if task is None or task.done():
        return
    # Ask the framework first: ExternalTermination unwinds the group chat
    # cleanly and lets the turns already collected survive. Cancelling the task
    # outright is the fallback for a debate wedged somewhere AutoGen's
    # termination cannot reach.
    magi.cancel()
    with contextlib.suppress(asyncio.CancelledError, TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
    if not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def _on_audio(audio: bytes, stt: STTService | None, draft: Draft, bus: Bus) -> None:
    """One PTT press: transcribe it and append it to the draft.

    Every outcome tells the console something specific, because the operator's
    next action differs: a hallucination means the recording was bad, an empty
    result means the button was released too early, and a failure means the
    node is broken. "Nothing appeared" for all three would leave them pressing
    the button again into the same wall.
    """
    await bus.publish(TOPIC_STT, {"state": "transcribing", "bytes": len(audio)})
    with get_tracer().start_as_current_span(SPAN_STT) as span:
        span.set_attribute("magi.audio_bytes", len(audio))
        try:
            text = await stt.transcribe(audio) if stt else ""
        except STTUnavailable as exc:
            await bus.publish(TOPIC_STT, {"state": "unavailable", "detail": str(exc)})
            return
        except Exception as exc:
            logger.exception("Transcription failed")
            span.record_exception(exc)
            await bus.publish(TOPIC_STT, {"state": "error", "detail": str(exc)})
            return

    if text == STTService.HALLUCINATION:
        await bus.publish(TOPIC_STT, {"state": "discarded", "detail": "no clear speech"})
        return
    if not text:
        await bus.publish(TOPIC_STT, {"state": "empty", "detail": "nothing was heard"})
        return

    line = await draft.add(text)
    if line is None:
        await bus.publish(TOPIC_STT, {"state": "full", "detail": "the draft is full"})
        return
    await bus.publish(TOPIC_STT, {"state": "ok", "text": text})
    await draft_mod.publish(bus, draft, TOPIC_DRAFT)


async def start_ui(
    bus: Bus,
    magi: Magi,
    personas: PersonaSet,
    settings: Settings,
    stt: STTService | None = None,
    draft: Draft | None = None,
) -> None:
    """Serve until cancelled. Runs under ``supervise``."""
    import uvicorn

    app = create_app(bus, magi, personas, settings, stt, draft)
    config = uvicorn.Config(
        app,
        host=settings.ui_host,
        port=settings.ui_port,
        log_level="warning",
        access_log=False,
    )
    try:
        await uvicorn.Server(config).serve()
    except SystemExit as exc:
        # uvicorn answers a busy port with `sys.exit(3)`. SystemExit derives
        # from BaseException, so it slips past `supervise`'s `except Exception`
        # and unwinds the whole TaskGroup as a forty-line asyncio traceback —
        # for what is an operator running the node twice. Translated into the
        # one sentence that is actually true.
        raise RuntimeError(
            f"cannot serve the console on {settings.ui_host}:{settings.ui_port} — "
            "the port is already in use (is MAGI already running? "
            "pkill -f 'python -m magi')"
        ) from exc
