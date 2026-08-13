"""Boot: config, personas, event bus, supervised services.

Nothing is wired yet beyond the boot banner — the orchestrator, speech and UI
services arrive in the order set out in ``CLAUDE.md`` § Status. The banner is
here from the start because it is what tells you, over ssh on a Pi, which
config a running node actually picked up: the answer is otherwise spread across
``.env``, real environment variables and ``config/magi.yaml``.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from magi import personas as personas_mod
from magi.bus import Bus
from magi.config import Settings
from magi.constants import PREFLIGHT_FRESH_S, TOPIC_STATUS
from magi.orchestrator import Magi
from magi.services import stream_view
from magi.services.draft import Draft
from magi.services.stt import STTService, is_available
from magi.services.telemetry import telemetry_service
from magi.setup.setup_logs import setup_logging
from magi.setup.setup_tracing import setup_tracing, shutdown_tracing
from magi.store.debates import DebateStore
from magi.supervision import supervise
from magi.ui.server import start_ui

logger = logging.getLogger("magi")


def _speech_installed() -> bool:
    from magi.services.stt import is_available

    return is_available()


def _banner(settings: Settings, personas: personas_mod.PersonaSet) -> None:
    logger.info("MAGI starting (AutoGen edition)")
    logger.info("  Node:      %s", settings.node_id)
    logger.info("  Engine:    %s", settings.engine)
    logger.info("  Inference: %s @ %s", settings.llm_backend, settings.llm_base_url)
    logger.info(
        "  Rounds:    max %d, timeout %.0fs", settings.max_rounds, settings.debate_timeout_s
    )
    for persona in personas.magi:
        logger.info("  %-10s %-14s %s", persona.name, persona.archetype, persona.model)
    # NOT "orchestrator", however the YAML key reads. The orchestrator is the
    # `Magi` class in orchestrator/magi.py — it drives the phases, tallies the
    # votes and decides the outcome. The *agent* named MAGI only writes the
    # sentences that get spoken, for an outcome it is handed as a constraint.
    # Printing "orchestrator" next to a model tag told the operator that a 12B
    # model was in charge of the verdict, which is the one thing this design
    # deliberately never does.
    #
    # The same persona also backs the JUDGE agent and, under autogen_selector,
    # the speaker-selection client — but the name in the left column is MAGI's,
    # so the role printed beside it is MAGI's.
    logger.info(
        "  %-10s %-14s %s", personas.orchestrator.name, "verdict writer",
        personas.orchestrator.model,
    )
    ui_display = "localhost" if settings.ui_host in ("0.0.0.0", "::") else settings.ui_host
    logger.info("  Console:   http://%s:%d", ui_display, settings.ui_port)
    logger.info(
        "  Tracing:   %s",
        f"{settings.otel_exporter} -> {settings.otel_endpoint}"
        if settings.otel_enabled
        else "disabled (benchmark-clean)",
    )
    logger.info("  Database:  %s", settings.db_path)
    logger.info(
        "  Speech:    %s",
        f"{settings.stt_model} ({settings.stt_language}), "
        f"preload={'on' if settings.stt_preload else 'off'}"
        if _speech_installed()
        else "not installed — PTT disabled (./setup_env.sh --voice)",
    )


async def _residency_warning(store: DebateStore, node_id: str) -> bool | None:
    """What the last pre-flight said about model residency, if it is still true.

    ``magi-preflight`` is a separate process that ``launch.sh`` runs seconds
    before this one, so the database is the only channel between them. Its
    verdict is accepted only while it is fresh: a node restarted a day later
    with ``--skip-checks`` must record "nobody looked" rather than inherit
    yesterday's all-clear, because the fact being carried is about the inference
    host's memory right now.
    """
    run = await store.latest_preflight(node_id)
    if run is None:
        logger.info("  Residency: unknown — no pre-flight recorded for this node")
        return None
    if run.age_s > PREFLIGHT_FRESH_S:
        logger.info(
            "  Residency: unknown — the last pre-flight was %.0f min ago",
            run.age_s / 60,
        )
        return None
    if run.residency_warning:
        logger.warning(
            "  Residency: pre-flight warned — %s. Latency figures from this run "
            "carry an asterisk and every debate row will say so.", run.detail,
        )
    return run.residency_warning


async def run() -> None:
    settings = Settings()
    setup_logging(settings.log_level)

    try:
        personas = personas_mod.load(settings.personas_file)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Cannot load %s: %s", settings.personas_file, exc)
        raise SystemExit(2) from exc

    _banner(settings, personas)

    provider = setup_tracing(settings)
    bus = Bus()

    # Opened at boot rather than on the first debate: a database that cannot be
    # created should say so while someone is still reading the log, not three
    # minutes into the first question anyone asks.
    store = DebateStore(settings.db_path)
    await store.open()
    residency = await _residency_warning(store, settings.node_id)

    magi = Magi(
        settings, personas, bus,
        tracer_provider=provider,
        store=store,
        residency_warning=residency,
    )

    # The draft is owned here, not by the UI app, because it is node state: the
    # kiosk and any other client are two views of the same composed question.
    draft = Draft()
    stt = STTService(settings)
    if settings.stt_preload and is_available():
        # Loading the model takes tens of seconds on a Pi. Paying it at boot,
        # where nobody is waiting, beats paying it on the first thing anyone
        # ever asks the machine to do.
        await stt.preload()

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                supervise("ui", lambda: start_ui(bus, magi, personas, settings, stt, draft))
            )
            # The terminal view of a debate, for whoever launched the node and is
            # watching it. Two conditions, both necessary: at least one advisor
            # asked to stream, and stdout is a terminal rather than the journal.
            # Not supervised — a broken renderer must not restart the node, and
            # losing the view costs nothing the console does not also show.
            if personas.streaming_names() and stream_view.wanted():
                logger.info(
                    "Streaming to this terminal: %s",
                    ", ".join(personas.streaming_names()),
                )
                tg.create_task(stream_view.render_stream(bus))
            if settings.telemetry_interval_s > 0:
                tg.create_task(
                    supervise(
                        "telemetry",
                        lambda: telemetry_service(
                            bus, TOPIC_STATUS, settings.telemetry_interval_s
                        ),
                    )
                )
    finally:
        # Spans batch, so the last debate's trace only reaches the backend if
        # the provider is shut down rather than killed with the process.
        shutdown_tracing()


def main() -> None:
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        sys.exit(130)


if __name__ == "__main__":
    main()
