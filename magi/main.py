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
from magi.constants import TOPIC_STATUS
from magi.orchestrator import Magi
from magi.services.draft import Draft
from magi.services.stt import STTService, is_available
from magi.services.telemetry import telemetry_service
from magi.setup.setup_logs import setup_logging
from magi.setup.setup_tracing import setup_tracing, shutdown_tracing
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
    logger.info(
        "  %-10s %-14s %s", personas.orchestrator.name, "orchestrator",
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
    magi = Magi(settings, personas, bus, tracer_provider=provider)

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
