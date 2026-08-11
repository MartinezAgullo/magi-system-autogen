"""Local speech-to-text via faster-whisper.

Runs **on the node**. The Pi never sends audio anywhere: it transcribes locally
and only text travels, which is the same split the rest of the system follows —
reasoning here, token generation on the Spark.

``faster_whisper`` is imported **inside** the function that needs it, never at
module level. It lives in the optional ``voice`` extra because it drags in
ctranslate2, which is the longest install in the tree on ARM; a module-level
import would make the whole daemon unstartable on a node that only wants to
hold debates. That is not hypothetical — the Pi was benchmarked for a week
without it.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

    from magi.config import Settings

logger = logging.getLogger(__name__)


class STTUnavailable(RuntimeError):
    """The ``voice`` extra is not installed on this node."""


def is_available() -> bool:
    """Whether speech recognition can run here at all.

    The console asks this to decide between a working PTT button and an
    honestly disabled one. Checking the spec rather than importing keeps the
    check free on a node that will never use it.
    """
    import importlib.util

    return importlib.util.find_spec("faster_whisper") is not None


class STTService:
    """Turn recorded audio into one line of text.

    A semaphore serialises transcriptions: the Pi has four cores, and two
    overlapping presses would have them fighting each other for the same model.
    Queueing is the right answer because the operator is speaking phrase by
    phrase anyway — the second press is a second sentence, not a race.
    """

    #: Returned instead of "" so the caller can tell "the model invented
    #: something" from "there was no speech". They need different messages:
    #: one is a bad recording, the other is a released-too-early button.
    HALLUCINATION = "__hallucination__"

    # Whisper treats this as a conditioning prompt and biases strongly towards
    # its vocabulary and register. Kept short and representative of what people
    # actually ask a deliberation system, so questions about architecture,
    # tradeoffs and risk transcribe cleanly.
    INITIAL_PROMPT = (
        "Should we migrate the monolith to microservices? "
        "What are the tradeoffs of adopting Kubernetes for a small team? "
        "Is it worth building our own authentication instead of using a provider? "
        "Consider the operational cost, the security risk, and the maintenance "
        "burden over the next two years. Argue the case against."
    )

    # What Whisper produces from silence or noise. Compared case-insensitively
    # with trailing punctuation stripped.
    _HALLUCINATIONS = {
        "thank you",
        "thanks for watching",
        "thanks for watching!",
        "you",
        "bye",
        "subscribe",
        "please subscribe to my channel",
        "transcribed by https://otter.ai",
    }

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: WhisperModel | None = None
        self._sem = asyncio.Semaphore(1)

    # ── public API ───────────────────────────────────────────────────────

    async def preload(self) -> None:
        """Load the model up front so the first press is not the slow one.

        Loading takes tens of seconds on a Pi. Paying that on boot, where
        nobody is waiting, beats paying it on the first thing the operator ever
        asks the machine to do.
        """
        if not is_available():
            logger.info("Speech not installed — skipping STT preload")
            return
        logger.info("Preloading STT model (%s) in a worker thread", self._settings.stt_model)
        await asyncio.to_thread(self._ensure_model)

    async def transcribe(self, audio: bytes) -> str:
        """Transcribe one recording. Returns text, ``""``, or ``HALLUCINATION``.

        *audio* is whatever the browser's ``MediaRecorder`` produced — WebM/Opus
        in practice. faster-whisper decodes it through ffmpeg, so nothing here
        needs to know the container.
        """
        if not is_available():
            raise STTUnavailable(
                "faster-whisper is not installed — run ./setup_env.sh --voice"
            )
        async with self._sem:
            logger.info("Transcribing %d bytes", len(audio))
            return await asyncio.to_thread(self._transcribe_sync, audio)

    # ── internals ────────────────────────────────────────────────────────

    def _ensure_model(self) -> WhisperModel:
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper %s (compute=%s, threads=%d)",
                self._settings.stt_model,
                self._settings.stt_compute_type,
                self._settings.stt_cpu_threads,
            )
            self._model = WhisperModel(
                self._settings.stt_model,
                device="cpu",
                compute_type=self._settings.stt_compute_type,
                cpu_threads=self._settings.stt_cpu_threads,
            )
            logger.info("STT model ready")
        return self._model

    def _transcribe_sync(self, audio: bytes) -> str:
        """Blocking. Always reached through ``asyncio.to_thread``."""
        model = self._ensure_model()
        segments, info = model.transcribe(
            io.BytesIO(audio),
            language=self._settings.stt_language,
            beam_size=self._settings.stt_beam_size,
            initial_prompt=self.INITIAL_PROMPT,
            # These four together are what stop a released-too-early press from
            # becoming a confident sentence about nothing. Whisper will happily
            # narrate silence otherwise, and a hallucinated line in the question
            # is worse than a missing one: the operator may not reread it.
            hallucination_silence_threshold=2.0,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())

        if text.lower().strip(".!?, ") in self._HALLUCINATIONS:
            logger.warning("STT hallucination filtered: %r", text[:120])
            return self.HALLUCINATION
        if not text:
            logger.info("STT found no speech")
            return ""
        logger.info("STT (%s, %d chars): %s", info.language, len(text), text[:120])
        return text
