"""OpenTelemetry setup.

Returns a provider or ``None``. ``None`` is the honest representation of
"tracing is off": it is what gets handed to AutoGen's runtime, and it means no
provider, no exporter and no span objects — not a provider quietly dropping
spans on the floor.

That distinction is not tidiness. This node's CPU and resident memory are the
object of study, and tracing consumes both. A benchmark run with
``MAGI_OTEL_ENABLED=0`` has to be genuinely uninstrumented, and the sibling repo
has to be able to be uninstrumented the same way, or the comparison measures
the observability stack.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)

from magi.constants import TRACER_NAME

if TYPE_CHECKING:
    from collections.abc import Sequence

    from opentelemetry.sdk.trace import ReadableSpan

    from magi.config import Settings

logger = logging.getLogger(__name__)

_provider: TracerProvider | None = None


class JsonLinesSpanExporter(SpanExporter):
    """Append spans to a file, one JSON object per line.

    For a node in the field with no collector reachable. Without it the choice
    would be between shipping spans to nothing and turning tracing off, and a
    run whose traces were silently discarded is worse than either — it looks
    instrumented and is not.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                for span in spans:
                    fh.write(span.to_json(indent=None) + "\n")
        except OSError:
            logger.exception("Could not write spans to %s", self._path)
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def setup_tracing(settings: Settings) -> TracerProvider | None:
    """Configure tracing once per process. Returns the provider, or ``None``."""
    global _provider

    if not settings.otel_enabled:
        logger.info("Tracing disabled (MAGI_OTEL_ENABLED=0) — benchmark-clean run")
        return None

    if _provider is not None:
        return _provider

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.instance.id": settings.node_id,
            # The engine is on the resource as well as on every debate span:
            # a whole process runs one engine, and having it here lets a
            # backend separate the two implementations without opening a span.
            "magi.engine": settings.engine,
        }
    )
    provider = TracerProvider(resource=resource)

    exporter: SpanExporter
    if settings.otel_exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=f"{settings.otel_endpoint}/v1/traces")
    elif settings.otel_exporter == "file":
        exporter = JsonLinesSpanExporter(settings.otel_file_path)
    else:
        exporter = ConsoleSpanExporter()

    # Batch, never Simple. Synchronous export would put network I/O on the
    # critical path of a voice interface, on the machine being measured.
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    _instrument_httpx(provider)

    _provider = provider
    logger.info(
        "Tracing on: %s -> %s (service %s)",
        settings.otel_exporter,
        settings.otel_endpoint if settings.otel_exporter == "otlp" else settings.otel_file_path,
        settings.otel_service_name,
    )
    return provider


def _instrument_httpx(provider: TracerProvider) -> None:
    """Auto-instrument outbound HTTP, which is every call to Ollama.

    Gives wall-clock per request underneath the per-turn spans, which is how a
    slow turn gets separated into "the model was slow" and "the node was slow".
    """
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    except Exception:
        logger.exception("Could not instrument httpx — HTTP spans will be missing")


def get_tracer() -> trace.Tracer:
    """A tracer, or a no-op one when tracing was never set up.

    Callers never branch on whether tracing is enabled: the OTel API returns
    non-recording spans from a no-op provider, so ``with tracer.start_as_current
    _span(...)`` costs almost nothing and reads the same either way.
    """
    return trace.get_tracer(TRACER_NAME)


def shutdown_tracing() -> None:
    """Flush and stop. Without this the last debate's spans die with the
    process, which is precisely the run someone is waiting to look at."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None
