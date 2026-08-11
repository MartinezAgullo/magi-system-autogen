"""Process-level setup: logging now, tracing when the orchestrator lands.

Separate from the services because these configure the process itself rather
than doing work in it, and because both have to run before anything else can be
trusted to report what it is doing.
"""
