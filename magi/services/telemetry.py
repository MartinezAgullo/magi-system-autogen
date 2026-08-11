"""What the node can say about itself.

Exists because the UI leans on Evangelion's status panels, and those panels are
only worth having if the numbers on them are real. A CONDITION GREEN flag that
is hardcoded green is set dressing; one driven by the actual SoC temperature and
throttle bits tells the operator something.

Everything here degrades to ``None`` off a Pi, so the dev machine renders the
same layout with the fields it cannot know left blank.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass

import psutil

logger = logging.getLogger(__name__)

# Bit flags from `vcgencmd get_throttled`. The low bits are "right now", the
# high bits are "has happened since boot" — a node that throttled during a
# debate and recovered still owes the operator that fact, because it explains a
# slow turn that looks otherwise inexplicable.
THROTTLE_BITS = {
    0: "UNDER-VOLTAGE",
    1: "FREQ-CAPPED",
    2: "THROTTLED",
    3: "TEMP-LIMIT",
}


@dataclass
class NodeStatus:
    """One reading. All optional: a MacBook has no ``vcgencmd``."""

    temp_c: float | None = None
    cpu_percent: float | None = None
    mem_percent: float | None = None
    uptime_s: float | None = None
    throttling: tuple[str, ...] = ()

    @property
    def condition(self) -> str:
        """The headline flag, in the anime's vocabulary.

        Thresholds are the Pi 5's own: it soft-throttles at 80 °C and hard-caps
        at 85 °C, so AMBER at 70 leaves room to notice before anything is lost.
        """
        if self.throttling:
            return "EMERGENCY"
        if self.temp_c is not None and self.temp_c >= 80:
            return "WARNING"
        if self.temp_c is not None and self.temp_c >= 70:
            return "CAUTION"
        return "GREEN"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["throttling"] = list(self.throttling)
        data["condition"] = self.condition
        return data


def _vcgencmd(*args: str) -> str | None:
    binary = shutil.which("vcgencmd")
    if binary is None:
        return None
    try:
        out = subprocess.run(  # noqa: S603 — fixed binary, fixed args
            [binary, *args], capture_output=True, text=True, timeout=3, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _read_temp() -> float | None:
    """SoC temperature, from sysfs first and ``vcgencmd`` as a fallback.

    sysfs is preferred because it needs no subprocess: this is polled every few
    seconds for the life of the node, and forking a process each time to read
    one number would be a silly way to spend a core we are trying to measure.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return int(fh.read().strip()) / 1000
    except (OSError, ValueError):
        pass

    raw = _vcgencmd("measure_temp")  # e.g. "temp=56.0'C"
    if raw and "=" in raw:
        with contextlib.suppress(ValueError):
            return float(raw.split("=")[1].rstrip("'C"))
    return None


def _read_throttling() -> tuple[str, ...]:
    raw = _vcgencmd("get_throttled")  # e.g. "throttled=0x0"
    if not raw or "=" not in raw:
        return ()
    try:
        bits = int(raw.split("=")[1], 16)
    except ValueError:
        return ()
    return tuple(label for bit, label in THROTTLE_BITS.items() if bits & (1 << bit))


def read_status() -> NodeStatus:
    """One synchronous reading. Cheap enough to call on a timer."""
    return NodeStatus(
        temp_c=_read_temp(),
        cpu_percent=psutil.cpu_percent(interval=None),
        mem_percent=psutil.virtual_memory().percent,
        uptime_s=_uptime(),
        throttling=_read_throttling(),
    )


def _uptime() -> float | None:
    try:
        with open("/proc/uptime") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        return None


async def telemetry_service(bus, topic: str, interval_s: float = 5.0) -> None:
    """Publish a reading every *interval_s* for the life of the node.

    Runs the read in a thread: it touches sysfs and may fork ``vcgencmd``, and
    blocking the event loop mid-debate to find out how warm the board is would
    be a poor trade.
    """
    while True:
        status = await asyncio.to_thread(read_status)
        await bus.publish(topic, status.to_dict())
        await asyncio.sleep(interval_s)
