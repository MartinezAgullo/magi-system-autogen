"""Bus behaviour, focused on the part that is easy to get wrong: replay_last is
what makes a browser opening mid-debate render current state instead of an
empty screen."""

from __future__ import annotations

import asyncio

from magi.bus import Bus


async def test_publish_reaches_every_subscriber():
    bus = Bus()
    a = bus.subscribe("debate")
    b = bus.subscribe("debate")

    await bus.publish("debate", {"round": 1})

    assert await a.get() == {"round": 1}
    assert await b.get() == {"round": 1}


async def test_replay_last_delivers_current_state_to_a_late_subscriber():
    bus = Bus()
    await bus.publish("debate", {"round": 2})

    late = bus.subscribe("debate", replay_last=True)

    assert late.get_nowait() == {"round": 2}


async def test_without_replay_last_a_late_subscriber_waits():
    bus = Bus()
    await bus.publish("debate", {"round": 2})

    late = bus.subscribe("debate")

    with_timeout = asyncio.wait_for(late.get(), timeout=0.05)
    try:
        await with_timeout
        raise AssertionError("late subscriber should not have received the old value")
    except TimeoutError:
        pass


async def test_unsubscribe_stops_delivery():
    bus = Bus()
    q = bus.subscribe("debate")
    bus.unsubscribe("debate", q)

    await bus.publish("debate", {"round": 1})

    assert q.empty()
