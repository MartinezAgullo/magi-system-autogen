"""The question under composition.

The behaviour that matters: a PTT press adds a line, a bad line can be dropped,
and only SEND spends three models' time. Firing a debate per press would mean a
mis-heard word costs a full deliberation and an answer to a question nobody
asked.
"""

from __future__ import annotations

from magi.services.draft import MAX_LINES, Draft


async def test_lines_accumulate_in_dictation_order():
    draft = Draft()

    await draft.add("Should we adopt Kubernetes")
    await draft.add("for our first product?")

    assert await draft.question() == "Should we adopt Kubernetes for our first product?"


async def test_lines_join_with_spaces_not_newlines():
    """The advisors get one question. A multi-line prompt invites a model to
    answer each line separately instead of treating them as one thought."""
    draft = Draft()
    await draft.add("first")
    await draft.add("second")

    assert "\n" not in await draft.question()


async def test_a_mis_heard_line_can_be_dropped():
    draft = Draft()
    await draft.add("Should we adopt Kubernetes")
    bad = await draft.add("for our fur spot duck")
    await draft.add("for our first product?")

    assert await draft.delete(bad.id)

    assert await draft.question() == "Should we adopt Kubernetes for our first product?"


async def test_ids_are_never_reused():
    """Deleting by id rather than position is what makes a delete safe while a
    press is still being transcribed: with positions, dropping "line 2" could
    remove whatever landed there in the meantime."""
    draft = Draft()
    first = await draft.add("one")
    await draft.delete(first.id)
    second = await draft.add("two")

    assert second.id != first.id


async def test_deleting_an_unknown_id_reports_it_rather_than_guessing():
    draft = Draft()
    await draft.add("one")

    assert not await draft.delete(999)
    assert await draft.question() == "one"


async def test_blank_transcriptions_are_not_lines():
    """Whisper returns "" for a released-too-early press. An empty line in the
    draft would be a delete the operator has to perform for nothing."""
    draft = Draft()

    assert await draft.add("   ") is None
    assert await draft.add("") is None
    assert (await draft.as_dict())["lines"] == []


async def test_the_draft_is_capped():
    """A stuck PTT button must not grow the prompt without bound behind the
    operator's back."""
    draft = Draft()
    for i in range(MAX_LINES):
        assert await draft.add(f"line {i}") is not None

    assert await draft.add("one too many") is None
    assert (await draft.as_dict())["full"]


async def test_clearing_empties_it():
    draft = Draft()
    await draft.add("one")

    await draft.clear()

    assert await draft.question() == ""
    assert not (await draft.as_dict())["full"]


async def test_as_dict_carries_what_the_console_renders():
    draft = Draft()
    line = await draft.add("Should we adopt Kubernetes?")

    data = await draft.as_dict()

    assert data["lines"] == [
        {"id": line.id, "text": "Should we adopt Kubernetes?", "timestamp": line.created_at}
    ]
    assert data["question"] == "Should we adopt Kubernetes?"
