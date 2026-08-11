"""Team construction — the only thing the two engines differ in."""

from __future__ import annotations

import pytest
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat

from magi.config import Settings
from magi.constants import SELECTOR_LABEL
from magi.orchestrator.clients import build_advisors
from magi.orchestrator.teams import SELECTOR_PROMPT, build_team
from magi.personas import PersonaSet
from magi.services.metrics import CallCounter


@pytest.fixture
def personas() -> PersonaSet:
    return PersonaSet.model_validate(
        {
            "common_prompt": "rules",
            "magi": [
                {"name": "MELCHIOR", "model": "model-a:1b", "system_prompt": "a"},
                {"name": "CASPAR", "model": "model-b:1b", "system_prompt": "b"},
            ],
            "orchestrator": {"name": "MAGI", "model": "model-a:1b", "system_prompt": "o"},
        }
    )


def _team(engine: str, personas: PersonaSet, counter=None):
    settings = Settings(engine=engine)
    advisors = build_advisors(settings, personas)
    return build_team(
        settings, personas, advisors, MaxMessageTermination(7), counter
    )


def test_roundrobin_engine_builds_a_roundrobin_team(personas):
    assert isinstance(_team("autogen_roundrobin", personas), RoundRobinGroupChat)


def test_selector_engine_builds_a_selector_team(personas):
    assert isinstance(_team("autogen_selector", personas), SelectorGroupChat)


def test_selector_calls_are_counted_under_their_own_label(personas):
    """The difference between the two engines IS the selector's calls. Folding
    them into the orchestrator's bucket would erase the measurement, since both
    happen to use the same model."""
    counter = CallCounter()
    team = _team("autogen_selector", personas, counter)

    client = team._model_client  # noqa: SLF001 — no public accessor
    assert client._magi_label == SELECTOR_LABEL  # noqa: SLF001


def test_the_selector_prompt_is_ours_not_autogens_role_play_default(personas):
    """AutoGen's default opens with 'You are in a role play game', which asks the
    selector to keep things entertaining. What is wanted is coverage."""
    assert "role play game" not in SELECTOR_PROMPT
    assert "{roles}" in SELECTOR_PROMPT
    assert "{participants}" in SELECTOR_PROMPT
    assert "{history}" in SELECTOR_PROMPT


def test_no_selector_func_shortcuts_the_call_being_measured(personas):
    """A selector_func would skip the LLM call this engine exists to price."""
    team = _team("autogen_selector", personas)

    assert team._selector_func is None  # noqa: SLF001


def test_both_engines_register_the_structured_message_type(personas):
    """Without registration a group chat rejects StructuredMessage[MagiTurn] at
    run time, from inside a message container. It must hold for BOTH engines,
    not just whichever one was tried by hand."""
    from autogen_agentchat.messages import StructuredMessage

    from magi.models import MagiTurn

    for engine in ("autogen_roundrobin", "autogen_selector"):
        team = _team(engine, personas)
        factory = team._message_factory  # noqa: SLF001
        assert factory.is_registered(
            StructuredMessage[MagiTurn]
        ), f"{engine} would reject StructuredMessage[MagiTurn] mid-debate"
