"""The persona file is the one piece of configuration a bad edit can silently
ruin — three advisors that are secretly the same advisor still produce a
confident unanimous verdict. These tests cover the failures that would not be
obvious from watching a debate."""

from __future__ import annotations

import textwrap

import pytest

from magi import personas as personas_mod
from magi.config import Settings

MINIMAL = """
common_prompt: |
  Shared rules.
magi:
  - name: MELCHIOR
    archetype: scientist
    model: nemotron3:33b
    system_prompt: |
      Be the scientist.
  - name: CASPAR
    archetype: skeptic
    model: qwen3:14b
    system_prompt: |
      Be the skeptic.
orchestrator:
  name: MAGI
  model: gemma3:12b
  system_prompt: |
    Write the verdict.
"""


def _write(tmp_path, body: str):
    path = tmp_path / "magi.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_loads_the_committed_default_file():
    """The file shipped in the repo must actually validate — otherwise every
    fresh clone fails at boot."""
    personas = personas_mod.load(Settings().personas_file)

    assert personas.names == ["MELCHIOR", "BALTHASAR", "CASPAR"]
    assert personas.orchestrator.name == "MAGI"
    # Lineage diversity is the premise of the whole system: three models from
    # one family agreeing proves nothing.
    assert len(personas.distinct_models()) >= 3


def test_common_prompt_is_prepended_not_replaced(tmp_path):
    personas = personas_mod.load(_write(tmp_path, MINIMAL))
    melchior = personas.magi[0]

    prompt = personas.system_prompt_for(melchior)

    assert prompt.startswith("Shared rules.")
    assert "Be the scientist." in prompt


def test_bare_model_name_is_normalised_to_latest(tmp_path):
    """Ollama resolves `gemma3` to `gemma3:latest`, so pre-flight would report a
    model as missing that is in fact pulled."""
    path = _write(tmp_path, MINIMAL.replace("gemma3:12b", "gemma3"))

    personas = personas_mod.load(path)

    assert personas.orchestrator.model == "gemma3:latest"


def test_distinct_models_includes_the_orchestrator(tmp_path):
    """The orchestrator runs at the end of every debate, so its model has to be
    resident too — leaving it out would make the residency warning wrong."""
    personas = personas_mod.load(_write(tmp_path, MINIMAL))

    assert personas.distinct_models() == {"nemotron3:33b", "qwen3:14b", "gemma3:12b"}


def test_duplicate_advisor_names_are_rejected(tmp_path):
    """Two advisors called CASPAR make `agrees_with` ambiguous and the
    transcript unreadable."""
    path = _write(tmp_path, MINIMAL.replace("name: MELCHIOR", "name: CASPAR"))

    with pytest.raises(ValueError, match="unique"):
        personas_mod.load(path)


def test_orchestrator_may_not_share_a_name_with_an_advisor(tmp_path):
    path = _write(tmp_path, MINIMAL.replace("name: MAGI", "name: CASPAR"))

    with pytest.raises(ValueError, match="collides"):
        personas_mod.load(path)


def test_a_single_advisor_is_not_a_debate(tmp_path):
    body = MINIMAL.replace(
        """  - name: CASPAR
    archetype: skeptic
    model: qwen3:14b
    system_prompt: |
      Be the skeptic.
""",
        "",
    )

    with pytest.raises(ValueError, match="at least"):
        personas_mod.load(_write(tmp_path, body))


def test_names_are_uppercased(tmp_path):
    """The model has to match the name exactly when filling in `agrees_with`,
    and it is also what gets printed on an 800x480 screen."""
    personas = personas_mod.load(_write(tmp_path, MINIMAL.replace("MELCHIOR", "melchior")))

    assert personas.magi[0].name == "MELCHIOR"


def test_per_persona_options_override_defaults(tmp_path):
    body = MINIMAL.replace(
        "magi:",
        """defaults:
  temperature: 0.6
  options:
    think: false
magi:""",
    ).replace(
        "    model: qwen3:14b\n",
        "    model: qwen3:14b\n    temperature: 0.9\n",
    )
    personas = personas_mod.load(_write(tmp_path, body))
    melchior, caspar = personas.magi

    assert personas.temperature_for(melchior) == 0.6
    assert personas.temperature_for(caspar) == 0.9
    assert personas.options_for(caspar) == {"think": False}


def test_missing_file_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError):
        personas_mod.load(tmp_path / "nope.yaml")
