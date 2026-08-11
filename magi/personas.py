"""Loads and validates ``config/magi.yaml``.

Personas are **data**. Names, models, sampling parameters and prompts all come
from this file, and ``MAGI_PERSONAS_FILE`` selects which one. Two reasons that
matters more than usual here:

* Comparing role formulations is part of the experiment, and a comparison that
  needs a code edit between runs will not get done.
* The planned dynamic-role extension generates personas per question. It has to
  be able to produce exactly what the YAML loader produces, so this module
  returns plain validated objects with no file coupling beyond :func:`load`.

Validation is strict on purpose. A typo in a model tag or a duplicated advisor
name is cheap to catch here and expensive to debug three rounds into a debate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from magi.constants import MIN_ADVISORS

logger = logging.getLogger(__name__)


class ModelOptions(BaseModel):
    """Backend-specific generation options passed through untouched.

    Deliberately open: ``think``, ``num_ctx`` and friends are Ollama's, not
    ours, and enumerating them here would mean editing this file every time a
    backend adds one.
    """

    model_config = {"extra": "allow"}


class Persona(BaseModel):
    """One advisor: a name, a model, and the prompt that gives it a point of view."""

    name: str
    archetype: str = ""
    model: str
    temperature: float | None = None
    max_tokens: int | None = None
    # Tri-state on purpose. True/False force reasoning on or off; None leaves
    # the model's own default alone, which has to stay expressible: nemotron3
    # produces no valid structured output at all with reasoning suppressed,
    # while gemma3 is measurably more reliable with it off. One global setting
    # cannot serve both.
    thinking: bool | None = None
    options: ModelOptions | None = None
    system_prompt: str

    @field_validator("name")
    @classmethod
    def _name_is_a_callsign(cls, v: str) -> str:
        # Uppercase single tokens, because the name is spoken, printed on an
        # 800x480 screen, and used as a message source the model must match
        # exactly when filling in `agrees_with`.
        v = v.strip()
        if not v or " " in v:
            raise ValueError("advisor name must be a single word with no spaces")
        return v.upper()

    @field_validator("model")
    @classmethod
    def _model_is_tagged(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("model tag must not be empty")
        # Ollama resolves a bare name to ":latest"; normalise so pre-flight can
        # compare against /api/tags without a special case.
        return v if ":" in v else f"{v}:latest"

    @field_validator("system_prompt")
    @classmethod
    def _prompt_is_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("system_prompt must not be empty")
        return v


class Orchestrator(Persona):
    """MAGI itself: writes the verdict, judges split votes, never counts."""


class PersonaSet(BaseModel):
    """A validated ``magi.yaml``."""

    version: int = 1
    defaults: dict = Field(default_factory=dict)
    common_prompt: str = ""
    magi: list[Persona]
    orchestrator: Orchestrator

    @model_validator(mode="after")
    def _check_roster(self) -> PersonaSet:
        if len(self.magi) < MIN_ADVISORS:
            raise ValueError(
                f"need at least {MIN_ADVISORS} advisors to hold a debate, "
                f"got {len(self.magi)}"
            )
        names = [p.name for p in self.magi]
        if len(set(names)) != len(names):
            raise ValueError(f"advisor names must be unique, got {names}")
        # The orchestrator is not one of the advisors: it must not be able to
        # appear in `agrees_with`, and a name collision would make the debate
        # transcript ambiguous about who said what.
        if self.orchestrator.name in names:
            raise ValueError(
                f"orchestrator name {self.orchestrator.name!r} collides with an advisor"
            )
        return self

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.magi]

    def distinct_models(self) -> set[str]:
        """Every model tag the run will need loaded, advisors and orchestrator.

        This is what pre-flight checks against ``/api/tags`` and what model
        residency is judged by — the orchestrator counts, since it runs at the
        end of every debate.
        """
        return {p.model for p in self.magi} | {self.orchestrator.model}

    def system_prompt_for(self, persona: Persona) -> str:
        """Full system prompt: the shared rules of debate, then the personality.

        Kept separate in the YAML so a rule change applies to all advisors at
        once. An advisor that had drifted out of sync with the others' rules
        would look like a personality difference and be impossible to spot.
        """
        if not self.common_prompt.strip():
            return persona.system_prompt.strip()
        return f"{self.common_prompt.strip()}\n\n{persona.system_prompt.strip()}"

    def temperature_for(self, persona: Persona) -> float | None:
        if persona.temperature is not None:
            return persona.temperature
        return self.defaults.get("temperature")

    def max_tokens_for(self, persona: Persona) -> int | None:
        if persona.max_tokens is not None:
            return persona.max_tokens
        return self.defaults.get("max_tokens")

    def thinking_for(self, persona: Persona) -> bool | None:
        """Whether this advisor reasons before answering, or ``None`` for the
        model's default.

        Budget interacts with this: a thinking model spends `max_tokens` on
        reasoning first, so an advisor with `thinking` on and a small budget
        returns nothing at all. Pre-flight catches that combination.
        """
        if persona.thinking is not None:
            return persona.thinking
        return self.defaults.get("thinking")

    def options_for(self, persona: Persona) -> dict:
        """Defaults merged with per-persona overrides, persona winning."""
        merged = dict(self.defaults.get("options") or {})
        if persona.options is not None:
            merged.update(persona.options.model_dump())
        return merged


def load(path: Path) -> PersonaSet:
    """Read and validate a persona file.

    Raises ``FileNotFoundError`` or ``ValueError`` — both are fatal at boot, and
    both are meant to be. There is no useful degraded mode for "the advisors are
    misconfigured".
    """
    if not path.exists():
        raise FileNotFoundError(f"persona file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    personas = PersonaSet.model_validate(raw)
    logger.debug(
        "Loaded %d advisors from %s: %s",
        len(personas.magi),
        path,
        ", ".join(f"{p.name}={p.model}" for p in personas.magi),
    )
    return personas
