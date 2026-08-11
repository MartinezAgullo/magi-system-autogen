"""The debate itself: agents, teams, termination, tally, verdict."""

from magi.orchestrator.consensus import Tally, tally
from magi.orchestrator.magi import Magi

__all__ = ["Magi", "Tally", "tally"]
