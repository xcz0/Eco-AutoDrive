"""Explicit recoverable failures at the evaluation-episode boundary."""

from __future__ import annotations

from eco_planner.evaluation.artifacts.models import FailurePhase


class EpisodeFailure(RuntimeError):
    """A classified episode failure that may be persisted before continuing the job."""

    def __init__(self, phase: FailurePhase, cause: Exception) -> None:
        if not isinstance(phase, FailurePhase):
            raise TypeError("episode failure phase must be a FailurePhase")
        if not isinstance(cause, Exception):
            raise TypeError("episode failure cause must be an Exception")
        self.phase = phase
        self.cause = cause
        super().__init__(f"{phase.value}: {cause}")
