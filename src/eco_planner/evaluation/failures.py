"""Explicit recoverable failures at the evaluation-episode boundary."""

from __future__ import annotations


class EpisodeFailure(RuntimeError):
    """A classified episode failure that may be persisted before continuing the job."""

    def __init__(self, stage: str, cause: Exception) -> None:
        if not isinstance(stage, str) or not stage:
            raise ValueError("episode failure stage must be a non-empty string")
        if not isinstance(cause, Exception):
            raise TypeError("episode failure cause must be an Exception")
        self.stage = stage
        self.cause = cause
        super().__init__(f"{stage}: {cause}")
