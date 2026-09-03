"""Traffic history state machine over immutable domain frames."""

from __future__ import annotations

from collections import deque

from eco_planner.contracts import TRAFFIC_HISTORY_FRAMES

from ..domain import TrafficFrame


class TrafficHistory:
    """Stores the latest fixed window of consecutive simulator snapshots."""

    def __init__(self) -> None:
        self._frames: deque[TrafficFrame] = deque(maxlen=TRAFFIC_HISTORY_FRAMES)
        self._artifact_participant_ids: dict[str, str] = {}

    @property
    def frames(self) -> tuple[TrafficFrame, ...]:
        return tuple(self._frames)

    @property
    def latest(self) -> TrafficFrame:
        if not self._frames:
            raise RuntimeError("traffic history is unavailable before reset")
        return self._frames[-1]

    def reset(self, frame: TrafficFrame) -> None:
        self._frames.clear()
        self._frames.append(frame)
        self._artifact_participant_ids.clear()
        _register_artifact_participant_ids(frame, self._artifact_participant_ids)

    def append(self, frames: tuple[TrafficFrame, ...]) -> None:
        previous_step = self.latest.simulator_step
        staged_artifact_ids = dict(self._artifact_participant_ids)
        for frame in frames:
            if frame.simulator_step != previous_step + 1:
                raise ValueError(
                    "traffic history simulator steps must be consecutive: "
                    f"expected {previous_step + 1}, got {frame.simulator_step}"
                )
            previous_step = frame.simulator_step
            _register_artifact_participant_ids(frame, staged_artifact_ids)
        self._frames.extend(frames)
        self._artifact_participant_ids = staged_artifact_ids

    def require_full(self) -> tuple[TrafficFrame, ...]:
        if len(self._frames) != TRAFFIC_HISTORY_FRAMES:
            raise RuntimeError(
                f"traffic history must contain exactly {TRAFFIC_HISTORY_FRAMES} frames; "
                f"received {len(self._frames)}"
            )
        return self.frames

    def artifact_participant_id(self, object_id: str) -> str:
        """Return the stable trace identity registered for a simulator participant."""

        return self._artifact_participant_ids[object_id]


def _register_artifact_participant_ids(frame: TrafficFrame, artifact_ids: dict[str, str]) -> None:
    unseen = [state for state in frame.participants if state.object_id not in artifact_ids]
    keyed = sorted(
        (
            (
                state.kind,
                *state.position_xy_m,
                state.heading_rad,
                *state.velocity_xy_mps,
                state.width_m,
                state.length_m,
            ),
            state.object_id,
        )
        for state in unseen
    )
    for previous, current in zip(keyed, keyed[1:], strict=False):
        if previous[0] == current[0]:
            raise RuntimeError(
                "new traffic participants have indistinguishable physical identity keys"
            )
    for _, object_id in keyed:
        artifact_ids[object_id] = f"participant-{len(artifact_ids):06d}"
