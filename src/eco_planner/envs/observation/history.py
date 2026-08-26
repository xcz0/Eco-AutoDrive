"""Traffic history state machine over immutable domain frames."""

from __future__ import annotations

from collections import deque

from eco_planner.envs.contracts import TRAFFIC_HISTORY_FRAMES
from eco_planner.envs.domain.traffic import TrafficFrame


class TrafficHistory:
    """Stores the latest fixed window of consecutive simulator snapshots."""

    def __init__(self) -> None:
        self._frames: deque[TrafficFrame] = deque(maxlen=TRAFFIC_HISTORY_FRAMES)

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

    def append(self, frames: tuple[TrafficFrame, ...]) -> None:
        previous_step = self.latest.simulator_step
        for frame in frames:
            if frame.simulator_step != previous_step + 1:
                raise ValueError(
                    "traffic history simulator steps must be consecutive: "
                    f"expected {previous_step + 1}, got {frame.simulator_step}"
                )
            previous_step = frame.simulator_step
        self._frames.extend(frames)

    def require_full(self) -> tuple[TrafficFrame, ...]:
        if len(self._frames) != TRAFFIC_HISTORY_FRAMES:
            raise RuntimeError(
                f"traffic history must contain exactly {TRAFFIC_HISTORY_FRAMES} frames; "
                f"received {len(self._frames)}"
            )
        return self.frames
