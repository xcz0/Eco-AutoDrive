"""Project lifecycle service independent of simulator and tensor frameworks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from eco_planner.envs.array_types import SingleObservation, TrajectoryArray
from eco_planner.envs.contracts import TRAFFIC_HISTORY_WARMUP_STEPS
from eco_planner.envs.domain.result import ExecutionResult
from eco_planner.envs.domain.traffic import TrafficFrame


class Simulator(Protocol):
    """Port used by lifecycle code; integrations own framework callbacks."""

    @property
    def vehicle_state(self) -> np.ndarray: ...

    def reset(self, *, scenario: str, seed: int) -> Mapping[str, object]: ...

    def execute(self, trajectory: TrajectoryArray) -> ExecutionResult: ...

    def close(self) -> None: ...


class ObservationService(Protocol):
    def reset(self, initial_frame: TrafficFrame) -> None: ...

    def commit(self, frames: tuple[TrafficFrame, ...]) -> None: ...

    def observe(self) -> SingleObservation: ...


@dataclass(frozen=True, slots=True)
class SlotReset:
    metadata: Mapping[str, object]
    warmup_initial_state: np.ndarray


class EnvironmentSlot:
    """Own the reset → warmup → observe → execute → commit lifecycle."""

    def __init__(self, simulator: Simulator, observation: ObservationService) -> None:
        self._simulator = simulator
        self._observation = observation

    def reset(self, *, scenario: str, seed: int, initial_frame: TrafficFrame) -> SlotReset:
        metadata = self._simulator.reset(scenario=scenario, seed=seed)
        self._observation.reset(initial_frame)
        return SlotReset(
            metadata=metadata, warmup_initial_state=self._simulator.vehicle_state.copy()
        )

    def warmup(self, stationary_trajectory: TrajectoryArray) -> Iterator[ExecutionResult]:
        for _ in range(TRAFFIC_HISTORY_WARMUP_STEPS):
            result = self.execute(stationary_trajectory)
            if result.status.max_step or result.status.arrive_dest:
                raise RuntimeError("traffic history warmup ended before the required frame count")
            yield result

    def observe(self) -> SingleObservation:
        return self._observation.observe()

    def execute(self, trajectory: TrajectoryArray) -> ExecutionResult:
        result = self._simulator.execute(trajectory)
        self._observation.commit(result.trace.traffic_frames)
        return result

    def close(self) -> None:
        self._simulator.close()
