"""Fixed-slot, process-isolated MetaDrive environments for batched planning."""

from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from multiprocessing.connection import Connection, wait
from time import perf_counter
from typing import Any, Literal

import numpy as np

from eco_planner.envs.array_types import SingleObservation
from eco_planner.envs.execution import TrajectoryExecutionRecord
from eco_planner.envs.metadrive_env import TrajectoryMetaDriveEnv
from eco_planner.envs.observation_adapter import (
    MetaDriveObservationAdapter,
    NoTrafficMetaDriveObservationAdapter,
)
from eco_planner.models.config import OfficialDiffusionPlannerConfig
from eco_planner.vector_worker_entry import worker_main

ObservationMode = Literal["traffic", "no_traffic"]


@dataclass(frozen=True, slots=True)
class VectorEnvScenario:
    """The immutable scenario identity assigned to one fixed environment slot."""

    name: str
    map: str
    seed: int


@dataclass(frozen=True, slots=True)
class VectorEnvTiming:
    """Worker and transport timings for one reset or step command."""

    environment_s: float
    observation_s: float
    ipc_send_s: float
    ipc_receive_s: float


@dataclass(frozen=True, slots=True)
class VectorEnvReset:
    """The single observation and reset metadata returned by one slot."""

    slot: int
    scenario: VectorEnvScenario
    observation: SingleObservation
    route_completion: float
    programmatic_lane_speed_limit_audit: Mapping[str, object]
    timing: VectorEnvTiming


@dataclass(frozen=True, slots=True)
class VectorEnvStep:
    """The execution result and next single observation returned by one slot."""

    slot: int
    observation: SingleObservation
    reward: float
    terminated: bool
    truncated: bool
    execution: TrajectoryExecutionRecord
    timing: VectorEnvTiming


@dataclass(frozen=True, slots=True)
class _WorkerTiming:
    environment_s: float
    observation_s: float


@dataclass(frozen=True, slots=True)
class _WorkerResponse:
    slot: int
    payload: object
    timing: _WorkerTiming


@dataclass(frozen=True, slots=True)
class _ReceivedResponse:
    response: _WorkerResponse
    ipc_send_s: float
    ipc_receive_s: float


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    slot: int
    operation: str
    traceback_text: str


@dataclass(frozen=True, slots=True)
class _WorkerLaunch:
    slot: int
    env_config: dict[str, Any]
    mode: ObservationMode
    model_config: OfficialDiffusionPlannerConfig
    map_query_radius_m: float
    history_warmup_steps: int


@dataclass(frozen=True, slots=True)
class _WorkerResetPayload:
    scenario: VectorEnvScenario
    observation: SingleObservation
    route_completion: float
    programmatic_lane_speed_limit_audit: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _WorkerStepPayload:
    observation: SingleObservation
    reward: float
    terminated: bool
    truncated: bool
    execution: TrajectoryExecutionRecord


class VectorMetaDriveWorkerError(RuntimeError):
    """A worker failed while executing one explicit vector-environment command."""


class VectorMetaDriveEnv:
    """Synchronously operate one process-isolated MetaDrive environment per fixed slot."""

    def __init__(
        self,
        env_configs: Sequence[Mapping[str, Any]],
        *,
        mode: ObservationMode,
        model_config: OfficialDiffusionPlannerConfig,
        map_query_radius_m: float,
        history_warmup_steps: int,
    ) -> None:
        if not env_configs:
            raise ValueError("VectorMetaDriveEnv requires at least one environment slot")
        if mode not in {"traffic", "no_traffic"}:
            raise ValueError("mode must be either 'traffic' or 'no_traffic'")
        if not isinstance(model_config, OfficialDiffusionPlannerConfig):
            raise TypeError("model_config must be an OfficialDiffusionPlannerConfig")
        if type(map_query_radius_m) not in {int, float} or map_query_radius_m <= 0.0:
            raise ValueError("map_query_radius_m must be a positive real scalar")
        if type(history_warmup_steps) is not int or history_warmup_steps < 0:
            raise ValueError("history_warmup_steps must be a non-negative integer")
        expected_warmup = model_config.time_len - 1 if mode == "traffic" else 0
        if history_warmup_steps != expected_warmup:
            raise ValueError(
                f"{mode} vector environments require history_warmup_steps={expected_warmup}"
            )

        context = _spawn_context()
        self._workers: list[tuple[Connection, mp.Process]] = []
        try:
            for slot, config in enumerate(env_configs):
                parent, child = context.Pipe()
                launch = _WorkerLaunch(
                    slot=slot,
                    env_config=dict(config),
                    mode=mode,
                    model_config=model_config,
                    map_query_radius_m=float(map_query_radius_m),
                    history_warmup_steps=history_warmup_steps,
                )
                process = context.Process(target=worker_main, args=(child, _launch_payload(launch)))
                process.start()
                child.close()
                self._workers.append((parent, process))
            self._receive("initialize", range(len(self._workers)), {})
        except BaseException:
            self.close()
            raise

    @property
    def num_envs(self) -> int:
        """Return the fixed number of worker slots."""

        return len(self._workers)

    def reset(self, scenarios: Sequence[VectorEnvScenario]) -> tuple[VectorEnvReset, ...]:
        """Reset every slot and return its single, CPU-resident observation."""

        if len(scenarios) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} scenarios, got {len(scenarios)}")
        self._validate_scenarios(scenarios)
        slots = tuple(range(self.num_envs))
        return tuple(
            self._reset_result(item)
            for item in self._receive("reset", slots, self._send("reset", slots, scenarios))
        )

    def reset_at(self, slot: int, scenario: VectorEnvScenario) -> VectorEnvReset:
        """Reset one slot without changing any other worker's episode state."""

        self._validate_slot(slot)
        self._validate_scenarios((scenario,))
        return self._reset_result(
            self._receive("reset", (slot,), self._send("reset", (slot,), (scenario,)))[0]
        )

    def step(self, trajectories: Sequence[np.ndarray]) -> tuple[VectorEnvStep, ...]:
        """Step every slot once with its corresponding ego-local trajectory."""

        if len(trajectories) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} trajectories, got {len(trajectories)}")
        slots = tuple(range(self.num_envs))
        return tuple(
            self._step_result(item)
            for item in self._receive("step", slots, self._send("step", slots, trajectories))
        )

    def step_at(self, slot: int, trajectory: np.ndarray) -> VectorEnvStep:
        """Step one slot without advancing any other worker's simulator."""

        self._validate_slot(slot)
        return self._step_result(
            self._receive("step", (slot,), self._send("step", (slot,), (trajectory,)))[0]
        )

    def close(self) -> None:
        """Close every worker-owned MetaDrive engine and join its process."""

        workers, self._workers = self._workers, []
        for connection, process in workers:
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in workers:
            if process.is_alive():
                try:
                    if connection.poll(10.0):
                        connection.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            process.join(timeout=10.0)
            if process.is_alive():
                process.terminate()
                process.join()
            connection.close()

    def __enter__(self) -> VectorMetaDriveEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _send(
        self, operation: str, slots: Sequence[int], payloads: Sequence[object]
    ) -> dict[int, float]:
        timings: dict[int, float] = {}
        for slot, payload in zip(slots, payloads, strict=True):
            started = perf_counter()
            self._workers[slot][0].send((operation, payload))
            timings[slot] = perf_counter() - started
        return timings

    def _receive(
        self, operation: str, slots: Sequence[int], send_timings: Mapping[int, float]
    ) -> tuple[_ReceivedResponse, ...]:
        pending = set(slots)
        responses: dict[int, _ReceivedResponse] = {}
        while pending:
            ready = wait([self._workers[slot][0] for slot in pending], timeout=0.1)
            if not ready:
                failed = next(
                    (slot for slot in pending if not self._workers[slot][1].is_alive()), None
                )
                if failed is not None:
                    raise VectorMetaDriveWorkerError(
                        f"MetaDrive worker slot {failed} exited during {operation} "
                        f"with exit code {self._workers[failed][1].exitcode}"
                    )
                continue
            for connection in ready:
                slot = next(slot for slot in pending if self._workers[slot][0] is connection)
                started = perf_counter()
                try:
                    response = connection.recv()
                except EOFError as error:
                    raise VectorMetaDriveWorkerError(
                        f"MetaDrive worker slot {slot} exited during {operation}"
                    ) from error
                receive_s = perf_counter() - started
                pending.remove(slot)
                if isinstance(response, _WorkerFailure):
                    raise VectorMetaDriveWorkerError(
                        f"MetaDrive worker slot {slot} failed during {response.operation}:\n"
                        f"{response.traceback_text}"
                    )
                if not isinstance(response, _WorkerResponse):
                    raise VectorMetaDriveWorkerError(
                        f"MetaDrive worker slot {slot} returned an invalid {operation} response"
                    )
                responses[slot] = _ReceivedResponse(
                    response=response,
                    ipc_send_s=send_timings.get(slot, 0.0),
                    ipc_receive_s=receive_s,
                )
        return tuple(responses[slot] for slot in slots)

    @staticmethod
    def _reset_result(received: _ReceivedResponse) -> VectorEnvReset:
        response = received.response
        payload = response.payload
        if not isinstance(payload, _WorkerResetPayload):
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {response.slot} returned reset data"
            )
        return VectorEnvReset(
            slot=response.slot,
            scenario=payload.scenario,
            observation=payload.observation,
            route_completion=payload.route_completion,
            programmatic_lane_speed_limit_audit=payload.programmatic_lane_speed_limit_audit,
            timing=VectorEnvTiming(
                response.timing.environment_s,
                response.timing.observation_s,
                received.ipc_send_s,
                received.ipc_receive_s,
            ),
        )

    @staticmethod
    def _step_result(received: _ReceivedResponse) -> VectorEnvStep:
        response = received.response
        payload = response.payload
        if not isinstance(payload, _WorkerStepPayload):
            raise VectorMetaDriveWorkerError(
                f"MetaDrive worker slot {response.slot} returned step data"
            )
        return VectorEnvStep(
            slot=response.slot,
            observation=payload.observation,
            reward=payload.reward,
            terminated=payload.terminated,
            truncated=payload.truncated,
            execution=payload.execution,
            timing=VectorEnvTiming(
                response.timing.environment_s,
                response.timing.observation_s,
                received.ipc_send_s,
                received.ipc_receive_s,
            ),
        )

    def _validate_slot(self, slot: int) -> None:
        if type(slot) is not int or not 0 <= slot < self.num_envs:
            raise IndexError(f"slot must be in [0, {self.num_envs})")

    @staticmethod
    def _validate_scenarios(scenarios: Sequence[VectorEnvScenario]) -> None:
        if not all(isinstance(scenario, VectorEnvScenario) for scenario in scenarios):
            raise TypeError("scenarios must contain only VectorEnvScenario values")


def _worker_main(connection: Connection, launch: _WorkerLaunch) -> None:
    env: TrajectoryMetaDriveEnv | None = None
    try:
        env = TrajectoryMetaDriveEnv(launch.env_config)
        adapter = _create_adapter(launch)
        connection.send(_WorkerResponse(launch.slot, None, _WorkerTiming(0.0, 0.0)))
        while True:
            operation, payload = connection.recv()
            if operation == "close":
                env.close()
                connection.send(_WorkerResponse(launch.slot, None, _WorkerTiming(0.0, 0.0)))
                return
            try:
                if operation == "reset":
                    response = _reset_worker(env, adapter, launch, payload)
                elif operation == "step":
                    response = _step_worker(env, adapter, launch.slot, payload)
                else:
                    raise ValueError(f"unknown vector environment operation {operation!r}")
                connection.send(response)
            except BaseException:
                connection.send(_WorkerFailure(launch.slot, operation, traceback.format_exc()))
    except BaseException:
        try:
            connection.send(_WorkerFailure(launch.slot, "initialize", traceback.format_exc()))
        finally:
            if env is not None:
                env.close()
    finally:
        connection.close()


def _worker_main_from_payload(connection: Connection, payload: Mapping[str, Any]) -> None:
    """Rebuild adapter-only model metadata after Windows spawn initialization."""

    launch = _WorkerLaunch(
        slot=payload["slot"],
        env_config=payload["env_config"],
        mode=payload["mode"],
        model_config=OfficialDiffusionPlannerConfig(
            **payload["model_config"], state_normalizer=None, observation_normalizer=None
        ),
        map_query_radius_m=payload["map_query_radius_m"],
        history_warmup_steps=payload["history_warmup_steps"],
    )
    _worker_main(connection, launch)


def _create_adapter(
    launch: _WorkerLaunch,
) -> MetaDriveObservationAdapter | NoTrafficMetaDriveObservationAdapter:
    if launch.mode == "traffic":
        return MetaDriveObservationAdapter(launch.model_config, launch.map_query_radius_m)
    return NoTrafficMetaDriveObservationAdapter(launch.model_config, launch.map_query_radius_m)


def _reset_worker(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter | NoTrafficMetaDriveObservationAdapter,
    launch: _WorkerLaunch,
    scenario: object,
) -> _WorkerResponse:
    if not isinstance(scenario, VectorEnvScenario):
        raise TypeError("reset requires a VectorEnvScenario")
    if scenario.map != env.config["map"]:
        raise ValueError(
            f"slot {launch.slot} is configured for map {env.config['map']!r}, not {scenario.map!r}"
        )
    started = perf_counter()
    env.reset(seed=scenario.seed)
    if isinstance(adapter, MetaDriveObservationAdapter):
        adapter.reset(env, env.initial_traffic_frame)
        _warmup_traffic(env, adapter, launch.history_warmup_steps)
    else:
        adapter.reset(env)
    environment_s = perf_counter() - started
    observation_started = perf_counter()
    observation = _build_observation(adapter, env)
    return _WorkerResponse(
        launch.slot,
        _WorkerResetPayload(
            scenario,
            observation,
            env.route_completion,
            env.programmatic_lane_speed_limit_audit,
        ),
        _WorkerTiming(environment_s, perf_counter() - observation_started),
    )


def _step_worker(
    env: TrajectoryMetaDriveEnv,
    adapter: MetaDriveObservationAdapter | NoTrafficMetaDriveObservationAdapter,
    slot: int,
    trajectory: object,
) -> _WorkerResponse:
    started = perf_counter()
    _, reward, terminated, truncated, info = env.step(trajectory)
    execution = info["trajectory_execution"]
    if not isinstance(execution, TrajectoryExecutionRecord):
        raise RuntimeError("TrajectoryMetaDriveEnv did not return a TrajectoryExecutionRecord")
    if isinstance(adapter, MetaDriveObservationAdapter):
        adapter.append_frames(execution.traffic_frames)
    environment_s = perf_counter() - started
    observation_started = perf_counter()
    return _WorkerResponse(
        slot,
        _WorkerStepPayload(
            _build_observation(adapter, env),
            float(reward),
            bool(terminated),
            bool(truncated),
            execution,
        ),
        _WorkerTiming(environment_s, perf_counter() - observation_started),
    )


def _build_observation(
    adapter: MetaDriveObservationAdapter | NoTrafficMetaDriveObservationAdapter,
    env: TrajectoryMetaDriveEnv,
) -> SingleObservation:
    if isinstance(adapter, MetaDriveObservationAdapter):
        observation, _ = adapter.build(env)
        return observation
    return adapter.build(env)


def _warmup_traffic(
    env: TrajectoryMetaDriveEnv, adapter: MetaDriveObservationAdapter, required_steps: int
) -> None:
    collected = 0
    initial_position = np.asarray(env.agent.position, dtype=np.float64).copy()
    while collected < required_steps:
        _, _, terminated, truncated, info = env.step(_stationary_trajectory())
        if terminated or truncated:
            raise RuntimeError("traffic history warmup ended before the required frame count")
        execution = info["trajectory_execution"]
        if not isinstance(execution, TrajectoryExecutionRecord):
            raise RuntimeError("TrajectoryMetaDriveEnv did not return a TrajectoryExecutionRecord")
        if (
            float(np.linalg.norm(execution.substep_states[:, :2] - initial_position, axis=1).max())
            >= 1e-3
        ):
            raise RuntimeError("ego moved during stationary traffic history warmup")
        adapter.append_frames(execution.traffic_frames)
        collected += execution.substep_states.shape[0]
    if collected != required_steps:
        raise RuntimeError("traffic history warmup overshot the required frame count")


def _stationary_trajectory() -> np.ndarray:
    trajectory = np.zeros((80, 4), dtype=np.float32)
    trajectory[:, 2] = 1.0
    return trajectory


def _spawn_context() -> mp.context.BaseContext:
    """Use the active virtual-environment interpreter for Windows worker imports."""

    if sys.platform == "win32":
        mp.set_executable(sys.executable)
    return mp.get_context("spawn")


def _launch_payload(launch: _WorkerLaunch) -> dict[str, Any]:
    model_config = {
        field.name: getattr(launch.model_config, field.name)
        for field in fields(OfficialDiffusionPlannerConfig)
        if field.name not in {"state_normalizer", "observation_normalizer"}
    }
    return {
        "slot": launch.slot,
        "env_config": launch.env_config,
        "mode": launch.mode,
        "model_config": model_config,
        "map_query_radius_m": launch.map_query_radius_m,
        "history_warmup_steps": launch.history_warmup_steps,
    }
