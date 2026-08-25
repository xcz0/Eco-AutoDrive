"""Child-process runtime for one fixed MetaDrive environment slot."""

from __future__ import annotations

import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection
from time import perf_counter
from typing import Any

from eco_planner.envs.observation import PlannerObservationSpec
from eco_planner.envs.slot import MetaDriveEnvSlot, ObservationMode
from eco_planner.envs.vector_metadrive import (
    VectorEnvScenario,
    _WorkerFailure,
    _WorkerResetPayload,
    _WorkerResponse,
    _WorkerStepPayload,
    _WorkerTiming,
)


@dataclass(frozen=True, slots=True)
class _WorkerLaunch:
    slot: int
    env_config: dict[str, Any]
    mode: ObservationMode
    observation_spec: PlannerObservationSpec
    map_query_radius_m: float
    history_warmup_steps: int


def worker_main_from_payload(connection: Connection, payload: Mapping[str, Any]) -> None:
    """Rebuild the typed worker launch after Windows spawn initialization."""

    launch = _WorkerLaunch(
        slot=payload["slot"],
        env_config=payload["env_config"],
        mode=payload["mode"],
        observation_spec=PlannerObservationSpec(**payload["observation_spec"]),
        map_query_radius_m=payload["map_query_radius_m"],
        history_warmup_steps=payload["history_warmup_steps"],
    )
    _worker_main(connection, launch)


def _worker_main(connection: Connection, launch: _WorkerLaunch) -> None:
    slot: MetaDriveEnvSlot | None = None
    try:
        slot = MetaDriveEnvSlot(
            launch.env_config,
            mode=launch.mode,
            observation_spec=launch.observation_spec,
            map_query_radius_m=launch.map_query_radius_m,
            history_warmup_steps=launch.history_warmup_steps,
        )
        connection.send(_WorkerResponse(launch.slot, None, _WorkerTiming(0.0, 0.0, 0.0)))
        while True:
            wait_started = perf_counter()
            operation, payload = connection.recv()
            wait_s = perf_counter() - wait_started
            if operation == "close":
                slot.close()
                connection.send(
                    _WorkerResponse(launch.slot, None, _WorkerTiming(0.0, 0.0, wait_s))
                )
                return
            try:
                if operation == "reset":
                    response = _reset_worker(slot, launch.slot, payload, wait_s)
                elif operation == "step":
                    response = _step_worker(slot, launch.slot, payload, wait_s)
                else:
                    raise ValueError(f"unknown vector environment operation {operation!r}")
                connection.send(response)
            except BaseException:
                connection.send(_WorkerFailure(launch.slot, operation, traceback.format_exc()))
    except BaseException:
        try:
            connection.send(_WorkerFailure(launch.slot, "initialize", traceback.format_exc()))
        finally:
            if slot is not None:
                slot.close()
    finally:
        connection.close()


def _reset_worker(
    slot: MetaDriveEnvSlot,
    slot_index: int,
    scenario: object,
    wait_s: float,
) -> _WorkerResponse:
    if not isinstance(scenario, VectorEnvScenario):
        raise TypeError("reset requires a VectorEnvScenario")
    started = perf_counter()
    reset = slot.reset(map_name=scenario.map, seed=scenario.seed)
    warmup_executions = tuple(slot.warmup())
    environment_s = perf_counter() - started
    observation_started = perf_counter()
    observation = slot.observe()
    return _WorkerResponse(
        slot_index,
        _WorkerResetPayload(
            scenario,
            observation.observation,
            reset.route_completion,
            reset.route_length_m,
            reset.warmup_initial_state,
            slot.vehicle_state,
            warmup_executions,
            observation.traffic_audit,
            reset.programmatic_lane_speed_limit_audit,
        ),
        _WorkerTiming(environment_s, perf_counter() - observation_started, wait_s),
    )


def _step_worker(
    slot: MetaDriveEnvSlot,
    slot_index: int,
    trajectory: object,
    wait_s: float,
) -> _WorkerResponse:
    started = perf_counter()
    step = slot.step(trajectory)
    environment_s = perf_counter() - started
    observation_started = perf_counter()
    observation = slot.observe()
    return _WorkerResponse(
        slot_index,
        _WorkerStepPayload(
            observation.observation,
            step.reward,
            step.terminated,
            step.truncated,
            step.execution,
            observation.traffic_audit,
        ),
        _WorkerTiming(environment_s, perf_counter() - observation_started, wait_s),
    )
