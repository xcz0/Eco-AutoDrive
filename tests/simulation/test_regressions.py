from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Binary, Composite, Unbounded

from eco_planner.envs.domain import (
    EnergyTrace,
    FASTSimEnergyConfig,
    FASTSimEnergyProvider,
    TrafficFrame,
    TrafficParticipantState,
)
from eco_planner.envs.metadrive import (
    EnvSlotReset,
    EnvSlotState,
    EnvSlotTiming,
    LocalRouteUnavailableError,
)
from eco_planner.envs.metadrive.observation import NoTrafficMetaDriveObservationPipeline
from eco_planner.envs.observation import PLANNER_OBSERVATION_FIELDS
from eco_planner.runtime.envs.torchrl import TorchRLMetaDriveEnv
from eco_planner.runtime.envs.vector import VectorMetaDriveEnv, VectorMetaDriveWorkerError
from eco_planner.runtime.envs.worker import TorchRLScenarioMetaDriveEnv, WorkerFailure


def _observation() -> TensorDict:
    fields = {
        name: torch.zeros(shape, dtype=torch.bool if dtype == np.dtype(np.bool_) else torch.float32)
        for name, (shape, dtype) in PLANNER_OBSERVATION_FIELDS.items()
    }
    return TensorDict(fields, batch_size=[])


def _frame(
    simulator_step: int,
    *,
    participants: tuple[TrafficParticipantState, ...] = (),
) -> TrafficFrame:
    return TrafficFrame(
        simulator_step=simulator_step,
        ego_center_xy_m=(0.0, 0.0),
        ego_heading_rad=0.0,
        ego_rear_wheelbase_m=1.0,
        participants=participants,
        static_objects=(),
    )


class _RouteRetrySlot:
    def __init__(self, failure: Exception) -> None:
        self._failure = failure
        self.reset_calls = 0
        self.recreate_calls = 0

    def reset(self, *, map_name: str, seed: int) -> EnvSlotReset:
        assert (map_name, seed) == ("S", 7)
        self.reset_calls += 1
        if self.reset_calls == 1:
            raise self._failure
        return EnvSlotReset(
            state=EnvSlotState(_observation(), None, np.zeros(7), 0.0),
            route_length_m=100.0,
            warmup_initial_state=np.zeros(7),
            warmup_steps=(),
            programmatic_lane_speed_limit_audit={},
            timing=EnvSlotTiming(0.0, 0.0),
        )

    def recreate_environment(self) -> None:
        self.recreate_calls += 1

    def close(self) -> None:
        pass


def test_torchrl_reset_retries_only_typed_local_route_failures() -> None:
    typed_slot = _RouteRetrySlot(LocalRouteUnavailableError("typed route failure"))
    typed_env = TorchRLMetaDriveEnv(typed_slot, map_name="S", seed=7)  # type: ignore[arg-type]

    output = typed_env._reset(None)

    assert output["observation"].batch_size == torch.Size([])
    assert typed_slot.reset_calls == 2
    assert typed_slot.recreate_calls == 1

    runtime_slot = _RouteRetrySlot(RuntimeError("no connected navigation route lanes exist"))
    runtime_env = TorchRLMetaDriveEnv(runtime_slot, map_name="S", seed=7)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="no connected navigation route lanes exist"):
        runtime_env._reset(None)

    assert runtime_slot.reset_calls == 1
    assert runtime_slot.recreate_calls == 0


def test_no_traffic_pipeline_rejects_participants_in_later_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = NoTrafficMetaDriveObservationPipeline(100.0)
    monkeypatch.setattr(pipeline._map_adapter, "reset", lambda env: None)
    env = SimpleNamespace(
        config={"traffic_density": 0.0, "random_traffic": False, "accident_prob": 0.0}
    )
    pipeline.reset(env, _frame(0))
    participant = TrafficParticipantState(
        object_id="late-vehicle",
        kind="vehicle",
        position_xy_m=(1.0, 0.0),
        heading_rad=0.0,
        velocity_xy_mps=(0.0, 0.0),
        width_m=2.0,
        length_m=4.0,
    )

    with pytest.raises(RuntimeError, match="late-vehicle"):
        pipeline.append_frames((_frame(1, participants=(participant,)),))


def test_step_failure_output_preserves_the_original_worker_traceback() -> None:
    worker = SimpleNamespace(
        observation_spec=Composite(
            observation=Unbounded(shape=(2,), dtype=torch.float32), shape=(), device="cpu"
        ),
        done_spec=Composite(
            done=Binary(1, shape=(1,), dtype=torch.bool, device="cpu"),
            terminated=Binary(1, shape=(1,), dtype=torch.bool, device="cpu"),
            truncated=Binary(1, shape=(1,), dtype=torch.bool, device="cpu"),
            shape=(),
            device="cpu",
        ),
        reward_spec=Unbounded(shape=(1,), dtype=torch.float32, device="cpu"),
        _operation_result=None,
    )
    try:
        raise RuntimeError("sentinel worker failure")
    except RuntimeError:
        output = TorchRLScenarioMetaDriveEnv._failure_output(worker, "step")

    assert isinstance(output, TensorDict)
    assert output["reward"].shape == (1,)
    assert output["reward"].dtype is torch.float32
    assert isinstance(worker._operation_result, WorkerFailure)
    assert worker._operation_result.operation == "step"
    assert "sentinel worker failure" in worker._operation_result.traceback_text
    assert "AttributeError" not in worker._operation_result.traceback_text

    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    facade = SimpleNamespace(close=close)
    with pytest.raises(VectorMetaDriveWorkerError, match="slot 3 failed during step") as error:
        VectorMetaDriveEnv._raise_worker_failure(facade, 3, worker._operation_result)

    assert closed is True
    assert "sentinel worker failure" in str(error.value)


def _fastsim_config() -> FASTSimEnergyConfig:
    return FASTSimEnergyConfig(
        vehicle_resource="2012_Ford_Fusion.yaml",
        grade=0.0,
        ambient_temperature_k=295.15,
        initial_elevation_m=121.92,
    )


def _short_trace() -> EnergyTrace:
    return EnergyTrace(
        time_s=np.arange(6, dtype=np.float64),
        speed_mps=np.arange(6, dtype=np.float64),
        step_distance_m=np.arange(1, 6, dtype=np.float64),
    )


def test_fastsim_conventional_vehicle_matches_locked_numeric_result() -> None:
    metrics = FASTSimEnergyProvider(_fastsim_config()).measure(_short_trace())

    assert metrics.metric == "fastsim_fuel_energy"
    assert metrics.distance_m == 15.0
    assert metrics.energy_j == pytest.approx(116_855.316236982)
    assert metrics.energy_wh == pytest.approx(metrics.energy_j / 3_600.0)
    assert metrics.energy_j_per_km == pytest.approx(metrics.energy_j * 1_000.0 / 15.0)
    assert metrics.fuel_ml is None


def test_fastsim_stationary_trace_keeps_idle_energy_without_intensity() -> None:
    trace = EnergyTrace(
        time_s=np.asarray([0.0, 1.0, 2.0]),
        speed_mps=np.zeros(3, dtype=np.float64),
        step_distance_m=np.zeros(2, dtype=np.float64),
    )

    metrics = FASTSimEnergyProvider(_fastsim_config()).measure(trace)

    assert metrics.energy_j is not None and metrics.energy_j > 0.0
    assert metrics.energy_j_per_km is None
    assert metrics.energy_wh_per_km is None
