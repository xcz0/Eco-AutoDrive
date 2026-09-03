"""Planner-facing MetaDrive interfaces."""

# ruff: noqa: I001

# Torch must load before MetaDrive/Panda3D on Windows to avoid DLL initialization failures.
from eco_planner.envs.domain import TrajectoryExecutionRecord, WorldTrajectory
from eco_planner.envs.observation import (
    PlannerObservationSpec,
    TrafficObservationAudit,
)
from eco_planner.envs.metadrive.slot import (
    EnvSlotObservation,
    EnvSlotReset,
    EnvSlotStep,
    MetaDriveEnvSlot,
)
from eco_planner.envs.torchrl.adapter import TorchRLMetaDriveEnv
from eco_planner.envs.runtime.vector import (
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
    operation_results,
)
from eco_planner.envs.torchrl.worker import (
    VectorEnvScenario,
    VectorEnvTiming,
    WorkerResetResult,
    WorkerStepResult,
)

__all__ = [
    "PlannerObservationSpec",
    "TrafficObservationAudit",
    "TrajectoryExecutionRecord",
    "WorldTrajectory",
    "EnvSlotObservation",
    "EnvSlotReset",
    "EnvSlotStep",
    "MetaDriveEnvSlot",
    "TorchRLMetaDriveEnv",
    "VectorEnvScenario",
    "VectorEnvTiming",
    "VectorMetaDriveEnv",
    "VectorMetaDriveWorkerError",
    "operation_results",
    "WorkerResetResult",
    "WorkerStepResult",
]
