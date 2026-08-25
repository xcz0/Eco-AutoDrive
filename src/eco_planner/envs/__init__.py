"""Planner-facing MetaDrive interfaces."""

# ruff: noqa: I001

# Torch must load before MetaDrive/Panda3D on Windows to avoid DLL initialization failures.
from eco_planner.envs.observation import PlannerObservationSpec, collate_observations
from eco_planner.envs.observation_adapter import TrafficObservationAudit
from eco_planner.envs.execution import TrajectoryExecutionRecord
from eco_planner.envs.metadrive_env import TrajectoryMetaDriveEnv
from eco_planner.envs.slot import (
    EnvSlotObservation,
    EnvSlotReset,
    EnvSlotStep,
    MetaDriveEnvSlot,
)
from eco_planner.envs.torchrl_env import TorchRLMetaDriveEnv
from eco_planner.envs.torchrl_parallel_env import (
    TorchRLParallelScenario,
    TorchRLScenarioMetaDriveEnv,
    create_torchrl_parallel_env_poc,
)
from eco_planner.envs.vector_metadrive import (
    VectorEnvReset,
    VectorEnvScenario,
    VectorEnvStep,
    VectorEnvTiming,
    VectorMetaDriveEnv,
    VectorMetaDriveWorkerError,
)

__all__ = [
    "collate_observations",
    "PlannerObservationSpec",
    "TrafficObservationAudit",
    "TrajectoryMetaDriveEnv",
    "TrajectoryExecutionRecord",
    "EnvSlotObservation",
    "EnvSlotReset",
    "EnvSlotStep",
    "MetaDriveEnvSlot",
    "TorchRLMetaDriveEnv",
    "TorchRLParallelScenario",
    "TorchRLScenarioMetaDriveEnv",
    "create_torchrl_parallel_env_poc",
    "VectorEnvReset",
    "VectorEnvScenario",
    "VectorEnvStep",
    "VectorEnvTiming",
    "VectorMetaDriveEnv",
    "VectorMetaDriveWorkerError",
]
