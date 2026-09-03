"""Framework-independent environment facts, metrics, and energy providers."""

from .energy import MetaDriveFuelProxyProvider
from .execution import TrajectoryExecutionRecord
from .fastsim import FASTSimEnergyConfig, FASTSimEnergyProvider
from .metrics import (
    EnergyMetricName,
    EnergyMetricProvider,
    EnergyMetrics,
    EnergyTrace,
    TransitionMetricInput,
    TransitionMetrics,
    derive_transition_metrics,
    minimum_time_to_collision_s,
)
from .traffic import (
    ParticipantKind,
    StaticObjectKind,
    StaticTrafficObjectState,
    TrafficFrame,
    TrafficParticipantState,
)
from .trajectory import WorldTrajectory, to_world_trajectory

__all__ = [
    "EnergyMetricName",
    "EnergyMetricProvider",
    "EnergyMetrics",
    "EnergyTrace",
    "FASTSimEnergyConfig",
    "FASTSimEnergyProvider",
    "MetaDriveFuelProxyProvider",
    "StaticTrafficObjectState",
    "TrajectoryExecutionRecord",
    "TransitionMetricInput",
    "TransitionMetrics",
    "TrafficFrame",
    "TrafficParticipantState",
    "WorldTrajectory",
    "derive_transition_metrics",
    "minimum_time_to_collision_s",
    "to_world_trajectory",
    "ParticipantKind",
    "StaticObjectKind",
]
