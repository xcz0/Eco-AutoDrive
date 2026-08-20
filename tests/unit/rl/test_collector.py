from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from eco_planner.envs import TrafficFrame, TrajectoryExecutionRecord
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl.collector import collect_rollout_episode
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.rollout import PPOTrainingDecision
from eco_planner.rl.runtime import RolloutAudit, RolloutDecision


def test_collector_aligns_one_substep_observations_actions_and_tail_bootstrap(
    monkeypatch,
) -> None:
    class FakeEnv:
        instances: list[FakeEnv] = []

        def __init__(self, config: dict[str, object]) -> None:
            self.config = config
            self.step_count = 0
            self.trajectories: list[np.ndarray] = []
            FakeEnv.instances.append(self)

        def reset(self, *, seed: int) -> tuple[None, dict[str, object]]:
            assert seed == 9
            return None, {}

        @property
        def route_completion(self) -> float:
            return self.step_count * 0.01

        def step(self, trajectory: np.ndarray):
            self.trajectories.append(trajectory.copy())
            self.step_count += 1
            return None, 0.25, False, False, _info()

        def close(self) -> None:
            return None

    class FakeAdapter:
        def __init__(self, *_: object) -> None:
            self.env: FakeEnv | None = None

        def reset(self, env: FakeEnv) -> None:
            self.env = env

        def build(self, env: FakeEnv) -> dict[str, torch.Tensor]:
            return {"marker": torch.tensor([env.step_count], dtype=torch.float32)}

    class FakeRuntime:
        noise_seed = 4
        policy_action_seed = 5
        planner_config = SimpleNamespace()

        def __init__(self) -> None:
            self.observation_markers: list[int] = []
            self.bootstrap_marker: int | None = None

        def new_noise_generator(self) -> torch.Generator:
            return torch.Generator().manual_seed(self.noise_seed)

        def new_policy_generator(self) -> torch.Generator:
            return torch.Generator().manual_seed(self.policy_action_seed)

        def decide(self, observation, diffusion_generator, policy_generator) -> RolloutDecision:
            marker = int(observation["marker"].item())
            self.observation_markers.append(marker)
            context = _context(float(marker))
            trajectory = np.zeros((1, 11, 80, 4), dtype=np.float32)
            trajectory[..., 2] = 1.0
            base = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
            audit = RolloutAudit(
                prediction=trajectory,
                initial_noise=torch.zeros((1, 11, 80, 4), dtype=torch.float32),
                policy_context=context,
                base_action=base,
                guidance_action=2.0 * base - 1.0,
                old_joint_guidance_log_prob=torch.tensor([0.5], dtype=torch.float32),
                old_value=torch.tensor([float(marker)], dtype=torch.float32),
                beta_alpha=torch.full((1, 2), 2.0),
                beta_beta=torch.full((1, 2), 2.0),
                diffusion_rng_state=diffusion_generator.get_state(),
                policy_rng_state=policy_generator.get_state(),
            )
            return _FakeRolloutDecision(audit)

        def bootstrap_value(self, observation, diffusion_generator) -> torch.Tensor:
            self.bootstrap_marker = int(observation["marker"].item())
            return torch.tensor([float(self.bootstrap_marker)], dtype=torch.float32)

    monkeypatch.setattr("eco_planner.rl.collector.TrajectoryMetaDriveEnv", FakeEnv)
    monkeypatch.setattr(
        "eco_planner.rl.collector.NoTrafficMetaDriveObservationAdapter", FakeAdapter
    )
    runtime = FakeRuntime()
    episode = collect_rollout_episode(
        ScenarioConfig(name="fake", map="S", seed=9),
        runtime,
        {"trajectory_execution_steps": 1},
        mode="no_traffic",
        map_query_radius_m=100.0,
        history_warmup_steps=0,
        max_transitions=2,
    )

    assert runtime.observation_markers == [0, 1]
    assert runtime.bootstrap_marker == 2
    assert episode.tail_kind == "rollout_limit"
    assert episode.tail_bootstrap_value.item() == 2.0
    assert episode.training_trajectory["state_value"].squeeze(-1).tolist() == [0.0, 1.0]
    assert len(FakeEnv.instances[0].trajectories) == 2
    assert all(trajectory.shape == (80, 4) for trajectory in FakeEnv.instances[0].trajectories)


def _context(marker: float) -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.full((1, 2, 4), marker, dtype=torch.float32),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4), dtype=torch.float32),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4), dtype=torch.float32),
    )


class _FakeRolloutDecision:
    def __init__(self, audit: RolloutAudit) -> None:
        self._audit = audit

    @property
    def ego_trajectory(self) -> np.ndarray:
        return self._audit.ego_trajectory

    @property
    def training_decision(self) -> PPOTrainingDecision:
        return PPOTrainingDecision.from_policy_output(
            self._audit.policy_context,
            self._audit.guidance_action,
            self._audit.old_joint_guidance_log_prob,
            self._audit.old_value,
        )

    def audit_result(self) -> RolloutAudit:
        return self._audit


def _info() -> dict[str, object]:
    states = np.zeros((1, 7))
    return {
        "trajectory_execution": TrajectoryExecutionRecord(
            start_center=np.zeros(2),
            start_heading=0.0,
            world_centers=np.zeros((80, 2)),
            world_headings=np.zeros(80),
            substep_states=states,
            target_centers=states[:, :2],
            target_headings=states[:, 2],
            position_errors_m=np.zeros(1),
            heading_errors_rad=np.zeros(1),
            substep_rewards=np.asarray([0.25]),
            substep_dense_rewards=np.asarray([0.25]),
            substep_terminated=np.asarray([False]),
            substep_truncated=np.asarray([False]),
            traffic_frames=(TrafficFrame(1, (0.0, 0.0), 0.0, 1.0, (), ()),),
            route_completion=0.0,
            arrive_dest=False,
            out_of_road=False,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            max_step=False,
        )
    }
