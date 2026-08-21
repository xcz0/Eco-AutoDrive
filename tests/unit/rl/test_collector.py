from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from eco_planner.envs import TrajectoryExecutionRecord
from eco_planner.envs.traffic_state import TrafficFrame
from eco_planner.evaluation.config import ScenarioConfig
from eco_planner.rl import collector
from eco_planner.rl.collector import collect_rollout_episode, collect_vector_rollout_episodes
from eco_planner.rl.policy import ExplorationPolicyContext
from eco_planner.rl.rollout import build_training_decision
from eco_planner.rl.runtime import RolloutAudit, RolloutDecision


def test_stationary_trajectory_satisfies_execution_contract() -> None:
    trajectory = collector._stationary_trajectory()

    assert trajectory.shape == (80, 4)
    assert trajectory.dtype == np.float32
    assert np.isfinite(trajectory).all()
    assert np.all(np.linalg.norm(trajectory[:, 2:4], axis=-1) > 0.0)


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
            return _observation(float(env.step_count))

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
            marker = int(observation["ego_current_state"][0, 0].item())
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
            self.bootstrap_marker = int(observation["ego_current_state"][0, 0].item())
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
    assert episode.training["state_value"].squeeze(-1).tolist() == [0.0, 1.0]
    assert len(FakeEnv.instances[0].trajectories) == 2
    for trajectory in FakeEnv.instances[0].trajectories:
        assert trajectory.shape == (80, 4)
        assert trajectory.dtype == np.float32
        assert np.isfinite(trajectory).all()
        assert np.all(np.linalg.norm(trajectory[:, 2:4], axis=-1) > 0.0)


def test_vector_collector_keeps_other_slot_rng_and_gae_boundaries_after_reset(monkeypatch) -> None:
    class FakeVectorEnv:
        terminal_first_slot = True
        instances: list[FakeVectorEnv] = []

        def __init__(self, *_: object, **__: object) -> None:
            self.step_counts = [0, 0]
            self.reset_counts = [0, 0]
            self.step_batches: list[int] = []
            FakeVectorEnv.instances.append(self)

        def __enter__(self) -> FakeVectorEnv:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def reset(self, scenarios):
            return tuple(self._reset(slot) for slot in range(len(scenarios)))

        def reset_at(self, slot: int, _scenario):
            self.reset_counts[slot] += 1
            self.step_counts[slot] = 0
            return self._reset(slot)

        def step(self, trajectories):
            self.step_batches.append(len(trajectories))
            results = []
            for slot in range(len(trajectories)):
                self.step_counts[slot] += 1
                terminated = (
                    self.terminal_first_slot
                    and slot == 0
                    and self.reset_counts[slot] == 0
                    and self.step_counts[slot] == 1
                )
                results.append(
                    SimpleNamespace(
                        observation=_observation(float(slot * 10 + self.step_counts[slot])),
                        reward=0.25,
                        terminated=terminated,
                        truncated=False,
                        execution=_info()["trajectory_execution"],
                    )
                )
            return tuple(results)

        def _reset(self, slot: int):
            return SimpleNamespace(observation=_observation(float(slot * 10)), route_completion=0.0)

    class FakeBatchDecision:
        def __init__(self, decisions: list[_FakeRolloutDecision]) -> None:
            self._decisions = decisions
            self.ego_trajectories = np.tile(
                collector._stationary_trajectory(), (len(decisions), 1, 1)
            )

        def slot(self, index: int) -> _FakeRolloutDecision:
            return self._decisions[index]

    class FakeRuntime:
        planner_config = SimpleNamespace()

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []
            self.bootstrap_batch_sizes: list[int] = []

        def decide_batch(self, observation, diffusion_generators, policy_generators):
            batch = observation["ego_current_state"].shape[0]
            self.batch_sizes.append(batch)
            decisions = []
            for slot in range(batch):
                diffusion_state = diffusion_generators[slot].get_state()
                policy_state = policy_generators[slot].get_state()
                noise = torch.rand((), generator=diffusion_generators[slot]).item()
                action = torch.rand((1, 2), generator=policy_generators[slot])
                base = 0.2 + 0.6 * action
                marker = float(observation["ego_current_state"][slot, 0])
                audit = RolloutAudit(
                    prediction=np.tile(collector._stationary_trajectory(), (1, 11, 1, 1)),
                    initial_noise=torch.full((1, 11, 80, 4), noise),
                    policy_context=_context(marker),
                    base_action=base,
                    guidance_action=2.0 * base - 1.0,
                    old_joint_guidance_log_prob=torch.tensor([0.5]),
                    old_value=torch.tensor([marker]),
                    beta_alpha=torch.full((1, 2), 2.0),
                    beta_beta=torch.full((1, 2), 2.0),
                    diffusion_rng_state=diffusion_state,
                    policy_rng_state=policy_state,
                )
                decisions.append(_FakeRolloutDecision(audit))
            return FakeBatchDecision(decisions)

        def bootstrap_value_batch(self, observation, diffusion_generators):
            self.bootstrap_batch_sizes.append(observation["ego_current_state"].shape[0])
            return torch.tensor(
                [torch.rand((), generator=generator).item() for generator in diffusion_generators]
            )

    monkeypatch.setattr("eco_planner.rl.collector.VectorMetaDriveEnv", FakeVectorEnv)

    def collect(terminal_first_slot: bool):
        FakeVectorEnv.terminal_first_slot = terminal_first_slot
        runtime = FakeRuntime()
        episodes = collect_vector_rollout_episodes(
            (
                ScenarioConfig(name="first", map="S", seed=0),
                ScenarioConfig(name="second", map="S", seed=1),
            ),
            runtime,
            {"trajectory_execution_steps": 1},
            mode="no_traffic",
            map_query_radius_m=100.0,
            history_warmup_steps=0,
            transitions_per_slot=3,
            stopped_speed_threshold_mps=0.1,
            diffusion_generators=(
                torch.Generator().manual_seed(10),
                torch.Generator().manual_seed(11),
            ),
            policy_generators=(
                torch.Generator().manual_seed(20),
                torch.Generator().manual_seed(21),
            ),
            noise_seeds=(100, 101),
            policy_action_seeds=(200, 201),
        )
        return episodes, runtime

    reset_episodes, reset_runtime = collect(True)
    baseline_episodes, baseline_runtime = collect(False)

    assert reset_runtime.batch_sizes == [2, 2, 2]
    assert reset_runtime.bootstrap_batch_sizes == [2]
    assert FakeVectorEnv.instances[0].step_batches == [2, 2, 2]
    assert [len(slot) for slot in reset_episodes] == [2, 1]
    assert reset_episodes[0][0].tail_kind == "terminated"
    assert reset_episodes[0][0].training["next", "done"][-1].item()
    assert reset_episodes[0][1].tail_kind == "rollout_limit"
    assert baseline_runtime.batch_sizes == [2, 2, 2]

    reset_slot = reset_episodes[1][0].audit
    baseline_slot = baseline_episodes[1][0].audit
    for name in ("base_action", "initial_noise", "diffusion_rng_state", "policy_rng_state"):
        torch.testing.assert_close(reset_slot[name], baseline_slot[name])


def _context(marker: float) -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.full((1, 2, 4), marker, dtype=torch.float32),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4), dtype=torch.float32),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4), dtype=torch.float32),
    )


def _observation(marker: float) -> dict[str, torch.Tensor]:
    ego = torch.zeros(10, dtype=torch.float32)
    ego[0] = marker
    return {
        "ego_current_state": ego,
        "neighbor_agents_past": torch.zeros((32, 21, 11), dtype=torch.float32),
        "static_objects": torch.zeros((5, 10), dtype=torch.float32),
        "lanes": torch.zeros((70, 20, 12), dtype=torch.float32),
        "lanes_speed_limit": torch.zeros((70, 1), dtype=torch.float32),
        "lanes_has_speed_limit": torch.zeros((70, 1), dtype=torch.bool),
        "route_lanes": torch.zeros((25, 20, 12), dtype=torch.float32),
        "route_lanes_speed_limit": torch.zeros((25, 1), dtype=torch.float32),
        "route_lanes_has_speed_limit": torch.zeros((25, 1), dtype=torch.bool),
    }


class _FakeRolloutDecision:
    def __init__(self, audit: RolloutAudit) -> None:
        self._audit = audit

    @property
    def ego_trajectory(self) -> np.ndarray:
        return self._audit.ego_trajectory

    @property
    def training_decision(self) -> object:
        return build_training_decision(
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
