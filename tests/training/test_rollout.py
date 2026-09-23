from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from eco_planner.envs import TrajectoryExecutionRecord, TrajectoryExecutionResult
from eco_planner.envs.domain import (
    EnergyMetrics,
    TrafficFrame,
    TransitionMetricInput,
    TransitionMetrics,
)
from eco_planner.planning.policy import ExplorationPolicyContext, policy_context_tensordict
from eco_planner.reward import RewardComponents, RewardDiagnostics, RewardResult
from eco_planner.rl.artifacts import (
    ENERGY_ROLLOUT_ARTIFACT_FIELDS,
    write_rollout_episode,
)
from eco_planner.rl.rollout import (
    ExecutionTransitionAudit,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
)
from eco_planner.rl.rollout.collector import _execution_transition_audit


def _context() -> ExplorationPolicyContext:
    return ExplorationPolicyContext(
        scene_tokens=torch.zeros((1, 2, 4)),
        scene_padding_mask=torch.zeros((1, 2), dtype=torch.bool),
        navigation_tokens=torch.zeros((1, 1, 4)),
        navigation_padding_mask=torch.zeros((1, 1), dtype=torch.bool),
        reference_trajectory=torch.zeros((1, 80, 4)),
    )


def _transition():
    context = _context()
    training_decision = build_training_decision(
        context,
        torch.tensor([[-0.5, 0.5]]),
        torch.tensor([0.5]),
        torch.tensor([1.0]),
    )
    decision_audit = policy_context_tensordict(context).update(
        dict(
            prediction=torch.zeros((1, 11, 80, 4)),
            initial_noise=torch.zeros((1, 11, 80, 4)),
            base_action=torch.tensor([[0.25, 0.75]]),
            guidance_action=torch.tensor([[-0.5, 0.5]]),
            old_joint_guidance_log_prob=torch.tensor([[0.5]]),
            state_value=torch.tensor([[1.0]]),
            beta_alpha=torch.full((1, 2), 2.0),
            beta_beta=torch.full((1, 2), 2.0),
            diffusion_rng_state=torch.ones((1, 5), dtype=torch.uint8),
            policy_rng_state=torch.ones((1, 5), dtype=torch.uint8),
        )
    )
    execution_audit = ExecutionTransitionAudit(
        reward_result=_reward_result(0.25),
        substep_results=(_reward_result(0.25),),
        route_completion_delta=0.1,
        distance_m=1.0,
        speed_mps=2.0,
        stopped=False,
        collision=False,
        wrong_direction=False,
        position_error_m=0.0,
        heading_error_rad=0.0,
        arrive_dest=False,
        out_of_road=False,
        crash_vehicle=False,
        crash_object=False,
        crash_building=False,
        crash_human=False,
        crash_sidewalk=False,
        terminated=False,
        truncated=False,
    )
    provenance = RolloutProvenance(0, 1, 2, 0)
    return training_decision, decision_audit, execution_audit, provenance


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize(
    "terminated,truncated,tail,bootstrap",
    [
        (True, False, "terminated", 0.0),
        (False, True, "truncated", 5.0),
        (False, False, "rollout_limit", 5.0),
        (True, True, "terminated", 0.0),
    ],
)
def test_rollout_constructs_next_values_and_tail_without_mutating_decisions(
    length,
    terminated,
    truncated,
    tail,
    bootstrap,
) -> None:
    builder = RolloutEpisodeBuilder()
    decisions = []
    snapshots = []
    for index in range(length):
        training, audit, execution, provenance = _transition()
        training["state_value"] = torch.tensor([[float(index + 1)]])
        audit["state_value"] = training["state_value"].clone()
        if index == length - 1:
            execution = replace(execution, terminated=terminated, truncated=truncated)
        decisions.append((training, audit))
        snapshots.append((training.clone(), audit.clone()))
        builder.append(training, audit, execution, replace(provenance, planning_cycle_index=index))

    episode = builder.finish(tail, torch.tensor([bootstrap]))

    assert episode.training["next", "state_value"].flatten().tolist() == [
        *range(2, length + 1),
        bootstrap,
    ]
    assert episode.training["next", "done"].flatten().tolist() == [False] * (length - 1) + [True]
    assert episode.training["next", "terminated"][-1].item() == terminated
    assert episode.training["next", "truncated"][-1].item() == truncated
    assert episode.training["old_joint_guidance_log_prob"].shape == torch.Size([length])
    assert episode.transition_count == length
    assert episode.reward_profile == "plannerrft_energy_v1"
    assert episode.audit["planning_cycle_index"].flatten().tolist() == list(range(length))
    for actual, expected in zip(decisions, snapshots, strict=True):
        for value, snapshot in zip(actual, expected, strict=True):
            assert (value == snapshot).all()
    audit_snapshot = episode.audit.clone()
    episode.training["next", "reward"].zero_()
    episode.training["next", "terminated"].logical_not_()
    assert (episode.audit == audit_snapshot).all()


@pytest.mark.parametrize(
    "profile",
    [
        "plannerrft_energy_v1",
        "plannerrft_energy_band_lam1_v1",
        "plannerrft_energy_band_lam2_v1",
        "plannerrft_energy_band_lam4_v1",
        "plannerrft_energy_band_lam8_v1",
        "plannerrft_energy_band_lam64_v1",
        "plannerrft_no_energy_v1",
        "plannerrft_no_energy_calibrated_v1",
    ],
)
def test_rollout_artifact_uses_the_explicit_reward_profile_schema(tmp_path: Path, profile) -> None:
    builder = RolloutEpisodeBuilder()
    training, audit, execution, provenance = _transition()
    execution = replace(
        execution,
        reward_result=replace(execution.reward_result, profile_name=profile),
        substep_results=tuple(
            replace(result, profile_name=profile) for result in execution.substep_results
        ),
    )
    builder.append(training, audit, execution, provenance)
    episode = builder.finish("rollout_limit", torch.tensor([5.0]))
    artifact = tmp_path / "episode.npz"

    write_rollout_episode(artifact, episode)

    with np.load(artifact, allow_pickle=False) as arrays:
        assert set(arrays.files) == set(ENERGY_ROLLOUT_ARTIFACT_FIELDS)
        assert str(arrays["reward_profile"]) == profile
        assert set(arrays.files) == set(
            """
            scene_tokens scene_padding_mask navigation_tokens navigation_padding_mask
            reference_trajectory base_action guidance_action old_joint_guidance_log_prob
            state_value beta_alpha beta_beta initial_noise diffusion_rng_state policy_rng_state
            reward_total reward_base_total reward_safety_gate reward_component_ttc
            reward_component_progress reward_component_comfort reward_component_speed
            reward_component_energy route_completion_delta distance_m speed_mps stopped
            collision wrong_direction position_error_m heading_error_rad arrive_dest
            out_of_road crash_vehicle crash_object
            crash_building crash_human crash_sidewalk terminated truncated map_seed noise_seed
            policy_action_seed planning_cycle_index step_distance_m native_step_energy_ml
            native_episode_energy_ml executed_fuel_proxy_step_energy_ml
            executed_fuel_proxy_ml_per_km energy_distance_valid reward_diagnostic_collision_score
            reward_diagnostic_drivable_score reward_diagnostic_wrong_direction_score
            has_ttc_candidate min_ttc_s route_progress_delta_m speed_limit_mps overspeed_mps
            longitudinal_acceleration_mps2 lateral_acceleration_mps2 jerk_mps3 yaw_rate_radps
            reward_substep_count reward_substep_safety_gate
            reward_substep_component_ttc reward_substep_component_progress
            reward_substep_component_comfort reward_substep_component_speed
            reward_substep_component_energy reward_substep_route_progress_delta_m
            reward_substep_longitudinal_acceleration_mps2
            reward_substep_lateral_acceleration_mps2 reward_substep_jerk_mps3
            reward_substep_yaw_rate_radps reward_substep_executed_fuel_proxy_step_energy_ml
            reward_substep_step_distance_m reward_substep_energy_distance_valid
            reward_profile tail_kind tail_bootstrap_value
        """.split()
        )
        booleans = set(
            """
            scene_padding_mask navigation_padding_mask stopped collision wrong_direction
            arrive_dest out_of_road
            crash_vehicle crash_object crash_building crash_human crash_sidewalk terminated
            truncated energy_distance_valid has_ttc_candidate reward_substep_energy_distance_valid
        """.split()
        )
        integers = {
            "map_seed",
            "noise_seed",
            "policy_action_seed",
            "planning_cycle_index",
            "reward_substep_count",
        }
        for key in episode.audit.keys():
            expected_dtype = (
                np.bool_
                if key in booleans
                else np.int64
                if key in integers
                else np.uint8
                if key.endswith("rng_state")
                else np.float32
            )
            assert arrays[key].dtype == expected_dtype
        assert arrays["state_value"].shape == (1, 1)
        assert arrays["old_joint_guidance_log_prob"].shape == (1, 1)
        assert arrays["diffusion_rng_state"].shape == (1, 5)
        assert arrays["executed_fuel_proxy_ml_per_km"].item() == 50.0
        assert arrays["reward_component_progress"].item() == 0.5
        assert arrays["reward_diagnostic_wrong_direction_score"].item() == 1.0
        np.testing.assert_array_equal(arrays["reward_total"], episode.training["next", "reward"])


def test_batch_and_slot_audit_share_one_deferred_payload() -> None:
    from types import SimpleNamespace

    from tensordict import cat

    from eco_planner.rl.rollout.decision import BatchRolloutDecision
    from eco_planner.runtime.contracts import HostTrajectories
    from tests.training.test_ppo import _policy_config

    training, audit, _, _ = _transition()
    host = cat([audit, audit]).to_dict()
    resolutions = []

    def resolve():
        resolutions.append(True)
        return host

    diffusion_states = (torch.ones(5, dtype=torch.uint8), torch.full((5,), 2, dtype=torch.uint8))
    policy_states = (torch.full((5,), 3, dtype=torch.uint8), torch.full((5,), 4, dtype=torch.uint8))
    decision = BatchRolloutDecision(
        HostTrajectories(np.zeros((2, 80, 4), dtype=np.float32)),
        SimpleNamespace(resolve=resolve, timing=None),
        diffusion_states,
        policy_states,
        _policy_config().model_copy(update={"hidden_dim": 4, "cross_attention_heads": 1}),
        cat([training, training]),
    )
    slot = decision.slot(1)
    assert slot.training_decision.batch_size == torch.Size([1])
    assert slot.ego_trajectory.shape == (80, 4)
    assert not resolutions

    slot_audit = slot.audit_result()
    batch_audit = decision.audit_result()
    assert slot.audit_result() is slot_audit
    assert resolutions == [True]
    assert (slot_audit == batch_audit[1:2]).all()
    assert "prediction" in slot_audit
    assert slot_audit["state_value"].shape == (1, 1)
    assert slot_audit["old_joint_guidance_log_prob"].shape == (1, 1)
    torch.testing.assert_close(slot_audit["diffusion_rng_state"][0], diffusion_states[1])
    torch.testing.assert_close(slot_audit["policy_rng_state"][0], policy_states[1])


def _reward_result(total: float, *, safety_gate: float = 1.0) -> RewardResult:
    return RewardResult(
        profile_name="plannerrft_energy_v1",
        total=total,
        base_total=total,
        safety_gate=safety_gate,
        components=RewardComponents(1.0, 0.5, 1.0, 1.0, 0.5),
        diagnostics=RewardDiagnostics(
            collision_score=1.0,
            drivable_score=1.0,
            wrong_direction_score=1.0,
            has_ttc_candidate=False,
            min_ttc_s=10.0,
            route_progress_delta_m=1.0,
            speed_mps=2.0,
            speed_limit_mps=10.0,
            overspeed_mps=0.0,
            longitudinal_acceleration_mps2=0.0,
            lateral_acceleration_mps2=0.0,
            jerk_mps3=0.0,
            yaw_rate_radps=0.0,
            step_distance_m=1.0,
            native_step_energy_ml=0.0,
            native_episode_energy_ml=0.0,
            executed_fuel_proxy_step_energy_ml=0.05,
            executed_fuel_proxy_ml_per_km=50.0,
            energy_distance_valid=True,
        ),
    )


def _substep_metrics(
    index: int,
    *,
    stopped: bool = False,
    collision: bool = False,
    wrong_direction: bool = False,
) -> TransitionMetrics:
    position = (float(index), 0.0)
    return TransitionMetrics(
        input=TransitionMetricInput(
            previous_position_xy_m=(0.0, 0.0),
            position_xy_m=position,
            previous_velocity_xy_mps=(0.0, 0.0),
            velocity_xy_mps=(0.0, 0.0),
            previous_acceleration_xy_mps2=(0.0, 0.0),
            heading_rad=0.0,
            yaw_rate_radps=0.0,
            route_progress_delta_m=0.0,
            route_heading_rad=0.0,
            speed_limit_mps=10.0,
            ego_width_m=2.0,
            ego_length_m=4.0,
            traffic_frame=TrafficFrame(index, position, 0.0, 1.0, (), ()),
            target_position_xy_m=position,
            target_heading_rad=0.0,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            crash_sidewalk=False,
            out_of_road=False,
            native_step_energy_ml=0.0,
            native_episode_energy_ml=0.0,
            timestep_s=0.1,
        ),
        speed_mps=2.0 + index,
        longitudinal_acceleration_mps2=0.0,
        lateral_acceleration_mps2=0.0,
        jerk_mps3=0.0,
        step_distance_m=1.0,
        position_error_m=float(index),
        heading_error_rad=float(index) / 10.0,
        route_heading_error_rad=0.0,
        wrong_direction=wrong_direction,
        stopped=stopped,
        collision=collision,
        energy=EnergyMetrics("metadrive_fuel_proxy", 1.0, None, 0.1),
    )


def _execution_result(
    metrics: tuple[TransitionMetrics, ...], *, route_completion: float = 0.5
) -> TrajectoryExecutionResult:
    count = len(metrics)
    return TrajectoryExecutionResult(
        execution=TrajectoryExecutionRecord(
            start_center=np.zeros(2),
            start_heading=0.0,
            world_centers=np.zeros((80, 2)),
            world_headings=np.zeros(80),
            substep_states=np.zeros((count, 7)),
            target_centers=np.zeros((count, 2)),
            target_headings=np.zeros(count),
            substep_terminated=np.zeros(count, dtype=np.bool_),
            substep_truncated=np.zeros(count, dtype=np.bool_),
            traffic_frames=(),
            route_completion=route_completion,
            arrive_dest=False,
            out_of_road=False,
            crash_vehicle=False,
            crash_object=False,
            crash_building=False,
            crash_human=False,
            max_step=False,
        ),
        metrics=metrics,
        terminated=False,
        truncated=False,
    )


class _ScriptedEvaluator:
    def __init__(self, results: tuple[RewardResult, ...]) -> None:
        self._results = list(results)

    def __call__(self, metrics: TransitionMetrics) -> RewardResult:
        return self._results.pop(0)


def test_transition_audit_aggregates_multiple_substeps() -> None:
    metrics = (
        _substep_metrics(0),
        _substep_metrics(1, stopped=True, wrong_direction=True),
        _substep_metrics(2, collision=True),
    )
    evaluator = _ScriptedEvaluator(
        (
            _reward_result(1.0, safety_gate=1.0),
            _reward_result(2.0, safety_gate=0.5),
            _reward_result(4.0, safety_gate=0.25),
        )
    )

    audit = _execution_transition_audit(
        _execution_result(metrics, route_completion=0.5),
        0.2,
        terminated=False,
        truncated=False,
        reward_evaluator=evaluator,
    )

    assert audit.reward_result.total == pytest.approx(7.0)
    assert audit.reward_result.safety_gate == pytest.approx(0.25)
    assert audit.route_completion_delta == pytest.approx(0.3)
    assert audit.distance_m == pytest.approx(3.0)
    assert audit.speed_mps == pytest.approx(3.0)
    assert audit.stopped is True
    assert audit.collision is True
    assert audit.wrong_direction is True
    assert audit.position_error_m == pytest.approx(1.0)
    assert audit.heading_error_rad == pytest.approx(0.1)


def test_transition_audit_rejects_mismatched_substep_count() -> None:
    metrics = (_substep_metrics(0), _substep_metrics(1))
    result = _execution_result(metrics)
    mismatched = replace(
        result, execution=replace(result.execution, substep_states=np.zeros((1, 7)))
    )
    evaluator = _ScriptedEvaluator((_reward_result(1.0), _reward_result(1.0)))

    with pytest.raises(RuntimeError, match="match executed substeps"):
        _execution_transition_audit(
            mismatched,
            0.0,
            terminated=False,
            truncated=False,
            reward_evaluator=evaluator,
        )
