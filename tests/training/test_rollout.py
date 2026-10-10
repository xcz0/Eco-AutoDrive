from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from eco_planner.envs import TrajectoryExecutionRecord, TrajectoryExecutionResult
from eco_planner.envs.domain import (
    EnergyMetrics,
    TrafficFrame,
    TransitionMetricInput,
    TransitionMetrics,
)
from eco_planner.planning.policy import ExplorationPolicyContext
from eco_planner.reward import (
    PlannerRFTRewardResult,
    RewardComponents,
    RewardDiagnostics,
    RewardResult,
)
from eco_planner.rl.artifacts import (
    ENERGY_ROLLOUT_ARTIFACT_FIELDS,
    write_rollout_episode,
)
from eco_planner.rl.rollout import (
    ExecutionTransitionAudit,
    RolloutEpisodeBuilder,
    RolloutProvenance,
    build_training_decision,
    build_training_transition,
)
from eco_planner.rl.rollout.collector import _EpisodeLifecycle, _execution_transition_audit
from eco_planner.rl.rollout.decision import RolloutDecision
from eco_planner.runtime.contracts import HostTrajectories
from tests.training.helpers import _decision_audit


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_update_storage_preserves_interleaved_scenarios_and_disjoint_episode_views(
    monkeypatch, device
) -> None:
    from eco_planner.rl.rollout import contracts

    concatenate = contracts.cat

    def audit_cat(items):
        assert "next" not in items[0].keys(), "compact training must not concatenate episodes"
        return concatenate(items)

    monkeypatch.setattr(contracts, "cat", audit_cat)
    storage = TensorDict({}, batch_size=[8])
    builders = [
        RolloutEpisodeBuilder(training_storage=storage, training_start=slot * 4)
        for slot in range(2)
    ]
    schedules = [
        {1: ("terminated", 0.0), 2: ("truncated", 5.0), 3: ("rollout_limit", 7.0)},
        {0: ("terminated", 0.0), 3: ("rollout_limit", 9.0)},
    ]
    episodes = [[], []]
    snapshots = []
    for index in range(4):
        for slot in range(2):
            training, audit, execution, provenance = _transition()
            value = float(slot * 4 + index + 1)
            training["state_value"].fill_(value)
            audit["state_value"].fill_(value)
            training = training.to(device)
            tail, bootstrap = schedules[slot].get(index, (None, None))
            execution = replace(
                execution,
                terminated=tail == "terminated",
                truncated=tail == "truncated",
            )
            builders[slot].append(
                build_training_transition(
                    training,
                    execution.reward_result,
                    terminated=execution.terminated,
                    truncated=execution.truncated,
                ),
                audit,
                execution,
                replace(
                    provenance, map_seed=slot, planning_cycle_index=builders[slot].transition_count
                ),
            )
            # The sole collection write owns a snapshot, independent of the decision.
            training["state_value"].fill_(-100.0)
            if tail is not None:
                episode = builders[slot].finish(tail, torch.tensor([bootstrap]))
                episodes[slot].append(episode)
                snapshots.append((episode.training, episode.training.clone()))
                builders[slot] = builders[slot].next_episode()

    assert storage["state_value"].flatten().tolist() == list(range(1, 9))
    assert storage["next", "state_value"].flatten().tolist() == [2, 0, 5, 7, 0, 7, 8, 9]
    assert storage["next", "done"].flatten().tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        True,
    ]
    offset = 0
    for slot, group in enumerate(episodes):
        for episode in group:
            assert episode.audit["map_seed"].flatten().tolist() == [slot] * episode.transition_count
            for key, field in episode.training.items(include_nested=True, leaves_only=True):
                assert field.data_ptr() == storage[key][offset:].data_ptr()
                assert (
                    field.untyped_storage().data_ptr() == storage[key].untyped_storage().data_ptr()
                )
            offset += episode.transition_count
    for training, snapshot in snapshots:
        assert (training == snapshot).all()


def test_update_storage_rejects_broadcasting_or_casting_decisions() -> None:
    storage = TensorDict({}, batch_size=[2])
    builder = RolloutEpisodeBuilder(training_storage=storage)
    training, audit, execution, provenance = _transition()
    transition = build_training_transition(
        training, execution.reward_result, terminated=False, truncated=False
    )
    builder.append(transition, audit, execution, provenance)
    for invalid in (
        transition.clone().set("scene_tokens", torch.zeros(1, 1, 4)),
        transition.clone().set("state_value", torch.zeros(1, 1, dtype=torch.float64)),
    ):
        with pytest.raises(ValueError, match="differs from update storage"):
            builder.append(invalid, audit, execution, provenance)
    for key in ("guidance_action", ("next", "reward")):
        with pytest.raises(ValueError, match="fields differ from update storage"):
            builder.append(transition.exclude(key), audit, execution, provenance)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_update_storage_preserves_next_value_dtype_promotion(device, dtype):
    compact = RolloutEpisodeBuilder(training_storage=TensorDict({}, batch_size=[2]))
    separate = RolloutEpisodeBuilder()
    values = torch.tensor([[1.125], [2.375]], device=device, dtype=dtype)
    bootstrap = torch.tensor([5.001], dtype=torch.float32, device=device)
    for value in values:
        training, audit, execution, provenance = _transition()
        training = training.to(device)
        training["state_value"] = value.reshape(1, 1)
        transition = build_training_transition(
            training, execution.reward_result, terminated=False, truncated=False
        )
        compact.append(transition, audit, execution, provenance)
        separate.append(transition, audit, execution, provenance)

    compact_episode = compact.finish("rollout_limit", bootstrap)
    separate_episode = separate.finish("rollout_limit", bootstrap)

    expected = torch.cat((values[1:], bootstrap.reshape(1, 1)))
    actual = compact_episode.training["next", "state_value"]
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert (compact_episode.training == separate_episode.training).all()


@pytest.mark.parametrize("physical_slots", [1, 2, 3])
def test_collector_assigns_compact_storage_in_logical_order_across_worker_groups(
    monkeypatch, physical_slots
) -> None:
    from types import SimpleNamespace

    from eco_planner.rl.rollout import VectorRolloutCollector

    # Exercise collect's real grouping with deterministic episode producers.
    collector = object.__new__(VectorRolloutCollector)
    collector._specs = (0, 1, 2)
    collector._scenarios = (0, 1, 2)
    collector._physical_slot_count = physical_slots

    def initialize(specs, scenarios, diffusion, policy, noise_seeds, policy_seeds):
        return [
            SimpleNamespace(index=index, lifecycle=_EpisodeLifecycle(0.0)) for index in specs
        ], None

    def collect_group(states, observation, *, transitions_per_slot, **kwargs):
        groups = []
        for state in states:
            episodes = []
            for index in range(transitions_per_slot):
                training, audit, execution, provenance = _transition()
                value = float(state.index * transitions_per_slot + index)
                training["state_value"].fill_(value)
                audit["state_value"].fill_(value)
                terminated = index == 0
                execution = replace(execution, terminated=terminated)
                state.lifecycle.builder.append(
                    build_training_transition(
                        training, execution.reward_result, terminated=terminated, truncated=False
                    ),
                    audit,
                    execution,
                    replace(provenance, map_seed=state.index),
                )
                if terminated or index == transitions_per_slot - 1:
                    episodes.append(
                        state.lifecycle.finish(
                            "terminated" if terminated else "rollout_limit",
                            torch.tensor([0.0 if terminated else 9.0]),
                        )
                    )
            groups.append(tuple(episodes))
        return tuple(groups)

    monkeypatch.setattr(collector, "_initialize_group", initialize)
    monkeypatch.setattr(collector, "_collect_group", collect_group)
    kwargs = dict(
        transitions_per_slot=3,
        diffusion_generators=tuple(torch.Generator() for _ in range(3)),
        policy_generators=tuple(torch.Generator() for _ in range(3)),
        noise_seeds=(0, 1, 2),
        policy_action_seeds=(3, 4, 5),
    )
    storage = TensorDict({}, batch_size=[9])
    groups = collector.collect(**kwargs, training_storage=storage)

    assert storage["state_value"].flatten().tolist() == list(range(9))
    assert [[episode.transition_count for episode in group] for group in groups] == [[1, 2]] * 3
    for index, group in enumerate(groups):
        assert (
            group[0].training["state_value"].data_ptr()
            == storage["state_value"][index * 3 :].data_ptr()
        )
        assert all(episode.audit["map_seed"].unique().item() == index for episode in group)
    snapshot = storage.clone()
    collector.collect(**kwargs, training_storage=TensorDict({}, batch_size=[9]))
    assert (storage == snapshot).all()
    with pytest.raises(ValueError, match="empty, independently owned"):
        collector.collect(**kwargs, training_storage=storage)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "key,value,message",
    [
        ("scene_tokens", float("nan"), "finite"),
        ("navigation_tokens", float("inf"), "finite"),
        ("reference_trajectory", float("nan"), "finite"),
        ("guidance_action", float("nan"), "finite"),
        ("old_joint_guidance_log_prob", float("inf"), "finite"),
        ("state_value", float("nan"), "finite"),
        (("next", "state_value"), float("inf"), "finite"),
        (("next", "reward"), float("nan"), "finite"),
        ("guidance_action", -1.0, "strictly inside"),
        ("guidance_action", 1.0, "strictly inside"),
        ("guidance_action", -2.0, "strictly inside"),
        ("guidance_action", 2.0, "strictly inside"),
        (("next", "done"), False, "final GAE boundary"),
        (("next", "terminated"), False, "zero bootstrap"),
    ],
)
def test_episode_rejects_invalid_training_independently_of_valid_audit(
    device, key, value, message
) -> None:
    from tests.training.helpers import _episode

    episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    training = episode.training.clone().to(device)
    training[key].fill_(value)

    with pytest.raises(ValueError, match=message):
        replace(
            episode, training=training, tail_bootstrap_value=episode.tail_bootstrap_value.to(device)
        )


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("bootstrap", [float("nan"), float("inf"), 1.0])
def test_episode_rejects_invalid_bootstrap_independently_of_next_value(device, bootstrap) -> None:
    from tests.training.helpers import _episode

    episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    with pytest.raises(ValueError, match="bootstrap"):
        replace(
            episode,
            training=episode.training.to(device),
            tail_bootstrap_value=torch.tensor([bootstrap], device=device),
        )


@pytest.mark.parametrize(
    "bootstrap",
    [torch.zeros(1, dtype=torch.float64), torch.zeros(2), torch.zeros(1, requires_grad=True)],
)
def test_episode_rejects_invalid_bootstrap_structure(bootstrap) -> None:
    from tests.training.helpers import _episode

    episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    with pytest.raises(ValueError, match="detached float32 with shape"):
        replace(episode, tail_bootstrap_value=bootstrap)


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("scene_tokens", float("nan"), "finite"),
        ("reward_total", float("inf"), "finite"),
        ("base_action", 0.0, "strictly inside"),
        ("guidance_action", 1.0, "strictly inside"),
        ("beta_alpha", 0.0, "strictly positive"),
    ],
)
def test_episode_rejects_invalid_audit_independently_of_valid_training(key, value, message) -> None:
    from tests.training.helpers import _episode

    episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    audit = episode.audit.clone()
    audit[key].fill_(value)
    with pytest.raises(ValueError, match=message):
        replace(episode, audit=audit)


@pytest.mark.gpu
def test_episode_validation_uses_one_cuda_predicate_transfer_without_scalar_reads(monkeypatch):
    from tests.training.helpers import _episode

    episode = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    training = episode.training.to("cuda")
    bootstrap = episode.tail_bootstrap_value.to("cuda")
    transfers = []
    original_cpu = torch.Tensor.cpu
    original_bool = torch.Tensor.__bool__
    original_item = torch.Tensor.item
    original_equal = torch.equal

    def cpu(value, *args, **kwargs):
        if value.device.type == "cuda":
            transfers.append((value.shape, value.dtype))
        return original_cpu(value, *args, **kwargs)

    def boolean(value):
        assert value.device.type != "cuda", "validation must not branch on a CUDA tensor"
        return original_bool(value)

    def item(value, *args, **kwargs):
        assert value.device.type != "cuda", "validation must not read CUDA scalars"
        return original_item(value, *args, **kwargs)

    def equal(left, right):
        assert left.device.type != "cuda" and right.device.type != "cuda"
        return original_equal(left, right)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "__bool__", boolean)
    monkeypatch.setattr(torch.Tensor, "item", item)
    monkeypatch.setattr(torch, "equal", equal)
    replace(episode, training=training, tail_bootstrap_value=bootstrap)

    assert len(transfers) == 1
    shape, dtype = transfers[0]
    assert len(shape) == 1
    assert dtype == torch.bool


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_training_transition_needs_only_scalar_reward_on_collection_device(device) -> None:
    decision, _, _, _ = _transition()
    decision = decision.to(device)
    snapshot = decision.clone()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None

    transition = build_training_transition(
        decision, RewardResult(total=-0.125), terminated=False, truncated=True
    )

    assert (decision == snapshot).all()
    assert "next" not in decision.keys()
    for key, dtype, expected in (
        ("reward", torch.float32, -0.125),
        ("terminated", torch.bool, False),
        ("truncated", torch.bool, True),
    ):
        value = transition["next", key]
        assert value.device.type == device
        assert value.dtype == dtype
        assert value.shape == (1, 1)
        assert not value.requires_grad
        assert value.item() == expected
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


def test_episode_finish_does_not_read_reward_or_flags_from_audit(monkeypatch) -> None:
    from eco_planner.rl.rollout import contracts

    decision, decision_audit, execution, provenance = _transition()
    audit = contracts.build_rollout_audit(decision_audit, execution, provenance)
    monkeypatch.setattr(contracts, "build_rollout_audit", lambda *args: audit)
    builder = RolloutEpisodeBuilder()
    builder.append(
        build_training_transition(
            decision, execution.reward_result, terminated=False, truncated=False
        ),
        decision_audit,
        execution,
        provenance,
    )
    # Deliberately perturb the audit projection to expose any reverse dependency.
    audit["reward_total"].fill_(99.0)
    audit["terminated"].fill_(True)
    audit["truncated"].fill_(True)

    episode = builder.finish("rollout_limit", torch.tensor([5.0]))

    assert episode.training["next", "reward"].item() == execution.reward_result.total
    assert not episode.training["next", "terminated"].item()
    assert not episode.training["next", "truncated"].item()


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
    decision_audit = _decision_audit(context)
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


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
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
    device,
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
        training = training.to(device)
        if index == length - 1:
            execution = replace(execution, terminated=terminated, truncated=truncated)
        decisions.append((training, audit))
        snapshots.append((training.clone(), audit.clone()))
        transition = build_training_transition(
            training,
            execution.reward_result,
            terminated=execution.terminated,
            truncated=execution.truncated,
        )
        builder.append(
            transition, audit, execution, replace(provenance, planning_cycle_index=index)
        )

    episode = builder.finish(tail, torch.tensor([bootstrap]))

    assert episode.training["next", "state_value"].flatten().tolist() == [
        *range(2, length + 1),
        bootstrap,
    ]
    assert episode.training["next", "done"].flatten().tolist() == [False] * (length - 1) + [True]
    assert episode.training["next", "terminated"][-1].item() == terminated
    assert episode.training["next", "truncated"][-1].item() == truncated
    for training_key, audit_key in (
        ("reward", "reward_total"),
        ("terminated", "terminated"),
        ("truncated", "truncated"),
    ):
        assert torch.equal(episode.training["next", training_key].cpu(), episode.audit[audit_key])
    assert all(
        value.device.type == device
        for value in episode.training.values(include_nested=True, leaves_only=True)
    )
    assert episode.tail_bootstrap_value.device.type == device
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
    builder.append(
        build_training_transition(
            training,
            execution.reward_result,
            terminated=execution.terminated,
            truncated=execution.truncated,
        ),
        audit,
        execution,
        provenance,
    )
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
    from tests.training.helpers import _policy_config

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


def _reward_result(total: float, *, safety_gate: float = 1.0) -> PlannerRFTRewardResult:
    return PlannerRFTRewardResult(
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
    def __init__(self, results: tuple[PlannerRFTRewardResult, ...]) -> None:
        self._results = list(results)

    def __call__(self, metrics: TransitionMetrics) -> PlannerRFTRewardResult:
        return self._results.pop(0)


@pytest.mark.parametrize("substep_count", [1, 5])
@pytest.mark.parametrize(
    "terminated,truncated,tail,bootstrap",
    [
        (True, False, "terminated", 0.0),
        (False, True, "truncated", 5.0),
        (False, False, "rollout_limit", 5.0),
        (True, True, "terminated", 0.0),
    ],
)
def test_collector_projects_one_reward_evaluation_to_training_and_audit(
    substep_count, terminated, truncated, tail, bootstrap
) -> None:
    training, audit, _, _ = _transition()
    decision = RolloutDecision(HostTrajectories(np.zeros((1, 80, 4))), lambda: audit, training)
    metrics = tuple(_substep_metrics(index) for index in range(substep_count))
    results = tuple(_reward_result(0.1 * (index + 1)) for index in range(substep_count))
    evaluator = _ScriptedEvaluator(results)
    lifecycle = _EpisodeLifecycle(previous_route_completion=0.0)

    kind = lifecycle.append(
        decision,
        _execution_result(metrics),
        map_seed=0,
        noise_seed=1,
        policy_action_seed=2,
        reward_evaluator=evaluator,
        terminated=terminated,
        truncated=truncated,
        collection_limit=True,
    )
    assert kind == tail
    episode = lifecycle.finish(kind, torch.tensor([bootstrap]))

    # The finite scripted sequence fails if the collector evaluates any substep twice.
    assert not evaluator._results
    expected_reward = torch.tensor([[sum(result.total for result in results)]])
    assert torch.equal(episode.training["next", "reward"], expected_reward)
    assert torch.equal(episode.audit["reward_total"], expected_reward)
    for key in ("terminated", "truncated"):
        assert torch.equal(episode.training["next", key], episode.audit[key])
    assert episode.audit["reward_substep_count"].item() == substep_count


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
