from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import torch
from pydantic import TypeAdapter

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.envs.domain import (
    TransitionMetrics,
)
from eco_planner.reward import (
    PlannerRFTRewardResult,
    RewardProfileConfig,
    RewardProfileName,
    RewardResult,
    create_reward_evaluator,
)
from eco_planner.rl.artifacts import (
    TrainingUpdateSummary,
    build_update_summary,
    read_rollout_episode,
    write_rollout_episode,
)
from eco_planner.rl.rollout import (
    RolloutEpisodeBuilder,
    build_training_transition,
)
from eco_planner.rl.rollout.collector import _EpisodeLifecycle, _execution_transition_audit
from eco_planner.rl.rollout.decision import RolloutDecision
from eco_planner.runtime.contracts import HostTrajectories
from tests.training.rollout_helpers import (
    _execution_result,
    _reward_result,
    _substep_metrics,
    _transition,
)
from tests.training.summary_helpers import build_ppo_update_report


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float16),
        ("cpu", torch.bfloat16),
        pytest.param("cuda", torch.bfloat16, marks=pytest.mark.gpu),
    ],
)
def test_update_storage_preserves_next_value_dtype_promotion(device, dtype):
    compact = RolloutEpisodeBuilder()
    transitions = []
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
        transitions.append(transition)
        compact.append_audit_batch((compact,), audit, (execution,), (provenance,))
        separate.append(transition, audit, execution, provenance)

    from tensordict import cat

    collected = cat(transitions)
    collected["next", "state_value"] = torch.empty_like(
        collected["state_value"], dtype=torch.promote_types(dtype, torch.float32)
    )
    compact_episode = compact.finish("rollout_limit", bootstrap, training=collected)
    separate_episode = separate.finish("rollout_limit", bootstrap)

    expected = torch.cat((values[1:], bootstrap.reshape(1, 1)))
    actual = compact_episode.training["next", "state_value"]
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert (compact_episode.training == separate_episode.training).all()


@pytest.mark.parametrize(
    "device,key,value,message",
    [
        ("cpu", "scene_tokens", float("nan"), "finite"),
        ("cpu", ("next", "reward"), float("inf"), "finite"),
        ("cpu", "guidance_action", -1.0, "strictly inside"),
        ("cpu", "guidance_action", 1.0, "strictly inside"),
        ("cpu", ("next", "done"), False, "final GAE boundary"),
        ("cpu", ("next", "terminated"), False, "zero bootstrap"),
        pytest.param("cuda", "state_value", float("nan"), "finite", marks=pytest.mark.gpu),
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
    from eco_planner.rl.rollout import assembly

    decision, decision_audit, execution, provenance = _transition()
    audit = assembly.build_rollout_audit(decision_audit, execution, provenance)
    monkeypatch.setattr(assembly, "build_rollout_audit", lambda *args: audit)
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


@pytest.mark.parametrize(
    "device,length,terminated,truncated,tail,bootstrap",
    [
        ("cpu", 3, True, False, "terminated", 0.0),
        ("cpu", 3, False, True, "truncated", 5.0),
        ("cpu", 3, False, False, "rollout_limit", 5.0),
        ("cpu", 3, True, True, "terminated", 0.0),
        ("cpu", 1, False, False, "rollout_limit", 5.0),
        pytest.param("cuda", 3, False, True, "truncated", 5.0, marks=pytest.mark.gpu),
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


@pytest.mark.parametrize("profile_name", get_args(RewardProfileName))
def test_reward_profile_name_reaches_rollout_artifact_and_summary(
    tmp_path: Path, profile_name: RewardProfileName
) -> None:
    config = TypeAdapter(RewardProfileConfig).validate_python(
        load_resolved_yaml_mapping(CONFIG_ROOT / f"components/reward/{profile_name}.yaml")
    )
    evaluator = create_reward_evaluator(config)
    result = evaluator(_substep_metrics(0))
    assert result.profile_name == profile_name

    training, audit, _, _ = _transition()
    lifecycle = _EpisodeLifecycle(previous_route_completion=0.0)
    tail = lifecycle.append(
        RolloutDecision(HostTrajectories(np.zeros((1, 80, 4))), lambda: audit, training),
        _execution_result((_substep_metrics(0),)),
        map_seed=0,
        noise_seed=1,
        policy_action_seed=2,
        reward_evaluator=evaluator,
        terminated=False,
        truncated=False,
        collection_limit=True,
    )
    episode = lifecycle.finish(tail, torch.tensor([5.0]))
    assert episode.reward_profile == profile_name

    artifact = tmp_path / "episode.npz"
    write_rollout_episode(artifact, episode)
    with np.load(artifact, allow_pickle=False) as arrays:
        assert str(arrays["reward_profile"]) == profile_name
    restored = read_rollout_episode(artifact)
    assert restored.reward_profile == profile_name

    report = build_ppo_update_report(sample_count=restored.transition_count)
    update = build_update_summary(0, (restored,), report)
    assert update.reward_profile == profile_name
    assert (
        TrainingUpdateSummary.model_validate_json(update.model_dump_json()).reward_profile
        == profile_name
    )


def test_rollout_artifact_preserves_audit_and_nonterminal_bootstrap(tmp_path: Path) -> None:
    builder = RolloutEpisodeBuilder()
    training, audit, execution, provenance = _transition()
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
        for key, value in episode.audit.items():
            expected = value.numpy()
            assert arrays[key].dtype == expected.dtype
            np.testing.assert_array_equal(arrays[key], expected)
        np.testing.assert_array_equal(arrays["reward_total"], episode.training["next", "reward"])

    restored = read_rollout_episode(artifact)
    assert restored.tail_kind == episode.tail_kind
    torch.testing.assert_close(restored.tail_bootstrap_value, episode.tail_bootstrap_value)
    for key in episode.training.keys(include_nested=True, leaves_only=True):
        assert restored.training[key].dtype == episode.training[key].dtype
    assert (restored.training == episode.training).all()
    assert (restored.audit == episode.audit).all()


def test_slot_audit_matches_batch_fields_and_rng_states() -> None:
    from types import SimpleNamespace

    from tensordict import cat

    from eco_planner.rl.rollout.decision import BatchRolloutDecision
    from eco_planner.runtime.contracts import HostTrajectories
    from tests.training.helpers import _policy_config

    training, audit, _, _ = _transition()
    host = cat([audit, audit]).to_dict()
    diffusion_states = (torch.ones(5, dtype=torch.uint8), torch.full((5,), 2, dtype=torch.uint8))
    policy_states = (torch.full((5,), 3, dtype=torch.uint8), torch.full((5,), 4, dtype=torch.uint8))
    decision = BatchRolloutDecision(
        HostTrajectories(np.zeros((2, 80, 4), dtype=np.float32)),
        SimpleNamespace(resolve=lambda: host, timing=None),
        diffusion_states,
        policy_states,
        _policy_config().model_copy(update={"hidden_dim": 4, "cross_attention_heads": 1}),
        cat([training, training]),
    )
    slot = decision.slot(1)
    assert slot.training_decision.batch_size == torch.Size([1])
    assert slot.ego_trajectory.shape == (80, 4)
    slot_audit = slot.audit_result()
    batch_audit = decision.audit_result()
    assert (slot_audit == batch_audit[1:2]).all()
    assert "prediction" in slot_audit
    assert slot_audit["state_value"].shape == (1, 1)
    assert slot_audit["old_joint_guidance_log_prob"].shape == (1, 1)
    torch.testing.assert_close(slot_audit["diffusion_rng_state"][0], diffusion_states[1])
    torch.testing.assert_close(slot_audit["policy_rng_state"][0], policy_states[1])


class _ScriptedEvaluator:
    def __init__(self, results: tuple[PlannerRFTRewardResult, ...]) -> None:
        self._results = list(results)

    def __call__(self, metrics: TransitionMetrics) -> PlannerRFTRewardResult:
        return self._results.pop(0)


# Reward aggregation precedes tail selection, so substep counts need not cross every tail.
@pytest.mark.parametrize(
    "substep_count,terminated,truncated,tail,bootstrap",
    [
        pytest.param(1, False, False, "rollout_limit", 5.0, id="single-substep"),
        pytest.param(5, True, False, "terminated", 0.0, id="multi-substep-terminated"),
        pytest.param(5, False, True, "truncated", 5.0, id="multi-substep-truncated"),
        pytest.param(5, False, False, "rollout_limit", 5.0, id="multi-substep-rollout-limit"),
        pytest.param(5, True, True, "terminated", 0.0, id="multi-substep-terminal-precedence"),
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
