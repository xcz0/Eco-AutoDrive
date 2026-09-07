"""Matched scalar-reward protocol manifest, jobs, and runner guards."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import DictConfig, MissingMandatoryValue

from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.evaluation import parse_evaluation_config
from eco_planner.experiments.scalar_reward.config import (
    ScalarRewardProtocolConfig,
    load_scalar_reward_protocol,
)
from eco_planner.experiments.scalar_reward.runner import (
    compose_a0_evaluation_config,
    compose_arm_training_config,
    compose_policy_evaluation_config,
)
from eco_planner.jobs import compose_job_config
from eco_planner.models import Ddim5SamplerConfig
from eco_planner.rl.config import TrainingJobConfig

ComposeConfig = Callable[[str, list[str] | None], DictConfig]
PROTOCOL_PATH = Path(__file__).resolve().parents[2] / (
    "configs/experiments/scalar_reward/protocol.yaml"
)
CHECKPOINT_OVERRIDES = [
    "evaluation.policy_checkpoint.label=final",
    "evaluation.policy_checkpoint.path=checkpoints/run/policy-final.pt",
]


@pytest.fixture(autouse=True)
def without_machine_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)


def test_protocol_manifest_defines_disjoint_matched_pools() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    assert protocol.arms.a0.label == "frozen_diffusion_planner"
    assert protocol.arms.a1.reward_profile == "plannerrft_no_energy_v1"
    assert protocol.arms.a2.reward_profile == "plannerrft_energy_v1"
    assert protocol.update0_evaluation == "diagnostic"
    assert {item.name for item in protocol.training_scenarios()} == {
        f"{prefix}_s{seed}" for prefix in ("s", "sc") for seed in range(8)
    }
    assert {item.name for item in protocol.held_out_scenarios()} == {
        f"{prefix}_s{seed}" for prefix in ("s", "sc") for seed in range(16, 24)
    }
    assert not (protocol.training_pairs() & protocol.held_out_pairs())


def test_protocol_manifest_rejects_overlapping_train_and_eval_pools() -> None:
    raw = load_resolved_yaml_mapping(PROTOCOL_PATH)
    raw["evaluation"]["map_seeds"] = [0, 16, 17]

    with pytest.raises(ValueError, match="overlap"):
        ScalarRewardProtocolConfig.model_validate(raw)


def test_a0_heldout_job_matches_the_protocol() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    _, parsed = compose_a0_evaluation_config(protocol)

    assert parsed.runtime.seed == 760_025
    assert parsed.evaluation.mode == "no_traffic"
    assert parsed.evaluation.history_warmup_steps == 0
    assert parsed.evaluation.evaluated_horizon_steps == 300
    assert isinstance(parsed.sampler, Ddim5SamplerConfig)
    assert parsed.policy_checkpoint is None
    assert parsed.env["num_scenarios"] == 24
    assert {(item.map, item.seed) for item in parsed.scenarios} == protocol.held_out_pairs()


def test_a0_runner_rejects_seed_mismatch_against_the_composed_job() -> None:
    raw = load_resolved_yaml_mapping(PROTOCOL_PATH)
    raw["evaluation"]["seed"] = 760_026
    protocol = ScalarRewardProtocolConfig.model_validate(raw)

    with pytest.raises(ValueError, match="runtime.seed"):
        compose_a0_evaluation_config(protocol)


def test_arm_training_composition_pins_the_matched_protocol() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    assert protocol.training.seeds == [0, 1, 2]
    for seed in protocol.training.seeds:
        _, a1 = compose_arm_training_config(protocol, "a1", seed)
        _, a2 = compose_arm_training_config(protocol, "a2", seed)

        assert a1.reward.name == "plannerrft_no_energy_v1"
        assert a2.reward.name == "plannerrft_energy_v1"
        for training in (a1, a2):
            assert isinstance(training, TrainingJobConfig)
            assert training.runtime.seed == seed
            assert training.training.replay_id == 0
            assert isinstance(training.sampler, Ddim5SamplerConfig)
            pairs = {(item.map, item.seed) for item in training.scenarios}
            assert pairs == protocol.training_pairs()


def test_arm_training_composition_passes_through_ppo_overrides() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    _, training = compose_arm_training_config(
        protocol, "a1", 0, ["ppo.learning_rate=1.6301e-5"]
    )

    assert training.ppo.learning_rate == pytest.approx(1.6301e-5)


def test_arm_training_composition_rejects_protocol_violations() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    with pytest.raises(ValueError, match="reward profile"):
        compose_arm_training_config(protocol, "a1", 0, ["components/reward=plannerrft_energy_v1"])
    with pytest.raises(ValueError, match="runtime.seed"):
        compose_arm_training_config(protocol, "a2", 0, ["runtime.seed=1"])
    with pytest.raises(ValueError, match="num_scenarios"):
        compose_arm_training_config(protocol, "a1", 0, ["env.num_scenarios=4"])
    with pytest.raises(ValueError, match="training seed"):
        compose_arm_training_config(protocol, "a1", 7)


def test_policy_evaluation_composition_matches_the_training_arm() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)
    _, training = compose_arm_training_config(protocol, "a1", 0)

    _, policy_job = compose_policy_evaluation_config(
        protocol, "a1", "initial", Path("runs/a1/policy-initial.pt")
    )

    assert policy_job.policy_checkpoint is not None
    assert policy_job.policy_checkpoint.label == "initial"
    assert policy_job.policy_checkpoint.path == "runs/a1/policy-initial.pt"
    assert policy_job.policy == training.policy
    assert policy_job.model == training.model
    assert policy_job.map_query_radius_m == training.map_query_radius_m


def test_policy_heldout_job_requires_checkpoint_overrides() -> None:
    protocol = load_scalar_reward_protocol(PROTOCOL_PATH)

    config = compose_job_config(protocol.evaluation.policy_job)

    with pytest.raises(MissingMandatoryValue, match="policy_checkpoint.label"):
        parse_evaluation_config(config)


def test_policy_checkpoint_constraints_reject_unmatched_components(
    compose_config: ComposeConfig,
) -> None:
    guidance = compose_config(
        "jobs/evaluation/no_traffic_heldout_policy",
        ["components/guidance=none", *CHECKPOINT_OVERRIDES],
    )
    with pytest.raises(ValueError, match="orthogonal_policy"):
        parse_evaluation_config(guidance)

    sampler = compose_config(
        "jobs/evaluation/no_traffic_heldout_policy",
        ["components/sampler=dpm10", *CHECKPOINT_OVERRIDES],
    )
    with pytest.raises(ValueError, match="ddim5"):
        parse_evaluation_config(sampler)

    stochastic = compose_config(
        "jobs/evaluation/no_traffic_heldout_policy",
        ["sampler.ddim_stochasticity=0.5", *CHECKPOINT_OVERRIDES],
    )
    with pytest.raises(ValueError, match="ddim_stochasticity"):
        parse_evaluation_config(stochastic)

    dangling_policy = compose_config(
        "jobs/evaluation/no_traffic_heldout_policy",
        ["~evaluation.policy_checkpoint"],
    )
    with pytest.raises(ValueError, match="policy component requires"):
        parse_evaluation_config(dangling_policy)

    missing_policy = compose_config(
        "jobs/evaluation/no_traffic_heldout_policy",
        ["~policy", *CHECKPOINT_OVERRIDES],
    )
    with pytest.raises(ValueError, match="requires the policy component"):
        parse_evaluation_config(missing_policy)
