"""Training summary aggregation and persistence, independent of PPO updates."""

import math
from dataclasses import replace

import pytest
import torch

from eco_planner.rl.artifacts.metrics import build_update_summary
from eco_planner.rl.artifacts.summaries import TrainingUpdateSummary

from .diagnostic_helpers import build_ppo_update_report
from .helpers import _episode
from .reward_helpers import _five_substep_parity_case, _no_energy_config


def test_unequal_episode_summary_uses_transition_weights_and_ratio_of_totals():
    short = _episode(reward=0.25, terminated=True, truncated=False, bootstrap=0.0)
    training = torch.cat([short.training, short.training], dim=0)
    training["next", "terminated"][0] = False
    training["next", "done"][0] = False
    audit = torch.cat([short.audit, short.audit], dim=0)
    audit["speed_mps"][:] = 10.0
    audit["step_distance_m"][:] = 1.0
    audit["executed_fuel_proxy_step_energy_ml"][:] = 1.0
    long = replace(short, training=training, audit=audit)
    short.audit["speed_mps"][:] = 1.0
    short.audit["step_distance_m"][:] = 10.0
    short.audit["executed_fuel_proxy_step_energy_ml"][:] = 2.0
    short.audit["state_value"][:] = 1.0
    long.audit["state_value"][:] = 4.0
    short.audit["guidance_action"][:] = torch.tensor([-0.5, 0.5])
    long.audit["guidance_action"][:] = torch.tensor([0.5, -0.5])
    short.audit["beta_alpha"][:] = torch.tensor([2.0, 4.0])
    long.audit["beta_alpha"][:] = torch.tensor([5.0, 1.0])
    short.audit["beta_beta"][:] = torch.tensor([3.0, 6.0])
    long.audit["beta_beta"][:] = torch.tensor([6.0, 3.0])
    report = build_ppo_update_report(sample_count=3)
    result = build_update_summary(0, (short, long), report)
    assert result.episode_count == 2
    assert result.sample_count == 3
    assert result.mean_episode_length == 1.5
    assert result.mean_state_value == 3.0
    assert result.std_state_value == pytest.approx(math.sqrt(2.0))
    expected = {
        "beta_alpha_mean": (4.0, 2.0),
        "beta_alpha_min": (2.0, 1.0),
        "beta_alpha_max": (5.0, 4.0),
        "beta_beta_mean": (5.0, 4.0),
        "beta_beta_min": (3.0, 3.0),
        "beta_beta_max": (6.0, 6.0),
        "action_mean": (1.0 / 6.0, -1.0 / 6.0),
        "action_std": (math.sqrt(2.0) / 3.0,) * 2,
        "action_min": (-0.5, -0.5),
        "action_max": (0.5, 0.5),
    }
    for name, value in expected.items():
        assert getattr(result, name) == pytest.approx(value)
    assert result.mean_speed_mps == 7.0
    assert result.executed_fuel_proxy_ml_per_km == pytest.approx(4000.0 / 12.0)
    short.audit["step_distance_m"][:] = 0.0
    long.audit["step_distance_m"][:] = 0.0
    assert build_update_summary(0, (short, long), report).executed_fuel_proxy_ml_per_km is None


def test_summary_and_reader_preserve_transition_component_sums_above_one():
    episodes = tuple(
        _five_substep_parity_case(_no_energy_config(), count=count)[0] for count in (1, 5)
    )
    summary = build_update_summary(0, episodes, build_ppo_update_report(sample_count=2))
    restored = TrainingUpdateSummary.model_validate_json(summary.model_dump_json())
    assert restored.reward_component_means.ttc > 1.0
    assert restored == summary
    for component in ("ttc", "progress", "comfort", "speed", "energy"):
        expected = (
            sum(episode.audit[f"reward_component_{component}"].item() for episode in episodes) / 2
        )
        assert getattr(restored.reward_component_means, component) == pytest.approx(expected)
