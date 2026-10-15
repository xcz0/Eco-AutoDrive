"""Named PPO aggregation preserves metric identity and reduction semantics."""

import pytest
import torch

from eco_planner.planning.policy import ExplorationPolicy
from eco_planner.rl.optimization import ppo
from eco_planner.rl.optimization.metrics import PPOMetrics

from .helpers import _policy_config, _ppo_config, _training_episode


def test_named_metrics_ignore_input_order_and_average_evaluated_minibatches():
    metrics = PPOMetrics(("loss", "kl"), torch.device("cpu"))
    metrics.update({"loss": torch.tensor(2.0, requires_grad=True), "kl": torch.tensor(10.0)})
    metrics.gradient(torch.tensor(3.0))
    # The triggering KL minibatch is evaluated but has no optimizer/gradient step.
    metrics.update({"kl": torch.tensor(30.0), "loss": torch.tensor(6.0)})
    assert metrics.compute() == {
        "loss": 4.0,
        "kl": 20.0,
        "maximum_pre_clip_gradient_norm": 3.0,
    }
    assert all(metric.mean_value.dtype == torch.float64 for metric in metrics.means.values())
    assert all(metric.weight.dtype == torch.float64 for metric in metrics.means.values())
    assert all(not metric.mean_value.requires_grad for metric in metrics.means.values())


@pytest.mark.parametrize("names", [("loss",), ("loss", "kl", "extra")])
def test_named_metrics_reject_mismatched_fields_before_updating(names):
    metrics = PPOMetrics(("loss", "kl"), torch.device("cpu"))
    with pytest.raises(ValueError, match="match the configured names"):
        metrics.update(dict.fromkeys(names, torch.tensor(100.0)))
    metrics.update({"loss": torch.tensor(2.0), "kl": torch.tensor(10.0)})
    assert metrics.compute() == {
        "loss": 2.0,
        "kl": 10.0,
        "maximum_pre_clip_gradient_norm": 0.0,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_named_metrics_reject_nonfinite_results(value):
    metrics = PPOMetrics(("loss",), torch.device("cpu"))
    with pytest.raises((RuntimeError, FloatingPointError)):
        metrics.update({"loss": torch.tensor(value)})
        metrics.compute()


def test_named_metrics_reject_nonfinite_gradient_maximum():
    metrics = PPOMetrics(("loss",), torch.device("cpu"))
    metrics.update({"loss": torch.tensor(1.0)})
    metrics.gradient(torch.tensor(float("inf")))
    with pytest.raises(FloatingPointError, match="must be finite"):
        metrics.compute()


@pytest.mark.parametrize("reverse_names", [False, True])
def test_ppo_report_preserves_each_metric_identity(monkeypatch, reverse_names):
    if reverse_names:
        monkeypatch.setattr(
            ppo, "_PPO_UPDATE_METRIC_NAMES", tuple(reversed(ppo._PPO_UPDATE_METRIC_NAMES))
        )
    values = {
        "loss_objective": 1.0,
        "loss_critic": 2.0,
        "loss_entropy": 4.0,
        "kl_approx": 8.0,
        "clip_fraction": 16.0,
        "entropy": 32.0,
        "explained_variance": 64.0,
    }
    updater = ppo.PPOUpdater(ExplorationPolicy(_policy_config()), _ppo_config())

    def distinct_metrics(module, args, losses):
        for name, value in values.items():
            losses[name] = losses[name] * 0.0 + value

    episodes = tuple(
        _training_episode(reward=reward, terminated=True, truncated=False, bootstrap=0.0)
        for reward in (1.0, 2.0)
    )
    with updater.loss_module.register_forward_hook(distinct_metrics):
        report = updater.update(episodes)
    assert report.mean_policy_loss == 1.0
    assert report.mean_value_loss == 2.0
    assert report.mean_entropy_loss == 4.0
    assert report.mean_total_loss == 7.0
    assert report.mean_approximate_kl == 8.0
    assert report.mean_clip_fraction == 16.0
    assert report.mean_entropy == 32.0
    assert report.mean_explained_variance == 64.0
    assert report.maximum_pre_clip_gradient_norm == 0.0
