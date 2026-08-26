from __future__ import annotations

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from eco_planner.rl import parse_training_config
from eco_planner.rl.policy import ExplorationPolicy
from eco_planner.rl.trainer import train


def _training_config():
    config_dir = Path(__file__).parents[3] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        config = compose(
            config_name="jobs/training/ppo_smoke",
            overrides=["runtime.seed=0", "training.replay_id=0"],
        )
    return parse_training_config(config)


def test_training_closes_vector_workers_when_collection_fails(tmp_path, monkeypatch) -> None:
    config = _training_config()

    class FakeRuntime:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self.policy = ExplorationPolicy(config.policy)

        @staticmethod
        def new_noise_generator(seed: int) -> torch.Generator:
            return torch.Generator().manual_seed(seed)

        @staticmethod
        def new_policy_generator(seed: int) -> torch.Generator:
            return torch.Generator().manual_seed(seed)

        @staticmethod
        def frozen_planner_hash() -> str:
            return "f" * 64

    class FailingCollector:
        instance: FailingCollector | None = None

        def __init__(self, *_: object, **__: object) -> None:
            self.closed = False
            FailingCollector.instance = self

        def __enter__(self) -> FailingCollector:
            return self

        def __exit__(self, *_: object) -> None:
            self.closed = True

        def collect(self, **_: object):
            raise RuntimeError("collection failed")

    monkeypatch.setattr(
        "eco_planner.rl.trainer.create_fabric_rollout_runtime",
        lambda *_args, **_kwargs: FakeRuntime(),
    )
    monkeypatch.setattr("eco_planner.rl.trainer.VectorRolloutCollector", FailingCollector)
    deterministic = torch.are_deterministic_algorithms_enabled()
    matmul_precision = torch.get_float32_matmul_precision()
    try:
        with pytest.raises(RuntimeError, match="collection failed"):
            train(config, tmp_path / "run")
    finally:
        torch.use_deterministic_algorithms(deterministic)
        torch.set_float32_matmul_precision(matmul_precision)

    assert FailingCollector.instance is not None
    assert FailingCollector.instance.closed
