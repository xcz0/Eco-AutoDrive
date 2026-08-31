"""Internal composition and execution boundary for configured jobs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from eco_planner._repository import CONFIG_ROOT
from eco_planner.evaluation.artifacts import JobSummary
from eco_planner.evaluation.config import parse_evaluation_config
from eco_planner.evaluation.runner import run_evaluation
from eco_planner.rl.artifacts import TrainingRunSummary
from eco_planner.rl.config import parse_training_config
from eco_planner.rl.trainer import TrainingUpdateObserver, train


def compose_job_config(config_name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose one job through the shared Hydra lifecycle boundary."""

    if GlobalHydra.instance().is_initialized():
        return compose(config_name=config_name, overrides=list(overrides))
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT.resolve())):
        return compose(config_name=config_name, overrides=list(overrides))


def run_evaluation_job(config: DictConfig, output_dir: Path) -> JobSummary:
    """Parse and execute one composed evaluation job."""

    return run_evaluation(parse_evaluation_config(config), output_dir)


def run_training_job(
    config: DictConfig,
    output_dir: Path,
    *,
    update_observer: TrainingUpdateObserver | None = None,
) -> TrainingRunSummary:
    """Persist, parse, and execute one composed PPO training job."""

    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output_dir / "resolved_config.yaml", resolve=True)
    return train(parse_training_config(config), output_dir, update_observer=update_observer)
