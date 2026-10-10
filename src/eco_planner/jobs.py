"""Internal Hydra composition and execution boundary for configured jobs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf, open_dict

from eco_planner._repository import CONFIG_ROOT
from eco_planner.configuration import resolve_config_mapping, with_machine_resource_override
from eco_planner.evaluation import JobSummary, parse_evaluation_config
from eco_planner.rl.artifacts import TrainingRunSummary
from eco_planner.rl.config import parse_training_config
from eco_planner.runtime.resources import require_resource_profile

if TYPE_CHECKING:
    from eco_planner.rl.trainer import TrainingUpdateObserver


def compose_job_config(config_name: str, overrides: Sequence[str] = ()) -> DictConfig:
    """Compose one job through the shared Hydra lifecycle boundary."""

    resolved_overrides = with_machine_resource_override(overrides)
    if GlobalHydra.instance().is_initialized():
        config = compose(config_name=config_name, overrides=resolved_overrides)
    else:
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_ROOT.resolve())):
            config = compose(config_name=config_name, overrides=resolved_overrides)
    with open_dict(config):
        config.hydra = {"overrides": {"task": list(resolved_overrides)}}
    return config


def run_evaluation_job(
    config: DictConfig, output_dir: Path, *, overrides: Sequence[str] | None = None
) -> JobSummary:
    """Parse and execute one composed evaluation job."""

    parsed = parse_evaluation_config(config)
    require_resource_profile(parsed.resources)
    from eco_planner.evaluation import run_evaluation

    write_job_configuration(config, output_dir, overrides=overrides)
    return run_evaluation(parsed, output_dir)


def run_training_job(
    config: DictConfig,
    output_dir: Path,
    *,
    update_observer: TrainingUpdateObserver | None = None,
    overrides: Sequence[str] | None = None,
) -> TrainingRunSummary:
    """Persist, parse, and execute one composed PPO training job."""

    parsed = parse_training_config(config)
    require_resource_profile(parsed.resources)
    from eco_planner.rl.trainer import train

    write_job_configuration(config, output_dir, overrides=overrides)
    return train(parsed, output_dir, update_observer=update_observer)


def write_job_configuration(
    config: DictConfig, output_dir: Path, *, overrides: Sequence[str] | None = None
) -> None:
    """Persist resolved job values and the actual composition overrides separately."""

    declared = (
        overrides if overrides is not None else OmegaConf.select(config, "hydra.overrides.task")
    )
    if declared is None:
        raise ValueError("job persistence requires explicit invocation overrides")
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(resolve_config_mapping(config)), output_dir / "resolved_config.yaml"
    )
    OmegaConf.save(OmegaConf.create(list(declared)), output_dir / "overrides.yaml")
