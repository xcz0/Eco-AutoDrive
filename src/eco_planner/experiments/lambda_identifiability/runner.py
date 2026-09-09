"""Collect one update-0 batch, then diagnose all lambda arms offline."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

from eco_planner._repository import REPOSITORY_ROOT
from eco_planner.artifacts import write_json, write_npz
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.experiments.lambda_identifiability.diagnostics import analyze
from eco_planner.experiments.scalar_reward.config import load_scalar_reward_protocol
from eco_planner.experiments.scalar_reward.runner import compose_arm_training_config
from eco_planner.rl.artifacts import (
    policy_state_hash,
    write_rollout_episode,
    write_training_runtime_metadata,
)
from eco_planner.rl.optimization import PPOUpdater, save_exploration_policy_checkpoint
from eco_planner.rl.reward.config import PlannerRFTNoEnergyRewardConfig
from eco_planner.rl.rollout import VectorRolloutCollector, create_fabric_rollout_runtime
from eco_planner.rl.trainer import _derive_rollout_seeds


class IdentifiabilityConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    protocol: str
    training_seed: StrictInt = Field(ge=0)
    lambdas: list[StrictFloat] = Field(min_length=2)
    quantiles: list[StrictFloat] = Field(min_length=2)
    overrides: list[str]

    @model_validator(mode="after")
    def validate_axes(self) -> IdentifiabilityConfig:
        if self.lambdas[0] != 0 or sorted(set(self.lambdas)) != self.lambdas:
            raise ValueError("lambdas must start at zero and be strictly increasing")
        if (
            self.quantiles[0] != 0
            or self.quantiles[-1] != 1
            or sorted(set(self.quantiles)) != self.quantiles
        ):
            raise ValueError("quantiles must increase from zero to one")
        return self


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    study = IdentifiabilityConfig.model_validate(load_resolved_yaml_mapping(config_path))
    protocol = load_scalar_reward_protocol(Path(to_absolute_path(study.protocol)))
    resolved, config = compose_arm_training_config(
        protocol,
        "a1",
        study.training_seed,
        study.overrides,
    )
    if not isinstance(config.reward, PlannerRFTNoEnergyRewardConfig):
        raise ValueError("fixed batch collection requires the R0 profile")
    if config.resources is None:
        raise ValueError("diagnostic requires explicit resource profile")
    if config.training.resume_checkpoint_path is not None:
        raise ValueError("update-0 diagnostic cannot resume a trained policy")
    output_dir.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(resolved, output_dir / "resolved_config.yaml")
    OmegaConf.save(OmegaConf.create(study.model_dump()), output_dir / "diagnostic_config.yaml")
    # New experiment files can be untracked at run time; preserve their actual source too.
    sources = [
        *Path(__file__).parent.glob("*.py"),
        REPOSITORY_ROOT / "scripts/experiments/lambda_identifiability.py",
    ]
    for source in sources:
        target = output_dir / "source" / source.relative_to(REPOSITORY_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if config.training.deterministic:
        torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("high")
    start = time.perf_counter()
    noise_seeds, policy_seeds = _derive_rollout_seeds(config.runtime.seed, len(config.scenarios))
    runtime = create_fabric_rollout_runtime(
        config.runtime,
        config.sampler,
        config.guidance,
        config.policy,
        Path(to_absolute_path(config.model.args_path)),
        Path(to_absolute_path(config.model.checkpoint_path)),
        policy_seeds[0],
        planner_compile_mode=config.training.planner_compile_mode,
    )
    updater = PPOUpdater(runtime.policy, config.ppo)
    write_training_runtime_metadata(output_dir / "runtime_metadata.json", runtime, config.resources)
    initial_hash = policy_state_hash(runtime.policy)
    planner_hash = runtime.frozen_planner_hash()
    save_exploration_policy_checkpoint(output_dir / "policy-initial.pt", runtime.policy)
    print("Collecting one update-0 batch (no optimizer steps).", flush=True)
    with VectorRolloutCollector(
        config.scenarios,
        runtime,
        config.env,
        mode=config.training.mode,
        map_query_radius_m=config.map_query_radius_m,
        history_warmup_steps=config.training.history_warmup_steps,
        physical_slot_count=config.resources.rollout_worker_count,
        torch_threads_per_worker=config.resources.torch_threads_per_worker,
        reward_profile=config.reward,
    ) as collector:
        slots = collector.collect(
            transitions_per_slot=config.training.transitions_per_environment,
            stopped_speed_threshold_mps=config.training.stopped_speed_threshold_mps,
            diffusion_generators=tuple(runtime.new_noise_generator(s) for s in noise_seeds),
            policy_generators=tuple(runtime.new_policy_generator(s) for s in policy_seeds),
            noise_seeds=noise_seeds,
            policy_action_seeds=policy_seeds,
        )
    episodes, training_payload, sample_index = [], [], []
    for slot, slot_episodes in enumerate(slots):
        for number, episode in enumerate(slot_episodes):
            write_rollout_episode(
                output_dir / "updates/update-000" / f"slot-{slot}-episode-{number}.npz",
                episode,
            )
            episodes.append(episode)
            training_payload.append(episode.training.cpu().to_dict())
            sample_index.extend(
                {
                    "scenario_index": slot,
                    "scenario": config.scenarios[slot].name,
                    "map": config.scenarios[slot].map,
                    "map_seed": config.scenarios[slot].seed,
                    "episode_index": number,
                    "planning_cycle_index": int(cycle),
                }
                for cycle in episode.audit["planning_cycle_index"].reshape(-1).tolist()
            )
    torch.save(training_payload, output_dir / "training-batch.pt")
    write_json(output_dir / "sample_index.json", {"samples": sample_index})
    print(
        f"Collected {len(sample_index)} transitions; computing six matched actor backwards.",
        flush=True,
    )
    summary, arrays = analyze(
        updater,
        episodes,
        config.reward,
        study.lambdas,
        study.quantiles,
        np.asarray([sample["scenario_index"] for sample in sample_index], dtype=np.int64),
    )
    if policy_state_hash(runtime.policy) != initial_hash:
        raise RuntimeError("initial policy changed")
    if runtime.frozen_planner_hash() != planner_hash:
        raise RuntimeError("frozen planner changed")
    if any(p.grad is not None for p in runtime._planner.parameters()):
        raise RuntimeError("diagnostic created planner gradients")
    summary.update(
        {
            "status": "completed",
            "training_seed": study.training_seed,
            "noise_seeds": noise_seeds,
            "policy_action_seeds": policy_seeds,
            "initial_policy_hash": initial_hash,
            "final_policy_hash": initial_hash,
            "frozen_planner_hash": planner_hash,
            "policy_unchanged": True,
            "planner_unchanged": True,
            "planner_gradients_absent": True,
            "episode_count": len(episodes),
            "wall_time_s": time.perf_counter() - start,
            "batch_source": "new single update-0 collection; not historical E-032 replay",
        }
    )
    write_npz(output_dir / "diagnostics.npz", arrays)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    return {
        "status": "completed",
        "output_dir": str(output_dir),
        "sample_count": len(sample_index),
        "optimizer_steps": 0,
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Lambda identifiability: fixed update-0 batch",
        "",
        "New batch; actor objective only; no optimizer steps. No automatic gate threshold.",
        "",
        summary["undefined_reason"],
        "",
        "Component std uses population variance; advantage std uses sample variance.",
        "",
        "| Component | Mean | Std | Quantiles |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, stats in summary["components"].items():
        lines.append(
            f"| {name} | {stats['mean']:.9g} | {stats['std']:.9g} | {stats['quantiles']} |"
        )
    lines += [
        "",
        "| Lambda | Raw A mean | Raw A std | Norm A std | Head norm | Trunk norm |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in summary["arms"]:
        lines.append(
            f"| {arm['lambda']:g} | {arm['raw_advantage']['mean']:.9g} | "
            f"{arm['raw_advantage']['std']:.9g} | "
            f"{arm['normalized_advantage']['std']:.9g} | "
            f"{arm['gradient_norms']['actor_head']:.9g} | "
            f"{arm['gradient_norms']['shared_trunk']:.9g} |"
        )
    lines += [
        "",
        "| Lambda i → j | Pearson | Spearman | Sign flip | Head cosine | Norm j/i |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in summary["pairs"]:
        grad = pair["gradients"]["actor_head"]
        lines.append(
            f"| {pair['lambda_i']:g} → {pair['lambda_j']:g} | {pair['pearson']} | "
            f"{pair['spearman']} | {pair['sign_flip_fraction']} | "
            f"{grad['cosine']} | {grad['norm_ratio_j_over_i']} |"
        )
    lines += [
        "",
        "Full arm statistics, dimension-specific gradients and per-scenario matched "
        "differences: [summary.json](summary.json). Per-transition values and all gradient "
        "vectors: [diagnostics.npz](diagnostics.npz), indexed by "
        "[sample_index.json](sample_index.json).",
        "",
        "These measurements concern this batch and initial policy only. They do not "
        "establish learned behavioral separation or select Task B/C.",
        "",
    ]
    return "\n".join(lines)
