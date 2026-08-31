"""Run the fixed-seed energy benchmark study without retrying failed episodes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, field_validator

from eco_planner._repository import CONFIG_ROOT, REPOSITORY_ROOT
from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping

DEFAULT_STUDY = CONFIG_ROOT / "studies" / "energy" / "matrix.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class ExecutionSpec(_StrictModel):
    simulator_hz: Literal[10]  # pyright: ignore[reportInvalidTypeForm]
    planner_hz: Literal[2]  # pyright: ignore[reportInvalidTypeForm]
    physics_world_step_size_s: Literal[0.02]  # pyright: ignore[reportInvalidTypeForm]
    decision_repeat: Literal[5]
    trajectory_execution_steps: Literal[5]


class SamplerSpec(_StrictModel):
    name: Literal["ddim5"]
    ddim_stochasticity: Literal[0.0]  # pyright: ignore[reportInvalidTypeForm]


class EnergyMetricSpec(_StrictModel):
    name: Literal["metadrive_fuel_proxy"]
    implementation: Literal["recompute_metadrive_base_vehicle_formula_on_executed_trace"]
    formula: str = Field(min_length=1)
    unit: Literal["mL"]
    sampling_interval_s: Literal[0.1]  # pyright: ignore[reportInvalidTypeForm]
    interpretation: str = Field(min_length=1)


class VehicleSpec(_StrictModel):
    random_agent_model: Literal[False]


class GuidanceProfileSpec(_StrictModel):
    id: Literal[
        "baseline",
        "longitudinal_negative",
        "longitudinal_zero",
        "longitudinal_positive",
    ]
    config: str = Field(min_length=1)
    longitudinal_scale: StrictFloat | None


class ScenarioSpec(_StrictModel):
    name: str = Field(min_length=1)
    map: str = Field(min_length=1)
    seed: StrictInt = Field(ge=0)
    traffic_condition: str = Field(min_length=1)


class EvaluationJobSpec(_StrictModel):
    id: str = Field(min_length=1)
    config_name: str = Field(min_length=1)
    scenarios: tuple[ScenarioSpec, ...]

    @field_validator("scenarios", mode="before")
    @classmethod
    def tuple_scenarios(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class EnergyStudyConfig(_StrictModel):
    version: Literal[1]
    planner_noise_seed: StrictInt = Field(ge=0)
    execution: ExecutionSpec
    sampler: SamplerSpec
    energy_metric: EnergyMetricSpec
    vehicle_config: VehicleSpec
    guidance_profiles: tuple[GuidanceProfileSpec, ...]
    jobs: tuple[EvaluationJobSpec, ...]

    @field_validator("guidance_profiles", "jobs", mode="before")
    @classmethod
    def tuple_items(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    def validate_study_contract(self) -> None:
        profile_ids = tuple(profile.id for profile in self.guidance_profiles)
        if profile_ids != (
            "baseline",
            "longitudinal_negative",
            "longitudinal_zero",
            "longitudinal_positive",
        ):
            raise ValueError("energy study guidance profiles are incomplete or out of order")
        names = [scenario.name for job in self.jobs for scenario in job.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("energy study scenario names must be unique")
        required = {
            "cruise",
            "curve",
            "intersection",
            "merge_lane_change",
            "speed_limit_50_30_50",
            "traffic_follow",
        }
        if set(names) != required:
            raise ValueError("energy study does not contain its required fixed scenarios")


def load_energy_study(path: Path) -> EnergyStudyConfig:
    config = EnergyStudyConfig.model_validate(load_resolved_yaml_mapping(path))
    config.validate_study_contract()
    return config


def _validate_resolved_config(
    resolved: dict[str, object], study: EnergyStudyConfig, job: EvaluationJobSpec
) -> None:
    expected_scenarios = [
        {"name": scenario.name, "map": scenario.map, "seed": scenario.seed}
        for scenario in job.scenarios
    ]
    if resolved["scenarios"] != expected_scenarios:
        raise RuntimeError(f"resolved scenarios disagree with energy study job {job.id!r}")
    runtime = resolved["runtime"]
    if not isinstance(runtime, dict) or runtime.get("seed") != study.planner_noise_seed:
        raise RuntimeError("resolved planner noise seed disagrees with energy study")
    sampler = resolved["sampler"]
    if not isinstance(sampler, dict) or (
        sampler.get("name") != study.sampler.name
        or sampler.get("ddim_stochasticity") != study.sampler.ddim_stochasticity
    ):
        raise RuntimeError("resolved sampler disagrees with energy study")
    environment = resolved["env"]
    if not isinstance(environment, dict):
        raise TypeError("resolved env config must be a mapping")
    for name, expected in (
        ("trajectory_execution_steps", study.execution.trajectory_execution_steps),
        ("decision_repeat", study.execution.decision_repeat),
        ("physics_world_step_size", study.execution.physics_world_step_size_s),
    ):
        if environment.get(name) != expected:
            raise RuntimeError(f"resolved env.{name} disagrees with energy study timing")
    if environment.get("random_agent_model") != study.vehicle_config.random_agent_model:
        raise RuntimeError("resolved vehicle randomization disagrees with energy study")


def _collect_run(
    study: EnergyStudyConfig,
    job: EvaluationJobSpec,
    guidance: GuidanceProfileSpec,
    run_dir: Path,
    returncode: int,
) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    resolved_path = run_dir / "resolved_config.yaml"
    if not summary_path.is_file() or not resolved_path.is_file():
        return {
            "job": job.id,
            "guidance": guidance.id,
            "returncode": returncode,
            "status": "launcher_failure",
            "output_dir": str(run_dir),
            "episodes": [],
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resolved = load_resolved_yaml_mapping(resolved_path)
    _validate_resolved_config(resolved, study, job)
    declared = {scenario.name: scenario for scenario in job.scenarios}
    episodes = []
    for episode in summary["episodes"]:
        scenario_name = episode["scenario"]["name"]
        if scenario_name not in declared:
            raise RuntimeError(f"summary contains undeclared scenario {scenario_name!r}")
        episodes.append(
            {
                "scenario_metadata": declared[scenario_name].model_dump(mode="json"),
                "evaluation": episode,
            }
        )
    return {
        "job": job.id,
        "guidance": guidance.id,
        "returncode": returncode,
        "status": summary["status"],
        "output_dir": str(run_dir),
        "episodes": episodes,
    }


def run_study(study_path: Path, output_root: Path) -> int:
    study = load_energy_study(study_path)
    output_root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(study_path), output_root / "study_manifest.yaml", resolve=True)
    records: list[dict[str, object]] = []
    failed = False
    for job in study.jobs:
        for guidance in study.guidance_profiles:
            run_dir = output_root / job.id / guidance.id
            command = [
                sys.executable,
                "-m",
                "scripts.evaluate",
                f"--config-name={job.config_name}",
                f"components/sampler={study.sampler.name}",
                f"components/guidance={guidance.config}",
                f"hydra.run.dir={run_dir.as_posix()}",
            ]
            completed = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
            record = _collect_run(study, job, guidance, run_dir, completed.returncode)
            records.append(record)
            write_json(output_root / "matrix_summary.json", {"runs": records})
            failed = failed or completed.returncode != 0 or record["status"] != "completed"
    return 1 if failed else 0
