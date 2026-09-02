"""Run the fixed-seed energy benchmark study without retrying failed episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, field_validator

from eco_planner._repository import CONFIG_ROOT
from eco_planner.artifacts import write_json
from eco_planner.configuration import load_resolved_yaml_mapping
from eco_planner.evaluation import load_job_summary
from eco_planner.jobs import compose_job_config, run_evaluation_job

DEFAULT_STUDY = CONFIG_ROOT / "studies" / "energy" / "matrix.yaml"


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)


class EnergyMetricSpec(_StrictModel):
    name: Literal["metadrive_fuel_proxy"]
    implementation: Literal["recompute_metadrive_base_vehicle_formula_on_executed_trace"]
    formula: str = Field(min_length=1)
    unit: Literal["mL"]
    sampling_interval_s: float = Field(gt=0.0)
    interpretation: str = Field(min_length=1)


class GuidanceProfileSpec(_StrictModel):
    id: Literal[
        "baseline",
        "longitudinal_negative",
        "longitudinal_zero",
        "longitudinal_positive",
    ]
    config: str = Field(min_length=1)
    longitudinal_scale: StrictFloat | None


class EvaluationJobSpec(_StrictModel):
    id: str = Field(min_length=1)
    config_name: str = Field(min_length=1)


class EnergyStudyConfig(_StrictModel):
    version: Literal[1]
    energy_metric: EnergyMetricSpec
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


def load_energy_study(path: Path) -> EnergyStudyConfig:
    config = EnergyStudyConfig.model_validate(load_resolved_yaml_mapping(path))
    config.validate_study_contract()
    return config


def _collect_run(
    job: EvaluationJobSpec,
    guidance: GuidanceProfileSpec,
    run_dir: Path,
    returncode: int,
) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return {
            "job": job.id,
            "guidance": guidance.id,
            "returncode": returncode,
            "status": "launcher_failure",
            "output_dir": str(run_dir),
            "episodes": [],
        }
    summary = load_job_summary(summary_path)
    episodes = []
    for episode in summary.episodes:
        episodes.append(
            {
                "scenario_metadata": {
                    **episode.scenario.model_dump(mode="json"),
                    "traffic_condition": _traffic_condition(
                        episode.evaluation_mode, episode.traffic_density
                    ),
                },
                "evaluation": episode.model_dump(mode="json"),
            }
        )
    return {
        "job": job.id,
        "guidance": guidance.id,
        "returncode": returncode,
        "status": summary.status,
        "output_dir": str(run_dir),
        "episodes": episodes,
    }


def _traffic_condition(mode: str, traffic_density: float) -> str:
    if mode == "no_traffic":
        return "no_traffic"
    if mode != "traffic":
        raise ValueError(f"unsupported evaluation mode {mode!r}")
    return f"low_density_trigger_{traffic_density:g}"


def run_study(study_path: Path, output_root: Path) -> int:
    study = load_energy_study(study_path)
    output_root.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.load(study_path), output_root / "study_manifest.yaml", resolve=True)
    records: list[dict[str, object]] = []
    failed = False
    for job in study.jobs:
        for guidance in study.guidance_profiles:
            run_dir = output_root / job.id / guidance.id
            config = compose_job_config(
                job.config_name,
                (f"components/guidance={guidance.config}",),
            )
            summary = run_evaluation_job(config, run_dir)
            returncode = 1 if summary.status == "failed" else 0
            record = _collect_run(job, guidance, run_dir, returncode)
            records.append(record)
            write_json(output_root / "matrix_summary.json", {"runs": records})
            failed = failed or returncode != 0 or record["status"] != "completed"
    return 1 if failed else 0
