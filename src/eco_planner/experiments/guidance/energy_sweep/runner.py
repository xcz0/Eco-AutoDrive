from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from eco_planner.analysis import publish
from eco_planner.artifacts import write_json
from eco_planner.evaluation import load_job_summary
from eco_planner.jobs import compose_job_config, run_evaluation_job

from .config import (
    EvaluationJobSpec,
    GuidanceProfileSpec,
    load_energy_study,
)


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


def run_study(study_path: Path, output_root: Path, *, figures: bool = True) -> int:
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
    publish("energy-sweep", output_root, output_root, figures=figures)
    return 1 if failed else 0
