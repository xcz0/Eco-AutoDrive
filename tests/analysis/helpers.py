"""Shared synthetic job data and report artifact assertions."""

import re
from pathlib import Path

from eco_planner.artifacts import read_json
from eco_planner.evaluation.artifacts.models import (
    CheckpointSummary,
    EvaluationWorkload,
    InferenceRuntimeSummary,
    JobSummary,
    WorkloadScenario,
)
from tests.evaluation.helpers import _episode


def assert_report(output: Path, *, figures: bool) -> None:
    report = (output / "report.md").read_text(encoding="utf-8")
    assert (output / "analysis.json").is_file()
    links = re.findall(r"\]\(<?([^)>]+)>?\)", report)
    for link in links:
        assert (output / link).exists(), link
    files = read_json(output / "analysis.json")["figures"]
    assert bool(files) == figures
    for relative in files:
        path = output / relative
        assert path.stat().st_size > 100
        if relative.endswith(".svg"):
            assert "<svg" in path.read_text(encoding="utf-8")
        else:
            assert path.read_bytes().startswith(b"\x89PNG")


def job(energy=2.0):
    episode = _episode(seed=0, distance_m=100.0, energy_ml=energy)
    return JobSummary(
        status="completed",
        runtime=InferenceRuntimeSummary(
            requested_accelerator="cpu",
            resolved_accelerator="cpu",
            requested_precision="32-true",
            resolved_precision="32-true",
            device="cpu",
            seed=0,
            world_size=1,
        ),
        checkpoint=CheckpointSummary(ema_tensor_count=1, parameter_count=1),
        sampler=episode.sampler,
        guidance=episode.guidance,
        workload=EvaluationWorkload(
            mode="traffic",
            profile="fixture",
            history_warmup_steps=0,
            evaluated_horizon_steps=10,
            scenarios=(WorkloadScenario(name="traffic", map="S", seed=0),),
            video_enabled=False,
        ),
        episodes=(episode,),
    )
