"""Offline protocol checks over persisted typed evaluation summaries."""

from eco_planner.evaluation.artifacts.models import JobSummary

from .config import ComparisonProtocol


def validate_evaluation(protocol: ComparisonProtocol, summary: JobSummary) -> None:
    if {
        (e.scenario.map_sequence, e.scenario.seed) for e in summary.episodes
    } != protocol.held_out_pairs():
        raise ValueError("evaluation does not cover the held-out pool")
    if summary.runtime.seed != protocol.evaluation.seed:
        raise ValueError("evaluation seed differs from protocol")
    if (
        summary.workload.evaluated_horizon_steps != protocol.evaluation.horizon_steps
        or summary.sampler.name != protocol.evaluation.sampler
        or summary.sampler.ddim_stochasticity != 0
    ):
        raise ValueError("evaluation horizon/sampler differs from protocol")
