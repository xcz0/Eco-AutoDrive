"""Task H evaluation-diagnostics manifest, composition, and summary diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from eco_planner.analysis.training import beta_probe_statistics, paired_beta_deltas

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = REPOSITORY_ROOT / "configs" / "experiments" / "comparison" / "default.yaml"
STUDY_PATH = (
    REPOSITORY_ROOT / "configs" / "experiments" / "training" / "evaluation-diagnostics.yaml"
)
STOCHASTIC_OVERRIDES = [
    "evaluation.policy_checkpoint.action_mode=sample",
    "evaluation.policy_checkpoint.policy_action_seed=810001",
]


@pytest.fixture(autouse=True)
def without_machine_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MACHINE_NAME", raising=False)


def test_beta_probe_statistics_aggregates_affine_beta_moments() -> None:
    probe = {
        "alpha": [[2.0, 2.0], [2.0, 2.0]],
        "beta": [[2.0, 2.0], [2.0, 2.0]],
    }

    statistics = beta_probe_statistics(probe)

    assert statistics["context_count"] == 2
    assert statistics["beta_mean"]["mean"] == pytest.approx([0.0, 0.0])
    assert statistics["concentration"]["mean"] == pytest.approx([4.0, 4.0])
    assert statistics["variance"]["mean"] == pytest.approx([0.2, 0.2])


def test_paired_beta_deltas_reports_per_dimension_mean_and_rms() -> None:
    reference = {"alpha": [[2.0, 2.0]], "beta": [[2.0, 2.0]]}
    stress = {"alpha": [[1.0, 2.0]], "beta": [[2.0, 2.0]]}

    deltas = paired_beta_deltas(reference, stress)

    assert deltas["concentration"]["mean_delta_per_dimension"] == pytest.approx([-1.0, 0.0])
    assert deltas["beta_mean"]["mean_delta_per_dimension"][1] == pytest.approx(0.0)
    assert deltas["beta_mean"]["mean_delta_per_dimension"][0] < 0.0
    assert deltas["variance"]["rms"] > 0.0


def _heldout_values(speed: float, energy: float) -> dict[str, Any]:
    return {
        "mean_speed_mps": speed,
        "distance_m": 100.0,
        "route_completion": 0.95,
        "energy_total_ml": energy * 0.1,
        "energy_ml_per_km": energy,
        "arrive_dest_fraction": 0.8125,
        "collision_count": 0,
        "out_of_road_count": 0,
        "stopped_fraction": 0.0,
        "simulated_seconds": 30.0,
    }


def _unit_records() -> tuple[dict[int, dict[str, dict[str, Any]]], dict[int, dict[str, Any]]]:
    probe = {"alpha": [[2.0, 2.0]], "beta": [[2.0, 2.0]]}
    records = {
        0: {
            "initial": {
                "checkpoint": "runs/r0-seed-0/policy-initial.pt",
                "deterministic": _heldout_values(10.0, 47.0),
                "stochastic": {
                    810001: _heldout_values(10.0, 47.0),
                    810002: _heldout_values(10.0, 47.0),
                },
            },
            "r0": {
                "checkpoint": "runs/r0-seed-0/policy-final.pt",
                "deterministic": _heldout_values(9.9, 46.9),
                "stochastic": {
                    810001: _heldout_values(9.9, 46.9),
                    810002: _heldout_values(9.9, 46.9),
                },
            },
            "rstress": {
                "checkpoint": "runs/rstress-seed-0/policy-final.pt",
                "deterministic": _heldout_values(10.1, 47.1),
                "stochastic": {
                    810001: _heldout_values(10.1, 47.1),
                    810002: _heldout_values(10.3, 47.2),
                },
            },
        }
    }
    beta_records = {
        0: {arm: {"probe_before": probe, "probe_after": probe} for arm in ("r0", "rstress")}
    }
    return records, beta_records
