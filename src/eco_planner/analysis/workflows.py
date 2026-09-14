"""Recompute distributions and paired diagnostics from persisted workflow arrays."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .io import read_json
from .statistics import (
    advantage_comparison,
    gradient_comparison,
    paired_difference,
    rmse,
    statistics,
)


def fixed(source: Path) -> dict:
    summary = read_json(source / "summary.json")
    samples = read_json(source / "sample_index.json")["samples"]
    if len(samples) != summary["sample_count"]:
        raise ValueError("sample index length differs from summary")
    identities = [
        (r["scenario_index"], r["episode_index"], r["planning_cycle_index"]) for r in samples
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate sample index")
    with np.load(source / "diagnostics.npz", allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    scenario = np.asarray([r["scenario_index"] for r in samples])
    np.testing.assert_array_equal(arrays["scenario_index"], scenario)
    quantiles = summary["quantiles"]
    distributions = {}
    for name, values in arrays.items():
        if name == "scenario_index" or "__gradient_" in name:
            continue
        if len(values) != len(samples):
            raise ValueError(f"sample axis differs for {name}")
        distributions[name] = {
            "all": statistics(
                values,
                quantiles,
                ddof=1
                if "advantage" in name
                else summary["value_target_ddof"]
                if "value_target" in name
                else 0,
            ),
            "per_scenario": {
                str(i): statistics(values[scenario == i], quantiles) for i in np.unique(scenario)
            },
        }
    pairs = []
    for a, b in combinations(summary["arms"], 2):
        first, second = a["label"], b["label"]
        if summary["kind"] == "reward":
            left, right = arrays[f"{first}__reward_total"], arrays[f"{second}__reward_total"]
            difference, _ = paired_difference(left, right, scenario, quantiles)
            pairs.append(
                {"reference": first, "comparison": second, "reward_difference": difference}
            )
            continue
        for credit in summary["credit_forms"]:
            for form in summary["advantage_forms"]:
                key = {
                    "raw": "raw_advantage",
                    "center": "center_advantage",
                    "z": "normalized_advantage",
                }[form]
                left, right = (
                    arrays[f"{first}__{credit}__{key}"],
                    arrays[f"{second}__{credit}__{key}"],
                )
                difference, _ = paired_difference(left, right, scenario, quantiles)
                prefix = f"{first}__{credit}__gradient_{form}_"
                groups = summary["gradient_groups"]
                pairs.append(
                    {
                        "reference": first,
                        "comparison": second,
                        "credit_form": credit,
                        "advantage_form": form,
                        **advantage_comparison(left, right),
                        "rmse": rmse(left, right),
                        "matched_difference": difference,
                        "gradients": {
                            g: gradient_comparison(
                                arrays[prefix + g],
                                arrays[f"{second}__{credit}__gradient_{form}_{g}"],
                            )
                            for g in groups
                        },
                    }
                )
    result = {"recorded": summary, "distributions": distributions, "pairs": pairs}
    if summary["kind"] == "reward":
        result["audit"] = read_json(source / "audit.json")
        with np.load(source / "audit.npz", allow_pickle=False) as audit:
            np.testing.assert_array_equal(audit["scenario_index"], scenario)
            cycles = np.asarray([r["planning_cycle_index"] for r in samples])
            np.testing.assert_array_equal(audit["planning_cycle_index"], cycles)
            result["audit_distributions"] = {
                key: {
                    "all": statistics(audit[key], quantiles),
                    "per_scenario": {
                        str(i): statistics(audit[key][scenario == i], quantiles)
                        for i in np.unique(scenario)
                    },
                    "per_planning_cycle": {
                        str(i): statistics(audit[key][cycles == i], quantiles)
                        for i in np.unique(cycles)
                    },
                }
                for key in audit.files
                if key not in ("scenario_index", "planning_cycle_index")
            }
    return result


def training(source: Path) -> dict:
    # All Torch/checkpoint measurements are persisted by diagnose/grid; reporting stays lightweight.
    summary = read_json(source / "summary.json")
    if summary["kind"] not in ("training-grid", "training-diagnostics", "training-evaluation"):
        raise ValueError("source is not a current training workflow result")
    return summary
