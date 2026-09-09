"""Recompute fixed-batch descriptive evidence without reward, GAE or backward execution."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from .io import read_arrays, read_json
from .statistics import (
    advantage_comparison,
    gradient_comparison,
    paired_difference,
    rmse,
    statistics,
)

FORMS = {"raw": "raw_advantage", "center": "center_advantage", "z": "normalized_advantage"}


def recompute(source: Path, *, sample_source: Path | None = None) -> dict[str, Any]:
    result = deepcopy(read_json(source / "summary.json"))
    arrays = read_arrays(source / "diagnostics.npz")
    samples = read_json((sample_source or source) / "sample_index.json")["samples"]
    ids = np.asarray([sample["scenario_index"] for sample in samples])
    keys = [(s["scenario_index"], s["episode_index"], s["planning_cycle_index"]) for s in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate fixed-batch sample index")
    if len(samples) != result["sample_count"] or not np.array_equal(ids, arrays["scenario_index"]):
        raise ValueError("diagnostics scenario order differs from sample_index")
    ablation = "credit_forms" in result
    quantiles = [float(q) for q in result["arms"][0]["reward"]["quantiles"]]
    # JSON serialization sorts keys lexically, so recover the numerical quantile order.
    quantiles.sort()

    def vector(key: str, *, gradient: bool = False) -> np.ndarray:
        value = arrays[key]
        if value.ndim != 1 or (not gradient and value.size != len(samples)):
            raise ValueError(f"invalid diagnostic array shape: {key}")
        if not np.isfinite(value).all():
            raise ValueError(f"nonfinite diagnostic array: {key}")
        return value

    def describe(entry: dict, prefix: str, *, ablation_form: bool = False) -> None:
        for key in (
            "reward",
            "raw_advantage",
            "center_advantage",
            "normalized_advantage",
            "value_target",
        ):
            if key in entry:
                entry[key] = statistics(
                    vector(prefix + key),
                    quantiles,
                    ddof=1 if "advantage" in key or ablation_form else 0,
                )
        for form_or_group, norms in entry.get("gradient_norms", {}).items():
            if isinstance(norms, dict):
                for group in norms:
                    norms[group] = float(
                        np.linalg.norm(
                            vector(
                                f"{prefix}gradient_{form_or_group}_{group}", gradient=True
                            ).astype(np.float64)
                        )
                    )
            else:
                entry["gradient_norms"][form_or_group] = float(
                    np.linalg.norm(
                        vector(f"{prefix}gradient_{form_or_group}", gradient=True).astype(
                            np.float64
                        )
                    )
                )

    for key in result.get("components", {}):
        result["components"][key] = statistics(vector(key), quantiles)
    labels = []
    for i, arm in enumerate(result["arms"]):
        labels.append(arm.get("label", arm.get("lambda")))
        prefix = f"arm_{arm['label']}_" if ablation else f"arm_{i}_"
        describe(arm, prefix)
        for credit, entry in arm.get("credit_forms", {}).items():
            describe(entry, f"arm_{arm['label']}__{credit}__", ablation_form=True)

    def compare(entry: dict, left: str, right: str, form: str, *, flat: bool = False) -> None:
        x, y = vector(left + FORMS[form]), vector(right + FORMS[form])
        entry.update(advantage_comparison(x, y))
        metric = "normalized_advantage_rmse" if form == "z" else "advantage_rmse"
        entry[metric] = rmse(x, y)
        for group in entry["gradients"]:
            suffix = f"gradient_{group}" if flat else f"gradient_{form}_{group}"
            entry["gradients"][group] = gradient_comparison(
                vector(left + suffix, gradient=True), vector(right + suffix, gradient=True)
            )
        for key in entry.get("matched_differences", {}):
            entry["matched_differences"][key], _ = paired_difference(
                vector(left + key), vector(right + key), ids, quantiles
            )

    for pair in result["pairs"]:
        if ablation:
            credit = pair["credit_form"]
            left, right = f"arm_r0__{credit}__", f"arm_energy_only__{credit}__"
        else:
            i = labels.index(pair["arm_i"] if "arm_i" in pair else pair["lambda_i"])
            j = labels.index(pair["arm_j"] if "arm_j" in pair else pair["lambda_j"])
            left, right = f"arm_{i}_", f"arm_{j}_"
        compare(pair, left, right, "z", flat="advantage_forms" not in result)
        for form, entry in pair.get("advantage_forms", {}).items():
            compare(entry, left, right, form)
    for form, entry in result.get("endpoint_forms", {}).items():
        compare(entry, "arm_0_", f"arm_{labels.index('energy_only')}_", form)
    return result


def calibration(source: Path) -> dict[str, Any]:
    original = recompute(source / "original", sample_source=source)
    calibrated = recompute(source / "calibrated", sample_source=source)
    audit = read_json(source / "audit.json")
    arrays = read_arrays(source / "audit.npz")
    samples = read_json(source / "sample_index.json")["samples"]
    ids = np.asarray([s["scenario_index"] for s in samples])
    cycles = np.asarray([s["planning_cycle_index"] for s in samples])
    if not np.array_equal(ids, arrays["scenario_index"]) or not np.array_equal(
        cycles, arrays["planning_cycle_index"]
    ):
        raise ValueError("calibration audit index differs from sample_index")
    q = sorted(float(v) for v in original["arms"][0]["reward"]["quantiles"])
    # Audit limits and attribution are experimental evidence; only distributions are recomputed.
    distributions = {}
    for key, values in arrays.items():
        if key in ("scenario_index", "planning_cycle_index"):
            continue
        values = values.reshape(-1)
        if len(values) != len(ids):
            raise ValueError(f"audit sample count mismatch: {key}")
        distributions[key] = {
            "all": statistics(values, q),
            "per_scenario": {str(i): statistics(values[ids == i], q) for i in np.unique(ids)},
            "per_planning_cycle": {
                str(i): statistics(values[cycles == i], q) for i in np.unique(cycles)
            },
        }
    return {
        "original": original,
        "calibrated": calibrated,
        "audit": audit,
        "distributions": distributions,
        "experiment": read_json(source / "summary.json"),
    }
