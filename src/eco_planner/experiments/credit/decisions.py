"""Configuration and pure diagnostics for objective decomposition."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from eco_planner.analysis.statistics import rmse as rmse


class GateThresholds(BaseModel):
    """Configured objective engineering thresholds; not claimed as general theory."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    endpoint_max_actor_head_cosine: StrictFloat = Field(gt=-1.0, lt=1.0)
    min_normalized_advantage_rmse: StrictFloat = Field(ge=0.0)
    min_sign_flip_fraction: StrictFloat = Field(ge=0.0, le=1.0)
    min_stress_fraction_of_endpoint_separation: StrictFloat = Field(gt=0.0, le=1.0)


def _angular_separation(cosine_value: float) -> float:
    return float(np.arccos(np.clip(cosine_value, -1.0, 1.0)))


def evaluate_gate(
    endpoint_pair: dict[str, Any],
    endpoint_forms: dict[str, Any],
    stress_pairs: list[dict[str, Any]],
    thresholds: GateThresholds,
) -> dict[str, Any]:
    """Apply Configured objective thresholds and attribution rules to measured pair metrics."""

    def head_cosine(pair: dict[str, Any]) -> float:
        value = pair["gradients"]["actor_head"]["cosine"]
        if value is None:
            raise RuntimeError("actor-head gradient cosine is undefined for the gate")
        return float(value)

    endpoint_cosine = head_cosine(endpoint_pair)
    endpoint_rmse = float(endpoint_pair["normalized_advantage_rmse"])
    endpoint_sign_flip = float(endpoint_pair["sign_flip_fraction"])
    endpoint_identifiable = endpoint_cosine <= thresholds.endpoint_max_actor_head_cosine and (
        endpoint_rmse >= thresholds.min_normalized_advantage_rmse
        or endpoint_sign_flip >= thresholds.min_sign_flip_fraction
    )
    endpoint_separation = _angular_separation(endpoint_cosine)
    stress: list[dict[str, Any]] = []
    for pair in stress_pairs:
        value = head_cosine(pair)
        separation = _angular_separation(value)
        stress.append(
            {
                "lambda": pair["lambda_j"],
                "actor_head_cosine": value,
                "angular_separation_rad": separation,
                "fraction_of_endpoint_separation": separation / endpoint_separation
                if endpoint_separation > 0
                else None,
                "norm_ratio_j_over_i": pair["gradients"]["actor_head"]["norm_ratio_j_over_i"],
            }
        )
    separations = [entry["angular_separation_rad"] for entry in stress]
    nondecreasing = all(b >= a for a, b in pairwise(separations))
    increases_overall = separations[-1] > separations[0]
    reaches_fraction = any(
        entry["fraction_of_endpoint_separation"] is not None
        and entry["fraction_of_endpoint_separation"]
        >= thresholds.min_stress_fraction_of_endpoint_separation
        for entry in stress
    )
    failure_reasons: list[str] = []
    if endpoint_cosine > thresholds.endpoint_max_actor_head_cosine:
        failure_reasons.append("endpoint actor-head cosine above threshold")
    if (
        endpoint_rmse < thresholds.min_normalized_advantage_rmse
        and endpoint_sign_flip < thresholds.min_sign_flip_fraction
    ):
        failure_reasons.append(
            "endpoint normalized-advantage RMSE and sign-flip fraction below thresholds"
        )
    if not nondecreasing:
        failure_reasons.append("stress angular distance is not nondecreasing in lambda")
    if not increases_overall:
        failure_reasons.append("stress angular distance does not increase overall")
    if not reaches_fraction:
        failure_reasons.append("no stress arm reaches the required fraction of endpoint separation")

    def separable(form: str) -> bool:
        value = endpoint_forms[form]["gradients"]["actor_head"]["cosine"]
        return value is not None and float(value) <= thresholds.endpoint_max_actor_head_cosine

    if endpoint_identifiable:
        attribution = (
            "relative_scale_lambda_parameterization_too_weak" if not reaches_fraction else None
        )
    elif separable("raw") or separable("center"):
        attribution = "normalization_suppressed_identifiability"
    else:
        attribution = "objective_batch_collinearity"
    return {
        "thresholds": thresholds.model_dump(),
        "endpoint": {
            "actor_head_cosine": endpoint_cosine,
            "normalized_advantage_rmse": endpoint_rmse,
            "sign_flip_fraction": endpoint_sign_flip,
            "angular_separation_rad": endpoint_separation,
            "identifiable": endpoint_identifiable,
        },
        "stress": stress,
        "stress_angular_distance_nondecreasing": nondecreasing,
        "stress_angular_distance_increases_overall": increases_overall,
        "stress_reaches_fraction_of_endpoint_separation": reaches_fraction,
        "failure_reasons": failure_reasons,
        "passed": not failure_reasons,
        "attribution": attribution,
    }


CREDIT_FORMS = ("standard_gae", "reward_only_gae", "discounted_return")


class AttributionThresholds(BaseModel):
    """Configured endpoint thresholds for temporal-credit attribution."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    endpoint_max_actor_head_cosine: StrictFloat = Field(gt=-1.0, lt=1.0)
    min_normalized_advantage_rmse: StrictFloat = Field(ge=0.0)
    min_sign_flip_fraction: StrictFloat = Field(ge=0.0, le=1.0)


def _endpoint_identifiable(pair: dict[str, Any], thresholds: AttributionThresholds) -> bool:
    value = pair["gradients"]["actor_head"]["cosine"]
    if value is None:
        raise RuntimeError("actor-head gradient cosine is undefined for the attribution gate")
    return float(value) <= thresholds.endpoint_max_actor_head_cosine and (
        float(pair["normalized_advantage_rmse"]) >= thresholds.min_normalized_advantage_rmse
        or float(pair["sign_flip_fraction"]) >= thresholds.min_sign_flip_fraction
    )


def evaluate_attribution(
    pairs: Sequence[dict[str, Any]], thresholds: AttributionThresholds
) -> dict[str, Any]:
    """Apply the Configured credit attribution rules to the measured endpoint pairs."""
    by_form = {pair["credit_form"]: pair for pair in pairs}
    if set(by_form) != set(CREDIT_FORMS):
        raise ValueError("attribution requires exactly one endpoint pair per credit form")
    endpoint = {
        form: {
            "actor_head_cosine": float(by_form[form]["gradients"]["actor_head"]["cosine"]),
            "normalized_advantage_rmse": float(by_form[form]["normalized_advantage_rmse"]),
            "sign_flip_fraction": float(by_form[form]["sign_flip_fraction"]),
            "identifiable": _endpoint_identifiable(by_form[form], thresholds),
        }
        for form in CREDIT_FORMS
    }
    standard_ok = endpoint["standard_gae"]["identifiable"]
    reward_only_ok = endpoint["reward_only_gae"]["identifiable"]
    return_only_ok = endpoint["discounted_return"]["identifiable"]
    if standard_ok:
        attribution = None
    elif reward_only_ok:
        attribution = "critic_gae_common_term_dominated"
    elif return_only_ok:
        attribution = "temporal_credit_structure_sensitivity"
    else:
        attribution = "reward_batch_collinearity"
    return {
        "thresholds": thresholds.model_dump(),
        "endpoint": endpoint,
        "endpoint_identifiable_under_standard_gae": standard_ok,
        "attribution": attribution,
        "decision": "Batch attribution only; the PPO training definition is unchanged",
    }
