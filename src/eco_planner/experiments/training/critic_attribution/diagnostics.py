"""Explicit design and materiality criterion for offline critic attribution."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator

ArmLabel = Literal["r0", "rstress"]
CreditForm = Literal["standard_gae", "reward_only_gae", "discounted_return"]
AdvantageForm = Literal["raw", "center", "z"]

_ARM_REWARD_PROFILES: dict[str, str] = {
    "r0": "plannerrft_no_energy_calibrated_v1",
    "rstress": "plannerrft_energy_band_lam64_v1",
}

_MATERIALITY_GROUPS = (
    ("actor_head", "min_actor_head_cosine"),
    ("lateral", "min_lateral_cosine"),
    ("longitudinal", "min_longitudinal_cosine"),
)


class AttributionRun(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    label: str = Field(pattern=r"^[a-z0-9_-]+$")
    arm: ArmLabel
    training_seed: int = Field(ge=0)
    path: str
    reward_profile: str


class MaterialityThresholds(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    min_actor_head_cosine: StrictFloat
    min_longitudinal_cosine: StrictFloat
    min_lateral_cosine: StrictFloat
    max_advantage_sign_flip_fraction: StrictFloat

    @model_validator(mode="after")
    def valid_ranges(self) -> MaterialityThresholds:
        cosines = (
            self.min_actor_head_cosine,
            self.min_longitudinal_cosine,
            self.min_lateral_cosine,
        )
        if any(value < -1.0 or value > 1.0 for value in cosines):
            raise ValueError("cosine thresholds must lie in [-1, 1]")
        if not 0.0 <= self.max_advantage_sign_flip_fraction <= 1.0:
            raise ValueError("sign-flip threshold must lie in [0, 1]")
        return self


class CriticAttributionConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    runs: list[AttributionRun] = Field(min_length=1)
    baseline_credit_form: Literal["standard_gae"]
    comparison_credit_forms: list[CreditForm] = Field(min_length=1)
    advantage_form: AdvantageForm
    update_indices: list[int] = Field(min_length=1)
    provenance_rtol: StrictFloat
    required_run_count: int = Field(ge=1)
    thresholds: MaterialityThresholds

    @model_validator(mode="after")
    def validate_design(self) -> CriticAttributionConfig:
        labels = [run.label for run in self.runs]
        if len(labels) != len(set(labels)):
            raise ValueError("run labels must be unique")
        keys = [(run.arm, run.training_seed) for run in self.runs]
        if len(keys) != len(set(keys)):
            raise ValueError("run arm/seed pairs must be unique")
        for run in self.runs:
            if run.reward_profile != _ARM_REWARD_PROFILES[run.arm]:
                raise ValueError("run reward profile does not match its arm")
        if len(self.runs) != self.required_run_count:
            raise ValueError("run count must equal required_run_count")
        if "standard_gae" in self.comparison_credit_forms:
            raise ValueError("standard_gae is the baseline credit form")
        if len(self.comparison_credit_forms) != len(set(self.comparison_credit_forms)):
            raise ValueError("comparison credit forms must be unique")
        if any(index < 0 for index in self.update_indices):
            raise ValueError("update indices must be nonnegative")
        if self.update_indices != sorted(set(self.update_indices)):
            raise ValueError("update indices must strictly increase")
        if self.provenance_rtol <= 0.0:
            raise ValueError("provenance tolerance must be positive")
        return self


def evaluate_materiality(
    comparisons: dict[str, list[dict]],
    thresholds: MaterialityThresholds,
) -> dict:
    """Apply the pre-registered all-checkpoint materiality criterion.

    ``comparisons`` maps each comparison label to a flat list of per-checkpoint
    records with ``advantage`` and ``gradients``. A checkpoint passes only when
    every required cosine is at or above its threshold and the advantage
    sign-flip fraction is at or below its threshold. Undefined (zero-vector)
    cosines are recorded but excluded from the pass/fail decision.
    """

    gates: dict[str, dict] = {}
    overall_pass = True
    for label, records in comparisons.items():
        failures: list[dict] = []
        excluded: list[dict] = []
        for record in records:
            advantage = record["advantage"]
            sign_flip = advantage["sign_flip_fraction"]
            if sign_flip is None or sign_flip > thresholds.max_advantage_sign_flip_fraction:
                failures.append(
                    {
                        "run": record["run"],
                        "update_index": record["update_index"],
                        "check": "advantage_sign_flip_fraction",
                        "value": sign_flip,
                    }
                )
            for group, field in _MATERIALITY_GROUPS:
                cosine = record["gradients"][group]["cosine"]
                if cosine is None:
                    excluded.append(
                        {
                            "run": record["run"],
                            "update_index": record["update_index"],
                            "check": group,
                        }
                    )
                    continue
                if cosine < getattr(thresholds, field):
                    failures.append(
                        {
                            "run": record["run"],
                            "update_index": record["update_index"],
                            "check": f"{group}_cosine",
                            "value": cosine,
                        }
                    )
        passed = not failures
        overall_pass = overall_pass and passed
        gates[label] = {
            "passed": passed,
            "checkpoint_count": len(records),
            "failures": failures,
            "excluded_undefined": excluded,
        }
    return {
        "verdict": (
            "critic_not_material_to_actor_direction"
            if overall_pass
            else "critic_material_candidate"
        ),
        "comparisons": gates,
    }
