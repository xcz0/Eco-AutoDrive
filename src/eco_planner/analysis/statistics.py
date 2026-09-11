"""Float64 statistics with explicit ddof and undefined-vector semantics."""

import statistics as descriptive_statistics
from collections.abc import Sequence
from math import isfinite
from typing import Any, TypedDict, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from scipy.stats import bootstrap, pearsonr, spearmanr


def statistics(value: np.ndarray, quantiles: Sequence[float], *, ddof: int = 0) -> dict:
    x = np.asarray(value, dtype=np.float64).reshape(-1)
    if x.size <= ddof or not np.isfinite(x).all():
        raise ValueError("statistics require enough finite samples")
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=ddof)),
        "quantiles": {
            str(q): float(v) for q, v in zip(quantiles, np.quantile(x, quantiles), strict=True)
        },
    }


class ScenarioBootstrapConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid", allow_inf_nan=False)
    confidence_level: float = Field(ge=0.95, le=0.95)
    n_resamples: StrictInt = Field(gt=0)
    bootstrap_seed: StrictInt = Field(ge=0)


def scenario_effect(delta: np.ndarray, config: ScenarioBootstrapConfig) -> dict:
    """Scenario uncertainty conditional on one trained policy and available pairs."""
    x = np.asarray(delta, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("scenario bootstrap requires finite paired deltas")
    result = {
        "estimate": float(x.mean()) if x.size else None,
        "sample_count": int(x.size),
        "ci95": None,
        "ci_crosses_zero": None,
        "unavailable_reason": "fewer than two available scenario pairs" if x.size < 2 else None,
    }
    if x.size >= 2:
        interval = bootstrap(
            (x,),
            np.mean,
            method="percentile",
            confidence_level=config.confidence_level,
            n_resamples=config.n_resamples,
            rng=np.random.default_rng(config.bootstrap_seed),
        ).confidence_interval
        low, high = float(interval.low), float(interval.high)
        result["ci95"] = [low, high]
        result["ci_crosses_zero"] = low <= 0 <= high
    return result


def cosine(x: np.ndarray, y: np.ndarray) -> float | None:
    x, y = x.astype(np.float64), y.astype(np.float64)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return None if denominator == 0 else float(np.dot(x, y) / denominator)


def advantage_comparison(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = x.astype(np.float64).reshape(-1), y.astype(np.float64).reshape(-1)
    defined = x.size >= 2 and np.any(x != x[0]) and np.any(y != y[0])
    return {
        "pearson": float(cast(Any, pearsonr(x, y))[0]) if defined else None,
        "spearman": float(cast(Any, spearmanr(x, y))[0]) if defined else None,
        "sign_flip_fraction": float(np.mean(x * y < 0)),
        "zero_fraction_i": float(np.mean(x == 0)),
        "zero_fraction_j": float(np.mean(y == 0)),
    }


def rmse(x: np.ndarray, y: np.ndarray) -> float:
    delta = x.astype(np.float64) - y.astype(np.float64)
    return float(np.sqrt(np.mean(np.square(delta))))


def gradient_comparison(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    norm_x = np.linalg.norm(x.astype(np.float64))
    norm_y = np.linalg.norm(y.astype(np.float64))
    return {
        "cosine": cosine(x, y),
        "norm_ratio_j_over_i": None if norm_x == 0 else float(norm_y / norm_x),
    }


def paired_difference(
    reference: np.ndarray,
    comparison: np.ndarray,
    scenario_ids: np.ndarray,
    quantiles: Sequence[float],
) -> tuple[dict, np.ndarray]:
    if reference.shape != comparison.shape or reference.size != scenario_ids.size:
        raise ValueError("paired arrays and scenario index must have matching shapes")
    delta = comparison.astype(np.float64) - reference.astype(np.float64)
    return {
        "all": statistics(delta, quantiles),
        "per_scenario": {
            str(slot): statistics(delta[scenario_ids == slot], quantiles)
            for slot in np.unique(scenario_ids)
        },
    }, delta


class Measurement(TypedDict):
    samples: list[float]
    median: float
    minimum: float
    maximum: float


def measurement(samples: Sequence[float]) -> Measurement:
    values = [float(value) for value in samples]
    if not values:
        raise ValueError("benchmark measurement requires at least one sample")
    if not all(isfinite(value) for value in values):
        raise ValueError("benchmark measurement samples must be finite")
    if any(value < 0.0 for value in values):
        raise ValueError("benchmark measurement samples must be non-negative")
    return {
        "samples": values,
        "median": descriptive_statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }
