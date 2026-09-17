"""Recompute critic-attribution advantage comparisons from persisted arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import read_json
from .statistics import advantage_comparison

_ADVANTAGE_FIELDS = (
    "pearson",
    "spearman",
    "sign_flip_fraction",
    "zero_fraction_i",
    "zero_fraction_j",
)


def _same(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= 1e-9


def recompute(source: Path) -> dict:
    summary = read_json(source / "summary.json")
    if summary.get("kind") != "training-critic-attribution":
        raise ValueError("source is not a training critic-attribution result")
    samples = read_json(source / "sample_index.json")["samples"]
    with np.load(source / "diagnostics.npz", allow_pickle=False) as data:
        arrays = {key: data[key].copy() for key in data.files}
    credit_forms = summary["credit_forms"]
    baseline, comparison_forms = credit_forms[0], credit_forms[1:]
    recorded: dict[tuple[str, str, int], dict[str, Any]] = {}
    for run in summary["runs"]:
        for update in run["updates"]:
            for label, comparison in update["comparisons"].items():
                recorded[(label, run["label"], update["update_index"])] = {
                    **comparison,
                    "losses": update["losses"],
                    "critic": update["critic"],
                }
    checkpoints: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for sample in samples:
        identity = (sample["run"], sample["update_index"])
        if identity in seen:
            raise ValueError("duplicate sample record")
        seen.add(identity)
        prefix, transition_count = sample["prefix"], sample["transition_count"]
        for credit in sample["credit_forms"]:
            value = arrays.get(f"{prefix}__{credit}__raw_advantage")
            if value is None or value.shape != (transition_count,):
                raise ValueError(f"missing or mismatched advantage array for {prefix}/{credit}")
        for credit in comparison_forms:
            label = f"{baseline}_vs_{credit}"
            advantage = advantage_comparison(
                arrays[f"{prefix}__{baseline}__raw_advantage"],
                arrays[f"{prefix}__{credit}__raw_advantage"],
            )
            source_record = recorded[(label, sample["run"], sample["update_index"])]
            if any(
                not _same(advantage[field], source_record["advantage"][field])
                for field in _ADVANTAGE_FIELDS
            ):
                raise ValueError(f"recomputed advantage differs from recorded evidence: {prefix}")
            checkpoints.append(
                {
                    "run": sample["run"],
                    "arm": sample["arm"],
                    "training_seed": sample["training_seed"],
                    "update_index": sample["update_index"],
                    "comparison": label,
                    "advantage": advantage,
                    "gradients": source_record["gradients"],
                    "losses": source_record["losses"],
                    "critic": source_record["critic"],
                }
            )
    return {**summary, "checkpoints": checkpoints}
