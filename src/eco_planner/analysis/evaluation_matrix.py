"""Validate and summarize the fixed 20-episode MetaDrive traffic matrix."""

from __future__ import annotations

from pathlib import Path

from eco_planner.evaluation.analysis import summarize_matrix


def summarize_evaluation_matrix(matrix_root: Path, *, partial: bool) -> dict[str, object]:
    return summarize_matrix(matrix_root, partial=partial)
