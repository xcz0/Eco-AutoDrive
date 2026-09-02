"""Evaluation package facade and lightweight offline import coverage."""

from __future__ import annotations

import subprocess
import sys


def test_offline_reader_import_does_not_load_online_dependencies() -> None:
    script = """
import sys
from eco_planner.evaluation import build_matrix_report, load_job_summary, load_trace_artifact
assert callable(build_matrix_report)
assert callable(load_job_summary)
assert callable(load_trace_artifact)
assert "torch" not in sys.modules
assert "metadrive" not in sys.modules
assert "panda3d" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_public_api_declares_primary_interfaces() -> None:
    import eco_planner.evaluation as evaluation

    assert {
        "CompletedEpisodeSummary",
        "EvaluationJobConfig",
        "build_matrix_report",
        "load_job_summary",
        "parse_evaluation_config",
        "run_evaluation",
    } <= set(evaluation.__all__)
