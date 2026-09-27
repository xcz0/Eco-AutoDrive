"""Execution timing compares workloads; scientific metrics keep their domain owner."""

import subprocess
import sys

import pytest

from benchmarks.execution import build_report, write_report
from eco_planner.artifacts import read_json, write_json
from tests.analysis.test_reports import job


@pytest.fixture
def backend_jobs(tmp_path):
    roots = []
    for index, mode in enumerate(("serial", "job_level", "vector")):
        root = tmp_path / mode
        root.mkdir()
        summary = job(energy=2.0 + index)
        write_json(root / "summary.json", summary.model_dump(mode="json"))
        metadata = {
            **dict.fromkeys(
                (
                    "git_head",
                    "git_branch",
                    "platform",
                    "python",
                    "torch",
                    "lightning",
                    "metadrive",
                    "pydantic",
                ),
                "fixture",
            ),
            "git_status_short": [],
            "inference_runtime": summary.runtime.model_dump(mode="json"),
            "sampler": summary.sampler.model_dump(mode="json"),
            "guidance": summary.guidance.model_dump(mode="json"),
            "execution": {
                "mode": "parallel" if mode == "job_level" else "serial",
                "launcher": "joblib" if mode == "job_level" else "basic",
                "worker_count": 1,
                "vector_env_slots": 1 if mode == "vector" else None,
                "deterministic": True,
                "resolved_accelerator": "cpu",
                "process_id": 1,
                "logical_cpu_count": 1,
            },
            "cadence": {
                "simulator_step_s": 0.1,
                "closed_loop_execution_steps": 5,
                "decision_interval_s": 0.5,
            },
            "elapsed_seconds": float(index + 1),
            "cuda_memory": None,
        }
        write_json(root / "runtime_metadata.json", metadata)
        roots.append(root)
    return roots


def test_backend_report_and_offline_cli_preserve_performance_boundary(backend_jobs, tmp_path):
    source = tmp_path / "source"
    report = write_report(
        *backend_jobs,
        serial_wall_s=3.0,
        job_level_wall_s=2.0,
        vector_wall_s=1.0,
        output=source / "evaluation_modes.json",
        figures=False,
    )
    assert report["evaluation_modes"]["vector"]["job_elapsed_s"]["samples"] == [3.0]
    # Different energy outcomes are neither rejected nor turned into benchmark metrics.
    for mode in report["evaluation_modes"].values():
        assert "energy" not in mode and "success_rate" not in mode
    assert "does not establish numerical or behavioral parity" in (source / "report.md").read_text(
        encoding="utf-8"
    )
    script = """
import sys
class BlockExecution:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'metadrive', 'panda3d', 'matplotlib'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockExecution())
from scripts.benchmark_execution import main
sys.argv = ['benchmark', 'analyze', '--source-dir', sys.argv[1],
            '--output-dir', sys.argv[2], '--no-figures']
main()
"""
    output = tmp_path / "analysis"
    subprocess.run([sys.executable, "-c", script, str(source), str(output)], check=True)
    assert (output / "analysis.json").is_file()


@pytest.mark.parametrize("mismatch", ["topology", "workload", "schema"])
def test_backend_comparison_rejects_invalid_inputs(backend_jobs, mismatch):
    vector = backend_jobs[2]
    path = vector / "runtime_metadata.json"
    metadata = read_json(path)
    if mismatch == "topology":
        metadata["execution"]["vector_env_slots"] = None
    elif mismatch == "workload":
        summary_path = vector / "summary.json"
        summary = read_json(summary_path)
        summary["runtime"]["seed"] = 7
        metadata["inference_runtime"]["seed"] = 7
        write_json(summary_path, summary)
    else:
        metadata["elapsed_seconds"] = -1.0
    write_json(path, metadata)
    with pytest.raises(ValueError):
        build_report(*backend_jobs, serial_wall_s=3.0, job_level_wall_s=2.0, vector_wall_s=1.0)
