"""Artifact-to-report regression tests; synthetic fixtures are not research evidence."""

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from eco_planner.analysis.io import read_json, write_json
from eco_planner.analysis.runner import analyze
from eco_planner.analysis.statistics import (
    advantage_comparison,
    cosine,
    paired_difference,
    statistics,
)
from eco_planner.evaluation.artifacts.models import (
    CheckpointSummary,
    EvaluationWorkload,
    InferenceRuntimeSummary,
    JobSummary,
    PolicyCheckpointProvenance,
    WorkloadScenario,
)
from tests.characterization.test_experiment_outputs import _episode, _training_summary


def assert_report(output: Path, *, figures: bool) -> None:
    report = (output / "report.md").read_text(encoding="utf-8")
    assert (output / "analysis.json").is_file()
    links = re.findall(r"\]\(<?([^)>]+)>?\)", report)
    for link in links:
        assert (output / link).exists(), link
    files = read_json(output / "analysis.json")["figures"]
    assert bool(files) == figures
    for relative in files:
        path = output / relative
        assert path.stat().st_size > 100
        if relative.endswith(".svg"):
            assert "<svg" in path.read_text(encoding="utf-8")
        else:
            assert path.read_bytes().startswith(b"\x89PNG")


def write_batch(path, batch):
    path.mkdir(parents=True)
    summary, arrays = batch
    write_json(path / "summary.json", summary)
    np.savez(path / "diagnostics.npz", **arrays)
    write_json(
        path / "sample_index.json",
        {
            "samples": [
                {"scenario_index": i // 2, "episode_index": i % 2, "planning_cycle_index": 0}
                for i in range(4)
            ]
        },
    )


def test_shared_statistics_preserve_rank_ddof_and_undefined_semantics():
    x, y = np.array([1.0, 2.0, 2.0, 4.0]), np.array([4.0, 2.0, 2.0, 1.0])
    assert statistics(x, [0.0, 0.5, 1.0], ddof=1)["std"] == np.std(x, ddof=1)
    assert statistics(x, [0.0, 0.5, 1.0])["std"] == np.std(x)
    assert advantage_comparison(x, y)["spearman"] == pytest.approx(-1.0)
    assert cosine(np.zeros(4), x) is None
    assert advantage_comparison(np.ones(4), x)["pearson"] is None
    summary, delta = paired_difference(x, y, np.array([0, 0, 1, 1]), [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(delta, y - x)
    assert summary["per_scenario"]["0"]["mean"] == 1.5
    with pytest.raises(ValueError):
        statistics(np.array([1.0]), [0.0, 1.0], ddof=1)


def job(energy=2.0):
    episode = _episode(seed=0, distance_m=100.0, energy_ml=energy)
    return JobSummary(
        status="completed",
        runtime=InferenceRuntimeSummary(
            requested_accelerator="cpu",
            resolved_accelerator="cpu",
            requested_precision="32-true",
            resolved_precision="32-true",
            device="cpu",
            seed=0,
            world_size=1,
        ),
        checkpoint=CheckpointSummary(ema_tensor_count=1, parameter_count=1),
        sampler=episode.sampler,
        guidance=episode.guidance,
        workload=EvaluationWorkload(
            mode="traffic",
            profile="fixture",
            history_warmup_steps=0,
            evaluated_horizon_steps=10,
            scenarios=(WorkloadScenario(name="traffic", map="S", seed=0),),
            video_enabled=False,
        ),
        episodes=(episode,),
    )


def test_energy_matched_comparison_and_protocol_mismatch(tmp_path):
    from eco_planner.analysis.evaluation import paired

    source = tmp_path / "source"
    records = []
    for guidance, energy in (("baseline", 2.0), ("longitudinal_zero", 3.0)):
        path = source / "road" / guidance
        path.mkdir(parents=True)
        write_json(path / "summary.json", job(energy).model_dump(mode="json"))
        records.append({"job": "road", "guidance": guidance, "status": "completed"})
    write_json(source / "matrix_summary.json", {"runs": records})
    result = analyze("energy-sweep", source, tmp_path / "report")
    assert (
        result["evidence"]["comparisons"]["road/longitudinal_zero"]["statistics"]["energy_ml"][
            "mean"
        ]
        == 1.0
    )
    assert_report(tmp_path / "report", figures=True)
    altered = job().model_copy(
        update={"workload": job().workload.model_copy(update={"video_enabled": True})}
    )
    with pytest.raises(ValueError, match="workloads"):
        paired(job(), altered)
    altered = job().model_copy(
        update={"episodes": (job().episodes[0].model_copy(update={"noise_seed": 1}),)}
    )
    with pytest.raises(ValueError, match="pairs"):
        paired(job(), altered)


def test_failed_episode_is_visible_and_never_zero_filled(tmp_path):
    from eco_planner.analysis.evaluation import paired
    from eco_planner.analysis.reporting.experiments import experiment_figures
    from eco_planner.evaluation.artifacts.models import FailedEpisodeSummary

    original = job()
    episode = original.episodes[0].model_dump(mode="json")
    fields = {
        key: value for key, value in episode.items() if key in FailedEpisodeSummary.model_fields
    }
    fields.update(
        status="failed",
        trace_status="empty",
        termination={"type": "runtime_error", "detail": "fixture failure"},
        failure={
            "phase": "inference",
            "exception_type": "ValueError",
            "message": "fixture failure",
            "traceback": "fixture",
        },
    )
    failed = FailedEpisodeSummary.model_validate_json(json.dumps(fields))
    comparison = original.model_copy(update={"status": "failed", "episodes": (failed,)})
    result = paired(original, comparison)
    assert result["available_pair_count"] == 0
    assert result["unavailable_pair_count"] == 1
    assert result["statistics"]["energy_ml"] is None
    assert result["pairs"][0]["energy_ml_delta"] is None
    assert result["pairs"][0]["comparison"]["failure"]["message"] == "fixture failure"
    assert experiment_figures("energy-sweep", {"comparisons": {"failed": result}}, tmp_path)


@pytest.fixture
def training_summary():
    from tests.training.test_tracking import summary

    update = summary.__wrapped__()
    return _training_summary(0, 0).model_copy(update={"updates": (update,)})


def test_scalar_explicit_grouping_and_checkpoint_validation(tmp_path, training_summary):
    from eco_planner.experiments.comparison.inputs import load_comparison
    from eco_planner.experiments.protocol.config import DEFAULT_PROTOCOL

    source = tmp_path / "source"
    source.mkdir()
    protocol = OmegaConf.load(DEFAULT_PROTOCOL)
    protocol.evaluation.maps = ["S"]
    protocol.evaluation.map_seeds = [0]
    protocol.evaluation.seed = 0
    protocol.evaluation.horizon_steps = 10
    protocol.training.maps = ["C"]
    protocol.training.map_seeds = [1]
    protocol.training.seeds = [0]
    OmegaConf.save(protocol, source / "protocol.yaml")
    for name in ("a0", "a2"):
        (source / name).mkdir()
    write_json(source / "a0/summary.json", job().model_dump(mode="json"))
    policy = job(1.0).model_copy(
        update={
            "policy_checkpoint": PolicyCheckpointProvenance(
                label="final", path="policy.pt", policy_hash=training_summary.final_policy_hash
            )
        }
    )
    write_json(source / "a2/summary.json", policy.model_dump(mode="json"))
    write_json(source / "training.json", training_summary.model_dump(mode="json"))
    config = {
        "protocol": "protocol.yaml",
        "baseline_evaluation_dir": "a0",
        "runs": [
            {
                "arm": "a2",
                "training_summary": "training.json",
                "checkpoint_label": "final",
                "evaluation_dir": "a2",
            }
        ],
    }
    OmegaConf.save(OmegaConf.create(config), source / "comparison.yaml")
    result = analyze(
        "scalar-reward",
        source,
        tmp_path / "report",
        scalar_comparison=load_comparison(source / "comparison.yaml"),
    )
    assert result["evidence"]["runs"][0]["comparison"]["statistics"]["energy_ml"]["mean"] == -1.0
    assert_report(tmp_path / "report", figures=True)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    with pytest.raises(ValueError, match="separate"):
        analyze(
            "scalar-reward",
            unrelated,
            source / "report",
            scalar_comparison=load_comparison(source / "comparison.yaml"),
        )
    for field, invalid, reason in (
        ("training_seed", 99, "seed absent"),
        ("reward_profile", "plannerrft_no_energy_v1", "reward differs"),
        ("final_policy_hash", "f" * 64, "declared training"),
    ):
        write_json(
            source / "training.json",
            training_summary.model_copy(update={field: invalid}).model_dump(mode="json"),
        )
        with pytest.raises(ValueError, match=reason):
            load_comparison(source / "comparison.yaml")
    write_json(source / "training.json", training_summary.model_dump(mode="json"))
    protocol.evaluation.seed = 99
    OmegaConf.save(protocol, source / "protocol.yaml")
    with pytest.raises(ValueError, match="seed differs"):
        load_comparison(source / "comparison.yaml")
    protocol.evaluation.seed = 0
    OmegaConf.save(protocol, source / "protocol.yaml")
    config["runs"] *= 2
    OmegaConf.save(OmegaConf.create(config), source / "comparison.yaml")
    with pytest.raises(ValueError, match="duplicate"):
        analyze(
            "scalar-reward",
            source,
            tmp_path / "report",
            scalar_comparison=load_comparison(source / "comparison.yaml"),
        )


def test_sanity_keeps_failed_checks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    write_json(
        source / "sanity_report.json",
        {
            "status": "failed",
            "cases": {"case": {"components": {"energy": 0.5, "comfort": 1.0}}},
            "checks": [{"name": "expected", "passed": False}],
        },
    )
    result = analyze("reward-sanity", source, tmp_path / "report")
    assert result["evidence"]["status"] == "failed"
    assert_report(tmp_path / "report", figures=True)


@pytest.mark.parametrize("training", [False, True])
def test_scalar_run_uses_common_report_writer(tmp_path, training_summary, training):
    from eco_planner.analysis.runner import publish_scalar_run

    summary = training_summary if training else job()
    write_json(tmp_path / "summary.json", summary.model_dump(mode="json"))
    publish_scalar_run(tmp_path, training=training)
    assert_report(tmp_path, figures=True)


@pytest.mark.parametrize("figures", [False, True])
def test_publication_configures_plotting_only_when_requested(tmp_path, figures):
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "sanity_report.json", {"cases": {"fixture": {"components": {"x": 1.0}}}})
    write_json(source / "summary.json", job().model_dump(mode="json"))
    script = """
import sys
from pathlib import Path

figures = sys.argv[2] == 'True'
class BlockImports:
    def find_spec(self, fullname, path=None, target=None):
        blocked = ['torch', 'metadrive', 'panda3d', 'eco_planner.models']
        if not figures:
            blocked += ['matplotlib', 'seaborn', 'optuna.visualization']
        if any(fullname == root or fullname.startswith(root + '.') for root in blocked):
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
from eco_planner.analysis.runner import analyze, publish_scalar_run
source = Path(sys.argv[1])
analyze('reward-sanity', source, source.parent / 'report', figures=figures)
publish_scalar_run(source, training=False, figures=figures)
if figures:
    import matplotlib
    import matplotlib.pyplot as plt
    assert matplotlib.get_backend().lower() == 'agg'
    assert not plt.get_fignums()
else:
    assert not any(k == 'matplotlib' or k.startswith('matplotlib.') for k in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script, str(source), str(figures)], check=True)


def test_offline_imports_do_not_load_execution_modules():
    script = """
import sys
import scripts.experiments
import eco_planner.analysis.evaluation
import eco_planner.analysis.simple
import eco_planner.analysis.workflows
import eco_planner.analysis.reporting.guidance
from eco_planner.rl.artifacts import TrainingRunSummary
for root in ('torch', 'metadrive', 'panda3d', 'eco_planner.rl.trainer', 'eco_planner.models'):
    assert not any(k == root or k.startswith(root + '.') for k in sys.modules), root
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("module", ["eco_planner.analysis", "eco_planner.rl.artifacts"])
def test_offline_package_initialization_never_imports_torch(module):
    script = f"""
import importlib
import sys
class BlockExecutionImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('torch', 'metadrive', 'panda3d', 'eco_planner.models'):
            raise AssertionError('offline initialization imported ' + fullname)
sys.meta_path.insert(0, BlockExecutionImports())
importlib.import_module({module!r})
from eco_planner.rl.artifacts import TrainingRunSummary
assert TrainingRunSummary.__name__ == 'TrainingRunSummary'
"""
    subprocess.run([sys.executable, "-c", script], check=True)
