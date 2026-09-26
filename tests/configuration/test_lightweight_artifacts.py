"""I/O and import guarantees independent of experimental topic organization."""

import json
import subprocess
import sys

import numpy as np
import pytest
from pydantic import BaseModel, ValidationError

from eco_planner.artifacts import read_arrays, read_json, write_json, write_npz
from eco_planner.evaluation.artifacts.models import PolicyActionSummary
from eco_planner.planning.policy.statistics import beta_statistics


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("typed", [False, True])
def test_json_writer_rejects_nonfinite_without_overwriting(tmp_path, value, typed):
    class Payload(BaseModel):
        value: float

    target = tmp_path / "artifact.json"
    target.write_text("original", encoding="utf-8")
    payload = Payload(value=value) if typed else {"nested": [value]}
    with pytest.raises(ValueError):
        write_json(target, payload)
    assert target.read_text(encoding="utf-8") == "original"


def test_lightweight_io_roundtrip_and_failure_propagation(tmp_path):
    arrays = {"values": np.arange(6, dtype=np.float32).reshape(2, 3)}
    write_npz(tmp_path / "arrays.npz", arrays)
    actual = read_arrays(tmp_path / "arrays.npz")
    np.testing.assert_array_equal(actual["values"], arrays["values"])
    assert actual["values"].dtype == np.float32
    write_json(tmp_path / "data.json", {"label": "原始证据", "undefined": None})
    assert read_json(tmp_path / "data.json") == {"label": "原始证据", "undefined": None}
    with pytest.raises(FileNotFoundError):
        read_arrays(tmp_path / "missing.npz")
    np.savez(tmp_path / "objects.npz", objects=np.asarray([{}], dtype=object))
    with pytest.raises(ValueError, match="allow_pickle"):
        read_arrays(tmp_path / "objects.npz")
    (tmp_path / "data.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_json(tmp_path / "data.json")


def test_beta_moments_retain_affine_scale():
    assert beta_statistics(2.0, 3.0) == pytest.approx((-0.2, 5.0, 0.16))


def test_policy_action_provenance_roundtrip_and_validation():
    for mode, seeds in [("mean", ()), ("sample", (7, 7))]:
        value = PolicyActionSummary(action_mode=mode, policy_action_seeds=seeds)
        assert PolicyActionSummary.model_validate_json(value.model_dump_json()) == value
    for mode, seeds in [("mean", [7]), ("sample", []), ("sample", [-1])]:
        with pytest.raises(ValidationError):
            PolicyActionSummary.model_validate_json(
                json.dumps({"action_mode": mode, "policy_action_seeds": seeds})
            )


@pytest.mark.parametrize(
    "module",
    [
        "eco_planner.artifacts",
        "eco_planner.planning.policy.statistics",
        "eco_planner.planning.policy.config",
        "eco_planner.rl.artifacts.summaries",
        "eco_planner.rl.artifacts.metadata",
        "eco_planner.evaluation.artifacts.io",
        "eco_planner.evaluation.artifacts.report",
        "eco_planner.evaluation.metrics",
        "scripts.training",
        "scripts.evaluation",
    ],
)
def test_foundation_imports_are_lightweight(module):
    script = """
import importlib, sys
class BlockImports:
    def find_spec(self, fullname, path=None, target=None):
        roots = ('torch', 'metadrive', 'panda3d', 'matplotlib',
                 'eco_planner.rl.trainer', 'eco_planner.rl.rollout',
                 'eco_planner.runtime.envs', 'eco_planner.evaluation.episodes.rendering')
        if any(fullname == root or fullname.startswith(root + '.') for root in roots):
            raise AssertionError('unexpected execution import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
importlib.import_module(sys.argv[1])
"""
    subprocess.run([sys.executable, "-c", script, module], check=True)
