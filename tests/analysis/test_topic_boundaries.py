"""Independent offline topic APIs and their source ownership boundaries."""

import subprocess
import sys
from importlib import import_module

import pytest

from tests.analysis.routing import MODULES


@pytest.mark.parametrize("module", list(MODULES.values()))
def test_topic_imports_do_not_initialize_runs_or_plotting(module):
    script = """
import importlib, sys
class BlockImports:
    def find_spec(self, fullname, path=None, target=None):
        roots = ('torch', 'metadrive', 'panda3d', 'matplotlib', 'seaborn',
                 'eco_planner.jobs', 'eco_planner.runtime.metadata',
                 'eco_planner.rl.trainer', 'eco_planner.rl.rollout',
                 'eco_planner.rl.optimization.update_diagnostics',
                 'eco_planner.evaluation.episodes')
        runner = fullname.startswith('eco_planner.experiments.') and fullname.endswith('.runner')
        if runner or any(
            fullname == root or fullname.startswith(root + '.') for root in roots
        ):
            raise AssertionError('unexpected online/plotting import: ' + fullname)
sys.meta_path.insert(0, BlockImports())
module = importlib.import_module(sys.argv[1])
assert callable(module.analyze)
"""
    subprocess.run([sys.executable, "-c", script, module], check=True)


@pytest.mark.parametrize("module", [m for k, m in MODULES.items() if k != "scalar-reward"])
@pytest.mark.parametrize("relation", ["same", "child", "parent"])
def test_topics_reject_overlapping_output_before_reading_evidence(tmp_path, module, relation):
    source = tmp_path / "source"
    source.mkdir()
    output = {"same": source, "child": source / "report", "parent": tmp_path}[relation]
    with pytest.raises(ValueError, match="separate"):
        import_module(module).analyze(source, output, figures=False)
    assert list(source.iterdir()) == []
