"""Test adapters exercising each independent topic API with the shared regression fixtures."""

from importlib import import_module

MODULES = {
    "reward": "eco_planner.experiments.reward.analysis",
    "credit": "eco_planner.experiments.credit.analysis",
    "training": "eco_planner.experiments.training.analysis",
    "reward-sanity": "eco_planner.experiments.reward.sanity_analysis",
    "energy-sweep": "eco_planner.experiments.guidance.sweep_analysis",
    "scalar-reward": "eco_planner.experiments.comparison.analysis",
    "execution-backend": "benchmarks.execution_analysis",
    "guidance-control-authority": "eco_planner.experiments.guidance.authority.analysis",
    "guidance-horizon": "eco_planner.experiments.guidance.horizon.analysis",
    "guidance-deferral": "eco_planner.experiments.guidance.deferral.analysis",
    "guidance-decomposition": "eco_planner.experiments.guidance.decomposition.analysis",
    "guidance-execution-bridge": "eco_planner.experiments.guidance.execution_bridge.analysis",
    "training-critic-attribution": "eco_planner.experiments.training.critic_attribution.analysis",
}


def analyze(experiment, source, output, *, figures=True, scalar_comparison=None, source_file=None):
    kwargs = {"figures": figures}
    if experiment == "scalar-reward":
        kwargs["scalar_comparison"] = scalar_comparison
    if experiment == "execution-backend":
        kwargs["source_file"] = source_file
    return import_module(MODULES[experiment]).analyze(source, output, **kwargs)
