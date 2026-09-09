"""Read-only Optuna study evidence and native static visualizations."""

from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import TrialState

from .io import read_json


def load_study(source: Path, study_name: str) -> optuna.Study:
    database = (source / "study.db").resolve(strict=True)
    storage = optuna.storages.RDBStorage(
        f"sqlite:///file:{database.as_posix()}?mode=ro&uri=true",
        skip_table_creation=True,
        skip_compatibility_check=True,
    )
    return optuna.load_study(study_name=study_name, storage=storage)


def importance(study: optuna.Study, seed: int) -> tuple[dict[str, float] | None, str | None]:
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if len(completed) < 2:
        return None, "parameter importance requires at least two completed trials"
    if len({t.value for t in completed}) < 2:
        return None, "parameter importance requires varying completed objective values"
    parameters = set.intersection(*(set(t.params) for t in completed))
    if not any(len({t.params[p] for t in completed}) > 1 for p in parameters):
        return None, "parameter importance requires a varying shared parameter"
    return get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=seed)), None


def analyze(source: Path) -> tuple[dict[str, Any], optuna.Study, int]:
    from omegaconf import OmegaConf

    config = OmegaConf.load(source / "study_manifest.yaml")
    seed = int(config.sampler_seed)
    study = load_study(source, str(config.study_name))
    importances, reason = importance(study, seed)
    data: dict[str, Any] = {
        "study_name": study.study_name,
        "importance_seed": seed,
        "trials": [
            {
                "number": t.number,
                "state": t.state.name,
                "value": t.value,
                "parameters": dict(t.params),
                "user_attributes": dict(t.user_attrs),
            }
            for t in study.trials
        ],
        "parameter_importances": importances,
        "parameter_importance_error": reason,
        "stages": {},
        "diagnostics": {},
    }
    for stage in ("b", "c"):
        path = source / f"stage-{stage}" / "summary.json"
        if path.exists():
            data["stages"][stage] = read_json(path)
    for path in sorted((source / "diagnostics").glob("*/summary.json")):
        data["diagnostics"][path.parent.name] = read_json(path)
    data["training_curves"] = {}
    from eco_planner.rl.artifacts.summaries import TrainingRunSummary

    for path in sorted(source.glob("stage-*/**/summary.json")) + sorted(
        (source / "diagnostics").glob("*/*/summary.json")
    ):
        payload = read_json(path)
        # Stage summaries and training summaries coexist in the existing tree.
        if "updates" not in payload:
            continue
        training = TrainingRunSummary.model_validate_json(path.read_text(encoding="utf-8"))
        data["training_curves"][path.parent.relative_to(source).as_posix()] = {
            "update": [u.update_index for u in training.updates],
            "total_reward": [u.total_reward for u in training.updates],
            "mean_approximate_kl": [u.mean_approximate_kl for u in training.updates],
            "mean_episode_length": [u.mean_episode_length for u in training.updates],
        }
    return data, study, seed


def figures(data: dict, study: optuna.Study, seed: int, output: Path) -> list[str]:
    from optuna.visualization import matplotlib as plots

    from .reporting.plots import save

    files = []
    unavailable = data.setdefault("unavailable_figures", {})
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        unavailable["optimization-history"] = "no completed trials"
        unavailable["parallel-coordinate"] = "no completed trials"
    else:
        files += save(plots.plot_optimization_history(study).figure, output, "optimization-history")
        if any(t.params for t in completed):
            files += save(
                plots.plot_parallel_coordinate(study).figure, output, "parallel-coordinate"
            )
        else:
            unavailable["parallel-coordinate"] = "no trial parameters"
    if data["parameter_importances"]:
        # The same seeded native evaluator is used for JSON and the native Optuna figure.
        files += save(
            plots.plot_param_importances(
                study, evaluator=FanovaImportanceEvaluator(seed=seed)
            ).figure,
            output,
            "parameter-importance",
        )
    else:
        unavailable["parameter-importance"] = data["parameter_importance_error"]
    shared = set.intersection(*(set(t.params) for t in completed)) if completed else set()
    varying = sorted(p for p in shared if len({t.params[p] for t in completed}) > 1)
    if len(varying) >= 2:
        axes = plots.plot_contour(study, params=varying)
        ax = axes.flat[0] if isinstance(axes, np.ndarray) else axes
        files += save(ax.figure, output, "contour")
    else:
        unavailable["contour"] = "requires at least two varying shared parameters"
    return files
