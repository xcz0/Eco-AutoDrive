"""Native Optuna figures without modifying analysis evidence."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
from optuna.importance import FanovaImportanceEvaluator
from optuna.trial import TrialState

from .plots import save


@dataclass(frozen=True)
class FigureResult:
    files: list[str]
    unavailable: dict[str, str]


def figures(data: dict, study: optuna.Study, seed: int, output: Path) -> FigureResult:
    from optuna.visualization import matplotlib as plots

    files = []
    unavailable: dict[str, str] = {}
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
    return FigureResult(files, unavailable)
