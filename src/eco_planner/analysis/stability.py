"""Read-only Optuna study evidence and search statistics."""

import math
from pathlib import Path
from typing import Any

import optuna
from optuna.importance import FanovaImportanceEvaluator, get_param_importances
from optuna.trial import FrozenTrial, TrialState

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
    search_summary = summarize_search(study, seed, int(config.stage_b.top_config_count))
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
        "parameter_importances": search_summary["parameter_importances"],
        "parameter_importance_error": search_summary["parameter_importance_error"],
        "search_summary": search_summary,
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


def summarize_search(study: optuna.Study, seed: int, top_config_count: int) -> dict[str, object]:
    completed = sorted(
        (trial for trial in study.trials if trial.state == TrialState.COMPLETE),
        key=lambda item: (-(item.value or -math.inf), item.number),
    )
    importances, importance_error = importance(study, seed)
    counts = {
        state.name.lower(): sum(trial.state == state for trial in study.trials)
        for state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL)
    }
    return {
        "study_name": study.study_name,
        "trial_count": len(study.trials),
        "state_counts": counts,
        "stability_counts": {
            "stable": counts["complete"],
            "unstable": counts["pruned"] + counts["fail"],
        },
        "top_configs": [trial_payload(item) for item in completed[:top_config_count]],
        "parameter_importances": importances,
        "parameter_importance_error": importance_error,
        "stability_region": [trial_payload(item) for item in study.trials],
    }


def trial_payload(trial: FrozenTrial) -> dict[str, object]:
    return {
        "trial_number": trial.number,
        "state": trial.state.name.lower(),
        "value": trial.value,
        "parameters": dict(trial.params),
        "user_attributes": dict(trial.user_attrs),
    }
