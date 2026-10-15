"""Small persisted batches shared by offline diagnostic integration tests."""

import torch
from omegaconf import OmegaConf

from eco_planner.artifacts import write_json
from eco_planner.jobs import compose_job_config
from eco_planner.planning.policy import ExplorationPolicy, save_exploration_policy_checkpoint
from eco_planner.rl.config import parse_training_config
from eco_planner.rl.rollout.fixed_batch import write_batch
from tests.training.helpers import _policy_config, _ppo_config


def diagnostic_inputs(reward, *, minibatch_size=2):
    raw = compose_job_config(
        "jobs/training/ppo",
        [
            "components/resources=rtx3050_laptop",
            "runtime.seed=0",
            "runtime.accelerator=cpu",
            "training.replay_id=0",
            "training.transitions_per_environment=2",
            "training.update_count=1",
        ],
    )
    raw.policy = _policy_config().model_dump()
    raw.ppo = (
        _ppo_config()
        .model_copy(
            update={
                "batch_size": 4,
                "minibatch_size": minibatch_size,
                "scheduler_total_optimizer_steps": 2,
            }
        )
        .model_dump()
    )
    raw.reward = reward.model_dump()
    config = parse_training_config(raw)
    with torch.random.fork_rng():
        torch.manual_seed(0)
        policy = ExplorationPolicy(config.policy)
    return config, policy


def persist_diagnostic_batch(path, config, policy, slots):
    path.mkdir(parents=True)
    episodes, samples = write_batch(path, slots, config.scenarios)
    OmegaConf.save(OmegaConf.create(config.model_dump(mode="json")), path / "resolved_config.yaml")
    save_exploration_policy_checkpoint(path / "policy-initial.pt", policy)
    write_json(path / "runtime_metadata.json", {"fixture": "synthetic CPU"})
    return episodes, samples
