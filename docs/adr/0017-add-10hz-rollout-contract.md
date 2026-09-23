# Add a 10 Hz closed-loop rollout contract

> 关于 10 Hz 单点 rollout transition 与 2 Hz evaluation 分离 cadence 的决定已由 [ADR 0039](0039-unify-closed-loop-cadence.md) 取代；DDIM、buffer、bootstrap 与随机流约束继续有效。

PlannerRFT PPO-only collection uses one 0.1 s trajectory point as one MDP transition. The existing
evaluation path remains a separate 2 Hz/0.5 s receding-horizon contract that executes five points;
its baseline artifacts and conclusions are not reinterpreted as PPO rollouts.

Each rollout decision generates a DDIM reference from one frozen scene/navigation encoding, samples
one Beta guidance action from an independent policy RNG, and completes the guided pass using that
same encoding, initial noise and DDIM transition randomness. The buffer stores CPU-resident frozen
policy context, base/transformed action, old guidance log-probability, old value, initial noise,
pre-consumption RNG states, seeds, execution count and terminal data. It does not store DDIM denoise
transitions.

`terminated` never bootstraps. A pure time-limit `truncated` transition bootstraps from the old
critic value on the final simulator state, while the future-advantage recursion must still stop at
both terminal and truncation boundaries. A bounded collector tail uses the same bootstrap rule.

Stage 4 transports MetaDrive's returned single-substep reward under the explicit
`metadrive_builtin_v1` / `dimensionless_score` label. This is not a PlannerRFT or nuPlan parity
reward and does not close G-07; GAE, PPO optimization, training orchestration, persisted rollout
artifacts and multi-environment collection remain outside this decision.
