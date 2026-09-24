# Add a 10 Hz closed-loop rollout contract

> 关于 10 Hz 单点 rollout transition 与 2 Hz evaluation 分离 cadence 的决定已由
> [ADR 0039](0039-unify-closed-loop-cadence.md) 取代；DDIM、buffer、bootstrap 与随机流决定保留。

当时选择每个 0.1 s 轨迹点作为 PPO transition，而已有 evaluation 每次执行五点、0.5 s。
这允许独立建立 rollout 路径，而不重解释 evaluation 的历史产物；它不是当前正式 cadence。

采集选择冻结 context、共享 reference/guided 随机样本以及独立 policy RNG，并保存足够的
动作、概率、边界和随机状态用于 replay，而不保存完整 denoise chain。
Terminal 与可 bootstrap 的 truncation/collector tail 分开，避免 advantage 跨 episode 泄漏。
这些决定的详细软件要求归 training/artifacts contract。

Stage 4 曾只运输标记为 `metadrive_builtin_v1 / dimensionless_score` 的单子步 reward；
GAE、PPO、训练编排、持久化与多环境采集不在当时范围内，也没有由此关闭 G-07 或证明 parity。
Builtin smoke 的理由见 [ADR 0019](0019-use-metadrive-score-for-closed-loop-ppo-smoke-training.md)；
本篇不据历史标签声明当前 profile 可用。

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Training protocol](../research/protocols/training.md)、[Training contract](../contracts/training.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
