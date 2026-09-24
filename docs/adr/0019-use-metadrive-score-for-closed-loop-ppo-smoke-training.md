# Use the MetaDrive score for closed-loop PPO smoke training

在最终研究 reward 未明确前，当时选择 MetaDrive builtin score 验证真实 closed-loop PPO
能否串通 Exploration Policy、采集、GAE、更新与 checkpoint。这样可将基础设施失效与新的
reward-model 不确定性分开。Builtin 是当时 smoke 的唯一优化信号，不是最终科研目标。

这一选择只能支持端到端训练链路验证；score 上升不能自动解释为安全改善、能耗降低、
PlannerRFT parity 或最终目标改善。Dense/terminal score 与客观运行事实分开审计，
是为了不把未声明的指标静默加入 objective。

当时提出的两个 logical slots、每 slot 16 个 10 Hz transitions、少量 updates 是 smoke profile
示例，不是 PPO invariant。旧 10 Hz 条件由
[ADR 0039](0039-unify-closed-loop-cadence.md) 的 cadence 决定取代，不改写历史实验。

独立 diffusion/action RNG、冻结 planner，以及 policy export 与 training-state checkpoint
分离，使训练对象与恢复边界可追溯。现行 reward 家族及持久化保证引用对应规范；
本篇保留 builtin 的历史理由，不将它恢复为当前可用 profile。

规范归属：[Training protocol](../research/protocols/training.md)、[Training contract](../contracts/training.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
