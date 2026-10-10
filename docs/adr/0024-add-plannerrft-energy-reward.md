# Add PlannerRFT-style energy reward at the MetaDrive execution boundary

**Status:** Accepted and implemented
**Date:** 2026-08-26

Issue #59 当时需要可配置、可审计的 PlannerRFT-style reward，并保持 builtin smoke 的
既有数值含义。MetaDrive 运动学执行与 nuPlan scorer/车辆动力学不同，因此选择显式命名的
适配 profile，而不声称作者实现或 scorer parity。

当时决定在每个实际子步的 objective-neutral TransitionMetrics 上求 score，再向 PPO
提供 scalar reward，另存 profile-specific audit。最初采用 safety gate 与加权分量，
energy 的指数尺度只是 E-019 条件下的 smoke normalization，不是已验证的节能目标。
Native energy 与执行重算 proxy 被分开，是为避免上游 phase ordering 的观测混入 reward；
E-019 的观察只适用于其历史运行条件。

**历史所有权说明（#102）：** 本篇原方案由环境 slot/vector worker 接收 reward profile 并求值；
这是当时的位置决定，不能用作现行要求。后续收口分离客观 execution facts 与 collector-side
reward，迁移依据见 [execution contract 的来源说明](../contracts/execution.md#来源与待确认边界)；
[ADR 0039](0039-unify-closed-loop-cadence.md) 又明确了在线／离线共享子步归约。
这不是宣布本篇整体被取代，也不否定引入可审计 reward 的历史理由。

详细公式、阈值、profile 与 audit schema 由目标规范和显式配置拥有。
本决定支持真实 PPO update 的链路验证，不构成 A/B、舒适性、parity 或节能改善证据；
原先“不在实验完成前关闭 Issue #59”的边界保留为当时验收要求，不推断其当前状态。

规范归属：[Training protocol](../research/protocols/training.md)、[Execution contract](../contracts/execution.md)、[Training contract](../contracts/training.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
