# Unify planner-facing MetaDrive environment slots

**Status:** Accepted and implemented
**Date:** 2026-08-25

Serial evaluation、single-environment rollout 和 vector worker 曾分别装配环境、
交通历史与轨迹执行，重复的 reset/warmup/step 容易让换图和失败语义分叉。

因此选择单个物理环境 slot 统一拥有仿真生命周期、observation/history 与 execution；
evaluation 的证据分类和 RL 的 reward/GAE/PPO 留在调用方。这让共同物理边界可以复用，
又不把不同研究消费者的语义合并。Observation spec 与完整 planner config 分离，
也避免 worker 为局部适配重建模型配置。

[ADR 0023](0023-use-torchrl-parallel-env-for-vector-metadrive.md) 只取代自定义 IPC
client/worker protocol 部分，保留 slot 所有权；逻辑 scenario、RNG、episode 仍归各自编排。
本次统一没有实现 [ADR 0009](0009-prewarm-background-traffic-without-ego-interaction.md)
的 ego-independent prewarm，也没有改变交通、坐标、reward 或 timing 的定义。

规范归属：[Execution contract](../contracts/execution.md)、[Training contract](../contracts/training.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
