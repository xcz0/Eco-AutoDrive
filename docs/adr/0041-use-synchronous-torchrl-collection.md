# Use synchronous TorchRL collection at the parent boundary

**Status:** Accepted and implemented
**Date:** 2026-10-14

[Issue #99](https://github.com/xcz0/Eco-AutoDrive/issues/99) 的本次范围已明确接受：
TorchRL 承担同步 step/reset 和训练批次装配，项目继续拥有逻辑 scenarios、逐 slot RNG、
规划调用、reward/audit 与显式 bootstrap。

选择在父进程提供环境与 policy 适配，由同步 Collector 驱动已有的并行环境。
这样可以复用采集循环与 TensorDict stacking，而无需把 planner、GPU 或 reward profile
移入 CPU simulator workers。训练字段和环境状态的生命周期分开，避免 reset 处理上一决策的
policy context，也避免把完整原始 observation 重复存入训练批次。

逻辑场景分组与领域收尾仍由项目编排：通用 collector 的 frame lifecycle 不自行定义场景配额、
环境终止事实或 bootstrap。因而可以保留现行采集边界，而不要求将科研 runtime 改写为通用策略网络。

本决定取代 [ADR 0023](0023-use-torchrl-parallel-env-for-vector-metadrive.md) 中当时保留整个
项目采集循环的选择；不取代其 ParallelEnv 进程生命周期决定，也不要求迁移 PPOTrainer。
本次未进行性能测试，不据基础设施替换宣称吞吐或显存改善。

规范归属：[Training contract](../contracts/training.md)、
[Execution contract](../contracts/execution.md)、[Training protocol](../research/protocols/training.md)。

> 本篇保存设计理由；现行科研方法与软件保证由上述 Protocol/Contract 拥有。
