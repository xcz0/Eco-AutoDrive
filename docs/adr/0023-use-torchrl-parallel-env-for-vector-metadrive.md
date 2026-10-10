# Use TorchRL ParallelEnv for vector MetaDrive execution

**Status:** Accepted and implemented
**Date:** 2026-08-25

Windows spawn 下的多环境执行需要固定 slots、partial reset/step、共享 tensor 与可关闭的
worker 生命周期。原有 Process/Pipe/消息协议重复了成熟库机制，因此选择 TorchRL ParallelEnv，
由库负责进程、共享 buffer、mask 与关闭，项目只保留 scenario/slot 映射、顺序和领域结果。

本决定仅取代 [ADR 0022](0022-unify-metadrive-environment-slots.md) 的自定义 IPC 部分，
不取消物理 slot 所有权。Evaluation 的动态补位和 rollout 的逻辑 waves、独立 RNG、
episode/bootstrap 与 artifact 仍由各自编排拥有。

当时固定依赖的 partial mask 与 NonTensor 路径不兼容，故通过库的 remote-method channel
传回领域 sidecar，未新建项目 IPC。该 workaround 是当时实现背景，库版本、调用细节与
sidecar 装配不作为永久规范；失败传播和共享 buffer 隔离仍由 execution contract 保证。

没有改用通用 TorchRL Collector，是因为逐步 audit、每 scenario 精确配额、逐 slot 随机流、
延迟传输和独立 bootstrap 测量不适合其固定 frame lifecycle。保留项目 collector
不等于重新实现进程、GAE 或 PPO；这是复用机制与保留科研语义之间的边界。

规范归属：[Execution contract](../contracts/execution.md)、[Training contract](../contracts/training.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
