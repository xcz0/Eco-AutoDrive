# Consolidate runtime ownership and evaluation topology

**Status:** Accepted and implemented
**Date:** 2026-09-01

共享设备与 host-transfer 行为仍散落在调用方，评测配置还把 semantic topology 与机器容量
混在一起，导致换资源 profile 可能改变 job 含义。

因此选择共享 runtime 拥有设备传输与随机采样机制，各执行路径保留自己的编排；
job 显式选择 serial/vector/job-parallel，resource profile 只给容量。
这一分离让同一个研究 job 换机器时仍保持执行含义，也让 per-slot 随机流可以审计。

当时的包移动、façade/worker 拆分和 benchmark 空字段删除属于直接内部 cutover。
本篇不再逐项列 current topology；它保留的理由是环境领域不反向依赖 runtime，
推理不拥有 rollout decision/profiling，执行与 audit 传输可分别管理。

曾测量持久 CUDA audit stream，但在当时 workload 与机器状态变化下没有一致收益，
因此选择每次 deferred transfer 独立创建 stream。该观察只解释当时取舍，不成为通用性能结论
或永久 stream-lifetime 规范。此组织改动没有改变 planner、随机流、reward、终止或 checkpoint。

规范归属：[Execution contract](../contracts/execution.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
