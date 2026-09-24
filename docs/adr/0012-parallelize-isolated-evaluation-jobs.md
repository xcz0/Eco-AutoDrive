# Parallelize isolated evaluation jobs

**Status:** Accepted and implemented

MetaDrive 0.4.3 的进程内单例使多个独立闭环在线程间共享引擎不安全。为避免环境生命周期、
随机性、设备资源和输出所有权耦合，当时选择在隔离的 evaluation jobs 之间并行：
每个进程独立拥有仿真、单设备推理和 writer，单 job 内场景串行。
这细化了 [ADR 0010](0010-use-fabric-inference-runtime.md)，不改变单 episode 语义。

当时没有引入集中 GPU batching、多 GPU 调度或跨 episode 共享 runtime，优先保证执行模式
可比较。后续同 job vector 环境由 [ADR 0023](0023-use-torchrl-parallel-env-for-vector-metadrive.md)
另行决定，语义 topology 与容量分离见
[ADR 0028](0028-consolidate-runtime-ownership-and-evaluation-topology.md)；
不能把本篇的 job-level 选择解释为对后续 vector 模式的禁止。

Worker 数、线程、显存检查、视频、确定性和性能阈值属于相应执行条件与配置，
不作为本篇不断更新的运行清单。

**Tracking:** [GitHub Issue #5](https://github.com/xcz0/Eco-AutoDrive/issues/5)

规范归属：[Execution contract](../contracts/execution.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
