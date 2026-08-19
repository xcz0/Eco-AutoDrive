# Parallelize isolated evaluation jobs

**Status:** Accepted and implemented
**Tracking:** [GitHub Issue #5](https://github.com/xcz0/Eco-AutoDrive/issues/5)

MetaDrive 0.4.3 的引擎是进程内单例，因此多个独立闭环场景不能安全地在线程之间并行。即使为每个场景分别构造环境，在同一进程中复制 MetaDrive 闭环或推理 runtime，也会使环境生命周期、随机性、设备资源和输出所有权相互耦合。

因此，evaluation 的并行边界定义在相互隔离的评测作业之间，而不是单个评测作业内部。

每个并行进程独立拥有：

- 一个 MetaDrive 引擎；
- 一个单设备推理 runtime；
- 一个独立的 artifact writer；
- 独立的地图、规划噪声和其他随机状态。

单个作业内的场景继续串行执行。该决定细化 ADR 0010 的单设备推理边界，但不改变单个 episode 的仿真、规划或指标语义。

这种设计优先保证不同执行模式之间的可比较性。并行运行不应通过共享 MetaDrive 状态、跨场景 GPU batching 或改变 episode 内执行顺序来获得吞吐量。

具体 worker 数、CPU 线程预算、CUDA 显存 preflight、视频开关、确定性设置和性能验收阈值属于运行配置及当前系统契约，而不是本 ADR 的长期架构约束。

本决定不引入：

- 线程内 MetaDrive 并行；
- 集中式 GPU batch inference；
- 多 GPU 调度；
- 跨 episode 共享运行时状态。

这些能力如果以后需要，应作为独立架构决策处理。
