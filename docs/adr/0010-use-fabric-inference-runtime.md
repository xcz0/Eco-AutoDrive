# Use a single-device Fabric inference runtime

Fabric（不使用 Trainer）被选为单设备推理的统一装配边界，避免每个调用方分别处理设备、
forward precision 和全局 seeding。该决定延续了每个 Hydra job 独立管理仿真与产物的思路，
没有引入分布式 simulator 启动。

自动 CUDA mixed precision 是吞吐量与数值参考之间的显式取舍：它保留官方权重、模型层级、
normalization 和采样含义，但不等于严格 FP32 数值基线。这细化了
[ADR 0001](0001-preserve-official-baseline.md)，没有放宽官方模型语义。具体精度解析与复现范围
由 execution contract 定义，实际请求值和解析值由 artifact 保存。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
