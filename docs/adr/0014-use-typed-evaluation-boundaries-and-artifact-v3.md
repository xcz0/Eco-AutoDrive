# Use typed evaluation boundaries and Artifact v3

**Status:** Accepted and implemented
**Date:** 2026-08-13

`evaluation` 曾同时承担 Hydra 解析、运行时装配、单回合循环、原始 MetaDrive `info` 解包、产物
构造和旧 schema 兼容。大量动态字典和跨层字段访问使接口只能依赖调用约定，修改一个字段时需要在
多个消费者中重复维护。既然本仓库不要求旧接口兼容，本次采用破坏性结构重构，不改变评测算法、
仿真语义、采样器或指标。

配置边界使用现有 Pydantic v2。`parse_evaluation_config` 将 Hydra `DictConfig` 一次性解析为
`EvaluationJobConfig`；子配置模型严格、字段冻结并拒绝额外字段。`env` 是 MetaDrive 的开放配置面，
不复制其完整配置 schema，但本项目依赖的字段由顶层模型交叉验证。job、episode 和 runtime JSON
同样使用严格 Pydantic 模型，读取器和 writer 共享这些定义。

环境边界使用标准库 frozen dataclass `TrajectoryExecutionRecord`，集中解析轨迹数组、交通帧和终止
状态。Pydantic 不适合直接承载高维 NumPy trace，也不会替代 MetaDrive 的运行时对象；NPZ 因此继续
由 NumPy 保存，并由单一 `validate_trace_arrays` 定义完整字段、shape、dtype、有限性和跨数组关系。
这避免手写 JSON 字段验证，同时保留适合数组的成熟 NumPy 表示。

runner 只编排作业，episode 只负责一个闭环回合，runtime、execution、rendering、artifact schema
和 reader 各自保持单一边界。Artifact schema 提升为 v3，v1/v2 JSON 与 NPZ 均直接拒绝，不添加
兼容 adapter 或迁移层。该决定取代 ADR 0012 中关于 v1 只读兼容的部分；并行执行决定保持不变。

未引入 OpenCV 等绘图库：现有渲染只需在 NumPy RGB frame 上绘制两条短折线，引入新的二进制依赖
不会减少领域接口或手写 schema。也未引入通用 schema/序列化框架；Pydantic、dataclass 和 NumPy
已经分别覆盖配置与 JSON、环境记录、数组 artifact 三类边界。
