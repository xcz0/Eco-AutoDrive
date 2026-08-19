# Use typed evaluation boundaries

**Status:** Accepted and implemented
**Date:** 2026-08-13

Evaluation 曾同时承担 Hydra 配置解析、运行时装配、episode 执行、原始 MetaDrive 数据解析、持久化对象构造和离线读取。大量动态字典及跨层字段访问使接口依赖隐式调用约定，一个字段变化往往需要多个消费者同步修改。

因此，evaluation 的主要领域边界使用显式类型表示，而不是在内部传播未经约束的字典。

配置和结构化 JSON 边界使用 Pydantic。Hydra/OmegaConf 只存在于配置入口，在进入 evaluation 领域代码之前解析为严格的配置模型。项目自身控制的配置模型拒绝未知字段，并在需要时保持不可变；MetaDrive 的 `env` 配置仍作为开放的第三方配置面处理，而项目依赖的跨字段约束由项目配置模型验证。

环境执行边界使用标准库 frozen dataclass。该类对象适合表达已经解析完成、具有明确语义的运行时记录，同时不会把 MetaDrive 对象本身或配置框架带入后续层。

高维轨迹和 trace 数组继续使用 NumPy 表示并保存为 NPZ。Pydantic 不用于承载这些高维数组；数组字段、shape、dtype、有限性以及跨数组关系由集中的 NumPy validation boundary 验证。

因此三类数据分别使用最适合其性质的表示：

- Pydantic：配置和结构化 metadata；
- frozen dataclass：环境与执行记录；
- NumPy：高维数值 trace。

这一划分避免为所有数据建立一个通用序列化框架，也避免消费者重复实现项目特有的 schema 校验。

Runner 负责作业编排，episode 负责单次闭环执行，runtime 负责模型推理，execution 负责轨迹执行，artifact boundary 负责持久化，reader/analysis 负责离线消费。ADR 0020 进一步规定这些职责之间的包级依赖方向。

具体字段、数组形状、dtype 和当前配置模型属于 `docs/agents/system-contract.md`，不在本 ADR 中重复定义。
