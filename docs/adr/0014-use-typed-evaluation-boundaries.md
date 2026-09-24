# Use typed evaluation boundaries

**Status:** Accepted and implemented
**Date:** 2026-08-13

Evaluation 曾混合配置解析、运行装配、episode 执行、MetaDrive 数据解释、持久化和离线读取。
动态字典及跨层字段访问让接口依赖隐式约定，一个字段变化需要多个消费者同步修改。

因此选择在领域边界使用显式类型：Pydantic 适合配置与结构化 metadata，frozen dataclass
适合已解析的环境／执行记录，NumPy 适合高维 trace。没有为所有数据建立通用序列化框架，
也没有把第三方开放 env 配置面错误地当成项目封闭 schema。

集中边界验证使消费者能够使用已经解释好的数据，避免重复校验与含义分叉。
[ADR 0020](0020-restructure-evaluation-artifacts-and-analysis.md) 后续讨论包级依赖方向；
本篇不再维护 runner/runtime/reader 的模块职责表或具体字段清单。

规范归属：[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
