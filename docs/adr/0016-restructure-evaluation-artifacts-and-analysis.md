# Restructure evaluation artifacts and analysis

**Status:** Accepted and implemented
**Date:** 2026-08-14

`evaluation` 的在线运行、artifact 持久化和离线分析曾共处于一个平面包，且 Stage-2 的一次性验收逻辑
被安装为运行时代码。该结构模糊了依赖方向，也使已由 Pydantic 验证的 JSON 在分析端退化为 `dict` 后
再次进行字段校验。

本变更将运行时拆为 `runtime/`，artifact 边界拆为 `artifacts/`，离线 consumer 拆为 `analysis/`。
`artifacts` 不依赖 Lightning、MetaDrive、episode 或 runtime；`analysis` 只使用 typed artifacts 和
NumPy，绝不参与在线 episode 执行。`stage2_matrix.py`、对应脚本和测试已删除；它们是历史实验验收，
不是稳定的评测领域 API。

Artifact schema 升为 v4，唯一有意的 JSON breaking change 是失败字段 `stage` 改为受限的
`FailurePhase` 字段 `phase`。v1--v3 均直接拒绝，保持 ADR 0014 的无兼容层原则。JSON reader 返回
`JobSummary`、`EpisodeSummary` 或 `RuntimeMetadata`，避免消费者再次手写 schema 校验。

trace 的 shape、dtype、guided 字段和有限性由 `TraceFieldSpec`/`TRACE_FIELDS` 作为单一声明源；recording
allocation 与落盘 validation 都从该合同读取。继续使用 NumPy NPZ，不引入新的 array-schema 或统计依赖，
因为它们不会消除项目特有的跨数组不变量，且会扩大已经验证的评测性能边界。
