# Track PPO through Fabric and MLflow

## Context

PPO 已有 typed update summaries 与研究 artifacts；需要可检索的 run 索引和曲线，
又不希望 reward、rollout 或 TorchRL 数学依赖 tracking backend。

## Decision

选择 Fabric 与 Lightning 的 MLFlowLogger，由训练层 adapter 管理 run 和发布已有统计；
通用 detached reduction 复用 TorchMetrics，领域统计不移交给 tracking。
采用本地 SQLite 作为当时默认，避免另建 logger framework、Trainer、autologging 或 registry。

MLflow 只作索引与可视化，configs、summaries、rollout 和 checkpoints 仍是独立研究证据。
恢复时延续 run 与绝对 update index，目的是把同一训练历史连续呈现；
允许记录 invocation 差异不等于承诺跨配置精确续训。

## Consequences

Tracking identity 与 policy export 分离，缺失历史参数不能用恢复时配置补造，
logging failure 也不应静默关闭追踪。这些取舍保护了历史身份与失败可见性。
详细 run 续写／回填规则由 artifacts contract 拥有，精确恢复范围由 training contract 拥有。

规范归属：[Artifacts contract](../contracts/artifacts.md)、[Training contract](../contracts/training.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
