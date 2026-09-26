# Add a forward-only Exploration Policy

选择独立的 Exploration Policy，消费冻结 scene/navigation context 与 physical reference，
让 PPO 学习 guidance，而不直接改写预训练 planner。最初先实现 forward-only policy，
待 rollout/reward 边界明确后再接入闭环；这一阶段边界不是永久禁止接入 evaluation。

当时以 MLP-Mixer 编码 reference，用其查询 scene/navigation tokens，并共享 actor/value trunk。
独立横纵 Beta、仿射 guidance、正 concentration 参数化和对称初始化，使初始均值中性且保留探索。
这些架构与初始化选择是项目复现决定，不能描述为 PlannerRFT 已公开实现。

决定拒绝概率边界动作而非 clipping，以保持 replay 与 PPO 概率记账一致。
动作有效域、Jacobian 和 RNG 要求统一引用 training contract。
Policy export 与可恢复训练状态分离，避免一个用于推理的文件被误认为完整训练快照。

规范归属：[Training protocol](../research/protocols/training.md)、[Training contract](../contracts/training.md)、[Artifacts contract](../contracts/artifacts.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
