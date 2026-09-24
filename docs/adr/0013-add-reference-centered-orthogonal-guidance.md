# Add reference-centered orthogonal guidance

选择 reference-centered 正交 guidance，使横纵动作有明确的物理参照，且中性动作精确返回
同次 unguided reference。Reference 与 guided pass 共用冻结官方 EMA、编码和随机样本，
每周期刷新 reference，以免把场景或噪声差异误认为 guidance 效果。

切向／左法向目标、差分速度、centered energy-gradient delta、单位注入系数及 ego-only
梯度作用范围都是项目复现选择；PlannerRFT 未公开这些细节，不能宣称作者实现 parity。
保留未应用的邻车梯度审计，是为了使梯度屏蔽的效果可检查。

选择将 active guidance 限于标准高斯 DDIM5，保留 DPM10 和半尺度 DDIM 为 unguided 对照。
具体目标、梯度步骤、动作接口与 audit 字段分别由 Protocol/Contract 拥有。

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、[Data/model contract](../contracts/data-and-model.md)、[Execution contract](../contracts/execution.md)、[Artifacts contract](../contracts/artifacts.md)。

> #102 迁移阶段：上述新规范仍为 proposed，生效入口遵循 [AGENTS](../../AGENTS.md)。
> 本篇保存设计理由与历史决定，不作为第二套现行规范；本次收口不激活新 owner。
