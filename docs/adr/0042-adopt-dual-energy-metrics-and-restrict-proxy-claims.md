# Adopt dual energy metrics and restrict proxy claims

**Status:** Accepted
**Date:** 2026-10-16

E-061 证明 execution fuel proxy 是纯速度型估计量：直接输入只有子步结束速度与位移，
E-060 A2−A1 的强度改善 98.96%–100.09% 由距离加权速度水平载带，加减速时序、怠速、
制动与坡度结构性不可见。E-062 在完全相同的执行轨迹上以 FASTSim 与最小纵向动力学
proxy 复核，proxy 的节能方向未获任一物理模型跨 seed 支持（seed 0 两者与 proxy 反向，
CI 不含 0），且 proxy 对 A0→A1 的物理差距低估 5–8 倍。

在此证据下，proxy 单独继续作为正式评测的 energy 主指标会把已被反证的方向声明留在
主表；但立即替换同样缺乏依据：两个物理模型彼此跨 seed 不一致、FASTSim 判定对窗口
分辨率敏感、physics proxy 是 study-local 极简实现，且不存在已验证的在线可用物理
provider——reward 侧要求逐 0.5 s transition 的 mL 口径信号，而 FASTSim 按
[ADR 0032](0032-separate-online-proxy-and-offline-fastsim-energy.md) 不可逐 substep 重启。

因此采纳 **dual-metric**：proxy 保留为 reward 一致性指标与训练诊断，正式比较中继续
配对报告但一律按代理口径解释；支持物理能效声明的正式比较必须在同一完整执行轨迹上
附带离线物理复核（provider、车辆/环境配置、窗口映射显式 pin），各单位内独立配对、
永不混合或互标定；proxy 与物理复核方向冲突时报告分歧，不作物理能效声明。E-060
Gate F 保留为 proxy 口径历史裁定，不追溯修改，也不再作为物理节能证据引用。

离线复核进入默认 evaluation artifact 属实现改动，在线物理 reward provider 属新
实验族（新 λ 选择、新 gate、不继承 E-060 数值），两者均由后续 Issue 定义；本决策
不实施。FASTSim 的分辨率敏感性与 family 结构在 E-062 已显式登记，任何以 FASTSim
为 gate 的后续设计必须先 pin 并 stress 这些条件。

裁定依据与三方案对照：[Issue #113 Task 3 决策记录](../research/issue113-task-3-energy-metric-decision.md)；
证据：[E-061](../experiments/records/e-061-issue113-task-1-proxy-structural-proof-and-speed-attribution.md)、
[E-062](../experiments/records/e-062-issue113-task-2-fastsim-fixed-trace-review.md)。

规范归属：[Planning/evaluation protocol](../research/protocols/planning-and-evaluation.md)、
[Semantics](../research/semantics.md)、[Findings](../research/findings.md)。

> 本篇保存设计理由与历史决定；现行要求由上述 Protocol/Contract 拥有，读取路由见 [AGENTS](../../AGENTS.md)。
