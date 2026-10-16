# Issue #113 Task 3 — energy metric keep / dual / replace 裁定与迁移方案

**日期 / 类型 / 目的 / 状态**：2026-10-16 / 研究决策记录（Issue #113 Task 3，Gate P3）/ 基于 Task 1（E-061）与 Task 2（E-062）证据，对 `MetaDriveFuelProxyProvider` 作 keep / dual / replace 裁定，并界定迁移 contract、历史证据继承与后续 Issue 边界 / **已完成（2026-10-16；裁定见下文）**。

**决策性质**：本记录不产生新仿真运行或新 artifact；输入证据为已登记的 E-061/E-062 离线诊断及其冻结配置。长期设计理由保存于 [ADR 0042](../adr/0042-adopt-dual-energy-metrics-and-restrict-proxy-claims.md)；现行要求的改动落入 [Planning/evaluation protocol](protocols/planning-and-evaluation.md)，解释边界落入 [Semantics](semantics.md)，证据结论落入 [Findings](findings.md)。本决策不实施任何代码或评测管线改动；不重训、不重跑仿真、不修改 E-060 Gate F 历史裁定。

**研究条件**：E-060 final matched evidence——A0 frozen planner（git pin `3bb59d9`）、A1 calibrated R0、A2 λ=1，final 112 episodes（held-out S/SC map seeds 16–23、noise seed 760025、canonical k=5、DDIM5 stochasticity=0、deterministic Beta mean、no-traffic、horizon 300）。复用条件、policy hash 与 provenance 由 [E-061](../experiments/records/e-061-issue113-task-1-proxy-structural-proof-and-speed-attribution.md) 与 [E-062](../experiments/records/e-062-issue113-task-2-fastsim-fixed-trace-review.md) 记录拥有。

## Gate P3 输入摘要

### P1 结论（E-061）

1. **Parity 通过**：112/112 final episodes 的 trace→公式→summary 链浮点级一致（逐步误差 ≤6.98e-16、恒等式残差 ≤7.41e-16），独立重算 ΔI 与 E-060 登记 F2 值逐位一致。
2. **Proxy 是纯速度型估计量**：`I = 32.5 × Σ w_i exp(0.036 v_i)`，直接输入只有子步结束速度与位移。E-060 A2−A1 的 Δln I 由距离加权速度水平项解释 **98.96%–100.09%**，形状项 ≤1.04% 且符号不稳定（seed 0 CI 含 0、LOO 有翻转）；排序反事实证明加减速**时序**完全不可见。无超出公式输入的独立信息。
3. **当前 proxy 不能支持「独立于降速的真实节能」表述**——该命题在 proxy 下不可验证（而非证伪）。

### P2 provider 证据与失败清单（E-062）

三 provider 消费同一距离保持 1.0 s 窗口均值重构轨迹，A2−A1（Δln，16/16 pairs per seed per provider）：

| Seed | Δln proxy [CI] | Δln fastsim [CI] | Δln physics [CI] | Δ time-mean speed m/s |
| --- | --- | --- | --- | ---: |
| 0 | −0.012688 [−0.01504, −0.01034] | **+0.056099** [+0.03867, +0.07397] | **+0.052358** [+0.02516, +0.07821] | −0.338 |
| 1 | −0.012925 [−0.01522, −0.01059] | +0.0014336 [−0.02709, +0.02714] | −0.034112 [−0.07326, +0.00347] | −0.341 |
| 2 | −0.014332 [−0.01699, −0.01165] | −0.0010427 [−0.03293, +0.02905] | −0.040467 [−0.08903, +0.00651] | −0.378 |

- **主对比方向一致性：不通过。** proxy 三 seed 稳定负号（mean −0.0133）未获任一物理模型跨 seed 支持；seed 0 fastsim 与 physics 同向强烈为正而 proxy 为负（CI 不含 0）——A2 的降速在含怠速/辅助载荷的物理口径下转为每公里能耗上升。分歧呈 family 结构（S family seed 1/2 physics −0.08 ~ −0.10 与 proxy 同向；SC family 各 seed 反向）。
- **对照 A0→A1**：seed 1/2 三 provider 同向为正（训练策略在物理口径下每公里能耗高于 frozen planner），**proxy 低估该差距 5–8 倍**；seed 0 physics 微负、CI 跨 0。
- **能力边界（合成不变量，与真实分歧机制自洽）**：proxy 对动能循环率敏感性精确 0（fastsim +22.1%、physics +38.9%）、对 10 s idle dwell 燃料差精确 0（fastsim 57.6 kJ、physics 22.2 kJ）、对制动强度近盲（45.50 vs 45.27 mL/km；fastsim 794.1 vs 648.6、physics 572.3 vs 451.0 Wh/km）。

失败 / 偏离 / 敏感性清单：

1. 主分辨率 1.0 s 重放 **112/112 可跟随**，无排除；窗口距离相对残差 max 3.71e-16；确定性前向仿真 0.18 ms/episode。
2. 次分辨率 0.5 s（仅描述性，永不进入 Gate）：110/112，`a1/seed_0/s21` 与 `a1/seed_0/sc21` fail-closed 记录并从该 provider 0.5 s 统计排除。
3. 首次正式运行（尾部残窗保留真实短时长构造）主分辨率仅 89/112 且排除集中于 a2（23 个中 15 个）；经用户裁定改为残组并入最后完整窗口后重跑，23 个失败证明全部为映射伪影。被取代运行保留于 `outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review-superseded-run1-tail-partial/`。
4. 冻结配置注释中的预检 census 数字（111/88/75）因探针脚本未保留而不可复现，已作流程缺口显式登记。
5. **FASTSim 判定对窗口分辨率敏感**（seed 0 符号在 0.5 s 描述分辨率下翻转），且分歧呈 S/SC family 结构；两者均已在 E-062 显式呈现。

## 三方案对照与裁定

| 方案 | 证据对照 | 裁定 |
| --- | --- | --- |
| KEEP-PROXY（限定用途） | 与 P2 反证冲突：proxy 的 A2−A1 节能方向未获任一物理模型支持，且对 A0→A1 低估物理差距 5–8 倍。继续以其单独承载正式评测 energy 主指标，等于把已被反证的方向声明留在主表。 | **否决**为独立方案 |
| DUAL-METRIC | 在线 proxy 便宜、与 reward 严格一致、可逐子步在线计算；离线物理复核已证明可对完全相同 trace 确定性重放（112/112）并覆盖 proxy 结构性缺失项（怠速、动能循环、制动）。两套指标各自单位/版本/窗口明确，并列报告可同时保留 reward 一致性信息与物理审计能力。 | **采纳** |
| REPLACE / REDESIGN | 当前证据不足以免标定替换方向：fastsim 与 physics 彼此跨 seed 不一致（seed 1/2 符号相反），FASTSim 判定对窗口分辨率敏感，physics proxy 是 study-local 极简实现（无再生制动、恒定路径效率），且无任何已验证的**在线可用**物理 provider（reward 侧要求逐 0.5 s transition 给出 mL 口径，FASTSim 按 ADR 0032 不可逐 substep 重启）。#113 非目标亦排除在本 Issue 内实施。 | **否决**为当前裁定；登记为后续新实验族（见候选 B） |
| （停止规则路径）证据不足暂缓 | 该路径适用于 FASTSim 无法重算的情形。实际主分辨率 112/112 全部可跟随，分歧是实测结果而非重算能力缺口，暂缓不成立。 | **不适用** |

**裁定：DUAL-METRIC。** 具体含义：

1. `MetaDriveFuelProxyProvider` 保留为 reward E 分量的一致性指标与训练诊断：不改 [training protocol](protocols/training.md) 的 reward 定义、band 校准数值与 λ profile 家族，不重训 A2。
2. 正式比较中的 proxy 配对报告保留，但一律按**代理口径**解释（见声明语言规则）。
3. 支持物理能效声明的正式比较 **MUST** 附带同一完整执行轨迹的离线物理复核（FASTSim 及/或可审计的纵向动力学 proxy，配置显式 pin）；proxy 与物理复核方向冲突时 MUST 报告分歧并保留双方数值，此时不得作出物理能效声明。规范条文见 [planning/evaluation protocol](protocols/planning-and-evaluation.md#指标及累计窗口)。
4. E-060 Gate F PASSED 保留为 **proxy 口径历史裁定**（原协议、原 provider 下的 F1–F7 不追溯修改），不再作为物理节能证据引用。
5. REPLACE（在线物理 reward provider）登记为后续新实验族研究（候选 B），其证据继承边界见下文。

## 迁移 contract

### 现行规范的改变（protocol change，随本记录生效）

改动条文与位置：

| 改动 | 位置 |
| --- | --- |
| Dual-metric 要求：物理能效声明的离线复核义务、单位隔离、冲突语义、proxy 声明边界 | [Planning/evaluation protocol](protocols/planning-and-evaluation.md)「指标及累计窗口」FASTSim 段后新增段；「Matched comparison」paired energy 解释补一句 |
| Proxy 概念的解释边界强化（强度由距离加权速度水平决定；未获物理复核支持时不得表述为物理节能） | [Semantics](semantics.md)「执行、道路信息与能耗」表 |
| E-061/E-062 新证据条目；E-060/E-019 条目的对应修订 | [Findings](findings.md) |
| 长期设计理由 | [ADR 0042](../adr/0042-adopt-dual-energy-metrics-and-restrict-proxy-claims.md)（ADR 0032 边界保留，本决策是其预留「接入 reward／默认 evaluation 需新研究决定」决策口的裁定） |

声明语言规则：

| 观察 | 允许的表述 | 禁止的表述 |
| --- | --- | --- |
| proxy 强度下降（如 E-060 A2−A1） | 「execution proxy 口径的强度下降，由距离加权速度水平（降速）载带」 | 「节能」「能效改善」「物理能耗下降」及任何独立于降速的机制声明 |
| proxy 与离线物理复核同向 | 「在所 pin 的离线模型配置下获得方向一致的离线模型支持」 | 「实车节能」「已证明真实能耗下降」（离线模型支持 ≠ 实车实证） |
| proxy 与离线物理复核冲突 | 「proxy 方向未获该物理模型配置支持，分歧按 family／分辨率登记」 | 以任一模型单独定论；以冲突反推「A2 真实物理能耗上升」的正面结论（物理模型自身也是模型选择） |

### 不随本决策改变的部分

- Training protocol 的 reward 公式、E 分量（只使用 execution fuel proxy）、`plannerrft_*` profile 家族与 band 校准数值不变；λ=1 的提名与确认仍是**该 reward/provider 下**的有效历史证据。
- ADR 0032 的在线/离线分离边界与「不得相加／替换」要求不变；本决策未把 FASTSim 接入在线 reward。
- E-060 Gate F 的 F1–F7 margins 与裁定不追溯修改；E-058/E-059 候选选择 provenance 不变。

### 实现边界（归后续 Issue，本决策不实施）

- 离线物理复核进入默认 evaluation artifact 或标准评测工作流：实现改动，归候选 A。
- `physics_proxy` 从 study-local 实现升格为 domain `EnergyMetricProvider`：新实现 + 新契约，归候选 A/B 的设计决定。
- 任何在线物理 reward provider 及其重训：新实验族，归候选 B。

## 历史证据继承表

| 证据 | 支持什么 | 继承状态 |
| --- | --- | --- |
| [E-019](../experiments/records/e-019-metadrive-native-energy-proxy-comparison.md) | native `step_energy` 与 execution-trace proxy 的边界 | **保留** |
| E-038 / #94（E-033–E-041） | proxy reward 表示下的 objective 可辨识性、有效更新与 positive-control 机制 | **保留**为机制证据；不证明物理节能方向 |
| E-055 / E-056 | 旧 scheduler/policy family 的 dose-response 与失败 | **保留**为历史条件证据；不跨 family 作数值比较 |
| E-057–E-060 | 原 reward/provider 下的机械稳定性、行为分离与 proxy 效果；E-060 Gate F | **保留**为 proxy 口径 + 机械稳定性证据；E-060 不再作为物理节能证据引用 |
| E-061 / E-062 | proxy 结构证明与物理方向反证 | 本决策直接依据 |
| 任何物理能耗结论 | — | **须重新验证**；现有证据不包含 |
| 新 provider 下的 reward 最优 λ | — | **须重新验证**；λ=1 最优性是 provider 相对的 |
| 新 provider 下的 Gate F / 跨 family 数值 | — | **须重新验证**；离线重算旧策略 ≠ 已按新 reward 训练 |

## 后续 Issue 边界（本记录只登记，不创建）

### 候选 A — 正式评测的离线物理 co-metric 实现与配置 stress

- 将 E-062 式离线复核接入正式评测工作流（或建立标准离线复核 study 入口），使 dual-metric 要求在默认路径可执行；属实现改动，须独立验收。
- 对 FASTSim 配置做显式 stress 后才允许其进入任何 gate：窗口分辨率敏感性（seed 0 符号在 0.5 s 翻转）、车辆配置唯一性（当前仅 `2012_Ford_Fusion` conventional）、warm-engine 副本假设、平路/无坡度限制、ramp 前缀与窗口映射构造。
- 决定 physics proxy 是否升格为 domain provider；如升格，须补齐 fail-closed 契约与 regression 锁。

### 候选 B — 在线物理 reward provider 可行性研究（新实验族）

研究是否存在对加减速/制动/怠速敏感、可在 0.5 s transition 内在线计算且成本可接受的物理 reward 信号。**新 provider 与新 reward 构成新实验族：λ 选择、gate 与 E-060 数值均不继承**。任何新 provider 提案必须至少说明：

1. 输入 trace 与车辆/环境参数、起止窗口、运动学执行的解释限制（并非真实油门/刹车控制）；
2. energy total / intensity 的单位与零距离/停车/怠速处理；online/offline 一致性、distance parity、异常/缺失的 fail-closed 语义；
3. 参数来源、场景覆盖、物理敏感性测试、确定性/复现性、运行成本预算与可审计的 calibration 依据；
4. matched A0/A1/A2 评价仍须联合报告速度、进度、距离、到达/失败、停车和安全；**离线重算旧策略 ≠ 已按新 reward 训练**；
5. 新评测需要独立于旧 λ 选择过程的场景/seed 设计——历史 E-060 是 reused-selection evidence（held-out 16 场景曾用于 E-058/E-059 候选选择/验证），不自动变成 blind test。

本候选同时承接 #114 登记的 λ4/λ8 sustained negative longitudinal saturation 现象作为 reward-structure 研究输入。

## 复现与产物指针

本决策无新增运行，故不新增 experiment record；E-061/E-062 已在 records 登记且[实验 README](../experiments/README.md) 不维护历史索引。

- E-061：冻结配置 `configs/experiments/energy-proxy/analysis.yaml`；复现命令 `just exp energy proxy run --source-dir outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison --output-dir outputs/studies/energy-proxy/e-061-issue113-task-1-speed-attribution`；产物 `outputs/studies/energy-proxy/e-061-issue113-task-1-speed-attribution/`。
- E-062：冻结配置 `configs/experiments/energy-proxy/fastsim-review.yaml`（含 post-freeze 偏差注记）；复现命令 `just exp energy fastsim run --source-dir outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison --output-dir outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review`；产物 `outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review/` 及 superseded 首次运行目录。
- 统计口径（两记录共享）：per-training-seed scenario bootstrap（0.95 / 10000 / seed 0 / percentile / matched_scenario_delta），training seeds 永不重采样，无跨 seed 总体区间。

## Gate P3 裁定核验

- [x] P1 结论：proxy 纯速度型、改善 ≥99% 由距离加权速度水平载带、不支持独立于降速的节能表述（E-061）。
- [x] P2 provider 证据/失败列表：三 provider 方向不一致、能力边界不变量、fail-closed 与 superseded 清单（E-062）。
- [x] keep/dual/replace 推荐与理由：**DUAL-METRIC**（三方案对照表）。
- [x] 后续 Issue 边界：候选 A（评测 co-metric 实现与 FASTSim 配置 stress）、候选 B（在线物理 reward provider 新实验族）。
- [x] 可保留/须重验历史证据清单：见历史证据继承表。
- [x] 交付研究记录并将摘要回填 #80（2026-10-16 评论）。
