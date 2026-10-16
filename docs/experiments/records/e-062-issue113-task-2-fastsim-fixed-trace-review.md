# E-062 Issue #113 Task 2 — FASTSim 固定轨迹复核 + 最小物理 proxy

**日期 / 类型 / 目的 / 状态**：2026-10-16 / 诊断研究（Issue #113 Task 2，Gate P2）/ 在 E-060 固定执行轨迹上做离线物理模型评审：FASTSim 固定轨迹窗口重放与最小纵向动力学 proxy 对既有 fuel proxy 的方向一致性、能力边界与重放可靠性检验 / **已完成（2026-10-16；分析计划于运行前冻结，结果与 Gate P2 裁定见下文）**。

**代码与运行身份**：commit `ab6bbc2`（`ab6bbc2d32ee2de4d1e5dc84e79f58852d8a5c5d`）、branch `main`、dirty workspace——本诊断全部新代码即未提交 diff（`src/eco_planner/experiments/energy_proxy/{fastsim_review,fastsim_report,physics_proxy}.py`、`tests/analysis/test_energy_proxy_fastsim_review.py`、`configs/experiments/energy-proxy/fastsim-review.yaml`、本记录）+ 修改 `scripts/experiments.py`（CLI 注册）。source-mode 离线重算，不产生新仿真运行；证据源为 E-060 既有 artifact（只读）。

**环境与模型**：本机 `.venv`（Python 3.10、fastsim 3.1.0 pinned by `uv.lock`）；不加载 checkpoint，不运行 simulator。三个 provider 消费**同一**距离保持窗口均值重构轨迹：`metadrive_fuel_proxy`（域内在线公式，mL/km）、`fastsim_fuel_energy`（固定轨迹窗口重放，Wh/km）、`physics_proxy`（最小纵向动力学，Wh/km）。**单位永不混合或互标定**，只做 provider 内配对 Δln 与符号比较。

**研究条件**：与 E-061 相同的 E-060 final matched evidence——A0 frozen planner、A1 calibrated R0（E-057）、A2 λ=1（E-058），final label 112 episodes（16 场景 × {A0, A1×3 seeds, A2×3 seeds}），noise seed 760025。相关规范：[ADR 0032](../../adr/0032-separate-online-proxy-and-offline-fastsim-energy.md)、[Planning/evaluation protocol](../../research/protocols/planning-and-evaluation.md)、Issue #113、E-061 记录。

## 冻结的分析计划（预注册；先于任何正式结果产生）

冻结机制：机器可读配置 `configs/experiments/energy-proxy/fastsim-review.yaml` + 本节自然语言计划；运行前定稿。冻结要点：

- **轨迹映射**：原始 0.1 s 执行轨迹为运动学 waypoint 输出（子步加速度伪影至 ~47 m/s²），物理上不可跟随 → 全部 provider 消费同一**距离保持窗口均值**重构：主分辨率 1.0 s（标准 drive-cycle 分辨率），次分辨率 0.5 s（canonical planner cadence，仅描述性，**永不进入 Gate P2**）。
- **FASTSim 车辆与边界**：`2012_Ford_Fusion.yaml`（pinned by 域 provider 测试）、平路、295.15 K、初始海拔 121.92 m（锁定回归配置）。**Warm-engine 假设**（已批准并显式注册）：轨迹为已在巡航车辆的 mid-trip 片段（初速 ≈10.6 m/s），冷启动功率斜坡（`fc.pwr_out_max_init_watts` 21.75 kW，6 s 后解锁且在轨迹温和需求下永不解锁）以 study-local 车辆副本替换为全功率 `fc.pwr_out_max_watts` 130.5 kW；域 provider 本身不修改。
- **初始化前缀**：FASTSim 从静止启动 → 0.9 m/s²、0.5 s 步进物理 ramp 至首窗速度 + 4 窗 hold；前缀燃料经逐步累计记账（`history[-1] − history[n_prefix−1]`）从一切报告燃料中排除，并以 `history[-1] ≈ final state` 断言守卫累计语义。
- **Physics proxy**：最小可审计纵向动力学——点质量功率平衡，恒定路径效率 η = 传动 0.875 × 峰值发动机 0.36，制动完全耗散（无再生），aux 700 W 与 idle 0 W 按 η 计入燃料；参数全部取自 pinned 车辆资源（测试锁定）。Study-scoped，不改域契约。
- **Fail-closed 语义**：FASTSim 不可跟随的 episode 逐个记录该 provider 失败消息并从其配对统计排除，不静默替换、不坏样本跳过；每 provider 分母显式报告。
- **统计**：与 E-061 完全一致——per-training-seed scenario bootstrap（0.95 / 10000 / seed 0 / percentile / matched_scenario_delta），training seeds 永不重采样，无跨 seed 总体区间；对照 A0 按 A1 每个 seed 共享配对。S/SC family 分层与 leave-one-scenario-out 为稳健性描述。
- **合成诊断 profile**（与真实 held-out 结果分开报告）：7 个从静止出发、可行加速度的 profile（cruise、fluctuation fast/slow、stop_and_go±dwell、hard/gentle brake），经**未修改域 provider**（stock 冷启动车辆）重放；三项 provider 不变量——fluctuation 率敏感性（proxy 恒 0）、idle 可见性（proxy 恒 0）、制动强度响应。
- **Gate P2 三项裁定**（运行后回填）：(1) 固定轨迹 FASTSim 重放可靠性；(2) 主对比 A2−A1 上三 provider Δln 方向一致性；(3) provider 能力边界（怠速/波动率/制动强度可见性）。

## 配置与命令

- 冻结配置：`configs/experiments/energy-proxy/fastsim-review.yaml`（含 post-freeze 偏差注记，见下）。
- 复现命令：`just exp energy fastsim run --source-dir outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison --output-dir outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review`
- 新增测试：`tests/analysis/test_energy_proxy_fastsim_review.py`（15 项：冻结配置锁、窗口构造精确性、physics proxy 手算/KE 恒等式/制动耗散、ramp 前缀、FASTSim 窗口测量确定性与 fail-closed、诊断 profile 可行性与不变量、physics 参数对活体资源锁定、配对统计 fail-closed 分母与 baseline 配对）。

### 计划偏离（显式登记）

1. **预检探针不可复现（流程缺口）**：冻结配置注释中的 census 事实（1.0 s+warm 111/112、0.5 s+warm 88/112、1.0 s+stock 75/112）来自 2026-10-15 只读预检，但探针脚本未保留，其窗口尾部构造无法复原；正式管线下各候选构造（残窗真实时长 / 尾部并入 / 尾部拉伸）均不能复现这组数字（stock 75/112 与平路道路负载物理不符，说明探针构造含中等量级尖峰）。该缺口已在冻结配置追加 dated 注记并在此登记。
2. **尾部残窗构造偏离（首次正式运行后、经用户裁定）**：首次正式运行（2026-10-16，已保存为 `outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review-superseded-run1-tail-partial/`）将尾部不满整窗的残组保留真实短时长（如 0.1 s），使原始轨迹的运动学量化伪影以轨迹末端加速度尖峰形式重新进入（例：0.1 s 内 11.7→14.83 m/s ≈ 31 m/s²）——主分辨率仅 89/112 可跟随，且排除集中于 a2（23 个中 15 个），与处理组相关，会使 Gate P2 的 FASTSim 配对统计偏倚。经用户裁定改为**残组并入最后一个完整窗口**（合并窗保持累计距离与累计真实时长；距离保持与真实平均速度不变，不发射任何短于分析窗口的窗点）。三候选中唯一不歪曲物理内容者；冻结参数无一改动。重跑后主分辨率 112/112 全部可跟随，证明首次运行的 23 个失败全部为映射伪影而非物理不可行。
3. 预注册阶段已登记的预期可行性发现（原始 0.1 s 不可跟随 → 窗口映射 / warm engine / ramp 前缀）按计划执行，非偏离。

## 结果（2026-10-16 正式运行回填）

全部输出见 `outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review/`（`summary.json` / `report.md` / `analysis.json` / `figures/`）。

### 重放 census（裁定 1 输入）

- **主分辨率 1.0 s：112/112 可跟随**，无排除、无 unavailable；窗口距离相对残差 max **3.71e-16**（构造在 >1e-9 时 raise）；FASTSim 平均 0.182 ms/episode（确定性前向仿真）。
- 次分辨率 0.5 s（仅描述性）：110/112，不可跟随 a1/seed_0/s21 与 a1/seed_0/sc21（fail-closed 记录并从该 provider 0.5 s 统计排除）；残差 max 3.19e-16。
- Proxy 口径交叉核对：窗口均值下 proxy Δln（见下）与 E-061 原始步长分解（−0.012047/−0.012938/−0.014412）方向一致、量级差 ≤5%（窗口平滑的预期效应）。

### 主对比 A2−A1（final；difference = comparison − reference；16/16 pairs per seed per provider）

| Seed | Δln proxy [CI] | Δln fastsim [CI] | Δln physics [CI] | Δ time-mean speed m/s |
| --- | --- | --- | --- | ---: |
| 0 | −0.012688 [−0.01504, −0.01034] | **+0.056099** [+0.03867, +0.07397] | **+0.052358** [+0.02516, +0.07821] | −0.338 |
| 1 | −0.012925 [−0.01522, −0.01059] | +0.0014336 [−0.02709, +0.02714] | −0.034112 [−0.07326, +0.00347] | −0.341 |
| 2 | −0.014332 [−0.01699, −0.01165] | −0.0010427 [−0.03293, +0.02905] | −0.040467 [−0.08903, +0.00651] | −0.378 |

- 跨 seed 一致性：proxy 三 seed 同负号（mean −0.0133）；**fastsim 与 physics 均跨 seed 不一致**（fastsim +0.0561/+0.0014/−0.0010；physics +0.0524/−0.0341/−0.0405），fastsim/physics 的 seed 1/2 CI 与 physics seed 2 CI 跨 0。
- 三 provider 全部同号仅 seed 2；seed 0 中 fastsim 与 physics **同向强烈为正**而 proxy 为负——A2 的降速在物理模型口径下反而增加每公里能耗。
- Family 分层：S family 中 physics 种子 1/2 达 −0.080/−0.102（与 proxy 同向），SC family 各 seed 均为正（+0.012 ~ +0.066）；方向分歧呈 family 结构。
- 次分辨率 0.5 s（描述性，seed 0 仅 14/16 pairs）：fastsim −0.101/−0.053/−0.058、physics −0.112/−0.057/−0.063，与 proxy 同向——但 0.5 s 下动能循环内容与配对分母均不同，且 seed 0 fastsim 符号相对主分辨率翻转，**FASTSim 判定对窗口分辨率敏感**，此观察不改变以主分辨率为准的 Gate P2 输入。

### 对照对比 A0→A1（final；A0 按 A1 seed 共享配对）

proxy +0.0025/+0.0091/+0.0107，fastsim +0.0005/+0.0542/+0.0697，physics −0.0082/+0.0654/+0.0876：seed 1/2 三 provider 同向为正（训练策略在物理口径下每公里能耗高于 frozen planner，且 proxy 低估该差距 5–8 倍），seed 0 分歧（physics 微负、CI 跨 0）。

### 合成诊断 profile（与真实 held-out 结果分开；经未修改域 provider，stock 冷启动车辆）

| 不变量 | proxy | fastsim | physics |
| --- | ---: | ---: | ---: |
| fluctuation 率敏感性（同均值/同边际分布，快慢 2×） | **0.0** | +22.1% | +38.9% |
| idle 可见性（10 s dwell 燃料差） | **0.0**（精确） | 57.6 kJ | 22.2 kJ |
| 制动强度响应（hard vs gentle，Wh/km 或 mL/km） | 45.50 vs 45.27 | 794.1 vs 648.6 | 572.3 vs 451.0 |

三项冻结不变量全部按预注册成立：proxy 对动能循环率与怠速时长**结构性不可见**（精确 0），对制动强度近盲；fastsim/physics 对三者均响应。

### Gate P2 三项裁定

1. **固定轨迹重放可靠性：通过。** 主分辨率 112/112 可跟随、距离残差浮点级（3.7e-16）、确定性可离线复算（0.18 ms/episode）；warm-engine 副本与 ramp 前缀记账显式注册；次分辨率 110/112 且仅描述性。
2. **主对比方向一致性：不通过。** A2−A1 上 proxy 的稳定节能方向（三 seed Δln ≈ −0.013）**未获任一物理模型支持**：fastsim 与 physics 跨 seed 均不一致，且 seed 0 两者与 proxy 反向（+0.056/+0.052，CI 不含 0）；A2 的降速驾驶（Δ time-mean ≈ −0.34 ~ −0.38 m/s）在含怠速/辅助载荷的物理口径下转为每公里能耗上升。proxy 与物理模型在 S family（seed 1/2 physics −0.08 ~ −0.10）与 0.5 s 描述分辨率下局部同向，但不构成跨 seed 稳定支持。
3. **Provider 能力边界：proxy 结构性缺失在真实证据域被激活。** 合成不变量（idle 0、fluctuation 率 0、制动强度近盲）与真实配对分歧机制一致：A2 的速度水平下移同时降低气动/滚动项并提高每公里辅助/怠速时间与低负荷效率损失，后者超出 proxy 输入空间。fastsim/physics 覆盖这些项。

## 产物

- `outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review/`：`summary.json`（census、Gate P2 输入、provenance）、`report.md`、`analysis.json`（逐 episode 三 provider 证据）、`figures/primary-provider-ln-intensity-delta.{png,svg}`、`figures/primary-scenario-fastsim-ln-intensity-delta.{png,svg}`。
- `outputs/studies/energy-proxy/e-062-issue113-task-2-fastsim-fixed-trace-review-superseded-run1-tail-partial/`：被取代的首次运行（尾部残窗真实时长构造，89/112），仅作偏离登记证据保留。

## 结论边界

本诊断为只读离线重算：不重训、不重跑仿真、不修改 E-060 Gate F 历史裁定。结论只覆盖：(a) 固定轨迹 FASTSim 重放（2012_Ford_Fusion、warm-engine 副本、1.0 s 距离保持窗口映射、平路 295.15 K）与最小 physics proxy 在 E-060 final matched evidence 上的**方向一致性**——proxy 的 A2−A1 节能方向未获物理模型支持；(b) 三 provider 的能力边界（怠速、动能循环率、制动强度）。

不证明：真实车辆能耗、任一物理模型自身正确性（符号一致本应是 offline-model support 而非实车证据，分歧构成对 proxy 方向声明的反证而非对 A2 物理能耗的正面结论）、其他速度域/场景域外推、λ 调度机制。FASTSim 判定存在分辨率敏感性（seed 0 符号在 0.5 s 描述分辨率下翻转）与 family 结构（S 同向、SC 反向），均已在报告中显式呈现。fuel proxy 的 keep/dual/replace 决策属 Task 3，超出本诊断范围。
