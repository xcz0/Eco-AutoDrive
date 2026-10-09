# Issue #83 Task 2 — final A0/A1/A2 matched evaluation margins 预注册

状态：**预注册（已冻结）**。本文件判据在 Task 3（final A0/A1/A2 matched evaluation）的任何
正式评测或分析运行前锁定，此后保留原文不改；执行偏离只写入对应 E-xxx 实验记录的
「与预注册计划的偏离」。Gate F 在 Task 3 实验记录内逐条裁定（沿 E-055 Gate D / E-056
Gate M / E-058 D7 先例，不新增 gate runner）。

任务边界权威来源：[Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83) Task 2 /
Gate F（Parent #80）。上游输入：E-058（Gate S3 PASSED，λ=1 唯一提名）、E-059（Gate M
PASSED 7/7，λ=1 进入 final A2）。

## 1. 任务定义（来自 Issue #83 Task 2）

在看到 final A2 matched results 前冻结：

- matched evaluation noise bound；
- energy minimum-effect threshold；
- progress/completion non-inferiority margin；
- speed/stopped collapse margin；
- safety regression rule；
- bootstrap / paired comparison procedure；
- missing/failure run handling。

规则锁定后不根据结果调整（Issue 冻结研究 contract 明确禁止「根据 final A2 结果事后修改
non-inferiority / noise margins」）；Task 3 的实验记录逐条引用本文件裁定 Gate F。

比较对象（Issue 冻结 contract，臂身份在此登记，protocol YAML 由 Task 3 决定）：

```text
A0: final clean commit 重跑的 frozen planner 基线（不依赖 E-029 历史 artifact schema）
A1: calibrated R0（plannerrft_no_energy_calibrated_v1），training seeds {0,1,2}
A2: λ=1（plannerrft_energy_band_lam1_v1），training seeds {0,1,2}（E-059 Gate M 通过的唯一候选）
```

主评测 deterministic Beta mean；held-out S/SC map seeds 16–23；固定 policy-action seed
stochastic evaluation 仅作辅助诊断，不参与 Gate F 裁定。

## 2. 预注册信息边界（本预注册为何仍然有效）

Task 3 的 A1/A2 训练与 held-out 评测产物复用 E-057/E-058（E-059 已对同一 A2−A1 contrast
执行 final 分析），因此 A2−A1 效应量级在预注册时已可观测（intensity −1.22%~−1.45%、
speed −3.3%~−3.7%、progress −0.27%~−0.43%、stopped Δ=0、safety 0/0/0、arrival 13/16 与
A1 持平，E-058/E-059）。为消除事后调参空间，本文件遵守：

- F1/F2 逐字沿用既有预注册判据（E-031 实测噪声界，E-054 c2 制度化；E-058 §4.2 提名
  要求），不重述推导、不开新区间；
- F3 逐字沿用 E-058 §4.2「系统性完成率下降」的操作化（该口径曾落选 λ=2），不新设数字
  margin；
- 唯一新设数值是 F4 的 speed collapse 界 10%：高于全部已观测 λ 臂（λ=1~8）的 held-out
  speed 代价（最大 λ=8 seed 均值 −0.83 m/s，约 −8%，E-058），即「collapse」与「大幅降速
  trade-off」的显式分界——该界用于捕获 collapse 级行为退化，不用于 trade-off 幅度裁定
  （幅度由 #113 caveat 与 F3 承载）；同时距候选 λ=1 的观测值（−3.3%~−3.7%）≥ 2.7×，
  裁定对具体取值不敏感；
- F5 为严格化规则（任意新增 safety 事件即 regression），不含数值区间；
- 全部通过/不通过边界均远离已知观测值，不在已知结果附近选择阈值。

## 3. 冻结 margins（F1–F7）

### F1 — matched evaluation noise bound

**0.1% 相对效应**（E-031 实测：matched held-out 聚合差异 ≤ 0.1%；E-054 c2 制度化沿用）。
任何 paired 指标的 |相对效应| < 0.1% 按「噪声水平内不可分辨」解释。

### F2 — energy minimum-effect threshold（A2−A1 主判据）

metric `energy_ml_per_km`（execution-trace fuel proxy，mL/km），**final（100-update）
checkpoint**，逐字沿用 E-058 §4.2 提名口径：

1. 3/3 training seeds point estimate < 0（节能方向）；
2. 3/3 seeds scenario bootstrap 95% CI 不含 0；
3. seed 均值相对效应 ≥ 0.1%（F1 噪声界）。

三条全部满足 = effect 超过 pre-registered minimum-effect/noise threshold（Gate F 条件 2）。

### F3 — progress/completion non-inferiority margin（A2−A1）

逐字沿用 E-058 §4.2「系统性完成率下降」操作化。**系统性下降 = 同时满足**：

- route_completion final 效应 3/3 seeds CI 不含 0 且为负；
- arrive_dest 计数低于 A1 于 ≥2/3 seeds。

同时成立 → non-inferiority fail（Gate F 条件 5 不满足）；其余情形（point 为负但 CI 部分
含 0，或 arrival 持平 / 仅单 seed 下降）按 within margin 如实报告。max_step 截断计数
逐 seed 对齐报告（arrival 下降的直接表现，E-058 λ=2 先例）。

### F4 — speed/stopped collapse margin（A2−A1）

- **stopped_fraction**：任一 training seed 较 A1 增加（paired delta > 0）→ collapse；
- **mean_speed**：seed 均值相对下降 > 10% → collapse。

任一触发 → Gate F 条件 6 不满足。stopped 在当前 matched 评测中为稀有事件计数（两臂恒 0），
任何增加都是行为信号；speed 降幅 ≤ 10% 的降速属「降速载带」trade-off（#113 必带 caveat
报告），不构成 collapse。

### F5 — safety regression rule（A2−A1）

任一 training seed 出现**新增** collision / out-of-road / wrong-direction（vs A1 同 seed
matched 计数）→ materially regress → Gate F 条件 4 不满足。确定性 Beta mean 评测 + 固定
runtime seed 下 0→k 无噪声解释空间；单 seed 单场景亦触发（宁可误杀）。触发而其余条件全过
时，收口用语为「safety regression」，不得收口为「no improvement」。

### F6 — bootstrap / paired comparison procedure

逐字沿用 E-054→E-059 契约：

- 配对单位 `matched_scenario_delta`（comparison−reference），只在 jointly-completed 场景
  交集上计算，available / unavailable 计数显式报告；
- percentile scenario bootstrap：**95% CI / 10000 resamples / seed 0**，per (training
  seed, checkpoint)；无跨 seed 合并 CI，跨 seed 一致性由 3/3 方向 + direction_counts
  承载（CI 是 scenario bootstrap，条件于固定 policy 与可用场景——沿 E-059 局限 #5 口径）；
- 相对效应 per seed 计算（vs reference 臂 16 场景均值），与 0.1% 噪声界的比较用 seed 均值；
- 报告粒度（Issue Task 3 最终统计清单）：每个 training seed、跨 seed aggregate（point +
  direction_counts）、scenario type（S/SC）、paired scenario-level difference、A1−A0 与
  A2−A1 两个 contrast、bootstrap interval、failure / missing run 明细；
- 稳健性块（沿 E-059 §3 判据口径）：**R1**（S/SC family split：两 family 的 mean paired
  energy-intensity delta 均 < 0，3/3 seeds）与 **R2**（leave-one-scenario-out：
  sign_change_count=0，3/3 seeds × 任一 scenario）作用于 **A2−A1** final checkpoint、
  metric `energy_ml_per_km`；
- A1−A0 无方向预设：R1/R2 数值只报告，不设符号判据。

### F7 — missing / failure run handling

沿 E-056 / E-058 D10 与 E-059 语义：

- 训练或评测崩溃即停止：保留 log / partial checkpoints / `runtime_metadata.json` /
  `resolved_config.yaml`，不删目录、不调参重试（失败证据是一等产物）；
- 不插补：intersection < 16/16 时如实报告可用率，不伪造缺失场景；
- 任一臂 / seed / checkpoint 产物缺失或 unavailable → 该 contrast 标记 unavailable，
  对应 Gate F 条件按**未满足**处理（fail-closed），收口为「evaluation unavailable
  （缺失明细见记录）」，既不裁定通过、也不推断为 negative result；
- A0 rerun 因基础设施失败（OOM / IO 等，不触及评测语义）中断 → 保留日志与 partial 产物，
  修复后重跑并在记录登记重跑原因；评测语义内失败（policy 执行路径非有限 / 异常崩溃）→
  按 F7 停住收口，不重跑；
- 不通过跳过场景 / 坏样本剔除 / 缺失值伪造 / 静默重试掩盖失败。

## 4. Gate F 逐条件操作化（Issue 原文 → 判据 → 证据源）

| Gate F 条件（Issue 原文） | 操作化 | 证据源（Task 3 analysis 块） |
| --- | --- | --- |
| A2−A1 energy intensity < 0 | F2 条 1（3/3 seeds point < 0） | `a2-a1` final paired effects |
| effect exceeds pre-registered minimum-effect/noise threshold | F2 三条全部 | paired effects + 相对效应（seed 均值） |
| direction reproduces across training seeds and matched scenarios | F2 条 2 + R1 + R2 | paired effects + robustness 块 |
| safety does not materially regress | F5 | arm outcomes safety 块 |
| progress/completion stays within pre-registered non-inferiority margin | F3 | paired route_completion + arrival / termination counts |
| speed/stopped behavior does not cross the pre-registered collapse margin | F4 | paired mean_speed_mps / stopped_fraction |
| improvement is not explained by shorter distance, stopping, or failed termination | §5 confound 规则 | energy / distance / stopped / termination 全指标 |

## 5. Confound 判定规则（Issue「统一标记为 confounded / failed trade-off」逐条操作化）

| Issue confound 条款 | 操作化 |
| --- | --- |
| total energy 降但 energy intensity 不降 | 主 metric 是 intensity（mL/km）；total energy（`energy_ml`）与 `energy_distance_m` 仅辅助。intensity 不满足 F2 时，无论 total energy 如何，不得声称 energy improvement |
| energy 降主要因 distance/progress 降 | route_completion 超出 F3 margin → 自动 confounded（Gate F 条件 5 已 fail）；margin 内时报告 distance / energy_distance 降幅与 intensity 改善的比例（`energy_distance_m` 与 `distance_m` 同估计器，非独立 metric，沿 E-059 局限 #2） |
| energy 降主要因停车或大幅降速 | F4 触发即 confounded + Gate F 条件 6 fail；speed 降幅 ≤ 10% 的降速载带不触发，但必须以 #113 caveat 报告，**不得声称独立于降速机制的节能**（λ=1 机制已知为降速载带：speed −3.4%~−3.8% vs intensity −1.2%~−1.45%，progress 代价 −0.26%~−0.42%） |
| collision / OOR / wrong-direction / failed termination 增加 | F5 + termination 结构逐 seed 对比（arrive_dest / max_step 计数 vs A1）；arrival 下降或 max_step 截断增加 = failed termination 增加，与 F3 联动 |
| 仅单 seed / 少数 scenarios 改善 | F2 的 3/3 条件 + R1 + R2（沿 E-059 判据） |
| training reward 上升但 held-out 无改善 | 不以 training return 裁定（沿 E-058 §4.2）；training return 仅诊断报告 |
| 50 update 有效、100 update 方向反转或 collapse | final 裁定对象是 100-update checkpoint；update-050 → final 轨迹一并报告，point estimate 异号（任一 seed）= 方向不稳定，Gate F 条件 3 不满足 |

## 6. A1 vs A0 — learned behavioral effect 报告判据（单独报告）

- **stable learned behavioral effect** = 任一 paired 指标（`energy_ml_per_km` /
  `mean_speed_mps` / `route_completion` / `stopped_fraction` / `energy_ml`）满足 3/3
  seeds CI 不含 0 且 seed 均值相对效应 ≥ 0.1%（F1）；
- 否则报告「no measurable learned behavioral effect（≤ 噪声界）」（沿 E-031 口径）；
- A1 是否「更好」不作为 A2 / Gate F 的选择条件（Issue 明确；A1−A0 单独报告）。

## 7. 预设失败分支与收口用语

| 情形 | 预设收口 |
| --- | --- |
| F2 不满足（CI 含 0 / 效应 < 0.1% 噪声界 / 方向不正） | 「effect below pre-registered minimum-effect/noise threshold」，Gate F FAILED，不放宽 |
| F3 fail | 「final non-inferiority 不满足（progress/completion 系统性下降）」 |
| F4 fail | 「speed/stopped collapse margin 越界」 |
| F5 fail | 「safety regression」 |
| update-050 → final 方向反转 | 「direction instability at final checkpoint」，Gate F 条件 3 不满足 |
| 任一臂产物缺失 / unavailable | 「evaluation unavailable（缺失明细）」，fail-closed，不裁定通过、不推断 negative |
| A0 rerun 崩溃 | 按 F7 区分基础设施 / 语义失败；语义失败停住收口 |
| 判据间冲突或未预见情形 | 停止裁定，登记 Issue 并交使用方决策；不引入本文件之外的新标准 |

## 8. Non-goals

- 不执行任何训练 / 评测 / 分析（Task 3 另行执行；A0 rerun 需要 GPU 与沙箱外执行）；
- 不在看到 final 结果后调整 F1–F7 任何 margin；
- 不新增 gate runner / 阈值配置代码（沿 E-058 D7：Gate 在实验记录内裁定）；
- 不重选候选（A2 = λ=1，E-058/E-059 裁定）、不重训、不改 reward / cadence / energy
  provider（#113 独立研究）；
- 不把 training return 作为裁定依据；
- 不在本任务内裁定 Gate F（Task 3 记录内裁定）。
