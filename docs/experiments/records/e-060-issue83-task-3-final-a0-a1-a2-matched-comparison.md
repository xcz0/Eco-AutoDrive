# E-060 Issue #83 Task 3：final A0/A1/A2 matched evaluation / Gate F

[返回实验索引](../README.md) · [Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83) Task 3 / Task 4 ·
[E-059（Gate M，λ=1 进入 final A2）](e-059-issue83-task-1-reentry-lam1-multiseed-confirmation.md) ·
[E-058（Gate S3，λ=1 提名）](e-058-issue114-task4-candidate-revalidation.md) ·
[E-057（R0 训练产物）](e-057-issue114-task3-interior-stress.md) ·
[E-029（历史 A0 artifact，不作为比较基线）](e-029-issue81-a0-frozen-baseline.md) ·
[预注册 margins（F1–F7，已冻结）](../../research/issue83-task-2-final-evaluation-margins-plan.md)

**日期 / 类型 / 目的**：2026-10-09 / 正式 final A0/A1/A2 matched evaluation /
在 final clean commit 重跑 frozen planner 基线（A0），复用 E-057/E-058 的 A1（calibrated
R0）/A2（λ=1）产物，按 Task 2 预注册的 F1–F7 margins 裁定 Gate F，并完成 Task 4 provenance
收尾。

```text
Gate F（A2−A1，λ=1 vs calibrated R0，final checkpoint）：PASSED —— 7/7 条件满足
  F2：energy intensity −0.573 / −0.619 / −0.691 mL/km（−1.22% / −1.30% / −1.45%），
      3/3 seeds point < 0 且 CI 不含 0；seed 均值相对效应 −1.32% ≥ 0.1% 噪声界；
  方向复现：R1（S/SC 两 family 均 < 0，3/3）+ R2（LOO sign_change_count=0，3/3）
      + update-050 → final 同号单调加深（−0.30 量级 → −0.62 量级）；
  F3：progress/completion within margin（route_completion 仅 seed 0 CI 不含 0，
      arrival 13/16 与 A1 逐 seed 持平，max_step 截断逐值一致）；
  F4：无 speed/stopped collapse（stopped Δ=0；speed seed 均值 −3.46% < 10%）；
  F5：无 safety regression（collision/OOR/wrong-direction 0 vs 0，3/3）；
  F7：无缺失/失败 run（48/48 评测产物 available，全部 paired 16/16）。
  结论边界（#113 必带 caveat）：improvement 由降速载带（speed −3.4%~−3.7% vs
  intensity −1.2%~−1.5%，progress 代价 −0.27%~−0.43%），不得声称独立于降速机制的节能。
A1−A0（单独报告，§6 判据）：stable learned behavioral effect 存在——A1 较 frozen
  planner 速度上移（seed 均值 +1.98%）且能耗强度上移（+0.75%），progress/safety 不变；
  A1 是否「更好」不作为 A2/Gate F 的选择条件。
```

## 冻结契约与协议

与 E-057/E-058/E-059 相同的冻结契约：ADR 0040 Beta interior 参数化（`interior_epsilon=1e-4`）；
Issue #114 Task 1 scheduler-matched schedule（`update_count=100` 与
`scheduler_total_optimizer_steps=100` 单一 scheduler history）；canonical k=5（0.1 s × 5 =
0.5 s）；E-052 冻结 PPO control（epochs=1、max_gradient_norm=0.5、target_kl=0.006、
batch=minibatch=128、value_coefficient=0.1、gamma=0.9509900499000001、gae_lambda=0.7737809375、
lr=1.5e-4 cosine → ~0）；训练池 S/SC seeds 0–7、held-out S/SC seeds 16–23、runtime seed
760025、horizon 300、DDIM5（`ddim_stochasticity=0`）、deterministic Beta mean（
`policy_action_seed=null`）、execution-trace fuel proxy；bootstrap 0.95/10000/seed 0
（percentile，unit `matched_scenario_delta`，jointly-completed 交集）。

臂身份（预注册 §1 登记，protocol YAML 由本 Task 决定）：

```text
A0: final clean commit 重跑的 frozen planner 基线（guidance=none，无 policy checkpoint）
A1: calibrated R0（plannerrft_no_energy_calibrated_v1），training seeds {0,1,2}
A2: λ=1（plannerrft_energy_band_lam1_v1），training seeds {0,1,2}（E-059 Gate M 通过的唯一候选）
```

protocol `configs/experiments/comparison/calibrated-canonical-final-comparison.yaml`：arms
`{a0, a1, a2}`（a0 为唯一 frozen 臂，`reward_profile: null`）、contrasts `[[a0, a1], [a1, a2]]`
→ `a1-a0` / `a2-a1`；training/evaluation/bootstrap 块与
`calibrated-canonical-lam1-confirmation.yaml`（E-059，自身逐字来自 E-058）**逐字相同**（pin
测试断言 `protocol.training/evaluation/bootstrap == confirmation.*`），保证 E-057/E-058 复用
产物通过 `load_comparison` 的 matched resolved-conditions 硬校验。A0 评测由
`compose_a0_evaluation_config` 组装 `jobs/evaluation/no_traffic_heldout`（seed 760025、
S/SC 16–23、horizon 300、warmup 0、DDIM5、`num_scenarios=24`，runner 硬校验全契约）。

### 实现改动（Task 3 范围内，均有测试）

commit `3bb59d9`（clean，A0 rerun 与 analyze 的运行 pin）：新 protocol config + pin 测试
（`test_final_comparison_protocol_pins_frozen_baseline_and_matched_arms`：三块与 E-059
protocol 全等、frozen 臂声明、a1/a2 reward profile、contrasts、`compose_a0_evaluation_config`
seed 760025 且无 policy checkpoint、cadence 0.1 × 5 = 0.5 s、update_count/scheduler=100）。
无训练/评测/reward/PPO/cadence 语义改动；无新 gate runner（沿 E-058 D7 / Task 2 预注册
Non-goals，Gate F 在本记录内裁定）。

## 命令

```powershell
# A0 rerun（GPU，final clean commit 3bb59d9；runner 拒绝 frozen 臂的 checkpoint 参数）
just exp compare eval `
  --config configs/experiments/comparison/calibrated-canonical-final-comparison.yaml `
  --arm a0 `
  --output-dir outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison/heldout/a0-frozen-planner

# E-060 manifest（25 引用：baseline_evaluation_dir + 24 条目，手写；
# a1 条目指向 E-057（r0 改名 a1），a2 条目指向 E-058（lam1 改名 a2），
# update-025/update-050 checkpoint_hash 与 E-058/E-059 manifest 同 checkpoint 一致）
just exp compare analyze `
  --config outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison/comparison.yaml `
  --source-dir outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison `
  --output-dir outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison-analysis

# pin 测试 + 相关 protocol/A0 guard 测试
.venv/Scripts/pytest.exe tests/training/test_training_workflows.py tests/configuration/test_scalar_reward.py -q -m "not gpu and not simulator and not slow"   # 33 passed
just lint      # passed
just typecheck # 0 errors
```

## Matched guards（analysis 硬校验）

`load_comparison` 在 E-060 manifest 上整体重验并通过（无 loud-fail）：

- **baseline（A0）**：protocol 声明唯一 frozen 臂；`baseline_evaluation_dir` 指向本 commit
  重跑产物；`validate_evaluation` 核验 held-out pool / seed 760025 / horizon 300 / DDIM5 /
  `ddim_stochasticity=0`；无 policy checkpoint；
- **runs（24 条目）**：per-seed initial policy hash 跨臂一致（seed 0 `56bb97ab0551…`、
  seed 1 `f91a826f38a2…`、seed 2 `0febc900778c…`，与 E-038/E-040/E-048/E-057/E-058 一致）；
  frozen planner hash（`6a014ec56d24…`）前后不变；`probe_before` matched；同 seed 各臂
  resolved 训练条件除 reward/tracking 外全等（显式含 `update_count=100` 与
  `scheduler_total_optimizer_steps=100`）；held-out 契约逐 run 核验；update-025/050
  checkpoint pin hash 与评测 job 重算 hash 交叉核对（与 E-058 manifest 数值逐字一致）；
  arm 改名后 `training.reward_profile` 与 `protocol.arms[a1/a2].reward_profile` 匹配。

analyze pin `3bb59d9`（branch `issue83-engery-select`，`git_status_short=[]`）。

## A0 rerun（frozen planner 基线，本 commit 新产物）

`just exp compare eval --arm a0`（evaluate-frozen，pin `3bb59d9` clean；cuda:0 /
bf16-mixed / RTX A4000，serial，36.0 s，16/16 episodes completed）：

| arm | speed (m/s) | distance (m) | route compl. | arrival | stopped | col/OOR/WD | termination | intensity (mL/km) | total (mL) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| a0（frozen planner） | 9.989 (3.76–15.01) | 169.38 | 0.9460 | 13/16 | 0.0 | 0/0/0 | arrive 13 + max_step 3 | 47.047 | 7.839 |

与 E-029 历史 A0 artifact（同 frozen planner、同 held-out 契约）不做数值比较：E-060 的
A0 是当前 commit / 当前 artifact schema 的正式基线（预注册 §1 边界），非 E-029 的复用。

## A2−A1 final 结果（Gate F 主判据，deterministic Beta mean，16 场景）

final 方向计数（primary `energy_ml` 与 `energy_ml_per_km` 一致）：**lower 3/3、higher 0、
zero 0、unavailable 0**。

per-seed paired effect（mL/km，comparison−reference，scenario bootstrap 95% CI）与
update 轨迹：

| seed | update-025 | update-050 | final |
| --- | ---: | ---: | ---: |
| 0 | −0.038 [−0.049, −0.029] | −0.294 [−0.350, −0.238] | **−0.573 [−0.690, −0.457]** |
| 1 | −0.041 [−0.050, −0.032] | −0.290 [−0.344, −0.236] | **−0.619 [−0.736, −0.501]** |
| 2 | −0.038 [−0.048, −0.030] | −0.313 [−0.374, −0.252] | **−0.691 [−0.828, −0.555]** |

全部 CI 不含 0；25 → 50 → 100 同号单调加深（无方向反转）。**与 E-059 同 contrast 数值逐值
一致**（−0.573/−0.619/−0.691），A1/A2 产物复用的分析可复现性交叉确认。

final checkpoint 全指标（per seed，paired delta，CI 见 analysis.json）：

| metric | seed 0 | seed 1 | seed 2 | 相对效应（seed 均值） |
| --- | ---: | ---: | ---: | ---: |
| energy_ml_per_km | −0.573（CI 不含 0） | −0.619（不含 0） | −0.691（不含 0） | **−1.32%**（−1.22/−1.30/−1.45） |
| mean_speed_mps | −0.338（不含 0） | −0.341（不含 0） | −0.378（不含 0） | −3.46%（−3.36/−3.34/−3.68） |
| route_completion | −0.0040（**不含 0**） | −0.0026（含 0） | −0.0034（含 0） | −0.36%（−0.43/−0.27/−0.36） |
| stopped_fraction | 0.0 | 0.0 | 0.0 | Δ=0 |
| energy_ml | −0.129（不含 0） | −0.137（不含 0） | −0.158（不含 0） | −1.79%（−1.64/−1.73/−1.99） |
| distance_m | −1.069（不含 0） | −1.080（不含 0） | −1.320（不含 0） | −0.68%（−0.63/−0.63/−0.78） |

task / safety / termination（per arm × seed，16 场景）：A1 arrival 13/16 ×3、A2 arrival
13/16 ×3（逐 seed 持平，无下降）；max_step 截断 3 = 3（逐值一致）；stopped 0.0 全臂；
collision / OOR / wrong-direction 0/0/0 全臂全 seed；termination 结构逐 seed 一致
（arrive 13 + max_step 3）。

guidance 配对探针（`after`，fixed-context probe）：纵向 Beta mean delta
−0.284 / −0.268 / −0.296（seeds 0/1/2），横向 −0.002~+0.002 ≈ 0——learned effect 近纯纵向
（与 E-059 一致）。

## A1−A0 learned behavioral effect（单独报告，§6 判据）

A1−A0 无方向预设，只报告。final 方向计数（primary `energy_ml`）：**higher 3/3、lower 0、
unavailable 0**。

final per-seed paired effect（3/3 CI 不含 0 的指标加粗）：

| metric | seed 0 | seed 1 | seed 2 | seed 均值相对效应 |
| --- | ---: | ---: | ---: | ---: |
| energy_ml_per_km | **+0.105** [0.084, 0.125] | **+0.434** [0.332, 0.536] | **+0.519** [0.389, 0.648] | **+0.75%**（+0.22/+0.92/+1.10） |
| mean_speed_mps | **+0.071** [0.060, 0.081] | **+0.241** [0.181, 0.300] | **+0.281** [0.203, 0.358] | **+1.98%**（+0.71/+2.41/+2.81） |
| energy_ml | **+0.034** [0.009, 0.064] | **+0.090** [0.064, 0.119] | **+0.107** [0.085, 0.133] | +0.99%（+0.44/+1.15/+1.37） |
| route_completion | +0.0004（含 0） | +0.0016（含 0） | **+0.0022** [0.0002, 0.0042] | +0.15%（非 3/3） |
| stopped_fraction | 0.0 | 0.0 | 0.0 | Δ=0 |
| distance_m | +0.392（含 0） | **+0.674** | **+0.812** | +0.37%（非 3/3） |

**§6 裁定：stable learned behavioral effect 存在**——`energy_ml_per_km` /
`mean_speed_mps` / `energy_ml` 均满足 3/3 CI 不含 0 且 seed 均值相对效应 ≥ 0.1%（F1）。
方向：A1（calibrated R0，100 updates）较 frozen planner **速度上移、能耗强度上移**，幅度
随 seed 分化（seed 0 最小 +0.22%，seed 2 最大 +1.10%）；progress/arrival/stopped/safety
与 A0 持平（route_completion 仅 seed 2 CI 不含 0，+0.23%）。learned 分量随 update 增长
（intensity 相对效应：initial ≈ 0（CI 含 0）→ update-050 +0.31% → final +0.75%）。

结构说明：A1−A0 在 initial checkpoint 已有微小非零 delta（mean_speed +0.011 m/s，CI 不含
0；intensity CI 含 0）——这是 **initial policy 的 orthogonal guidance 相对纯 planner 的基线
偏移**（任何未训练 policy 检查点经 `orthogonal_policy` guidance 执行都与 guidance=none 的
A0 略有差异），不是 learned 分量；learned 分量由 initial → final 的加深轨迹分离（见上）。
a1−a0 的 guidance 配对探针结构性 unavailable（frozen 臂无 policy probe / training
summary），符合实现预期。

## Scenario-family 稳健性（R1/R2，final checkpoint）

**R1（family split，`energy_ml_per_km`）**：

| contrast | seed | S family | SC family |
| --- | --- | ---: | ---: |
| a2-a1 | 0 | −0.806 | −0.340 |
| a2-a1 | 1 | −0.854 | −0.384 |
| a2-a1 | 2 | −0.966 | −0.417 |
| a1-a0（只报告） | 0 | +0.111 | +0.098 |
| a1-a0（只报告） | 1 | +0.636 | +0.231 |
| a1-a0（只报告） | 2 | +0.780 | +0.257 |

a2-a1：3/3 seeds 两个 family 均 < 0 → **R1 PASS**（S 幅度约为 SC 的 2.2–2.3×，登记为量级
不对称，方向一致；与 E-059 数值一致）。a1-a0：两 family 均 > 0（3/3），只报告不设判据。

**R2（leave-one-scenario-out，`energy_ml_per_km`）**：a2-a1 3/3 seeds
`sign_change_count=0`，剔除任一场景后的 mean delta 范围 seed 0 [−0.593, −0.555]、seed 1
[−0.640, −0.596]、seed 2 [−0.714, −0.664] → **R2 PASS**。a1-a0 `sign_change_count=0`
（3/3，loo 范围全正），只报告。效应不由单一 scenario family 或少数 outlier episode 驱动
（update-025/050 同构，见 analysis.json robustness 块）。

## Gate F 判定（预注册 §4 映射）

| # | Gate F 条件（Issue 原文） | 操作化 | 证据 | 裁定 |
| --- | --- | --- | --- | --- |
| 1 | A2−A1 energy intensity < 0 | F2 条 1 | final −0.573/−0.619/−0.691 mL/km，3/3 point < 0；direction_counts lower 3/3 | ✓ |
| 2 | effect 超过 pre-registered minimum-effect/noise threshold | F2 三条 | 3/3 CI 不含 0 + seed 均值相对效应 −1.32% ≥ 0.1%（F1） | ✓ |
| 3 | direction 在 training seeds 与 matched scenarios 上复现 | F2 条 2 + R1 + R2 + 50→100 轨迹 | 3/3 CI；R1 S/SC 均 < 0（3/3）；R2 sign_change=0（3/3）；u050 → final 同号单调加深（−0.30 → −0.62 量级） | ✓ |
| 4 | safety 无 material regression | F5 | collision/OOR/wrong-direction 0 vs 0（3/3 seeds，逐场景配对）；termination 结构逐值一致 | ✓ |
| 5 | progress/completion 在 non-inferiority margin 内 | F3 | route_completion 仅 seed 0 CI 不含 0 且为负（1/3，非 3/3）；arrival 13/16 = 13/16（0/3 seeds 下降）；max_step 3 = 3 → 「系统性下降」双条件不成立，within margin | ✓ |
| 6 | speed/stopped 未越过 collapse margin | F4 | stopped_fraction paired delta = 0（3/3）；speed seed 均值相对下降 3.46% < 10% | ✓ |
| 7 | improvement 非较短距离 / 停车 / failed termination 所解释 | §5 confound 规则 | intensity 改善（−1.32%）大于 distance 收缩（−0.68%）；stopped Δ=0；arrival/max_step 逐值一致；route_completion 在 F3 margin 内 | ✓ |

**Gate F PASSED（7/7）**：在预注册 margins 下，λ=1 energy term 相对 calibrated R0 提供
超过噪声界的、跨 seed 与场景可复现的 energy-intensity improvement，且 progress/speed/
stopped/safety trade-off 均在 margin 内。

## Confound 检查（预注册 §5 逐条）

| Issue confound 条款 | 裁定 |
| --- | --- |
| total energy 降但 intensity 不降 | 未触发：intensity 与 total energy 同降（−1.32% / −1.79% seed 均值），主 metric intensity 满足 F2 |
| energy 降主要因 distance/progress 降 | 未触发：route_completion 在 F3 margin 内；distance 收缩 −0.68% 小于 intensity 改善 −1.32%（intensity 为 per-km metric，非 distance 差异的副产物；`energy_distance_m` 与 `distance_m` 同估计器，非独立 metric，沿 E-059 局限 #2） |
| energy 降主要因停车或大幅降速 | **降速载带成立但不触发 confound**：F4 未越界（speed −3.46% ≤ 10%）；#113 caveat 必带——不得声称独立于降速机制的节能 |
| collision / OOR / wrong-direction / failed termination 增加 | 未触发：0 vs 0；arrival/max_step 逐 seed 逐值一致 |
| 仅单 seed / 少数 scenarios 改善 | 未触发：3/3 CI + R1 + R2 |
| training reward 上升但 held-out 无改善 | 未触发：held-out 改善成立（training return 仅诊断，不裁定） |
| 50 update 有效、100 update 方向反转或 collapse | 未触发：u050 → final 同号单调加深，无反转 |

## 与预注册计划的偏离

1. **固定 policy-action seed stochastic 辅助诊断未运行**：预注册 §1 / Issue 均规定其
   「仅作辅助诊断，不参与 Gate F 裁定」，F1–F7 无任何判据引用 stochastic 评测；
   E-057/E-058/E-059 从未运行 stochastic evals。按最小范围收口，登记为未运行（非缺失：
   Gate F 裁定对象不含该评测，无 unavailable 语义）。
2. **臂改名**：manifest 条目从 E-059 复制后将 `r0`→`a1`、`lam1`→`a2`（预注册 §1「protocol
   YAML 由 Task 3 决定」授权；改名使 contrast key `a1-a0`/`a2-a1` 与预注册 §4/§6 术语一一
   对应）。`load_comparison` 按改名后的 `protocol.arms` 重新核验 reward profile 匹配，
   checkpoint hash 数值与 E-058/E-059 manifest 逐字一致。
3. 无其他偏离：A0 rerun 一次通过（无基础设施/语义失败，F7 重跑分支未触发）；manifest 一次
   通过 `load_comparison` 全部硬校验（无转录错误重试）；未调整任何 F1–F7 margin。

## 局限与结论边界

1. **降速载带（#113 必带 caveat）**：A2−A1 的 energy-intensity 降低（−1.2%~−1.45%）主要由
   降速载带（speed −3.3%~−3.7%，progress 代价 −0.27%~−0.43%）。**本记录的 Gate F PASSED
   不得解释或转述为「独立于降速机制的节能」**；是否值得追求非降速节能机制由 #113 /
   reward-structure 研究线决定。
2. **energy provider 是 execution-trace fuel proxy**（MetaDrive 代理模型），非真实动力系统能耗；
   绝对数值不外推，只支持 matched 条件下的相对比较。
3. **energy_distance_m 与 distance_m 同估计器**（paired delta 逐值相等），非独立 metric；
   主 metric 是 intensity（mL/km）。
4. **S vs SC 幅度不对称**（S ≈ 2.2–2.3× SC）：方向一致（R1）但幅度有结构性差异，不应把两个
   scenario family 混同解释。
5. **复用 provenance 边界**：A1/A2 训练与评测产物 pin 于 #114 的三个 commit
   （07b2860/986b015/73b9db4），跨 commit matched 由 `load_comparison` 硬校验保证；A0 是
   本 commit（3bb59d9）新产物。E-056（旧 policy family）、E-055（50-step schedule）数值不与
   本 contract 比较；E-029 历史 A0 artifact 不作为基线（schema/provenance 边界，预注册 §1）。
6. **CI 是 scenario bootstrap**（条件于固定 policy 与可用场景，48/48 paired 16/16），不是跨
   training seed 的总体区间；跨 seed 一致性由 3/3 方向 + R1/R2 承载。
7. **A1−A0 的 initial 偏移**：未训练 policy 的 orthogonal guidance 相对 guidance=none 存在
   微小基线偏移（speed +0.11%，intensity CI 含 0）；A1 的 learned 分量须由 initial → final
   轨迹解读（本记录已分离报告）。
8. **Gate F 裁定边界**：仅针对 λ=1（`plannerrft_energy_band_lam1_v1`）vs calibrated R0 在
   冻结 contract 下的 final checkpoint；不外推到其他 λ、其他 reward family、其他 cadence 或
   带交通场景。

## Task 4 provenance 收尾（Issue Task 4 清单逐项）

| 项 | 值 |
| --- | --- |
| Git commit / branch / clean | A0 rerun 与 analyze pin `3bb59d9`（`3bb59d90db4249a6c28eb30114f3fd9a7034adb5`），branch `issue83-engery-select`，`git_status_short=[]`（runtime_metadata 双产物核验）；本记录的提交 hash 于实验索引与 Issue #83 Task 3 完成评论登记（记录不自引用） |
| 环境 / 设备 / 依赖 | Windows-10 (10.0.26200)；Python 3.10.20；torch 2.12.1+cu126；lightning 2.6.6；metadrive 0.4.3（third_party editable）；pydantic 2.13.5；cuda:0 RTX A4000、bf16-mixed、serial、12 torch threads |
| resolved config / overrides | A0：`heldout/a0-frozen-planner/resolved_config.yaml`（`jobs/evaluation/no_traffic_heldout`，guidance=none，无 policy checkpoint）；A1/A2 训练：E-057/E-058 各 run `resolved_config.yaml`（protocol training overrides：`components/resources=rtx_a4000`、target_kl=0.006、batch=128、transitions=8、update_count=100、lr=1.5e-4、epochs=1、mgn=0.5、scheduler steps=100、value_coefficient=0.1、gamma=0.9509900499000001、gae_lambda=0.7737809375、gradient_diagnostics=true）；评测：E-057/E-058 held-out `resolved_config.yaml`（`no_traffic_heldout_policy`，action_mode=mean，policy_action_seed=null） |
| planner/policy checkpoint identity | frozen planner `checkpoints/DP-Origin/model.pth`（hash `6a014ec56d24…`，训练前后不变）；initial policy per seed `56bb97ab0551` / `f91a826f38a2` / `0febc900778c`（跨臂一致）；final policy：A1 `94c6dfd93a79` / `c239cc354925` / `4b5486d86ee1`，A2 `55eb67f1d6a1` / `24d37e198cc0` / `b5750d4d95f3`；中间 checkpoint（update-024/049）hash 见 manifest（与 E-058/E-059 逐字一致） |
| scenario / map / training / noise / action seeds | held-out S/SC map seeds 16–23（16 场景）；training 池 S/SC seeds 0–7；training seeds {0,1,2}（runtime seed = training seed）；replay_id=0；评测 runtime/noise seed 760025（per-scenario generator）；action：deterministic Beta mean（无 policy RNG）；训练 rollout noise/policy_action seeds 见各 run summary（per-environment 显式枚举） |
| canonical cadence | 0.1 s × 5 = 0.5 s（`eco_planner.contracts` pin，protocol guard 全等比较） |
| 命令 | 见「命令」节（compare eval a0 / compare analyze / pytest / lint / typecheck） |
| artifact 路径 | 见「产物」表 |
| 测试 / preflight | pin 测试 + `tests/training/test_training_workflows.py` + `tests/configuration/test_scalar_reward.py`：33 passed（`-m "not gpu and not simulator and not slow"`）；`just lint` passed；`just typecheck` 0 errors；`load_comparison` matched guards 整体验证通过（含 baseline 与 24 run 条目） |
| failure 状态 | 无失败：A0 rerun 一次完成（16/16 completed）；A1/A2 复用产物 E-057/E-058 已裁定完整；全部 paired contrast 16/16 available、unavailable 0；无 missing run（F7 无插补、无重试） |
| 可支持的结论 | 冻结 contract 下 λ=1 vs calibrated R0 的 final energy-intensity improvement（超过预注册噪声界、3/3 seeds + R1/R2 可复现、trade-off 在 margin 内）；A1 存在 stable learned behavioral effect（速度/强度上移） |
| 不可支持的结论 | 独立于降速机制的节能（#113 caveat）；真实动力系统能耗结论（fuel proxy 边界）；跨 policy family / cadence / scheduler / λ 外推；带交通场景外推；A1「更好/更差」作为 A2 选择条件（Issue 禁止） |

## 产物

| 项 | 路径 / pin |
| --- | --- |
| 预注册 margins（F1–F7） | `docs/research/issue83-task-2-final-evaluation-margins-plan.md`（Task 2 冻结，原文未改） |
| protocol | `configs/experiments/comparison/calibrated-canonical-final-comparison.yaml`（pin 测试 `test_final_comparison_protocol_pins_frozen_baseline_and_matched_arms`；commit `3bb59d9`） |
| A0 rerun | `outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison/heldout/a0-frozen-planner/`（16 场景 + summary/resolved_config/runtime_metadata/analysis/report/figures；pin `3bb59d9` clean） |
| E-060 manifest | `outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison/comparison.yaml`（baseline + 24 条目） |
| E-060 analysis | `outputs/studies/scalar-reward/e-060-issue83-task-3-final-matched-comparison-analysis/`（analysis.json 含 a1-a0/a2-a1 全 checkpoint paired effects + robustness 块、report.md、figures `a1-a0-seed-effects` / `a2-a1-seed-effects` / `training-rewards`；pin `3bb59d9` clean） |
| A2 训练 runs + evals | `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/`（pin `73b9db4`） |
| A1 训练 runs + evals + shared initial | `outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/`（train pin `07b2860`，eval pin `986b015`） |
| 机械诊断（15-run） | `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-diagnostics/` |
| 上游裁定 | E-058（Gate S3，λ=1 提名）；E-059（Gate M PASSED 7/7）；E-056（旧 family λ=4/8 FAILED，历史证据）；E-055（λ=4/8 nomination 已失效） |
