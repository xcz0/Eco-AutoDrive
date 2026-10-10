# E-059 Issue #83 Task 1 re-entry：λ=1 candidate multi-seed long-run confirmation / Gate M

[返回实验索引](../README.md) · [Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83) Task 1（re-entry）·
[E-056 首次 Task 1（Gate M FAILED）](e-056-issue83-task-1-candidate-multiseed-confirmation.md) ·
[E-058 候选提名（Gate S3，λ=1）](e-058-issue114-task4-candidate-revalidation.md) ·
[E-057 R0 训练产物](e-057-issue114-task3-interior-stress.md) ·
[预注册计划](../../research/issue83-task-1-reentry-lam1-confirmation-plan.md)

**日期 / 类型 / 目的**：2026-10-09 / 正式 candidate multi-seed long-run confirmation（re-entry）/
对 #114/E-058 提名的唯一候选 λ=1（`plannerrft_energy_band_lam1_v1`）对照 calibrated R0
（`plannerrft_no_energy_calibrated_v1`）执行 Issue #83 Task 1 的 Gate M 七项裁定。

```text
Gate M（λ=1）：PASSED —— 7/7 项满足
  3/3 seeds 完成、无 boundary collapse / KL runaway / invalid action；
  energy-intensity 效应 3/3 seeds 为负（CI 不含 0），S/SC 两 family 与
  leave-one-scenario-out 稳健（预注册 R1/R2），50→100 同号单调加深，
  trade-off 可解释（降速载带），provenance 完整。
  λ=1 进入 #83 final A2；E-056 的 λ=4/8 FAILED 裁定保持原样。
```

## 冻结契约与协议

与 E-056/E-057/E-058 相同的冻结契约：ADR 0040 Beta interior 参数化（`interior_epsilon=1e-4`，
α,β 恒 > 1+ε）；Issue #114 Task 1 scheduler-matched schedule（`training.update_count=100` 与
`ppo.scheduler_total_optimizer_steps=100` 单一 scheduler history，cosine 1.5e-4 → ~0）；
canonical k=5（0.1 s × 5 = 0.5 s）；E-052 冻结 PPO control（epochs=1、max_gradient_norm=0.5、
target_kl=0.006、batch=minibatch=128、value_coefficient=0.1、gamma=0.9509900499000001、
gae_lambda=0.7737809375）；`ppo.gradient_diagnostics=true`（measurement only）；训练池
S/SC seeds 0–7、held-out S/SC seeds 16–23、runtime seed 760025、horizon 300、DDIM5、
deterministic Beta mean、execution-trace fuel proxy；bootstrap 0.95/10000/seed 0
（percentile，unit `matched_scenario_delta`）。

protocol `configs/experiments/comparison/calibrated-canonical-lam1-confirmation.yaml`：arms
`{r0, lam1}`、contrast `[r0, lam1]`；training/evaluation/bootstrap 块与
`calibrated-canonical-candidate-revalidation.yaml`（E-058）**逐字相同**（pin 测试断言
`protocol.training/evaluation/bootstrap == revalidation.*`）。**本 protocol 下不执行任何新训练
或新评测**（预注册计划 D1）：λ=1 训练 runs 与 held-out evals 为 E-058 产物（pin `73b9db4`），
R0 训练 runs 为 E-057 产物（pin `07b2860`），R0/shared-initial held-out evals pin `986b015`；
机械诊断为 E-058 `-diagnostics`（15-run，含全部 6 个相关 run，mc_draws=4096、mc_seed=1000003）。

### 实现改动（Task 1 re-entry 范围内，均有测试）

commit `7548706`（clean，E-059 analyze 运行 pin）：`analysis/evaluation.py` 新增
`paired_robustness`（contrast effect 附加 `robustness` 块：per-map-family paired delta 统计 +
leave-one-scenario-out 均值范围与符号翻转计数，纯统计无重采样，样本 < 2 为 null）；
新 protocol config + pin 测试；contracts 注册（`docs/agents/contracts/experiments.md`）；
预注册计划。无训练/评测/reward/PPO/cadence 语义改动。

## 命令

```powershell
# E-059 manifest（24 条目，手写；r0 条目指向 E-057，lam1 条目指向 E-058，
# update-025/update-050 checkpoint_hash 与 E-058 manifest 同 checkpoint 一致）
just exp compare analyze `
  --config outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation/comparison.yaml `
  --source-dir outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation `
  --output-dir outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation-analysis
```

## Matched guards（analysis 硬校验）

`load_comparison` 在 E-059 manifest（24 条目）上整体重验并通过（无 loud-fail）：per-seed
initial policy hash 跨臂一致（seed 0 `56bb97ab0551…`、seed 1 `f91a826f38a2…`、seed 2
`0febc900778c…`，与 E-038/E-040/E-048/E-057/E-058 一致）；frozen planner hash 前后不变；
`probe_before` matched；同 seed 各臂 resolved 训练条件除 reward/tracking 外全等（显式含
`update_count` 与 `scheduler_total_optimizer_steps`）；held-out scenario pool、seed、horizon、
sampler、`ddim_stochasticity=0` 对照协议核验；update-025/050 checkpoint pin hash 与评测 job
重算 hash 交叉核对。E-059 analyze pin `7548706`（branch `issue83-engery-select`，
`git_status_short=[]`）。

## 训练与机械健康（6/6 完成；数值取自 E-058 `-diagnostics` artifacts）

6 个 run（r0/lam1 × seeds {0,1,2}）均完整完成 100 updates，无 crash、无失败 seed、无
invalid/non-finite 动作（E-057/E-058 已裁定 Gate S2；本表为 #83 Gate M 引用复核）。

| run | initial hash | final hash | min α/β | analytic KL median/max | MC KL median/max | k3 median | probe boundary mass after | u75–100 纵向 mean | ep. length |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r0-seed-0 | 56bb97ab0551 | 94c6dfd93a79 | 1.531/1.933 | 2.00e-5 / 1.51e-4 | 1.86e-5 / 1.80e-4 | 2.02e-5 | 0.0388 | +0.042 | 8.0 → 8.0 |
| r0-seed-1 | f91a826f38a2 | c239cc354925 | 1.974/1.574 | 2.19e-5 / 1.81e-4 | 1.90e-5 / 2.15e-4 | 2.03e-5 | 0.0378 | +0.172 | 8.0 → 8.0 |
| r0-seed-2 | 0febc900778c | 4b5486d86ee1 | 1.997/1.449 | 4.03e-5 / 4.36e-4 | 3.95e-5 / 4.52e-4 | 3.79e-5 | 0.0608 | +0.213 | 8.0 → 8.0 |
| lam1-seed-0 | 56bb97ab0551 | 55eb67f1d6a1 | 1.527/2.000 | 2.37e-5 / 2.72e-4 | 2.48e-5 / 2.89e-4 | 2.41e-5 | 0.0396 | −0.235 | 8.0 → 8.0 |
| lam1-seed-1 | f91a826f38a2 | 24d37e198cc0 | 1.916/1.587 | 2.64e-5 / 1.91e-4 | 2.75e-5 / 1.85e-4 | 2.67e-5 | 0.0376 | −0.088 | 8.0 → 8.0 |
| lam1-seed-2 | 0febc900778c | b5750d4d95f3 | 1.919/1.452 | 3.93e-5 / 2.99e-4 | 4.08e-5 / 3.11e-4 | 4.02e-5 | 0.0605 | −0.071 | 8.0 → 8.0 |

（probe boundary mass before 均为 0.020996；`boundary_distance=0.05`；ceiling 0.2 全部满足。）

PPO / behavior 诊断要点（E-058 `-diagnostics`，全部 6 runs）：

- **analytic Beta KL + MC/k3 cross-check**：三估计量同量级且互相印证（上表 median 列比值
  ≤ 1.08），max ≤ 4.52e-4 ≪ target_kl 0.006，无 KL runaway；`kl_early_stop_fraction=0`。
- **policy ratio / clip fraction**：ratio p95 median ≤ 1.0137、ratio change ≤ 0.0089、
  clip fraction median = 0.0（全部 6 runs）。
- **per-group gradient diagnostics**（六组，measurement only）：critic 组占主导
  （shared_trunk_critic median 43.0–47.7、value_head_critic 56.4–62.2），actor 组温和
  （actor_head_policy median 1.60–2.20、actor_head_entropy 0.048–0.054），与 E-052/E-057
  已登记的 critic/shared-trunk coupling 结构一致；pre-clip total norm median 71.0–76.4。
- **parameter delta（final vs initial）**：actor_head 0.031–0.041、value_head 0.094–0.097、
  shared_trunk 3.12–3.25；两臂同 seed 的 shared_trunk/value_head delta 几乎相同、
  actor_head 因 reward profile 不同而分化——更新幅度与冻结 control 预期一致。
- **Beta mean / variance / concentration / boundary**：final 纵向（dim1）Beta mean lam1
  −0.237/−0.087/−0.084 vs r0 +0.045/+0.176/+0.208；concentration 维持 4.0–5.0（无收窄
  塌缩），variance 0.16–0.19；min α/β 全程 > 1+ε（上表）。
- **guidance 分布**（E-059 guidance contrast，fixed-context probe 配对 lam1−r0）：`before`
  （matched-initial 守卫）= 0；`after` 纵向 Beta mean delta −0.284/−0.268/−0.296
  （seeds 0/1/2），横向 −0.002~+0.002 ≈ 0——λ=1 的 learned effect 是几乎纯纵向的 guidance
  下移，横向不动。
- **episode length / reset frequency**：first/last median 8.0 → 8.0（全部 6 runs，训练
  horizon 8 transitions/episode 语义下无异常 reset）。

## Held-out 结果（E-059 analysis，deterministic Beta mean，16 场景）

final 方向计数：**lower 3/3、higher 0、unavailable 0**。

per-seed paired energy-intensity effect（mL/km，comparison−reference，scenario bootstrap
95% CI）与 50→100 轨迹：

| seed | update-025 | update-050 | final |
| --- | ---: | ---: | ---: |
| 0 | −0.038 [−0.049, −0.029] | −0.294 [−0.350, −0.238] | **−0.573 [−0.690, −0.457]** |
| 1 | −0.041 [−0.050, −0.032] | −0.290 [−0.344, −0.236] | **−0.619 [−0.736, −0.501]** |
| 2 | −0.038 [−0.048, −0.030] | −0.313 [−0.374, −0.252] | **−0.691 [−0.828, −0.555]** |

全部 CI 不含 0；25 → 50 → 100 同号单调加深（无方向反转、无 behavior collapse）。

final checkpoint 的任务 / safety / energy（per arm × seed，16 场景均值）：

| arm-seed | speed (m/s) | distance (m) | route compl. | arrival | stopped | col/OOR/WD | termination | intensity (mL/km) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| r0-0 | 10.060 (3.93–14.92) | 169.77 | 0.9464 | 13/16 | 0.0 | 0/0/0 | arrive 13 + max_step 3 | 47.152 |
| lam1-0 | 9.722 (4.12–14.85) | 168.70 | 0.9424 | 13/16 | 0.0 | 0/0/0 | arrive 13 + max_step 3 | 46.579 |
| r0-1 | 10.229 (3.98–15.07) | 170.05 | 0.9477 | 13/16 | 0.0 | 0/0/0 | arrive 13 + max_step 3 | 47.481 |
| lam1-1 | 9.888 (3.83–15.04) | 168.97 | 0.9451 | 13/16 | 0.0 | 0/0/0 | arrive 13 + max_step 3 | 46.861 |
| r0-2 | 10.270 (3.98–15.07) | 170.19 | 0.9483 | 13/16 | 0.0 | 0/0/0 | arrive 13 + max_step 3 | 47.566 |
| lam1-2 | 9.891 (3.95–15.06) | 168.87 | 0.9449 | 13/16 | 0.0 | 0/0/0 | arrive 13 + max_step 3 | 46.874 |

（update-050 与 update-025 结构相同：arrival 13/16、stopped 0、safety 0/0/0、termination
一致，见 analysis.json。）

lam1−r0 相对效应（final，per seed）：energy intensity **−1.22% / −1.30% / −1.45%**；
total execution-trace proxy energy −0.129 / −0.137 / −0.158 mL（−1.64% / −1.73% / −1.99%）；
energy distance −0.63% / −0.63% / −0.78%；mean speed **−3.36% / −3.34% / −3.68%**；
route completion −0.43% / −0.27% / −0.36%（seed 0 CI 不含 0，seeds 1/2 CI 含 0）；
stopped Δ=0。

## Scenario-family 稳健性（预注册判据 R1/R2，final checkpoint）

**R1（family split）**：per-seed mean paired energy-intensity delta（mL/km）——

| seed | S family | SC family |
| --- | ---: | ---: |
| 0 | −0.806 | −0.340 |
| 1 | −0.854 | −0.384 |
| 2 | −0.966 | −0.417 |

3/3 seeds 两个 family 均 < 0 → **R1 PASS**。（S 幅度约为 SC 的 2.2–2.3×，登记为量级不对称，
方向一致；update-025/050 同构。）

**R2（leave-one-scenario-out）**：3/3 seeds `sign_change_count=0`；剔除任一场景后的 mean
delta 范围 seed 0 [−0.593, −0.555]、seed 1 [−0.640, −0.596]、seed 2 [−0.714, −0.664]，
全部为负 → **R2 PASS**。效应不由单一 scenario family 或少数 outlier episode 驱动
（update-025/050 的 family/LOO 同样 `sign_change_count=0`，见 analysis.json robustness 块）。

## Gate M 判定

| # | 条件 | 证据 | 裁定 |
| --- | --- | --- | --- |
| 1 | 3/3 seeds 完成；失败 seed 不删除 | 6/6 runs 完整 100 updates（E-057/E-058），无失败 seed | ✓ |
| 2 | 3/3 无 boundary collapse / KL runaway / invalid action | boundary mass ≤ 0.061（ceiling 0.2）；min α/β ≥ 1.527/1.449；KL max ≤ 4.52e-4 ≪ 0.006；无 invalid/non-finite | ✓ |
| 3 | energy-intensity effect 方向在 seed aggregate 中一致 | final −0.573/−0.619/−0.691 mL/km，3/3 CI 不含 0；direction_counts lower 3/3 | ✓ |
| 4 | effect 非单一 family / 少数 outlier 驱动 | R1：S/SC 均 < 0（3/3）；R2：LOO sign_change_count=0（3/3） | ✓ |
| 5 | checkpoint 50→100 无方向反转 / behavior collapse | −0.294/−0.290/−0.313 → −0.573/−0.619/−0.691 同号单调加深；机械健康全维持 | ✓ |
| 6 | progress / speed / stopped / safety trade-off 可解释 | 效应由降速载带（speed −3.3%~−3.7% vs intensity −1.2%~−1.45%，progress −0.27%~−0.43%，stopped Δ=0，safety 0/0/0，arrival/termination 与 R0 逐值一致）；机制与 #113 一致 | ✓ |
| 7 | provenance 完整、commit clean | 双 pin 复用（07b2860/986b015/73b9db4）+ E-059 manifest `load_comparison` 整体重验 + analyze pin `7548706` clean | ✓ |

**Gate M PASSED（7/7）。λ=1 进入 #83 final A2。**

## 与预注册计划的偏离

无执行偏离。不执行新训练 / 评测 / diagnose 属预注册 D1 的设计本身（产物复用边界，§2），
不是偏离；robustness 判据只作用于 final、update-025/050 仅诊断，按预注册 §3 执行。
E-059 manifest 一次通过 `load_comparison` 全部硬校验（无转录错误重试）。

## 局限与结论边界

1. **降速载带（#113 必带 caveat）**：λ=1 的 energy-intensity 降低（−1.2%~−1.45%）主要由
   降速载带（speed −3.3%~−3.7%，progress 代价 −0.27%~−0.43%）；final A2 结论解释不得声称
   独立于降速机制的节能。
2. **energy_distance_m 与 distance_m 同估计器**（paired delta 逐值相等），不是独立 metric；
   energy intensity（mL/km）是主 metric。
3. **S vs SC 幅度不对称**（S ≈ 2.2–2.3× SC）：方向一致（R1）但幅度有结构性差异，final A2
   解释时不应把两个 scenario family 混同。
4. **复用 provenance 边界**：训练 / 评测产物 pin 于 #114 的三个 commit
   （07b2860/986b015/73b9db4），跨 commit matched 由 `load_comparison` 硬校验保证，非单
   commit 产物；E-056（旧 policy family）与 E-058/E-059（ADR 0040 family）数值不可比；
   E-055（50-step schedule）不与本 contract 的任何 checkpoint 比较。
5. **CI 是 scenario bootstrap**（条件于固定 policy 与可用场景），不是跨 training seed 的
   总体区间；跨 seed 一致性由 3/3 方向与 R1/R2 判定承载。
6. **Gate M ≠ Gate F**：本记录不裁定 energy term 的 independent improvement；final
   A0/A1/A2 matched evaluation 与 margins 预注册（Task 2/3）未执行。
7. E-057 记录表格存在已知 min α/β 三位小数笔误（E-058 已登记），本表数值一律取
   artifact。

## 产物

| 项 | 路径 / pin |
| --- | --- |
| 预注册计划 | `docs/research/issue83-task-1-reentry-lam1-confirmation-plan.md`（判据 R1/R2 于 analyze 运行前锁定） |
| protocol | `configs/experiments/comparison/calibrated-canonical-lam1-confirmation.yaml`（pin 测试 `test_lam1_confirmation_protocol_pins_frozen_control_and_candidate`） |
| E-059 manifest | `outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation/comparison.yaml`（24 条目） |
| E-059 analysis | `outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation-analysis/`（analysis.json 含 robustness 块、report.md、figures；pin `7548706` clean） |
| λ=1 训练 runs + evals | `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/`（pin `73b9db4`） |
| R0 训练 runs + evals + shared initial | `outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/`（train pin `07b2860`，eval pin `986b015`） |
| 机械诊断（15-run） | `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-diagnostics/` |
| 上游裁定 | E-056（Gate M FAILED，λ=4/8）；E-057（Gate S1/S2）；E-058（Gate S3，λ=1 提名） |
