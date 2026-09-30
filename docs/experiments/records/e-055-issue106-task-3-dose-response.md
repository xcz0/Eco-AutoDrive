# E-055 Issue #106 Task 3：λ={0,1,2,4,8} formal dose-response / Gate D

[返回实验索引](../README.md) · Issue #106 Task 3 · [E-054 Gate T3](e-054-issue106-task-2-canonical-positive-control.md) ·
[E-053 Gate T2](e-053-issue106-task-1-gate-t2-reentry.md) ·
[E-052 冻结 control](e-052-issue105-phase-c-canonical-ppo-refreeze.md) ·
[E-048 canonical identifiability](e-048-issue83-task-1a-objective-identifiability-transfer.md)

**日期 / 类型 / 目的**：2026-09-29 / 正式 λ dose-response coarse sweep / Issue #106 Task 3：在 canonical k=5
（`0.1 s × 5 = 0.5 s`）+ E-052 冻结 PPO control 下，λ={0,1,2,4,8} 单冻结 training seed（0）、50 updates、
checkpoints 0/25/50，判定 dose-response 是否可重复、可解释、不被行为混淆（Gate D），并提名 ≤1–2 个候选
交 #83。λ=0（arm `r0`）**复用 E-054 `r0-seed-0`**（同协议 resolved 训练条件逐值一致，final hash
`39aeb2571cad…`）；λ=1/2/4/8 为本次 4 个新 matched training runs。Gate D 裁定在实验记录中给出（与
E-050/E-051/E-054 一致，analysis 层不做 gate 裁定）。

**裁定：Gate D PASSED（6/6）。** 四个非零 λ 相对 R0 的 held-out energy intensity 全部超过预注册 noise
bound（相对变化 −0.149%/−0.234%/−0.383%/−0.531% ≥ 0.1%，bootstrap CI 全部不含 0），objective-pressure
（z-advantage RMSE 0.0227→0.1502）、guidance（paired β-mean RMS 0.0207→0.0824）与 executed behavior
（intensity −0.07→−0.25 mL/km、mean speed −0.40%→−1.42%）随 λ 单调且方向一致（更慢、单位里程能耗更低），
与 E-054 λ=64（intensity −0.637%）方向连续；无停车/失败/少走混淆（stopped Δ=0，16/16 completed，
13/16 arrival 与 termination 结构全臂一致，distance Δ ≤ 0.77 m / 169.5 m）；无 Beta boundary / KL
runaway / invalid trajectory。**候选提名：λ=4 与 λ=8**（选择规则仅用 held-out matched evaluation，不使用
training return），handoff #83。

## 冻结契约与协议

Task 3 在 #106 冻结研究 contract 下执行（frozen Diffusion Planner / DDIM5；同一 initial policy；
training S/SC map seeds 0–7；held-out S/SC seeds 16–23；deterministic Beta mean 主评测；E-034 校准
Progress/Comfort；E-038 冻结 efficiency-band energy；canonical execution 0.5 s；execution-trace fuel
proxy；PPO control 取 E-052 冻结结果）。

新增 protocol `configs/experiments/comparison/calibrated-canonical-dose-response.yaml`：在
`calibrated-canonical.yaml`（E-054）基础上，arms 改为 `r0 / lam1 / lam2 / lam4 / lam8`（reward profiles
`plannerrft_no_energy_calibrated_v1` 与 `plannerrft_energy_band_lam{1,2,4,8}_v1`），training seeds 收缩为
`{0}`，contrasts 为 4 个 `[r0, lamN]`；training overrides 与 E-054 逐值相同（E-052 冻结 control +
canonical 调度），evaluation contract 逐值相同（`no_traffic_heldout`/`no_traffic_heldout_policy`，
S/SC seeds 16–23，seed 760025，horizon 300，ddim5），bootstrap 逐值相同。protocol pin 测试
`tests/training/test_training_workflows.py::test_canonical_dose_response_protocol_pins_frozen_control_and_lambda_arms`
校验 arms/contrasts/冻结 control/cadence。

为支持 checkpoint 0/25/50，comparison 机制扩展（Task 3 范围内的实现改动，均有测试）：

- `CheckpointLabel` 放宽为任意字符串；`ComparisonRun.checkpoint_hash` 允许为非 `initial`/`final` 标签
  pin 期望 policy hash，并与 evaluation job 重算 hash 交叉校验（trainer 不记录 per-update hash，重用的
  E-054 run 无法事后补造）；
- `paired` / `arm_outcomes` 扩展到全部 `_PAIRED_METRICS`（energy_ml、energy_distance_m、
  energy_ml_per_km、mean_speed_mps、distance_m、stopped_fraction、route_completion）并新增
  behavior/energy/termination outcome blocks（speed min/mean/max、terminal reason 计数）；
  report 新增 "Behavior / task and energy by checkpoint" 与 "Paired per-metric effects by checkpoint"
  两节（checkpoint 排序 initial < 中间 < final）；
- `extract_arm_metrics` 新增 `beta_series`（全部 50 updates 的 Beta 摘要）。

测试：`tests/analysis/test_scalar_effects.py`（新 metrics/中间 checkpoint 报告排序）、
`tests/analysis/test_reports.py`（中间 checkpoint hash pin 三态校验）、
`tests/training/test_effective_update.py`（beta_series）。

## 命令

```powershell
# 4 个新 matched training runs（lam1/lam2/lam4/lam8 × seed 0），canonical k=5 + E-052 冻结 control
just exp compare train --config configs/experiments/comparison/calibrated-canonical-dose-response.yaml `
  --arm lam1 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response/lam1-seed-0
# ... lam2 / lam4 / lam8 同理

# held-out 评测（deterministic Beta mean，16 场景）：shared initial（E-054 r0-seed-0 的 policy-initial.pt，
# 五臂共享同一 initial policy）+ 每臂 update-025（policy-update-024.pt，即 25 updates 后状态）+ final
just exp compare eval --config ...calibrated-canonical-dose-response.yaml `
  --arm r0 --checkpoint initial --checkpoint-path .../e-054-issue106-task-2-positive-control/r0-seed-0/policy-initial.pt `
  --output-dir .../e-055-issue106-task-3-dose-response/heldout/shared-initial
just exp compare eval --config ... `
  --arm lam8 --checkpoint update-025 --checkpoint-path .../e-055-.../lam8-seed-0/policy-update-024.pt `
  --output-dir .../heldout/lam8-seed-0-update-025
# ... 每臂 update-025 / final 同理；r0 final 复用 E-054 heldout/r0-seed-0-final（同 evaluation contract）

# 离线配对比较（manifest: comparison.yaml，update-025 条目 pin 重算 hash）
just exp compare analyze --config .../e-055-issue106-task-3-dose-response/comparison.yaml `
  --source-dir .../e-055-issue106-task-3-dose-response `
  --output-dir outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response-analysis

# 跨 λ objective 证据（E-048 冻结 source batch 重放，无 gate 阈值，Gate D 在记录内裁定）
just exp credit run --source-dir outputs/studies/scalar-reward/e-048-issue83-task-1a-source-batch `
  --config configs/experiments/credit/e-055-dose-response-objective.yaml `
  --output-dir outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response-objective

# 每臂训练机械健康（5 summaries，含 beta_series / analytic + seeded-MC KL）
just exp training diagnose --config .../e-055-issue106-task-3-dose-response/diagnose.yaml `
  --output-dir outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response-diagnostics
```

## Matched guards（runner/analysis 硬校验）

五臂同 seed 训练共享：initial policy hash `049697739ab5…`（五臂逐值一致，为 E-052/E-053/E-054 同一冻结
initial）、noise/policy-action seeds、frozen planner hash、`probe_before`；同 seed 除 reward profile 外
resolved 训练条件一致（`compare analyze` matched-condition 校验通过）。`initial` checkpoint 上五臂
held-out 配对 delta 全指标严格为 0（同一 policy 重放），即 0-update matched baseline。update-025 provenance
通过 manifest pin hash 与 evaluation job 重算 hash 交叉校验（r0 `8593e956…`、lam1 `494c1aba…`、
lam2 `f957a6c1…`、lam4 `c98481f1…`、lam8 `47d35a63…`）。λ=0 arm（E-054 r0-seed-0）final hash
`39aeb2571cad…` 与 E-052/E-053 逐值一致。

## Objective / PPO 证据

**跨 λ objective 分离**（`credit run`，E-048 canonical k=5 冻结 source batch，standard GAE，z 形式；
`r0` vs `lambda_N`）：

| λ | z-advantage RMSE | actor-head gradient cosine | endpoint 分离占比（RMSE/0.6641） |
| ---: | ---: | ---: | ---: |
| 1 | 0.022668 | 0.999427 | 3.4% |
| 2 | 0.044071 | 0.998020 | 6.6% |
| 4 | 0.083406 | 0.993932 | 12.6% |
| 8 | 0.150219 | 0.984544 | 22.6% |
| energy_only（端点） | 0.664142 | 0.910851 | 100%（=E-048 冻结值，重放逐值一致） |

objective-pressure 随 λ 单调增强；energy_only 端点与 E-048 冻结 Gate T1 数值逐值一致（重放验证）。

**训练机械健康**（`training diagnose`，4 新 runs + E-054 r0，各 50 updates，全部 completed）：min
α/β ≥ 1.776/1.985（≫0.1 floor）；boundary mass after ≤ 0.0225（<0.2 ceiling）；无 early stop，seeded-MC
post-update KL median 1.04e-6→1.40e-5、max ≤ 7.33e-5（≪ target_kl 0.006，无 runaway），analytic Beta KL
与 k1/k3 single-draw 同量级一致；ratio mean median ≤ 1.00009、p95 ≤ 1.00834、clip fraction 0；
0 collision/OOR，episode length 8→8 稳定。guidance probe RMS shift 随 λ 单调（0.0311→0.0842）；
beta_series 全程 α∈[1.782, 2.008]、concentration∈[3.972, 4.003]，无 boundary 漂移。纵向维 Beta mean
（beta_final dim1）随 λ 单调负移：r0 +0.0054 → lam1 −0.0233 → lam2 −0.0448 → lam4 −0.0791 →
lam8 −0.1089（与 E-054 λ=64 纵向主导一致）。

**训练内 scalar return**（仅记录，不用于选择）：first→last total reward r0 498.7→497.5、lam1
481.3→480.7、lam2 465.9→466.6、lam4 439.8→444.0、lam8 400.5→412.1；随 λ 下降是 λ 加权 energy penalty
的机械结果。

## Behavior / task / energy 证据（held-out，16 matched 场景，deterministic Beta mean）

配对 delta（lamN − r0， jointly-completed 16/16，scenario bootstrap 95% CI 10000 resamples seed 0）：

| checkpoint | λ | energy_ml Δ（CI） | intensity Δ (mL/km) | 相对 intensity | mean speed Δ (m/s) | 相对 speed | distance Δ (m) | stopped Δ |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| final | 1 | −0.025（[−0.036, −0.013]） | −0.07 | −0.149% | −0.040 | −0.40% | −0.32 | 0 |
| final | 2 | −0.027（[−0.043, −0.010]） | −0.11 | −0.234% | −0.063 | −0.63% | −0.23 | 0 |
| final | 4 | −0.040（[−0.060, −0.018]） | −0.18 | −0.383% | −0.103 | −1.03% | −0.30 | 0 |
| final | 8 | −0.071（[−0.094, −0.049]） | −0.25 | −0.531% | −0.142 | −1.42% | −0.77 | 0 |
| update-025 | 1 | −0.008（[−0.018, +0.005]） | −0.03 | −0.06% | −0.019 | −0.19% | −0.07 | 0 |
| update-025 | 2 | −0.022（[−0.032, −0.009]） | −0.06 | −0.13% | −0.033 | −0.33% | −0.28 | 0 |
| update-025 | 4 | −0.025（[−0.039, −0.010]） | −0.09 | −0.19% | −0.051 | −0.51% | −0.24 | 0 |
| update-025 | 8 | −0.028（[−0.043, −0.013]） | −0.11 | −0.23% | −0.065 | −0.64% | −0.25 | 0 |
| initial | 全部 | 0（严格 0，同一 policy） | 0 | 0 | 0 | 0 | 0 | 0 |

绝对水平（final）：mean speed 10.008（r0）→ 9.866（lam8）m/s，episode speed range ~[3.8, 15.0]；intensity
47.06（r0）→ 46.82（lam8）mL/km；energy total 7.85→7.78 mL；distance 169.6→168.9 m。任务结构全臂一致：
16/16 completed、13/16 arrive_dest、termination 13 `arrive_dest` + 3 `max_step`、0 collision/OOR/
wrong-direction、stopped fraction 全 0。E-054 λ=64（同 seed 0）intensity −0.637%、speed −1.75% 作为
λ→∞ 方向锚点与本次 λ=8（−0.531%/−1.42%）连续。

## Gate D 判定

```text
Verdict: PASSED（6/6）
```

| 条件 | 判据 | 实测 | 通过 |
| --- | --- | --- | :---: |
| d1 ≥2 非零 λ 超预注册 noise 的 objective/guidance/behavior effect | behavior 相对变化 ≥0.1%（E-054 c2 bound）；guidance β-mean RMS ≥0.01 | 4/4 λ 全部超过：intensity −0.149%…−0.531%、speed −0.40%…−1.42%、guidance RMS 0.0207…0.0824；final energy CI 全不含 0 | ✓ |
| d2 λ 增大时 objective-pressure / guidance / executed behavior 方向可解释 | 单调且方向一致 | z-adv RMSE 0.023→0.150、guidance RMS 0.021→0.082、纵向 β-mean 0.005→−0.109、intensity Δ −0.07→−0.25、speed Δ −0.040→−0.142 全单调；update-025 同序；与 λ=64 方向连续 | ✓ |
| d3 energy-intensity 不由少走/停车/失败混淆 | stopped/完成/到达/距离结构一致 | stopped Δ=0；16/16 completed；13/16 arrival 与 termination 结构逐臂一致；distance Δ ≤ 0.77 m/169.5 m（≤0.45%），且 mL/km 本身按距离归一 | ✓ |
| d4 无 Beta boundary / KL runaway / invalid trajectory | T2 同款机械判据 | min α/β ≥1.776/1.985；boundary ≤0.0225；post-KL max 7.33e-5 ≪ 0.006；无 early stop；clip 0；0 collision/OOR；ep-len 8 稳定 | ✓ |
| d5 非单调行为需解释 | 若存在须给 trade-off / mapping 解释 | final 行为单调，无需解释；观察项：update-025 λ=1 的 energy CI [−0.018,+0.005] 含 0——λ=1 剂量在 25 updates 时尚未超 noise，50 updates 时才分辨（剂量随训练时间展开），λ≥2 在 25 updates 已分辨 | ✓ |
| d6 ≤1–2 候选，不使用单一 training return / 偶然 endpoint | 提名规则声明 | 提名 **λ=4 与 λ=8**（见下）；规则仅用 held-out matched evaluation 的 resolved effect + 任务结构保持，未使用 training return（训练内 return 随 λ 下降为 λ 加权 penalty 机械结果，已排除） | ✓ |

## 候选提名与 #83 handoff

**候选：λ=4、λ=8**（`plannerrft_energy_band_lam4_v1` / `plannerrft_energy_band_lam8_v1`）。

提名规则（预注册约束内）：在 Gate D 通过的 λ 中，取 held-out matched evaluation 上 energy-intensity
效应最大且任务结构（arrival/termination/stopped/distance/safety）与 R0 保持一致、final bootstrap CI
不含 0 的至多两个 λ。λ=8（−0.531% intensity，CI [−0.094, −0.049] mL）与 λ=4（−0.383%，CI
[−0.060, −0.018] mL）为剂量上端两个 resolved 剂量，效应量接近 λ=64（−0.637%）而行为代价更小
（speed −1.42% vs −1.75%）。λ=1/λ=2 不提名：效应 ≤ ~2× noise bound（−0.149%/−0.234%）且 λ=1 在
update-025 尚未分辨。0-update matched results：五臂同一 initial policy，held-out 全指标 delta 严格为 0
（见上表 initial 行）。完整 provenance：训练
`outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response/lam{4,8}-seed-0/`（resolved_config/
summary/checkpoints 0/24/49 + initial/final）、held-out
`.../heldout/lam{4,8}-seed-0-{update-025,final}/`、分析 `...-analysis/analysis.json`。

按 #106 contract，多 seed confirmation、final A0/A1/A2 matched evaluation 与候选确认归 #83；本 Issue
candidate path 判定完成（T2/T3/D 全通过）。

## 局限与结论边界

- **单 training seed**：dose-response 的形状与单调性是单 seed（0）观测；跨 seed 可重复性由 #83 候选
  confirmation（多 seed）检验，本记录不做跨 seed 推断。
- **效应量级**：λ≤8 的 held-out intensity 效应 ≤0.53%，处于预注册 noise bound（0.1%）与 λ=64 锚点
  （−0.637%）之间；"dose-response 成立"指单调可分辨，不指已达实用节能幅度。
- **guidance mean/std 仅 initial/final probe**：训练中 guidance 只有 β_series（Beta 参数），mean/std
  probe 只在 update 0 前与 final 后各一次；25-update 处的 guidance 分布未直接 probe。
- **speed profile 为 episode 级 min/mean/max**：非逐时刻轨迹；更细行为结构在 analysis.json 原始
  episode 键中。
- **update-025 provenance**：trainer 不记录 per-update hash，中间 checkpoint 由 manifest pin hash 与
  evaluation job 重算 hash 交叉校验；E-054 r0-seed-0 的 update-025 同样以后验 pin 方式登记（未记录的
  事实不事后补造，pin 的是重算 hash 本身）。
- **λ=0 复用 E-054 r0-seed-0**：同协议 resolved 训练条件逐值一致（analyze 校验通过），非独立复跑；
  其 final held-out eval（E-054 `heldout/r0-seed-0-final`）直接复用（evaluation contract 逐值相同）。
- **credit 证据为 fixed-batch 重放**：optimizer_steps=0、policy_unchanged=true；objective 分离不建立
  learned behavior（与 E-048 相同边界）。

## 产物

| 产物 | 路径 |
| --- | --- |
| 协议 | `configs/experiments/comparison/calibrated-canonical-dose-response.yaml` |
| 训练 runs（4 新） | `outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response/lam{1,2,4,8}-seed-0/` |
| λ=0 run（复用） | `outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control/r0-seed-0/` |
| held-out evals | `outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response/heldout/`（+ E-054 `heldout/r0-seed-0-final`） |
| comparison manifest | `outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response/comparison.yaml` |
| 配对分析 + 报告 | `outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response-analysis/`（analysis.json / report.md / figures） |
| 跨 λ objective | `outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response-objective/`（config：`configs/experiments/credit/e-055-dose-response-objective.yaml`） |
| 训练诊断（5 臂） | `outputs/studies/scalar-reward/e-055-issue106-task-3-dose-response-diagnostics/`（config：study dir `diagnose.yaml`） |
