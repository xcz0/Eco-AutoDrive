# E-057 Issue #114 Task 3：previously failing candidate stress test / Gate S1-S2

[返回实验索引](../README.md) · Issue #114 Task 3 ·
[E-056 失败先例](e-056-issue83-task-1-candidate-multiseed-confirmation.md) ·
[ADR 0040 interior 参数化](../../adr/0040-beta-interior-concentration-parameterization.md) ·
[预注册计划](../../research/issue114-task3-interior-stress-plan.md)

**日期 / 类型 / 目的**：2026-10-08 / 正式 previously failing candidate stress test / Issue #114 Task 3：
在 ADR 0040 Beta interior policy family（α, β 恒 > 1 + `interior_epsilon`=1e-4）下，保持 energy
provider 与 reward 不变，重跑 E-056 失败矩阵：r0 / λ=4 / λ=8 × seeds {0,1,2} × 100 updates
（`calibrated-canonical-confirmation.yaml`，`T_max=100`，checkpoint 0/25/50/100），回答四个研究
问题并裁定 Gate S1（distribution stability）/ Gate S2（long-run mechanical stability）。判定标准
运行前预注册，不依结果调整。

**裁定：numerical collapse 结构性消除（Gate S2 PASSED，9/9 runs 机械稳定），但 behavioral
saturation 确认存在（Gate S1 FAILED：λ=8 boundary mass 3/3 seeds 超过预注册 0.2 ceiling；
λ=4/8 均呈现预注册的 sustained negative longitudinal saturation = 「另一种退化」）。**
该分离正是 Issue #114 要求的「明确区分 numerical collapse 与 behavioral saturation」：
第一层问题（β 参数化允许 singular boundary mode → 训练崩溃 / log-prob 非有限）已被
interior 参数化消除；第二层问题（数值不崩，但 policy 持续把纵向分布收窄并贴向 −1 边界）
依然存在，且 3/3 seeds 一致、非单 seed 异常。R0 对照全部通过。结果不恢复 #83 final A2；
Task 4（candidate provenance revalidation）现在具备执行条件，λ=8 的 ceiling violation 与
λ=4/8 的 saturation 是其直接输入；Task 5 触发条件（interior 约束通过但长期 mean
saturation）满足；现象作为 #113 reward-structure 研究的重要输入。

## 冻结契约与协议

复用 `configs/experiments/comparison/calibrated-canonical-confirmation.yaml`（E-056 协议原样，
未改任何配置）：arms `r0/lam4/lam8`，training seeds `{0,1,2}`，`training.update_count=100`、
`ppo.scheduler_total_optimizer_steps=100`（#114 Task 1 scheduler-matched contract）、
`ppo.gradient_diagnostics=true`，held-out contract（S/SC seeds 16–23、runtime seed 760025、
horizon 300、ddim5、bootstrap 0.95/10000/seed 0）与 E-055/E-056 逐值相同，contrasts
`[[r0,lam4],[r0,lam8]]`。与 E-056 的唯一差异是 policy family：`interior_epsilon=1e-4` 取代
`minimum_concentration`（commit `03a6c99`，ADR 0040），per-seed initial policy hash 随之改变
（新 family 的预期直接后果）。

## 命令

```powershell
# 9 matched training runs（全部完成；clean commit 07b2860）
just exp compare train --config configs/experiments/comparison/calibrated-canonical-confirmation.yaml `
  --arm r0 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/r0-seed-0
# ... 其余 arm × seed 同理；日志存 study 根 train-{arm}-seed-{n}.log

# 离线机械诊断（9 summaries，mc_draws=4096、mc_seed=1000003；commit 986b015）
just exp training diagnose `
  --config .../e-057-issue114-task3-interior-stress/diagnose.yaml `
  --output-dir outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress-diagnostics

# held-out 评测 30 次（每 seed shared initial ×3 + 每臂 update-025/update-050/final ×27）
just exp compare eval --config ...calibrated-canonical-confirmation.yaml `
  --arm r0 --checkpoint initial --checkpoint-path .../r0-seed-0/policy-initial.pt `
  --output-dir .../heldout/initial-seed-0
just exp compare eval --config ... `
  --arm lam8 --checkpoint update-050 --checkpoint-path .../lam8-seed-2/policy-update-049.pt `
  --output-dir .../heldout/lam8-seed-2-update-050
# ... 其余组合同理

# 离线配对比较（manifest pin 重算 hash）
just exp compare analyze `
  --config .../e-057-issue114-task3-interior-stress/comparison.yaml `
  --source-dir .../e-057-issue114-task3-interior-stress `
  --output-dir outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress-analysis
```

## Matched guards（runner/analysis 硬校验）

9 个训练 run 全部在 clean commit `07b2860`（预注册计划 commit，`git_status_short=[]`）执行，
cuda:0 / bf16-mixed；canonical cadence resolved `0.1 × 5 = 0.5 s`；frozen planner hash 前后
不变。Per-seed initial policy hash 各臂逐值一致：seed 0 `56bb97ab0551…`、seed 1
`f91a826f38a2…`、seed 2 `0febc900778c…`（均为新 family 新值，≠ E-056 旧值，预期内）。
`probe_before` 全 run 一致（boundary mass 0.020996）。diagnose / 30 次 eval / analyze 在
clean commit `986b015` 执行（差异仅诊断测量层修复，见「与预注册计划的偏离」；训练产物
未被触碰，hash 交叉校验全部通过）。

## 训练结果（9-run 状态表）

| run | initial hash | updates | 状态 | min α / min β | probe_after boundary mass (max) | 纵向 mean 终态 (guidance) |
| --- | --- | ---: | --- | --- | --- | --- |
| r0-seed-0 | `56bb97ab0551` | 100 | completed | 1.531 / 1.933 | 0.0388 | +0.045 |
| r0-seed-1 | `f91a826f38a2` | 100 | completed | 1.966 / 1.570 | 0.0388 | +0.176 |
| r0-seed-2 | `0febc900778c` | 100 | completed | 1.996 / 1.451 | 0.0610 | +0.208 |
| lam4-seed-0 | `56bb97ab0551` | 100 | completed | 1.104 / 2.002 | 0.172 | −0.600 |
| lam4-seed-1 | `f91a826f38a2` | 100 | completed | 1.191 / 1.580 | 0.152 | −0.582 |
| lam4-seed-2 | `0febc900778c` | 100 | completed | 1.108 / 1.454 | 0.167 | −0.593 |
| lam8-seed-0 | `56bb97ab0551` | 100 | completed | 1.039 / 2.000 | **0.242** | −0.683 |
| lam8-seed-1 | `f91a826f38a2` | 100 | completed | 1.046 / 1.631 | **0.243** | −0.685 |
| lam8-seed-2 | `0febc900778c` | 100 | completed | 1.030 / 1.507 | **0.258** | −0.697 |

E-056 的四个失败组合（lam4-seed-2 崩溃于 update 76、lam8-seed-0 崩溃于 update 70、
lam4-seed-0/1 boundary collapse）在本 family 下全部完整完成 100 updates。

## 诊断与机制分析

**Q1 — E-056 boundary collapse 是否被结构性消除：是。** 全部 9 runs 完成；min α/β 全程
> 1+ε（最低 1.030，interior floor 生效但从未穿透——λ 加权 energy pressure 仍把纵向 α 压向
floor，只是不再可能 < 1）；无 invalid action / non-finite log-prob / exact-boundary failure；
analytic / seeded-MC / k3 post-update KL 三估计器全部有限、同量级且无 runaway（MC median
1.86e-5（r0-seed-0）→1.81e-4（lam8-seed-2），max ≤ 1.49e-3，≪ target_kl 0.006）；clip
fraction median 0.0，ratio p95 median ≤ 1.027；六组 per-group gradient 有限（actor_head_policy
max 7.5–10.4）；LR 1.5e-4 → 3.7e-8（T_max=100 cosine），cumulative exposure 0.0100，
per-update 序列与 `final_learning_rate` 交叉校验通过。

**Q2 — longitudinal mean 是否仍持续向负边界靠近：是（λ=4/8，3/3 seeds）。** 纵向
（dim 1）Beta mean 自 update 0 的 0 单调走负：update 75–100 段均值 λ=4 = −0.575/−0.575/−0.581、
λ=8 = −0.676/−0.679/−0.689，全部满足预注册 saturation 判据（≤ −0.5）；50→100 段无反转、
持续加深。R0 对照保持微正（+0.045~+0.208）。lateral（dim 0）mean 无系统性边界趋近
（−0.24~+0.35，跨 arm/seed 形态一致），如实报告、不预设判据。

**Q3 — λ=4/8 是否存在「数值不崩，但行为饱和」的第二层问题：存在，且为本研究的主要发现。**
退化形态是分布**收窄并贴向 −1 边界**而非发散：纵向 concentration 从初始 4.0 升至
5.49–5.75（λ=4）/ 6.58–6.83（λ=8），variance 从 0.19 降至 0.097–0.100 / 0.066–0.070，
α 压向 interior floor（λ=8 min α 1.03–1.05）。probe_after boundary mass：λ=8 = 0.242/0.243/0.258
**3/3 seeds 超过预注册 0.2 ceiling**；λ=4 = 0.152–0.172 未超但逼近；probe guidance RMS shift
0.447–0.544（λ arms）vs 0.168–0.289（r0）。该形态 3/3 seeds 一致，非单 seed 异常。对照
E-056：同样 energy pressure 下旧 family 在 update 51/57 后 α<1 进入 singular regime
（boundary mass 0.47–0.53 或训练崩溃）；新 family 把同一 pressure 的表达从「崩溃」改为
「贴边收窄」——pressure 本身未被 parameterization 消除。

**Q4 — 50→100 的 guidance / speed / progress / energy effect 是否方向稳定：稳定。**
Held-out 配对对比（vs R0，16 场景，bootstrap 95% CI）在 update-050 与 final 同号且单调加深，
全部 seeds 一致（speed 与 mL/km 的 CI 两 checkpoint 均不含零）：

| contrast（seed 均值范围） | update-050 | final |
| --- | --- | --- |
| lam4−r0 mean_speed_mps | −0.42 ~ −0.52 m/s | −0.65 ~ −0.87 m/s |
| lam4−r0 energy_ml_per_km | −0.69 ~ −0.88 mL/km | −1.00 ~ −1.40 mL/km |
| lam4−r0 route_completion | −0.004 ~ −0.006 | −0.008 ~ −0.014 |
| lam8−r0 mean_speed_mps | −0.51 ~ −0.64 m/s | −0.70 ~ −0.91 m/s |
| lam8−r0 energy_ml_per_km | −0.80 ~ −1.05 mL/km | −1.04 ~ −1.42 mL/km |
| lam8−r0 route_completion | −0.007 ~ −0.008 | −0.010 ~ −0.033 |

stopped_fraction 差异恒为 0；训练期 collision/OOR 全程为 0；safety gate 均值 1.0。guidance
方向由 Q2 的 per-update 纵向 mean 轨迹承载（held-out contrast 指标集不含 guidance 项）。
energy-intensity 降低的主要载体是降速（与 #113 的「降速即节能」objective shortcut 假设
一致），progress 代价小幅增长（λ=8 final seed 1/2 的 route_completion CI 不含零）。

**残余稀有失败模式（机制边界，如实登记）**：float32 下 base draw < 2⁻²⁵ 时
`2u−1` 舍入到 −1.0 精确边界，`(g+1)/2` 回映射为 0.0，log-prob 双侧 −inf。训练路径
（`validate_args=True`）会 loud fail（`guidance action must be strictly inside (-1,1)`），
概率量级 ~1e-4/run（本次 9×12,800 draws 未触发）；诊断 MC 采样密度为训练的 4096 倍，
lam4-seed-2 update 86 命中一次（正是下述测量修复的触发点）。这是 interior family 下
与 E-056（α<1 密度奇异）不同的新边界机制：密度支持的精确边界 draw 已不可能，但 float32
表示粒度仍构成稀有 loud-fail 通道。

## Gate S1 / S2 判定

```text
Gate S2（long-run mechanical stability）: PASSED（r0/lam4/lam8 各 3/3）
Gate S1（distribution stability）: FAILED（λ arms；r0 通过）
```

| 条件 | r0 | λ=4 | λ=8 |
| --- | --- | --- | --- |
| 3/3 seeds 完成 | ✓ | ✓ | ✓ |
| α/β 永不进入 singular region（> 1+ε） | ✓ | ✓ | ✓ |
| 无 exact-boundary / non-finite failure | ✓ | ✓ | ✓ |
| KL（analytic/MC/k3）有限且无 runaway | ✓ | ✓ | ✓ |
| boundary mass ≤ 预注册 ceiling 0.2 | ✓（≤0.061） | ✓（0.152–0.172，逼近） | **✗（0.242–0.258，3/3）** |
| entropy/concentration 无另一种退化 | ✓ | **✗（收窄贴边 + mean 饱和 ≤ −0.5）** | **✗（同左，更甚）** |
| 50→100 无 distribution/behavior collapse | ✓ | ✓（无 collapse；saturation 持续加深如实记录） | ✓（同左） |
| 结果非单 seed 异常 | ✓ | ✓（3/3 一致） | ✓（3/3 一致） |

Gate S1 的失败项属于 behavioral saturation（预注册 §4.3：不构成 Task 3 失败本身，
而是 S1 退化项、Task 5 触发与 #113 的判断输入）；Gate S2 的全部机械项通过。
二者合并读法与 Issue #114 的层级一致：**第一层（numerical）已修复，第二层
（behavioral）未消除。**

## 与预注册计划的偏离

1. **诊断测量层 bug 修复（commit `986b015`），发生于训练之后、诊断期间。** 首次
   diagnose 在 lam4-seed-2 update 86 中止：seeded-MC post-update KL 的
   `FloatingPointError: post-update MC log ratios must be finite`。逐 draw 定位证实这是
   测量工具的 float32 精度伪影而非 policy 失败：合法内部 draw 1.45e-08（近 floor
   α=1.12 场景）经 guidance 空间往返后坍缩到精确边界，双侧 log-prob 同为 −inf →
   NaN；float64 下该 draw 的真实 log-ratio 完全有限（−0.48 vs −0.46）。修复将 MC
   log-ratio 评估改为 base 坐标系 float64（仿射 Jacobian 在 ratio 中相消，base 值
   无损）：估计量语义不变，真正的边界奇异 draw（旧 family collapse 形态）在 float64
   下仍非有限、`FloatingPointError` 守卫语义保持。回归测试
   `tests/training/test_effective_update.py::test_mc_log_ratio_stays_finite_when_float32_rounds_interior_draw_to_boundary`
   以本实验的精确失败值复现。训练产物不受影响（训练 pin `07b2860`，诊断/评测/分析
   pin `986b015`，provenance 分别登记）。该修复与 E-056 的 probe 修复（`2a22b3c`）
   同类：测量层缺陷不得使应报告的测量对象不可见。
2. **其余无偏离**：9 训练 + 30 评测 + analyze 与预注册一致执行；无崩溃、无 seed 删除、
   无参数调整。

## 局限与结论边界

- **结论范围**：本记录裁定的对象是「新 interior policy family 下 r0/λ=4/λ=8 在确认协议
  （seeds {0,1,2} × 100 updates、T_max=100、冻结 control）下的长程稳定性与行为形态」。
  不构成对其他 λ、reward representation 或 energy provider 的判断（#113 边界）。
- **E-055/E-056 为旧 family 历史证据**：跨 family 的数值不可直接比较（initial hash、
  参数化、可达分布族均不同）；E-055 的 λ=4/8 nomination 依 Issue #114 Task 4 需在新
  family 下重新确认，本记录提供其机械稳定性输入（λ=4 过 ceiling 但 saturation；λ=8
  超 ceiling）。
- **boundary mass 仅在 checkpoint 0/100 测量**（预注册 D2）：中间时点的边界质量由
  per-update β mean/concentration/min α 序列间接刻画；未测量 update 25/50 的 probe
  boundary mass。
- **saturation 的下游归因未做**：「降速即节能」shortcut 假设（#113）未被本协议检验；
  本记录只确认 policy 稳定但持续选择极端负纵向 guidance 并把它作为 #113 输入。
- **残余 float32 边界通道未在本实验触发**：其 ~1e-4/run 的概率估计基于本实验参数形态
  的解析近似，未做专门扫掠。

## 对后续任务的输入

- **Task 4（candidate provenance revalidation）**：现在可执行。新 family 下 λ=4/λ=8 均
  可完整训练；候选选择须按 Task 4 最低要求重做（同一 scheduler contract、新参数化、
  matched held-out 比较）。本记录的 ceiling/saturation 证据表明 λ=8 在预注册判据下不可
  作为稳定候选直接恢复，λ=4 满足机械稳定性但携带 saturation 形态——是否接受由 Task 4
  的 selection 标准裁定，不由本记录代答。
- **Task 5（optional reference-policy trust region）**：触发条件满足（interior 约束通过、
  仍长期 mean saturation），仍为二阶段 contingency，是否研究另行决策。
- **#113（energy proxy / reward validity）**：λ=4/8 的 energy-intensity 降低主要由降速
  载带、progress 代价小幅上升，与 objective shortcut 假设方向一致，作为 #113 的重要输入。
- **#83 re-entry（Gate S3）**：继续 blocked（Task 4 未完成）。

## 产物

| 产物 | 路径 |
| --- | --- |
| 训练 runs（9） | `outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/{r0,lam4,lam8}-seed-{0,1,2}/` |
| 训练日志 | `.../train-{arm}-seed-{n}.log` |
| 训练诊断（9 runs） | `outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress-diagnostics/`（config：study dir `diagnose.yaml`） |
| Held-out 评测（30） | `.../e-057-issue114-task3-interior-stress/heldout/`（initial-seed-{0,1,2}、{arm}-seed-{n}-{update-025,update-050,final}） |
| 配对比较 manifest | `.../e-057-issue114-task3-interior-stress/comparison.yaml`（中间 checkpoint pin 重算 hash） |
| 配对比较分析 | `outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress-analysis/` |
| 诊断 / 分析日志 | `.../e-057-issue114-task3-interior-stress/e-057-diagnose.log`、`e-057-analyze.log` |
| MC KL 精度修复 | commit `986b015`（`mc_log_ratio_draws` + 回归测试） |
| 预注册计划 | `docs/research/issue114-task3-interior-stress-plan.md`（状态：已执行） |
