# E-058 Issue #114 Task 4：candidate provenance revalidation / Gate S3

[返回实验索引](../README.md) · Issue #114 Task 4 ·
[E-057 stress test](e-057-issue114-task3-interior-stress.md) ·
[E-055 旧 family nomination](e-055-issue106-task-3-dose-response.md) ·
[ADR 0040 interior 参数化](../../adr/0040-beta-interior-concentration-parameterization.md) ·
[预注册计划](../../research/issue114-task4-candidate-revalidation-plan.md)

**日期 / 类型 / 目的**：2026-10-09 / 正式 candidate provenance revalidation / Issue #114 Task 4：
ADR 0040 Beta interior policy family（α, β 恒 > 1 + `interior_epsilon`=1e-4）改变了初始 actor
bias 与 `policy_state_hash`，E-055 的 λ=4/8 nomination 仅是旧 family 历史证据。本实验在
新 family + Task 1 scheduler contract（100 updates / T_max=100 / 单一 scheduler history）下
重新执行 λ={0,1,2,4,8} candidate comparison（seeds {0,1,2}，checkpoints 0/25/50/100，
matched held-out evaluation + bootstrap 95% CI），按预注册标准重裁定候选提名与 Gate S3
（#83 re-entry）。判定标准运行前冻结，不依结果调整。

**裁定：λ=1 为唯一被提名的候选（否决 gates 全过 + 提名要求全过，3/3 seeds）；λ=2 通过
否决 gates 但因任务结构不一致（route_completion 3/3 seeds 系统性下降，final CI 全不含 0）
未获提名；λ=4 被 sustained negative longitudinal saturation 否决、λ=8 被 boundary mass
ceiling + saturation 双重否决（均继承 E-057 判定）。Gate S3 PASSED：new policy family
frozen（ADR 0040）+ scheduler semantics matched（Task 1）+ candidate selection revalidated
（λ=1）+ candidate long-run stability passes（3/3 seeds 100-update 机械稳定、无 saturation）。
#83 解除 block，λ=1 进入 #83 final A2 路径。**

## 冻结契约与协议

新协议 `configs/experiments/comparison/calibrated-canonical-candidate-revalidation.yaml`
（E-058，pin 测试 `test_candidate_revalidation_protocol_pins_frozen_control_and_lambda_arms`）：
training/evaluation/bootstrap 块逐字复制 `calibrated-canonical-confirmation.yaml`（E-056/E-057
契约：E-052 冻结 control、`update_count=100`、`scheduler_total_optimizer_steps=100`、
`gradient_diagnostics=true`、held-out S/SC seeds 16–23 / runtime seed 760025 / horizon 300 /
ddim5、bootstrap 0.95/10000/seed 0），arms/contrasts 采用 E-055 dose-response 的 5 臂结构
（r0/lam1/lam2/lam4/lam8，contrasts `[[r0,lam1],[r0,lam2],[r0,lam4],[r0,lam8]]`）。

**E-057 产物复用（双 pin provenance）**：r0/lam4/lam8 的 9 个训练 run（pin `07b2860`）与
30 个 held-out evals（pin `986b015`）原位复用（manifest `../e-057-.../` 引用，产物不复制）；
lam1/lam2 × seeds {0,1,2} 新训练（6 runs）+ 18 个新 evals，pin `73b9db4`（预注册 commit，
`git_status_short=[]`）。复用臂与新臂的 matched 一致性由 `load_comparison` 硬校验验证
（见 Matched guards）。

## 命令

```powershell
# 6 个新 matched training runs（lam1/lam2 × seed 0/1/2，100 updates，T_max=100）
just exp compare train --config configs/experiments/comparison/calibrated-canonical-candidate-revalidation.yaml `
  --arm lam1 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/lam1-seed-0
# ... 其余 5 个组合同理；日志存 study 根 train-<arm>-seed-<n>.log

# 离线机械诊断（15 summaries：E-057 9 个 ../e-057-.../ 引用 + 本 study 6 个）
just exp training diagnose `
  --config .../e-058-issue114-task4-candidate-revalidation/diagnose.yaml `
  --output-dir outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-diagnostics

# held-out 评测新增 18 次（lam1/lam2 × seed × {update-025, update-050, final}；
# initial 复用 E-057 heldout/initial-seed-{0,1,2}，initial policy 按 seed 跨臂共享）
just exp compare eval --config ...calibrated-canonical-candidate-revalidation.yaml `
  --arm lam1 --checkpoint update-050 --checkpoint-path .../lam1-seed-0/policy-update-049.pt `
  --output-dir .../e-058-.../heldout/lam1-seed-0-update-050
# ... 其余 17 个组合同理

# 离线配对比较（60 条目 manifest：5 臂 × 3 seeds × 4 labels；中间 checkpoint pin 重算 hash）
just exp compare analyze `
  --config .../e-058-issue114-task4-candidate-revalidation/comparison.yaml `
  --source-dir .../e-058-issue114-task4-candidate-revalidation `
  --output-dir outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-analysis
```

## Matched guards（runner/analysis 硬校验）

新 lam1/lam2 runs 在 clean commit `73b9db4` 执行（cuda:0 / bf16-mixed）；canonical cadence
resolved `0.1 × 5 = 0.5 s`；frozen planner hash 前后不变。**跨 commit 复用的 matched 一致性
全部通过 `load_comparison` 硬校验**：per-seed initial policy hash 各臂逐值一致（seed 0
`56bb97ab0551…`、seed 1 `f91a826f38a2…`、seed 2 `0febc900778c…`，新 runs 与 E-057 runs
逐值相同）；`probe_before` 一致（boundary mass 0.020996）；resolved training conditions
逐值相等（modulo `reward`/`tracking`——`arm`/`protocol` 标签位于 `tracking.tags`，被
matched-condition 比较显式排除）；30 个中间 checkpoint 的 manifest pin hash 与 evaluation
job 重算 hash 交叉校验全部通过。LR history：全部 15 runs 同一 cosine 调度
（1.5e-4 → 3.7e-8，cumulative exposure 0.0076，复用臂与新臂 artifact 序列逐值一致）。

## 训练与诊断结果（15-run 状态表）

| run | updates | min α / min β | probe_after boundary mass | 纵向 mean u75–100 | MC KL max |
| --- | ---: | --- | --- | --- | --- |
| r0-seed-0* | 100 | 1.531 / 1.933 | 0.039 | +0.042 | 1.80e-4 |
| r0-seed-1* | 100 | 1.974 / 1.574 | 0.038 | +0.172 | 2.15e-4 |
| r0-seed-2* | 100 | 1.997 / 1.449 | 0.061 | +0.212 | 4.52e-4 |
| lam1-seed-0 | 100 | 1.527 / 2.000 | 0.040 | −0.235 | 2.89e-4 |
| lam1-seed-1 | 100 | 1.916 / 1.587 | 0.038 | −0.088 | 1.85e-4 |
| lam1-seed-2 | 100 | 1.919 / 1.452 | 0.061 | −0.072 | 3.11e-4 |
| lam2-seed-0 | 100 | 1.285 / 2.000 | 0.091 | −0.436 | 6.65e-4 |
| lam2-seed-1 | 100 | 1.684 / 1.574 | 0.039 | −0.314 | 3.72e-4 |
| lam2-seed-2 | 100 | 1.416 / 1.444 | 0.066 | −0.347 | 5.30e-4 |
| lam4-seed-0* | 100 | 1.105 / 2.000 | 0.172 | −0.593 | 1.01e-3 |
| lam4-seed-1* | 100 | 1.191 / 1.576 | 0.152 | −0.575 | 8.05e-4 |
| lam4-seed-2* | 100 | 1.111 / 1.454 | 0.167 | −0.582 | 1.05e-3 |
| lam8-seed-0* | 100 | 1.038 / 2.000 | **0.242** | −0.677 | 1.33e-3 |
| lam8-seed-1* | 100 | 1.051 / 1.626 | **0.243** | −0.679 | 1.16e-3 |
| lam8-seed-2* | 100 | 1.031 / 1.513 | **0.258** | −0.690 | 1.20e-3 |

\* 复用 E-057 runs（pin `07b2860`），数值取自本实验 15-run diagnose 对原产物的重新计算，
与 E-057 diagnostics artifact 逐值一致（E-057 记录表中个别 min α/β 第三位小数与其自身
artifact 有出入，如 r0-seed-1 1.966→1.974、lam8-seed-1 1.046→1.051；不影响该记录任何
gate 判定，本表以 artifact 为准）。

15/15 runs 完整完成，无 invalid action / non-finite log-prob / exact-boundary failure；
analytic / seeded-MC / k3 KL 全部有限、同量级、无 runaway（≪ target_kl 0.006）；clip
fraction median 0.0、ratio p95 median ≤ 1.020。

**纵向 mean 的 λ 剂量梯度（u75–100 段，seed 均值）单调且 3/3 seeds 同向**：
λ=0 +0.14 → λ=1 −0.13 → λ=2 −0.37 → λ=4 −0.58 → λ=8 −0.68；λ=2 的 seed 0（−0.436）距
预注册 saturation 阈值 −0.5 仅 0.064，λ=1 各 seed 距阈值 ≥ 0.265。probe guidance RMS
shift：lam1 0.165–0.254、lam2 0.278–0.361（λ=4/8 为 0.447–0.544）。lam2 纵向
concentration 较初始 +0.36（轻度收窄，guidance contrast），lam1 −0.02（无收窄）。

## Held-out 配对效应（vs R0，16 场景，bootstrap 95% CI）

energy-intensity（mL/km）point estimate [CI]（3 seeds；update-050 → final 同号且单调加深，
全部 CI 不含 0）：

| contrast | update-050（seed 0/1/2） | final（seed 0/1/2） | final 相对效应 |
| --- | --- | --- | --- |
| lam1−r0 | −0.294 / −0.290 / −0.313 | −0.573 / −0.619 / −0.691 | −1.22% / −1.30% / −1.45% |
| lam2−r0 | −0.508 / −0.525 / −0.578 | −0.849 / −1.036 / −1.177 | −1.80% / −2.18% / −2.47% |
| lam4−r0 | −0.687 / −0.764 / −0.881 | −0.995 / −1.318 / −1.405 | −2.11% / −2.78% / −2.95% |
| lam8−r0 | −0.803 / −0.910 / −1.050 | −1.040 / −1.345 / −1.424 | −2.20% / −2.83% / −2.99% |

final mean_speed（m/s）：lam1 −0.338/−0.341/−0.378；lam2 −0.529/−0.601/−0.682（全部
CI 不含 0）。final route_completion：lam1 −0.0040（CI 不含 0）/ −0.0026（含 0）/
−0.0034（含 0）；lam2 −0.0053 / −0.0067 / −0.0067（**3/3 CI 不含 0**）。stopped_fraction
差异全部 seed 恒为 0。任务结构（final，每 seed）：completed 16/16 全臂；arrival：r0
13/16 ×3，lam1 13/16 ×3（与 R0 持平），lam2 12/13/12（2/3 seeds 少 1 次 arrival，多 1 次
max_step 截断）；collision / out-of-road / wrong-direction：lam1/lam2/r0 全 0（lam4 出现
OOR 2/4/4、lam8 2/5/5——两否决臂的附加任务结构退化，如实记录）。

## 预注册标准裁定

### 否决 gates（§4.1，每 λ × 3 seeds 须全过；复用臂继承 E-057 同口径诊断）

| gate | r0（对照） | λ=1 | λ=2 | λ=4 | λ=8 |
| --- | --- | --- | --- | --- | --- |
| 3/3 seeds 完成 | ✓ | ✓ | ✓ | ✓ | ✓ |
| min α/β > 1+ε | ✓ | ✓ | ✓ | ✓ | ✓ |
| 无 invalid / non-finite / exact-boundary | ✓ | ✓ | ✓ | ✓ | ✓ |
| KL（analytic/MC/k3）有限无 runaway | ✓ | ✓ | ✓ | ✓ | ✓ |
| probe_after boundary mass ≤ 0.2 | ✓（≤0.061） | ✓（0.038–0.040） | ✓（0.039–0.091） | ✓（0.152–0.172） | **✗（0.242–0.258，3/3）** |
| 无 sustained saturation（u75–100 纵向 mean > −0.5） | ✓（微正） | ✓（−0.235~−0.072） | ✓（−0.436~−0.314） | **✗（−0.575~−0.593，3/3）** | **✗（−0.677~−0.690，3/3）** |

**λ=8、λ=4 被否决**（分别超 ceiling + saturation、saturation；均 3/3 seeds，继承 E-057）。
**λ=1、λ=2 通过全部否决 gates**，进入提名判定。

### 提名要求（§4.2，在通过否决 gates 的 λ 中）

| 要求 | λ=1 | λ=2 |
| --- | --- | --- |
| 仅 held-out matched evaluation（不用 training return） | ✓ | ✓ |
| final energy-intensity 3/3 seeds 同向（负） | ✓ | ✓ |
| final energy-intensity CI 3/3 不含 0 | ✓ | ✓ |
| 相对效应 ≥ 0.1% 噪声界 | ✓（1.22–1.45%，12–15×） | ✓（1.80–2.47%） |
| stopped_fraction 差异为 0 | ✓ | ✓ |
| route_completion 无系统性完成率下降 | ✓（3/3 point −0.26~−0.42%，2/3 CI 含 0；arrival 13/16 与 R0 持平） | **✗（3/3 CI 不含 0，−0.56~−0.71%；arrival 12/16 2/3 seeds）** |
| 无新增 termination 异常（collision/OOR/wrong-direction） | ✓（0/0/0） | ✓（0/0/0；arrival 减少表现为 max_step 截断） |
| safety 无退化 | ✓ | ✓ |
| 50→100 方向稳定（§4.3） | ✓（u050→final 同号单调加深，3/3） | ✓ |

**λ=2 因任务结构不一致未获提名**：route_completion 3/3 seeds 系统性下降（final CI 全不含
0，−0.56%~−0.71%），叠加 2/3 seeds arrival 13→12（max_step 截断 +1）。**λ=1 通过全部提名
要求，为唯一被提名的候选**（提名规则：多于一个通过时取 energy-intensity 效应最大者——
仅 λ=1 通过，无歧义）。

### Gate S3（#83 re-entry）

```text
new policy family frozen（ADR 0040, commit 03a6c99）        ✓
AND scheduler semantics matched（Task 1, commit 53a9a94）    ✓
AND candidate selection revalidated（λ=1，本实验）           ✓
AND candidate long-run stability passes（λ=1 3/3 seeds）     ✓
→ Gate S3 PASSED：#83 解除 block，λ=1 进入 #83 final A2 路径
```

## 与预注册计划的偏离

1. **manifest 初版两处转写错误，均在硬校验下即刻 loud-fail 后修正**：(a) `diagnose.yaml`
   初版 r0 条目漏写 `../e-057-.../` 前缀（FileNotFoundError）；(b) `comparison.yaml` 初版
   lam8-seed-1 update-050 的 pin hash 第 19 字符转写错误（`98a7…`→`98e7…`，
   `evaluation checkpoint does not match the declared training state`）。两者均为手写
   manifest 的转写笔误，不触及任何训练/评测产物，修正后全部校验通过——这正是 pin hash
   交叉校验的设计目的。
2. **其余无偏离**：6 训练 + 18 评测 + diagnose + analyze 与预注册一致执行；无崩溃、无
   seed 删除、无参数调整、无标准放宽。

## 局限与结论边界

- **结论范围**：本记录裁定的是「interior family + 100-update scheduler contract +
  预注册 selection 标准下 λ={0,1,2,4,8} 的候选资格」。λ=1 的提名是对该标准的裁定结果，
  不构成对 energy proxy 有效性或节能机理的判断（#113 边界）。
- **候选身份变更**：E-055（旧 family、50 updates、T_max=50）提名 λ=4/λ=8 且 λ=1 因
  update-025 未分辨而落选；本实验（新 family、100 updates）λ=1 在 update-050 与 final 均
  3/3 分辨（1.2–1.5% 相对效应，CI 不含 0）——效应随累计 exposure 增长（u050 −0.29~−0.31
  → final −0.57~−0.69 mL/km）。两代提名不可直接比较（family、schedule、horizon 均不同），
  E-055 仅保留为旧 family 历史证据。
- **λ=1 的效应载体**：与全部 λ 臂一致，energy-intensity 降低主要由降速载带
  （speed −3.4%~−3.8% vs intensity −1.2%~−1.5%，progress 代价 −0.26%~−0.42%）——
  「降速即节能」objective shortcut 假设（#113）在候选 λ=1 上同样成立，作为 #83 final
  A2 解释的必带 caveat。
- **λ=2 的边界位置**：λ=2 通过全部否决 gates 但纵向 mean（seed 0 −0.436）距 saturation
  阈值仅 0.064、boundary mass 0.091 为通过臂中最高、concentration 轻度收窄（+0.36）——
  形态介于 λ=1 与 λ=4 之间；其落选由任务结构裁定而非 distribution gates，如实记录。
- **saturation 阈值的来源**：−0.5 阈值与 0.2 ceiling 沿用 E-057（Task 3）预注册判据，
  本实验未新设阈值；λ=1 各 seed 距阈值 ≥ 0.265，裁定对该阈值的具体取值不敏感。

## 对后续任务的输入

- **#83（re-entry / final A2）**：Gate S3 PASSED，解除 block。候选为 **λ=1**（`arm lam1`，
  `plannerrft_energy_band_lam1_v1`，seeds {0,1,2} 训练产物与 held-out evals 已在
  E-058/E-057 study 目录）。final A2 执行时沿用本实验冻结契约；E-055 的 λ=4/8 nomination
  正式失效（provenance 边界由 ADR 0040 + 本记录关闭）。
- **Task 5（reference-policy trust region）**：触发条件（interior 约束通过但长期 mean
  saturation）在 λ≥4 臂上成立（E-057 + 本实验复确认）；λ=1/λ=2 未饱和。Task 5 是否研究
  另行决策，本实验不引入。
- **#113（energy proxy / reward validity）**：全部 λ 臂（含候选 λ=1）的 energy-intensity
  降低均由降速载带且随 λ 单调加深（speed seed 均值 −0.35/−0.60/−0.78/−0.83 m/s），
  shortcut 假设的方向性证据进一步积累。
- **Issue #114**：Task 4 输出判据满足；剩余未完成项为「明确 #83 是否可 re-entry」的
  issue 勾选写回（本记录 + Gate S3 裁定为权威依据）。

## 产物

| 产物 | 路径 |
| --- | --- |
| 协议 | `configs/experiments/comparison/calibrated-canonical-candidate-revalidation.yaml`（pin 测试：`tests/training/test_training_workflows.py`） |
| 预注册计划 | `docs/research/issue114-task4-candidate-revalidation-plan.md`（状态：已执行） |
| 新训练 runs（6） | `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/lam{1,2}-seed-{0,1,2}/`（pin `73b9db4`） |
| 复用训练 runs（9） | `outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/{r0,lam4,lam8}-seed-{0,1,2}/`（pin `07b2860`） |
| 训练日志（新） | `.../e-058-.../train-{arm}-seed-{n}.log` |
| 训练诊断（15 runs） | `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-diagnostics/`（config：study dir `diagnose.yaml`） |
| Held-out 评测（新增 18） | `.../e-058-.../heldout/lam{1,2}-seed-{n}-{update-025,update-050,final}`；initial 复用 `.../e-057-.../heldout/initial-seed-{0,1,2}`（pin `986b015`） |
| 配对比较 manifest | `.../e-058-.../comparison.yaml`（60 条目，中间 checkpoint pin 重算 hash） |
| 配对比较分析 | `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-analysis/` |
| 诊断 / 分析日志 | `.../e-058-.../e-058-diagnose.log`、`e-058-analyze.log`、`e-058-eval-*.log` |
