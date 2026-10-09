# Issue #114 Task 4 — candidate provenance revalidation 执行计划

状态：**预注册**（2026-10-09 运行前冻结；判定标准不依结果调整；执行后状态与偏离见
`docs/experiments/records/e-058-issue114-task4-candidate-revalidation.md`）。

任务边界、验收标准与冻结契约的权威来源是
[Issue #114](https://github.com/xcz0/Eco-AutoDrive/issues/114) Task 4（Parent #80）。
上游输入：Task 2 Beta interior parameterization（commit `03a6c99`，ADR 0040）、
Task 1 scheduler-matched contract（commit `53a9a94`）、Task 3 / E-057 stress test
（训练 pin `07b2860`，诊断/评测 pin `986b015`）。

## 1. 任务定义（来自 Issue #114 Task 4）

ADR 0040 改变了 Exploration Policy family（初始 actor bias、`policy_state_hash`、可达分布
族），E-055 的 λ=4/8 nomination 只能视为**旧 policy family 下的历史证据**。在新 family 与
Task 1 scheduler contract 冻结后，重新确认 candidate selection 是否成立：

```text
arms    = {r0, lam1, lam2, lam4, lam8}    # λ = 0/1/2/4/8，仅 λ 变化
seeds   = {0, 1, 2}
updates = 100                              # ppo.scheduler_total_optimizer_steps = 100
checkpoint = 0 / 25 / 50 / 100
```

Issue 最低要求：

- 同一 scheduler contract（100 updates / T_max=100 / 单一 scheduler history）；
- 新冻结 policy parameterization（`interior_epsilon=1e-4`，已是 live 默认）；
- λ={0,1,2,4,8} candidate comparison，不与 PPO/reward/provider 搜索混合；
- selection 仍依据 matched held-out behavior/energy，不按 training return；
- candidate 必须同时满足 short-horizon effect 与 long-run mechanical stability。

## 2. 现行机制勘察结论（复用可行性）

| 环节 | 入口 | 现状 |
| --- | --- | --- |
| 5 臂 matched 训练 | `just exp compare train` | 新 protocol YAML（本计划 Step 1）：E-055 dose-response 的 arm/contrast 结构 + E-057 的 100-update 契约逐字合并；5 个 reward profile 全部已存在 |
| E-057 产物复用 | `comparison.yaml` / `diagnose.yaml` manifest | 路径均为 `root / path` 普通拼接，`../e-057-.../` 跨 study 引用可行；E-055 复用 E-054 r0-seed-0 run + heldout final 有先例 |
| 跨 protocol matched 校验 | `load_comparison`（`comparison/inputs.py`） | `compose_arm_training_config` 注入的 `arm`/`protocol` 标签位于 `tracking.tags`，matched-condition 比较显式 pop `reward` 与 `tracking`——E-057 runs（confirmation 协议）在新 protocol 下校验通过的前提是新 YAML 的 training overrides / scenario pool / reward profile 逐值相同（本计划保证） |
| held-out 评测 | `just exp compare eval` | `CheckpointLabel` + `--checkpoint-path`；update-025=`policy-update-024.pt`、update-050=`policy-update-049.pt` |
| 离线机械诊断 | `just exp training diagnose` | 单 config 可列 15 个 summaries（跨 study `../` 引用同上可行）；`beta_series` 覆盖全部 100 updates，供 saturation 裁定 |
| 配对比较 + bootstrap | `just exp compare analyze` | manifest pin 重算 hash；bootstrap 0.95 / 10000 / seed 0 |

关键事实：

- E-057 的 9 个训练 run（r0/lam4/lam8 × seeds {0,1,2}，100 updates，新 family，pin
  `07b2860`）与 30 个 held-out evals（pin `986b015`）已满足 Issue 对 scheduler contract 与
  policy parameterization 的全部要求；
- per-seed initial policy hash（seed 0 `56bb97ab0551…`、seed 1 `f91a826f38a2…`、
  seed 2 `0febc900778c…`）：新 lam1/lam2 runs 在相同 seed 下必须逐值一致（policy init 自
  `03a6c99` 后未变，`986b015` 仅测量层），不一致时 manifest 校验 loud-fail（见 §6）；
- r0 不参与提名（λ=0 无 energy 效应，为对照臂）；lam1/lam2 是唯一需要新训练的臂。

## 3. 设计决策

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | **复用 E-057**：r0/lam4/lam8 的 9 个训练 run + 30 个 held-out evals 直接进入 E-058 manifest；仅新增 lam1/lam2 × 3 seeds 训练（6 runs）+ 18 evals | E-057 runs 已满足 Issue 全部契约且 pin 干净 commit；避免 9 个冗余 100-update run；matched 校验（resolved config / initial hash / seeds / probe 逐值）由 `load_comparison` 硬验证，不一致即 loud-fail；E-055 复用 E-054 r0 run 有先例（使用方决策） |
| D2 | 新建 protocol `calibrated-canonical-candidate-revalidation.yaml`：training/evaluation/bootstrap 块逐字复制 confirmation 协议，arms/contrasts 采用 dose-response 的 5 臂结构 | 保证新 lam1/lam2 runs 与 E-057 runs 的 resolved training conditions 逐值相同（matched 校验的前提）；仅扩展 arm 集与 contrast 集，不改任何训练/评测语义 |
| D3 | held-out eval：新增 lam1/lam2 × 3 seeds × {update-025, update-050, final} = 18 次；initial 复用 E-057 per-seed shared-initial evals（initial policy 按 seed 跨臂共享） | 与 E-057 D3 同口径；manifest 中 initial 条目按 seed 指向 E-057 `heldout/initial-seed-{n}` |
| D4 | 单一 `diagnose.yaml` 列 15 个 summaries（`../e-057-.../` 9 个 + 本 study 6 个；mc_draws=4096、mc_seed=1000003） | lam1/lam2 的 boundary mass / saturation / KL 需与 E-057 同口径诊断；E-055 单 config 多臂先例 |
| D5 | **sustained negative longitudinal saturation 为候选否决条件**（使用方决策，运行前冻结）：沿用 E-057 §4.3 预注册口径——update 75–100 段纵向（dim 1）Beta mean ≤ −0.5 即否决 | behavioral saturation 与 numerical collapse 显式区分（Task 3 主要发现）；「数值不崩但持续贴边收窄」的 policy 不构成可交付 #83 的稳定候选；阈值沿用已有预注册判据，不新设事后标准 |
| D6 | boundary mass ceiling 0.2 沿用 E-057 §4.2 口径，同为否决条件 | λ=8 已 3/3 seeds 超限（0.242/0.243/0.258，E-057），继承判定 |
| D7 | Gate S3 在 E-058 实验记录内裁定，不新增 gate runner | E-055 Gate D / E-056 Gate M / E-057 Gate S1-S2 先例 |
| D8 | 实验编号 E-058；study root `outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/`（仅存放新 lam1/lam2 runs、manifests 与日志；复用产物留在 E-057 study root） | 当前最高记录 E-057；复用产物不复制、原位引用（hash/provenance 不变） |
| D9 | 先 clean commit（本文件 + protocol + pin 测试）再正式运行；新 runs 全部 pin 该 commit；复用产物保留原 pin（`07b2860`/`986b015`），E-058 记录双 pin 登记 | 正式 run 必须 pin clean commit；复用产物的 provenance 不因重新引用而改变 |
| D10 | 训练崩溃即停止：保留 log / partial checkpoints / `runtime_metadata.json` / `resolved_config.yaml`，不删目录、不调参重试 | 失败证据是一等产物（E-056/E-057 先例） |

## 4. 预注册判定标准（运行前冻结）

r0 为对照臂，不参与提名。每个候选 λ 的判定分两层：**否决 gates**（disqualifying，
任一不满足 → 该 λ 不可作为候选）与**提名要求**（nomination，在通过否决 gates 的 λ 中）。
复用臂（lam4/lam8）的否决 gates 继承 E-057 同口径诊断判定，不重新训练；lam1/lam2 用
新 runs 的同口径诊断裁定。

### 4.1 否决 gates（每 λ × 3 seeds 须全部通过）

机械稳定（E-057 §4.1 同口径）：

- 3/3 seeds 完整完成 100 updates；
- 全部 update 的 `beta_alpha_min` / `beta_beta_min`（两维）> 1 + 1e-4；
- 无 invalid action / non-finite log-prob / exact-boundary failure；
- diagnose 的 analytic / seeded-MC / k3 KL 全部有限且无 runaway。

分布/行为稳定（E-057 §4.2/§4.3 同口径）：

- probe_after boundary mass（`boundary_distance=0.05`，scenario max 每维）≤ **0.2**
  （3/3 seeds）；
- **无 sustained negative longitudinal saturation**：update 75–100 段纵向（dim 1）Beta
  mean > **−0.5**（3/3 seeds；任一 seed ≤ −0.5 即否决）；lateral（dim 0）如实报告、
  不预设判据。

已知输入（E-057，继承判定）：lam4 boundary mass 0.152–0.172 通过 ceiling 但纵向 mean
−0.575~−0.581 触发 saturation 否决；lam8 boundary mass 0.242–0.258 超 ceiling 且纵向
mean −0.676~−0.689 双重否决。**lam4/lam8 在本标准下已出局；lam1/lam2 由新数据裁定，
不预判。**

### 4.2 提名要求（在通过否决 gates 的 λ 中，全部满足才提名）

- 仅用 held-out matched evaluation（vs R0，16 场景，bootstrap 95% CI / 10000 resamples /
  seed 0），**不使用 training return**；
- final energy-intensity（mL/km）效应 3/3 seeds 同向（负，即节能）；
- final energy-intensity bootstrap 95% CI 不含 0（3/3 seeds）；
- energy-intensity 相对效应（seed 均值）≥ **0.1%**（E-031 / E-054 c2 噪声界）；
- 任务结构与 R0 一致：stopped_fraction 差异为 0、route_completion 无系统性完成率下降、
  无新增 termination 异常（collision / out-of-road / wrong-direction）、safety 无退化；
- 至多提名 2 个 λ；多于一个通过时取 energy-intensity 效应（seed 均值）最大者。

### 4.3 50→100 方向稳定性（short-horizon effect 佐证）

- held-out 配对对比（λN−r0）的 speed / energy-intensity / route_completion effect 在
  update-050 与 final 的 point estimate 同号且 seed 间方向一致 → 方向稳定；
- 出现 update-050 → final 方向反转或 seed 间不一致 → 如实记录，并在提名判定中作为
  short-horizon effect 不满足处理（否决）。

### 4.4 Gate S3（#83 re-entry）裁定规则

```text
re-entry iff ≥1 λ 被提名（提名 = 否决 gates 全过 + 提名要求全过，
                          已含 100-update long-run mechanical stability）
```

- 有候选提名 → Gate S3 通过，#83 解除 block，被提名 λ 进入 #83 final A2 路径；
- 无 λ 被提名 → Gate S3 不通过，#83 保持 blocked，candidate path 以
  「interior family + 预注册标准下无候选」关闭（对齐 E-056 计划的关闭措辞）；
- Task 5（reference-policy trust region）触发条件状态单独登记（E-057 已确认
  「interior 约束通过但长期 mean saturation」成立；lam1/lam2 结果不改变该登记），
  Task 5 本身不在本任务执行。

## 5. 实现步骤

### Step 1 — 预注册（本文件）+ protocol + pin 测试 + clean commit

新建 `configs/experiments/comparison/calibrated-canonical-candidate-revalidation.yaml`
（confirmation 契约 + 5 臂）与 `tests/training/test_training_workflows.py` pin 测试。

### Step 2 — 正式训练（沙箱外，GPU + MetaDrive；pin 本 commit）

仅 6 个新 run，study root
`outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/`：

```powershell
# lam1/lam2 × seed 0/1/2（100 updates，canonical k=5 + E-052 冻结 control + T_max=100）
just exp compare train --config configs/experiments/comparison/calibrated-canonical-candidate-revalidation.yaml `
  --arm lam1 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/lam1-seed-0
# ... 其余 5 个 arm × seed 组合同理；日志存 study 根 train-<arm>-seed-<n>.log
```

先跑 lam1-seed-0 健康检查（probe_before 与 E-057 一致 0.020996、initial policy hash 与
E-057 seed 0 `56bb97ab0551…` 逐值一致、首批 update 正常），再顺序完成其余 5 个。

### Step 3 — 离线机械诊断（15 summaries，含复用 9 个）

```powershell
just exp training diagnose `
  --config outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation/diagnose.yaml `
  --output-dir outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-diagnostics
```

### Step 4 — held-out 评测（新增 18 次）+ 离线配对比较

```powershell
# lam1/lam2 × seed × {update-025, update-050, final}
just exp compare eval --config configs/experiments/comparison/calibrated-canonical-candidate-revalidation.yaml `
  --arm lam1 --checkpoint update-050 --checkpoint-path .../lam1-seed-0/policy-update-049.pt `
  --output-dir .../e-058-.../heldout/lam1-seed-0-update-050
# ... 其余 17 个组合同理（initial 不重评，复用 E-057 heldout/initial-seed-{0,1,2}）

# manifest comparison.yaml：5 臂 × 3 seeds × 4 labels；复用条目 ../e-057-.../ 引用；
# update-025/update-050 条目 pin 重算 hash
just exp compare analyze `
  --config .../e-058-issue114-task4-candidate-revalidation/comparison.yaml `
  --source-dir outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation `
  --output-dir outputs/studies/scalar-reward/e-058-issue114-task4-candidate-revalidation-analysis
```

### Step 5 — 判定与记录

- 按 §4 预注册标准裁定候选提名与 Gate S3，写
  `docs/experiments/records/e-058-issue114-task4-candidate-revalidation.md` 并更新
  `docs/experiments/README.md` 索引；
- 更新 Issue #114（Task 4 判据、Gate S3、输出判据勾选）与 #83（re-entry 决定及理由）；
- 本文件状态改为「已执行」。

## 6. 风险与检查点

| 风险 | 检查/应对 |
| --- | --- |
| 新 lam1/lam2 runs 的 initial policy hash 与 E-057 per-seed 不一致 | manifest matched 校验 loud-fail（可见失败，符合预期）；复用前提不成立时停下，与使用方确认是否全量重跑 15 runs，不静默降级 |
| 新 protocol 组合的 resolved training conditions 与 E-057 runs 不一致（代码/配置演进引入字段差异） | 同上 loud-fail；协议 training/evaluation 块逐字复制可最大程度避免 |
| lam1/lam2 训练崩溃 | 按 D10 保留证据并停止；该臂按否决 gates 不通过处理，如实记录 |
| lam1/lam2 效应低于噪声界 / CI 含 0 | 预期可能结局（E-055 下 λ=1/λ=2 效应 ≤ ~2× 噪声界）：无候选 → 按 §4.4 关闭路径裁定，不放宽标准 |
| lam1/lam2 出现 saturation | 按否决 gates 裁定出局，如实记录；不因此调整阈值 |
| 复用 run 的 checkpoint 文件被移动/删除 | manifest hash 交叉校验失败即停；复用产物原位不动（D8） |

## 7. Non-goals

- 不修改 PPO 超参、reward representation、band 阈值、cadence、λ 取值、energy provider（#113）；
- 不执行 #83 final A0/A1/A2（Gate S3 裁定后另行）；
- 不执行 Task 5 reference-policy trust region（仅登记触发条件状态）；
- 不做第二阶段 mean + concentration parameterization；
- 不通过 action clamp / validate_args=False / 跳过 transition 掩盖失败；
- 不把 training return 作为判断依据；
- 不重训 r0/lam4/lam8（E-057 产物复用，见 D1）。
