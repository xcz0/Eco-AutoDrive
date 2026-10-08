# Issue #114 Task 3 — Previously failing candidate stress test 执行计划

状态：**预登记**（2026-10-08，运行前冻结；执行结果见
`docs/experiments/records/e-057-issue114-task3-interior-stress.md`）。

任务边界、验收标准与冻结契约的权威来源是
[Issue #114](https://github.com/xcz0/Eco-AutoDrive/issues/114)（Parent #80；Task 1
scheduler-matched contract 与 Task 2 Beta interior parameterization 已完成：commit
`53a9a94` / `03a6c99`，ADR 0040）。本计划在正式运行前预注册判定标准；运行后不依结果调整。

## 1. 任务定义（来自 Issue #114 Task 3）

在新 Beta interior policy family（α, β 恒 > 1 + `interior_epsilon`=1e-4，ADR 0040）下，
保持 energy provider 与 reward 不变，重跑 E-056 失败矩阵：

```text
arms   = {r0, lam4, lam8}
seeds  = {0, 1, 2}
updates = 100          # ppo.scheduler_total_optimizer_steps = 100（同一 scheduler history）
checkpoint = 0 / 25 / 50 / 100
```

目的不是恢复 #83 final A2，而是回答四个研究问题：

1. 原 E-056 boundary collapse 是否被**结构性消除**（numerical collapse）；
2. longitudinal mean 是否仍持续向负边界靠近（behavioral saturation）；
3. λ=4/8 是否存在"数值不崩，但行为饱和"的第二层问题；
4. 50→100 的 guidance / speed / progress / energy effect 是否方向稳定。

必须报告：longitudinal/lateral Beta mean、variance、concentration；min alpha/beta；
boundary mass；policy ratio / clip fraction；analytic/MC/k3 KL；gradient groups /
parameter delta；per-update learning rate 与 cumulative exposure；mean speed /
progress / stopped / safety；total proxy energy / mL/km。

## 2. 现行机制勘察结论

| 环节 | 入口 | 现状 |
| --- | --- | --- |
| matched 多臂训练 | `just exp compare train` | 协议 `calibrated-canonical-confirmation.yaml` 已精确编码 Task 3 规格（seeds {0,1,2}、`update_count=100`、`T_max=100`、`gradient_diagnostics=true`），有 pin 测试守卫，无需改动 |
| 训练期指标持久化 | trainer per-update `summary.json` | β α/β mean/min/max、ratio/clip、mean speed/progress/stopped/safety、proxy energy、mL/km 均已持久化 |
| 离线机械诊断 | `just exp training diagnose` | `extract_arm_metrics` + `post_update_kl_series`（analytic/seeded-MC/k1/k3）+ `learning_rate_history`（Task 1 新增，含 cumulative exposure）+ per-group gradient + parameter delta；单一 diagnose config 可覆盖全部 9 个 run（E-055 先例） |
| boundary mass | in-run probe | 仅 update 0（`probe_before`）与 final（`probe_after`），`boundary_distance=0.05`、4096 draws、`diagnostic_seed=300000`；中间 checkpoint 不重算 probe（本轮明确不做，见 D2） |
| held-out 评测（含中间 checkpoint） | `just exp compare eval` | `CheckpointLabel` 任意字符串 + `--checkpoint-path`；`update-025`=`policy-update-024.pt`、`update-050`=`policy-update-049.pt` |
| 配对比较 + bootstrap | `just exp compare analyze` | manifest `comparison.yaml`，中间 checkpoint 条目 pin 重算 hash（evaluation job 重算 hash 交叉校验，E-055 先例） |

关键事实：

- **E-056 产物全部不可复用**：ADR 0040 改变初始 actor bias 与 `policy_state_hash`，
  且 `minimum_concentration` 字段已被 `interior_epsilon` 替换；9 个 run 必须全部重训。
- **新初始 policy hash 与 E-056 不同是预期行为**（policy family 变更的直接后果），
  不是失败；跨臂守卫仍要求同一 seed 各臂 initial hash / seeds / probe_before 逐值一致。
- checkpoint 语义：checkpoint N（N updates 后）= `policy-update-(N-1).pt`；
  checkpoint 0 = `policy-initial.pt`；checkpoint 100 = `policy-final.pt`。
- initial policy 按 training seed 共享（同 seed 各臂一致），故每 seed 只需一次
  shared-initial held-out eval（E-055 `heldout/shared-initial` 先例）。

## 3. 设计决策

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 复用 `calibrated-canonical-confirmation.yaml`，不改任何配置 | 协议已精确编码 Task 3 运行矩阵与 E-052 冻结 control；改协议即改实验语义 |
| D2 | 不新增中间 checkpoint boundary mass 分析代码 | per-update β mean/concentration/min αβ 连续序列 + probe 0/100 两点已可回答 Q2/Q3；避免为单一测量扩代码面（与使用方确认的范围决策） |
| D3 | held-out eval 完整覆盖 0/25/50/100：每 seed shared initial（3）+ 每臂 update-025 / update-050 / final（27），共 30 次 | Q4 的「effect」按 E-055 惯例需要 matched held-out 配对对比 + bootstrap CI，且 Issue 明确列出 4 个 checkpoint（与使用方确认的范围决策） |
| D4 | 单一 `diagnose.yaml` 覆盖 9 个 run（mc_draws=4096、mc_seed=1000003） | E-055 单 config 多臂先例；新 family 下预期无 collapse，9 个 summary 均可诊断 |
| D5 | Gate S1/S2 在实验记录内裁定，不新增 gate runner | 与 E-055 Gate D / E-056 Gate M 一致 |
| D6 | 实验编号 E-057；study root `outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/` | 当前最高记录 E-056；per-experiment 命名使 resolved config/artifact 可唯一恢复协议 |
| D7 | 先 clean commit（本文件）再正式运行 | 正式 run 必须 pin clean commit；9 个 run 与后续诊断 pin 同一 commit |
| D8 | 训练崩溃即停止：保留 log / partial checkpoints / `runtime_metadata.json` / `resolved_config.yaml`，不删目录、不调参重试 | Gate S2 已失败；失败证据是本任务的一等产物（E-056 先例） |

## 4. 预注册判定标准（运行前冻结）

### 4.1 Numerical collapse（Q1，结构性消除验证）

任一条件不满足 → numerical collapse 未消除：

- 9/9 runs 完整完成 100 updates（`summary.json` + `policy-final.pt` 齐备）；
- 全部 update 的 `beta_alpha_min` / `beta_beta_min`（两维）> 1 + 1e-4；
- 无 rollout host tensor non-finite / invalid action / exact-boundary failure；
- diagnose 的 analytic / seeded-MC / k3 KL 全部有限且无 runaway（无 FloatingPointError）。

### 4.2 Boundary mass ceiling（沿用 E-056 口径）

- probe `boundary_mass`（`boundary_distance=0.05`）在 `probe_before` 与 `probe_after`
  的 scenario max（每维）≤ **0.2**；超出即记为 boundary-mass ceiling violation。

### 4.3 Behavioral saturation（Q2/Q3，与 numerical collapse 显式区分）

不构成 Task 3 失败本身，但作为 Gate S1「entropy/concentration 不发生另一种退化」、
Task 5（reference-policy trust region）是否触发、以及 #113 的判断输入：

- longitudinal（dim 1）Beta mean 在 update 75–100 段的均值相对 update 0 持续偏负，
  且 guidance-mean 距 -1 boundary 的接近程度超过 **0.5**（即 mean ≤ -0.5）→ 记为
  sustained negative saturation；
- 或 probe_after longitudinal boundary mass > 0.2 → 同上；
- lateral（dim 0）不预设 saturation 判定，如实报告。

### 4.4 50→100 方向稳定性（Q4）

- held-out 配对对比（lam4−r0、lam8−r0）的 guidance / speed / progress / energy-intensity
  effect 在 update-050 与 final 的 point estimate 同号，且 seed 间方向一致 → 方向稳定；
- 出现 update-050 → final 方向反转或 seed 间不一致 → 如实记录为不稳定。

### 4.5 Gate S1 / Gate S2 逐项映射

- Gate S1（distribution stability）：α/β 永不进入 singular region（4.1 第二条，
  结构上已排除，验证性勾选）、3/3 seeds 无 invalid/non-finite（4.1）、boundary mass
  ≤ ceiling（4.2）、KL 有限无 runaway（4.1）、entropy/concentration 无另一种退化
  （4.3 如实裁定）、R0 与 energy arms 均可完整运行（4.1 第一条）。
- Gate S2（long-run mechanical stability）：3/3 seeds 完成（4.1）、无 Beta boundary
  collapse（4.1）、无 exact-boundary/non-finite failure（4.1）、50→100 无
  distribution/behavior collapse（4.3 + 4.4 联合裁定）、R0 保持健康、
  λ=4/8 结果不是单 seed 异常（4.4 seed 一致性）。

## 5. 实现步骤

### Step 1 — 预登记方案（本文件）+ clean commit

### Step 2 — 正式训练（沙箱外，GPU + MetaDrive；pin 本 commit）

study root：`outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/`。

```powershell
# 9 matched training runs（r0/lam4/lam8 × seed 0/1/2，100 updates，canonical k=5 + E-052 冻结 control）
just exp compare train --config configs/experiments/comparison/calibrated-canonical-confirmation.yaml `
  --arm r0 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/r0-seed-0
# ... 其余 8 个 arm × seed 组合同理；日志按 E-056 惯例存 study 根 train-<arm>-seed-<n>.log
```

先跑 r0-seed-0 健康检查（probe_before、新 family 初始 hash、首批 update 正常），
再顺序完成其余 8 个。运行后核对：每 seed 各臂 initial policy hash 逐值一致、
frozen planner hash 前后不变、resolved cadence `0.1×5=0.5 s`。

### Step 3 — 离线机械诊断（9 summaries）

```powershell
just exp training diagnose `
  --config outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress/diagnose.yaml `
  --output-dir outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress-diagnostics
```

### Step 4 — held-out 评测（30 次）+ 离线配对比较

```powershell
# 每 seed shared initial（3 次，--arm r0 组合，同 seed 各臂共享 initial policy）
just exp compare eval --config configs/experiments/comparison/calibrated-canonical-confirmation.yaml `
  --arm r0 --checkpoint initial `
  --checkpoint-path .../e-057-.../r0-seed-<n>/policy-initial.pt `
  --output-dir .../e-057-.../heldout/initial-seed-<n>
# 每臂 × seed 的 update-025（policy-update-024.pt）/ update-050（policy-update-049.pt）/ final（27 次）
just exp compare eval --config ... `
  --arm lam8 --checkpoint update-050 --checkpoint-path .../lam8-seed-2/policy-update-049.pt `
  --output-dir .../heldout/lam8-seed-2-update-050
# ... 其余组合同理

# manifest comparison.yaml：update-025/update-050 条目 pin evaluation job 重算 hash
just exp compare analyze `
  --config .../e-057-issue114-task3-interior-stress/comparison.yaml `
  --source-dir outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress `
  --output-dir outputs/studies/scalar-reward/e-057-issue114-task3-interior-stress-analysis
```

### Step 5 — 判定与记录

- 按 §4 预注册标准裁定 Gate S1 / Gate S2，写
  `docs/experiments/records/e-057-issue114-task3-interior-stress.md` 并更新
  `docs/experiments/README.md` 索引；
- 显式区分 numerical collapse 与 behavioral saturation 的结论；
- 更新 Issue #114：Task 3 输出判据、Gate S1/S2 勾选状态、Task 4（candidate
  provenance revalidation）的输入；本文件状态改为「已执行」。

## 6. 风险与检查点

| 风险 | 检查/应对 |
| --- | --- |
| 新 family 下仍出现崩溃 | 按 D8 保留证据并停止；如崩溃形态非 boundary（如 NaN gradient），如实记录为新失败模式，不混入 E-056 机制链 |
| probe_after boundary mass 超 0.2 但无数值失败 | 按 §4.3 记为 behavioral saturation 证据，不掩盖、不触发 Task 5（属后续单独决策） |
| 30 次评测 + 9×100-update KL/MC 重算计算量大 | 沙箱外 GPU 执行；mc_draws 沿用 4096 |
| 新初始 hash 与全部历史实验不同 | 预期行为（ADR 0040）；仅要求同 seed 跨臂一致 |
| λ arm effect 在 3 seeds 间不一致 | 这正是 Gate S2「不是单 seed 异常」要检验的对象；如实记录，不筛 seed |

## 7. Non-goals

- 不修改 PPO 超参、reward representation、band 阈值、cadence、λ、energy provider（#113）；
- 不恢复 #83 final A2、不执行 Task 4 candidate revalidation（仅在记录中给出其输入）；
- 不引入 Task 5 reference-policy trust region 或第二阶段 mean+concentration 参数化；
- 不通过 action clamp / validate_args=False / 跳过 transition 掩盖失败；
- 不把 training return 作为判断依据。
