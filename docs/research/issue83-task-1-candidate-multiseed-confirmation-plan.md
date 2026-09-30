# Issue #83 Task 1 — Candidate multi-seed long-run confirmation 执行计划

状态：**已执行**（2026-09-30，结果见
`docs/experiments/records/e-056-issue83-task-1-candidate-multiseed-confirmation.md`：Gate M FAILED，
两候选均未通过；实际执行与本计划的偏离——lam8-seed-1/2 未训练、held-out 评测未执行、diagnose 仅覆盖
r0——及理由见该记录「与预注册计划的偏离」）。本文件保留预登记方案原文。

任务边界、验收标准与冻结契约的权威来源是 [Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83)
（Parent #80；前置 #105 / #106 已完成关闭）。`#106` Candidate path 已 handoff 候选 **λ=4 与 λ=8**
（E-055）。

## 1. 任务定义（来自 Issue #83 Task 1）

对 calibrated R0 与每个 #106 surviving candidate，统一运行：

```text
training seeds = {0,1,2}
updates = 100
checkpoint = 0 / 50 / 100
```

50-update checkpoint 与 #106 coarse sweep 对齐；100-update checkpoint 只用于长期稳定性确认，**不用于
重新选择 PPO config 或 λ**。本 Issue 不重新搜索 PPO / reward representation / cadence / λ。

必须报告：analytic Beta KL + MC/k3 cross-check、policy ratio / clip fraction、actor/shared/value
gradient diagnostics、parameter delta、Beta mean/variance/concentration/boundary、guidance 分布、
episode length / reset frequency；route progress / completion / distance、mean speed / speed profile、
stopped fraction、collision / OOR / wrong-direction / termination type；total execution-trace proxy
energy、energy distance、energy intensity (mL/km)。

### Gate M — multi-seed stability

```text
[ ] 3/3 seeds 完成；失败 seed 不删除；
[ ] 3/3 seeds 无 Beta boundary collapse / KL runaway / invalid action；
[ ] candidate vs R0 的 energy-intensity effect 方向在 seed aggregate 中一致；
[ ] effect 不是只由单一 scenario family 或少数 outlier episode 驱动；
[ ] checkpoint 50→100 不发生方向性反转或 behavior collapse；
[ ] progress / speed / stopped / safety trade-off 可解释；
[ ] provenance 完整、commit clean。
```

某 candidate Gate M 失败则不进入 final A2，失败原因保留；全部失败则以
“candidate long-run instability / non-reproducibility” 收口。

## 2. 现行机制勘察结论

| 环节 | 入口 | 实现 |
| --- | --- | --- |
| matched 多臂训练 | `just exp compare train` | `experiments/comparison/runner.py::run_command` → `run_training_job` |
| held-out 评测（含中间 checkpoint） | `just exp compare eval` | `compose_policy_evaluation_config`（E-055 起 `CheckpointLabel` 可为 `update-NNN`，需 pin hash） |
| 配对比较 + bootstrap + 报告 | `just exp compare analyze` | `analysis/runner.py` + `analysis/evaluation.py::scalar_reward` |
| 每 update PPO/behavior 诊断 | `just exp training diagnose` | `experiments/training/runner.py::diagnose` → `extract_arm_metrics` / `post_update_kl_series` |
| 训练组合与协议守卫 | — | `experiments/protocol/composition.py::compose_arm_training_config` |

关键事实：

- **冻结 control 与候选已可由协议 compose**：E-052 冻结 control（`value_coefficient=0.1`、
  `gamma=0.9509900499000001`、`gae_lambda=0.7737809375`）与 calibrated-band λ profiles 已在
  `calibrated-canonical.yaml` / `calibrated-canonical-dose-response.yaml` 中编码。
- **中间 checkpoint 已支持**：E-055 为 `update-025` 扩展了 `CheckpointLabel` / `ComparisonRun.
  checkpoint_hash` 与报告排序；`update-050` 为 `policy-update-049.pt`。
- **initial policy 按 training seed 共享**（同一 seed 各臂一致），final/中间 checkpoint 按臂区分。
- **gradient diagnostics 缺口**：`extract_arm_metrics` 原先只报告 total pre-clip norm；Task 1 需要
  actor/shared/value per-group gradient diagnostics，因此本次扩展 `extract_arm_metrics` 在
  `ppo.gradient_diagnostics=true` 时额外报告 per-group 序列。
- canonical cadence 由 `ClosedLoopCadenceConfig` pin（`0.1 × 5 = 0.5 s`）；正式 run 必须 pin clean commit。

## 3. 设计决策

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 复用 `compare train/eval/analyze` + `training diagnose`，不新增专用 runner | ADR 0038 已整合 matched 工作流，E-055 已为其扩展中间 checkpoint；避免新增框架（AGENTS「控制工程复杂度」） |
| D2 | 新增独立 protocol `calibrated-canonical-confirmation.yaml` | 需求为 seeds `{0,1,2}`、`update_count=100`、arms `{r0,lam4,lam8}`，与现有协议语义不同 |
| D3 | `ppo.gradient_diagnostics=true` 写入 protocol overrides | 仅测量，不改更新语义；使 Task 1 必需的 per-group gradient diagnostics 可持久化 |
| D4 | r0 在 100 updates 下**重新训练**，不复用 E-054/E-055 r0 | E-054/E-055 r0 为 50 updates 且未开 gradient diagnostics，resolved 条件不匹配；Task 1 各臂需 matched |
| D5 | Gate M 在实验记录内裁定，不新增 gate runner | 与 E-055 Gate D 一致（analysis 层不做 gate 裁定）；其余 gate 亦如此 |
| D6 | 实验编号 E-056 | 当前最高记录 E-055；per-experiment 命名使 resolved config/artifact 可唯一恢复协议 |
| D7 | 先 clean commit 再正式运行 | Issue Gate I 惯例；无 CI 时在实验记录登记本地测试命令与结果 |

## 4. 实现步骤

### Step 1 — 预登记方案（本文件）

### Step 2 — protocol + 诊断报告

1. 新增 `configs/experiments/comparison/calibrated-canonical-confirmation.yaml`：arms `r0/lam4/lam8`，
   seeds `{0,1,2}`，E-052 冻结 control + canonical 调度 + `training.update_count=100` /
   `ppo.scheduler_total_optimizer_steps=100` / `ppo.gradient_diagnostics=true`，held-out contract 与
   E-055 逐值相同，contrasts `[[r0,lam4],[r0,lam8]]`。
2. `extract_arm_metrics` 新增 `gradient_diagnostics`（当 summary 中 per-update 值非 `null` 时报告
   per-group 序列；`GRADIENT_GROUPS` 移至 `update_diagnostics` 作为单一所有权）。
3. 回归测试：`tests/training/test_training_workflows.py`（protocol pin）、
   `tests/training/test_effective_update.py`（gradient diagnostics 存在/缺省）。

### Step 3 — 回归测试（CPU，沙箱内）

`just test-target tests/training/test_training_workflows.py tests/training/test_effective_update.py`、
`just lint`、`just typecheck`。

### Step 4 — 正式运行（沙箱外，GPU + MetaDrive；clean commit）

study root：`outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation/`。

```powershell
# 9 matched training runs（r0/lam4/lam8 × seed 0/1/2，100 updates，canonical k=5 + E-052 冻结 control）
just exp compare train --config configs/experiments/comparison/calibrated-canonical-confirmation.yaml `
  --arm r0 --training-seed 0 --output-dir outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation/r0-seed-0
# ... r0/lam4/lam8 × seed {0,1,2} 共 9 个 run

# held-out 评测（deterministic Beta mean，16 场景）：每个 seed 的 shared initial（3）+ 每臂
# update-050（policy-update-049.pt，9）+ final（9）
just exp compare eval --config .../calibrated-canonical-confirmation.yaml `
  --arm r0 --checkpoint initial --checkpoint-path .../r0-seed-0/policy-initial.pt `
  --output-dir .../heldout/initial-seed-0
just exp compare eval --config ... `
  --arm lam8 --checkpoint update-050 --checkpoint-path .../lam8-seed-2/policy-update-049.pt `
  --output-dir .../heldout/lam8-seed-2-update-050
# ... 每臂 x seed 的 update-050 / final 同理

# 离线配对比较（manifest: comparison.yaml，update-050 条目 pin 重算 hash）
just exp compare analyze --config .../e-056-.../comparison.yaml `
  --source-dir outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation `
  --output-dir outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation-analysis

# 每臂训练机械健康（9 summaries，含 beta_series / analytic + seeded-MC KL / per-group gradient）
just exp training diagnose --config .../e-056-.../diagnose.yaml `
  --output-dir outputs/studies/scalar-reward/e-056-issue83-task-1-candidate-multiseed-confirmation-diagnostics
```

运行后核对：每 seed 各臂 initial policy hash 一致（seed0 `049697739ab5…`、seed1 `feee6d85b681…`、
seed2 预期新值），frozen planner hash 前后不变；resolved cadence `0.1×5=0.5 s`；失败 seed 保留不删除。

### Step 5 — Gate M 判定与记录

- 从 `analysis.json`（lam4-r0 / lam8-r0 × checkpoint × seed）与 diagnose summary 裁定 Gate M：
  3/3 完成、无 boundary/KL runaway/invalid、跨 seed 方向一致、S vs SC 与 leave-one-scenario-out 稳健、
  50→100 无反转、trade-off 可解释、provenance clean。
- 新增 `docs/experiments/records/e-056-issue83-task-1-candidate-multiseed-confirmation.md` 并更新
  `docs/experiments/README.md` 索引；记录含 Issue Task 1 provenance 清单与 Gate M 表。
- 将新 protocol 登记到 `docs/agents/contracts/experiments.md` 的 protocol 列表。
- 在 Issue #83 评论 Gate M 结果并指向 E-056；本文件状态改为「已执行」。

## 5. 风险与检查点

| 风险 | 检查/应对 |
| --- | --- |
| 100-update 训练 / 21 次评测计算量大 | 沙箱外 GPU 执行；失败 seed 保留，不补跑伪造完整表格 |
| diagnostic 需重算 900 个 checkpoint 的 KL/MC | mc_draws 沿用 #106 的 4096；必要时说明成本 |
| gradient_diagnostics 使 confirmation runs 与 E-054/E-055 非 bit-identical | 各臂均在确认协议下重新训练并 matched；不宣称复用旧 r0 |
| 单 seed 历史效应（E-055）在 3 seeds 下不稳定 | 这正是 Gate M 要检验的对象；如实记录，不筛 seed/场景 |
| initial policy hash 与既有实验不一致 | 运行后核对；不一致即暂停排查 |

## 6. Non-goals

- 不重新搜索或修改 PPO 超参、reward representation、band 阈值、cadence、λ；
- 不做 final A0/A1/A2 matched evaluation 与 margins 预注册（#83 Task 2/3）；
- 不在看到 final A2 结果后调整 non-inferiority / noise margins；
- 不把 training return 作为候选或选择依据。
