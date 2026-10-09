# Issue #83 Task 1 Re-entry — λ=1 candidate multi-seed confirmation 执行计划

状态：**预注册**（2026-10-09，E-059 analyze 运行前锁定；执行后本行指向
`docs/experiments/records/e-059-issue83-task-1-reentry-lam1-multiseed-confirmation.md`）。本文件为
re-entry 预注册方案原文；原 Task 1 计划
（`issue83-task-1-candidate-multiseed-confirmation-plan.md`）以 E-056 执行后 Gate M FAILED
（λ=4/8 Beta boundary collapse），按 Issue 预设以 "candidate long-run instability /
non-reproducibility" 收口，其预登记文本与 E-056 证据保持原样。

## 1. Re-entry 背景与任务定义

#114 Task 4 / E-058 完成 candidate provenance revalidation，Gate S3 PASSED：ADR 0040 Beta
interior 参数化（`interior_epsilon=1e-4`）+ Issue #114 Task 1 scheduler-matched
100-update 契约下，λ=1 为唯一提名候选（final energy-intensity −1.22%~−1.45%，3/3 CI 不含
0；任务结构与 R0 一致；3/3 seeds 100-update 机械稳定）。E-055 的 λ=4/8 提名正式失效
（ADR 0040 family 变更的 provenance 边界）。

本计划对 Issue #83 Task 1 的 Gate M 逐项裁定 λ=1 候选。任务定义（seeds {0,1,2}、
updates 100、checkpoint 0/50/100、必须报告清单、Gate M 七项）与原计划 §1 相同，不重复；
本文件只登记 re-entry 特有的决策、判据与产物边界。

## 2. 产物复用边界（决策 D1）

**不执行任何新训练或新 held-out 评测。** 确认对象为 E-057/E-058 已产出的 matched 产物：

| 产物 | 来源与 pin |
| --- | --- |
| λ=1 训练 runs（seeds {0,1,2} × 100 updates，全 checkpoint、gradient diagnostics） | E-058，pin `73b9db4` |
| R0 训练 runs（seeds {0,1,2} × 100 updates，同上） | E-057，pin `07b2860` |
| held-out evals（initial / update-025 / update-050 / final × 16 场景，deterministic Beta mean） | E-057（含 r0 evals 与 shared initial，pin `986b015`）+ E-058（lam1 evals，pin `73b9db4`） |
| 离线机械诊断（analytic + seeded-MC/k3 post-update KL、Beta series、per-group gradients、parameter delta） | E-058 `-diagnostics`（15 runs，含 lam1/r0 全部 6 个相关 run） |

复用理由：

- `calibrated-canonical-candidate-revalidation.yaml`（E-058 protocol）的 training /
  evaluation / bootstrap 块与 E-056/E-057 confirmation contract **逐字相同**（E-058 预注册
  计划 D2 的设计决策），且同 seeds {0,1,2} × 100 updates × `scheduler_total_optimizer_steps=100`
  单一 scheduler history、`gradient_diagnostics=true`——满足 Task 1 的全部训练/评测条件；
- matched 校验在 E-059 manifest 上由 `load_comparison` 整体重验（per-seed initial policy
  hash、frozen planner hash、probe_before、resolved training conditions、scenario pool、
  eval seed/horizon/sampler、update-025/050 checkpoint pin hash 交叉核对）；
- 同 seed 同契约重训不产生新证据（确定性训练流）；Gate M 的检验对象是跨 seed 稳定性，
  不是同 seed 机械可重复性。

## 3. 预注册 scenario-family 稳健性判据（Gate M 第 4 项操作化）

原计划 Step 5 要求 "S vs SC 与 leave-one-scenario-out 稳健"，E-056 因稳定性前置失败未到达
该步，E-058 只做了 pooled（16 场景合并）bootstrap。本节判据在 E-059 analyze 运行前锁定，
适用于 `lam1-r0` contrast 的 **final**（100 updates）checkpoint、metric
`energy_ml_per_km`（matched scenario delta，comparison−reference）：

- **R1（family split）**：每个 training seed 下，S 与 SC 两个 map family 各自的 mean
  paired energy-intensity delta 均 < 0；3/3 seeds 全部满足。
- **R2（leave-one-out）**：每个 training seed 下，从 16 个 held-out 场景中逐个剔除任一
  场景后，剩余 15 场景的 mean paired energy-intensity delta 仍 < 0（符号不翻转）；
  3/3 seeds × 任一 scenario。

失败分支预设：R1 或 R2 任一失败 → Gate M 第 4 项不满足 → Gate M FAILED，λ=1 不进入
final A2，以 "effect driven by single scenario family / few outlier episodes" 收口，
全部数值与失败证据保留，不删不改。

update-050 的 family / LOO 数值一并报告（诊断），但 gate 条件只作用于 final；50→100
方向稳定性由 Gate M 第 5 项另行裁定（update-050 与 final 的 effect 同号且无 behavior
collapse）。E-058 已按其 §4.3 裁定 u050→final 同号单调加深；E-059 analyze 独立复核。

## 4. Gate M 七项 → 证据来源映射

| Gate M 条件 | 证据来源 |
| --- | --- |
| 3/3 seeds 完成；失败 seed 不删除 | E-057/E-058 训练 runs（6/6 完成，无 crash；by reference + E-059 manifest 校验） |
| 3/3 无 boundary collapse / KL runaway / invalid action | E-058 `-diagnostics`（boundary mass、min α/β、analytic/MC/k3 KL） |
| energy-intensity effect 方向跨 seed 一致 | E-059 analysis（`lam1-r0` final，3/3 seeds，CI 不含 0） |
| effect 非单一 family / 少数 outlier 驱动 | E-059 analysis 新增 robustness 块（判据 R1/R2） |
| 50→100 无方向反转 / behavior collapse | E-059 analysis（`lam1-r0` update-050 vs final） |
| progress / speed / stopped / safety trade-off 可解释 | E-059 analysis（arm outcomes + paired metrics）+ #113 降速载带 caveat |
| provenance 完整、commit clean | 双 pin 复用论证 + `load_comparison` 重验 + E-059 analyze pin clean commit |

## 5. 设计决策

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 复用 E-057/E-058 训练与评测产物，不重训不重评 | 见 §2；#114 handoff 明确产物已就位 |
| D2 | 新增 protocol `calibrated-canonical-lam1-confirmation.yaml`（arms r0/lam1、contrast `[r0, lam1]`，training/evaluation/bootstrap 与 revalidation 逐字相同） | 不改 E-056/E-057 已注册的 confirmation protocol（arms r0/lam4/lam8 是历史证据的 pin）；#83 re-entry 需要自己的 protocol pin |
| D3 | `scalar_reward` contrast effect 新增 `robustness` 块（per-family paired delta 统计 + leave-one-scenario-out 符号稳定性），measurement-only，无 bootstrap 扩展 | 原计划 Step 5 要求的稳健性证据需要落进正式 analysis 产物；Gate M 裁定在记录内给出（沿 D5 惯例） |
| D4 | E-059 manifest 手写 24 条目（r0/lam1 × seeds {0,1,2} × 4 labels），pin hash 取自 E-058 manifest 同 checkpoint 条目 | 沿 E-058 双 pin 模式；`load_comparison` 硬校验捕获转录错误 |
| D5 | 实验编号 E-059 | 当前最高记录 E-058 |
| D6 | 先 clean commit 再正式运行 | 惯例（E-056 D7） |

## 6. 实现步骤

### Step 1 — 预登记方案（本文件）

### Step 2 — 分析扩展 + protocol + 测试

1. `analysis/evaluation.py`：新增 `paired_robustness(pairs)`，`scalar_reward` 的每个
   contrast effect 附加 `robustness`（`families[metric][family] = {mean, count}`；
   `leave_one_out[metric] = {sample_count, full_mean, loo_mean_min, loo_mean_max,
   sign_change_count}`，样本 < 2 时为 null）。
2. 新增 `configs/experiments/comparison/calibrated-canonical-lam1-confirmation.yaml`。
3. 回归测试：`tests/analysis/test_reports.py`（robustness 分组 / LOO / 符号翻转 / 缺项）、
   `tests/training/test_training_workflows.py`（protocol pin + 与 revalidation 的
   training/evaluation/bootstrap 逐字相等断言）。
4. `docs/agents/contracts/experiments.md` 登记 protocol。

### Step 3 — 回归测试（CPU，沙箱外申请）

`just test-target tests/training/test_training_workflows.py tests/analysis/test_reports.py`、
`just lint`、`just typecheck`。

### Step 4 — E-059 执行（沙箱外；clean commit）

study root：`outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation/`。

```powershell
# 手写 24 条目 manifest（comparison.yaml），r0 条目指向 E-057、lam1 条目指向 E-058，
# update-025/update-050 pin hash 与 E-058 manifest 相同 checkpoint 一致
just exp compare analyze `
  --config outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation/comparison.yaml `
  --source-dir outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation `
  --output-dir outputs/studies/scalar-reward/e-059-issue83-task-1-reentry-lam1-confirmation-analysis
```

运行后核对：`load_comparison` matched guards 全过（含 per-seed initial hash
`56bb97ab…`/`f91a826f…`/`0febc900…`、checkpoint pin hash 交叉核对）；analysis 含
`lam1-r0` × {initial, update-025, update-050, final} × seeds {0,1,2} 的 effects 与
robustness 块。

### Step 5 — Gate M 判定与记录

- 从 E-059 analysis（effects + robustness）与 E-058 `-diagnostics` 裁定 Gate M 七项。
- 新增 `docs/experiments/records/e-059-issue83-task-1-reentry-lam1-multiseed-confirmation.md`
  并更新 `docs/experiments/README.md` 索引；记录含 Issue Task 1 provenance 清单、必须
  报告清单（PPO/behavior、task/safety、energy）与 Gate M 表。
- 在 Issue #83 评论 Gate M 结果并指向 E-059；本文件状态改为「已执行」。

## 7. 风险与检查点

| 风险 | 检查/应对 |
| --- | --- |
| manifest 转录错误 | `load_comparison` loud-fail（E-058 先例：缺前缀/错 hash 均被捕获）；修正后重跑 |
| R1/R2 失败 | 按预设失败分支收口，不调整判据 |
| 复用产物与 Task 1 条件不匹配 | 由 D2 逐字相等断言 + `load_comparison` matched guards 消除；不匹配即中止排查 |
| 记录数值与 E-057 记录笔误冲突 | 一律以 artifact 数值为准（E-058 已登记 E-057 记录存在三位小数笔误） |

## 8. Non-goals

- 不重训、不重评、不新增 diagnose（E-058 `-diagnostics` 已覆盖全部相关 run）；
- 不重新搜索或修改 PPO 超参、reward representation、band 阈值、cadence、λ；
- 不做 final A0/A1/A2 matched evaluation 与 margins 预注册（#83 Task 2/3，另行执行；
  Task 3 的 A0 rerun 需要 GPU）；
- 不把 training return 作为裁定依据；不在本任务内裁定 Gate F。
