# Issue #83 Task 1B — Effective-update transfer 执行计划

状态：**已执行**（工作文档）。执行结果与 provenance 见
[E-049 实验记录](../../experiments/records/e-049-issue83-task-1b-effective-update-transfer.md)；
本文件保留预登记方案，不在此维护结果。

任务边界、验收标准与冻结契约的权威来源是 [Issue #83](https://github.com/)（Parent #80；历史前置
#94/#98 已关闭；execution protocol 来自 #96 Task E / ADR 0039）。本文档记录本次 Task 1B 的
预登记执行方案；执行完成后由 `docs/experiments/records/` 的实验记录取代，不在本文件维护结果。

## 1. 任务定义（来自 Issue #83 Task 1B）

目的：**不重新搜索 reward/PPO**，在 ADR 0039 canonical cadence（k=5 × 0.1 s = 0.5 s decision
interval）的 MDP 时间语义下，验证 #94 冻结的 **E-039 candidate PPO control**（calibrated R0 +
`lr=1.5e-4 / epochs=1 / max_gradient_norm=0.5 / target_kl=0.006`）是否仍产生**有效且稳定**的
per-update 更新。

协议：

1. 固定 calibrated R0（`plannerrft_no_energy_calibrated_v1`）与 E-039 candidate PPO config；
2. 在 canonical cadence 下做**一次** matched transfer run，`update_count=50`；
3. **不做** lr / epochs / max_gradient_norm 网格搜索；
4. 复用 #94 Task F 的 **Gate F** 七条判据，作为本次 **Gate T2**。

**Gate T2（≡ Gate F，阈值冻结自 E-039）**：

```text
c1 median seeded-MC post-update KL >= 1e-6
c2 >= 90% updates KL <= target_kl=0.006 且无持续 runaway（tail-10 median <= 10x 总 median）
c3 policy ratio 出现可测变化（median max(|ratio_mean-1|, ratio_std) >= 1e-4）
c4 deterministic Beta mean batch-level RMS shift >= 0.01
c5 无 Beta boundary collapse（min alpha/beta >= 0.1 且 probe boundary mass <= 0.2）
c6 无 behavioral collapse（collision+OOR <= 2 且 episode-length tail 保留 >= 50%）
c7 至少一个 held-out 非安全指标相对变化 >= 0.1%（E-031 matched evaluation noise bound）
```

通过 = E-039 control 成功 transfer。失败时**不在 #83 内重新做 optimizer hyperparameter
search**，应回到 #80 新建/指定独立 cadence-PPO 诊断任务。

## 2. 现行机制勘察结论

| 环节 | 入口 | 实现 |
| --- | --- | --- |
| 单臂训练 + Gate F 判定 | `just exp training grid` | `scripts/experiments.py:76` → `experiments/training/grid.py::run` |
| Gate F 判据 c1–c6 | — | `experiments/training/decisions.py::evaluate_update_gate` |
| Gate F 判据 c7 | — | `experiments/training/decisions.py::evaluate_heldout_change` |
| 训练诊断量 | — | `rl/optimization/update_diagnostics.py::{extract_arm_metrics, post_update_kl_series}` |
| 训练组合与协议守卫 | — | `experiments/protocol/composition.py::compose_arm_training_config` |

关键事实：

- **E-039 candidate 已在现有 protocol 中编码**：`configs/experiments/comparison/calibrated.yaml`
  arm `r0` = `plannerrft_no_energy_calibrated_v1`，training overrides 为
  `lr=0.00015 / epochs=1 / max_gradient_norm=0.5 / target_kl=0.006 / batch_size=128 /
  transitions_per_environment=8 / update_count=50 / scheduler_total_optimizer_steps=50`，
  base job `jobs/training/ppo_conservative`（`minibatch_size=128`，故 batch==minibatch）。
  该组合即 Issue #83 Task 1B 的「calibrated R0 + E-039 candidate」。
- **canonical cadence**：`configs/components/ppo_training.yaml` 已 pin `0.1 s × 5 = 0.5 s`，
  `TrainingJobConfig.cadence`（`rl/config.py`）在 typed boundary 拒绝非 canonical 值并在
  resolved config 中显式保存。
- **Gate F 阈值**（冻结自 E-039）：`configs/experiments/training/grid.yaml` 第 25–43 行；
  本次 Task 1B 配置必须与其逐值一致，不因 cadence 变化调整。
- **Gate F 已覆盖全部 Gate T2 条目**：c1=KL floor、c2=within-target/runaway、c3=ratio 变化、
  c4=Beta mean RMS shift、c5=boundary collapse、c6=behavioral collapse、c7=held-out 超噪声；
  训练内非有限值由 `post_update_kl_series` 与 PPO 的 finite guard 直接抛错。
- **`training grid` 支持单臂**：`GridConfig` 只要求非空且排序唯一；单元素笛卡尔积
  （1×1×1）不构成搜索，产物 `summary.json → update_gate_passed` 即 Gate T2 裁定。注意
  runner 的 held-out c7 只对通过 c1–c6 的候选臂评测：若 c1–c6 未全通过则不进入 held-out。

## 3. 设计决策

| # | 决策 | 理由 |
| --- | --- | --- |
| D1 | 复用 `training grid` runner，配置为单元素 grid | Gate F 已完整实现；单臂 grid 不构成搜索，避免新增 factory/runner 抽象（AGENTS「控制工程复杂度」） |
| D2 | 独立 study manifest，不改 `grid.yaml` | `grid.yaml` 指向 `comparison/default.yaml` arm `a1`（未校准 R0）且为 24-arm 搜索清单；Task 1B 需 calibrated R0 单臂，语义不同 |
| D3 | protocol 用 `experiments/comparison/calibrated.yaml` arm `r0` | 该 arm 精确等于 calibrated R0 + E-039 candidate，provenance 可由 resolved config 唯一恢复 |
| D4 | Gate 阈值逐值复制 `grid.yaml` | Gate T2 ≡ 冻结 Gate F；不因 canonical cadence 调整判据 |
| D5 | 实验编号 E-049 | 当前最高记录 E-048；per-experiment 命名使 resolved config/artifact 可唯一恢复协议 |
| D6 | 先 clean commit 再正式运行 | Issue Gate I 惯例：正式实验从 clean commit 运行；无 CI 时在实验记录登记本地测试命令与结果 |

## 4. 实现步骤

### Step 1 — 预登记方案（本文件）

### Step 2 — study manifest

新增 `configs/experiments/training/e-049-effective-update-transfer.yaml`：

```yaml
version: 1
study_name: effective_update_transfer
protocol: experiments/comparison/calibrated.yaml
arm: r0
training_seed: 0
update_count: 50
base_overrides:
- components/resources=rtx_a4000
grid:
  learning_rates:
  - 0.00015
  epochs:
  - 1
  max_gradient_norms:
  - 0.5
gate:            # 与 configs/experiments/training/grid.yaml 第 25-43 行逐值一致
  target_kl: 0.006
  kl_floor: 1.0e-06
  within_target_kl_fraction: 0.9
  runaway_tail_updates: 10
  runaway_tail_factor: 10.0
  ratio_change_floor: 0.0001
  guidance_rms_shift_floor: 0.01
  beta_parameter_floor: 0.1
  boundary_mass_ceiling: 0.2
  episode_length_retention_floor: 0.5
  collision_budget: 2
  heldout_relative_change_floor: 0.001
  heldout_metrics:
  - mean_speed_mps
  - distance_m
  - route_completion
  - energy_total_ml
  - energy_ml_per_km
selection: lowest_learning_rate_then_epochs_then_max_gradient_norm
mc_draws: 4096
mc_seed: 1000003
```

不新增 `src/` 代码：`training grid` 已实现 Gate F 全部判据与 matched held-out c7。

### Step 3 — 回归测试（CPU，沙箱内）

在 `tests/training/test_training_workflows.py` 或 `test_effective_update.py` 增加测试，加载实际
manifest 并断言：

1. `load_training_grid` 通过，`grid.combinations()` 恰为单一 `(1.5e-4, 1, 0.5)`；
2. `compose_arm_training_config(protocol, "r0", 0)` 解析出的 parsed 训练条件 =
   `reward.name == plannerrft_no_energy_calibrated_v1`、`lr=1.5e-4`、`epochs=1`、
   `max_gradient_norm=0.5`、`target_kl=0.006`、`batch_size == minibatch_size == 128`、
   `optimizer_steps_per_update == 1`、`update_count=50`；
3. cadence = `0.1 × 5 = 0.5 s`（`ClosedLoopCadenceConfig`）；
4. Gate 阈值逐值等于 `grid.yaml` 冻结值；
5. `arm_label` 为 `lr1.5000e-04-epochs1-mgn0.5`。

运行 `just test-target tests/training/test_training_workflows.py tests/training/test_effective_update.py`、
`just lint`、`just typecheck`；必要时 `just test`（smoke）。

### Step 4 — 正式运行（沙箱外，GPU + MetaDrive）

前置：Step 1–3 改动 commit clean 并记录 hash。

```powershell
just exp training grid `
  --config configs/experiments/training/e-049-effective-update-transfer.yaml `
  --output-dir outputs/studies/scalar-reward/e-049-issue83-task-1b-effective-update-transfer
```

runner 依次：训练 `r0` 单臂 50 updates（canonical k=5）→ 提取 c1–c6 → **若**该臂通过 c1–c6，
再做 matched initial/final held-out 评测得 c7 → 写 `summary.json`（含 `update_gate_passed`）。
若 c1–c6 未全通过则不进行 held-out 评测，c7 未评估。

运行后核对：resolved config 的 cadence（0.1×5=0.5）、reward profile
`plannerrft_no_energy_calibrated_v1`、seeds（training/runtime seed 0、replay 0）；initial policy
hash 应与 E-038/E-040/E-048 一致（`049697739a…`）——不一致即停下检查。

### Step 5 — 判定与记录

- **Gate T2 判定**：读 `summary.json → update_gate_passed` 与 `arms[0].gate`；
  c1–c7 全通过即 E-039 control 成功 transfer。
- 通过 → Task 1B 完成，进入 Task 1C（canonical positive-control bridge，另行规划）。
- 失败 → 停止 #83 正式 sweep，记录失败发生在哪条判据；按 Issue 要求不重新搜索 optimizer
  hyperparameter，应回到 #80 指定独立 cadence-PPO 诊断任务。
- 新增 `docs/experiments/records/e-049-issue83-task-1b-effective-update-transfer.md` 并更新
  `docs/experiments/README.md` 索引；记录含 Issue Task 5 provenance 清单与 **E-039(k=1) vs
  E-049(k=5) Gate F 逐条对照**。
- 文档写回：本次不改变实现契约（仅新增 study manifest），无需修改 `docs/agents/contracts/`；
  预登记方案状态改为「已执行」。
- Issue #83 评论：Gate T2 结果 + 指向实验记录。

## 5. 风险与检查点

| 风险 | 检查/应对 |
| --- | --- |
| canonical k=5 下 per-update KL/ratio/RMS 量级变化，Gate F 阈值可能不再匹配 | 这正是 transfer gate 要检验的对象；阈值冻结，如实记录，不因结果调整 |
| 单 minibatch/epoch 下 torchrl 训练内 `kl_approx` 结构性 ≈0 | 主估计量是 seeded-MC post-update KL（E-039 方法学结论），c1/c2 用它判定 |
| initial policy hash 与 E-039/E-048 不一致 | 运行后核对；不一致则暂停排查（架构不应变） |
| 50 updates 训练的 held-out c7 仅由速度类指标边际过线（E-039 即如此） | 按冻结阈值如实记录；不额外筛 seed 或指标 |
| 训练需要 GPU+仿真 | 按沙箱要求申请沙箱外执行；不重建环境 |

## 6. Non-goals

- 不重新搜索或修改 PPO 超参、reward representation、band 阈值、cadence；
- 不做 lr/epochs/max_gradient_norm 网格搜索（那是 E-039 已完成的工作）；
- 不启动 Task 1C 或 λ={1,2,4,8} 正式 sweep；
- 不复用 E-039/E-040 旧 k=1 产物作为已迁移结论（ADR 0039 改变了 MDP 时间语义）。
