# E-049 Issue #83 Task 1B：effective-update transfer 到 canonical k=5 cadence

[返回实验索引](../README.md) · [Task 1B 执行计划](../../research/issue83-task-1b-effective-update-transfer-plan.md) ·
[Task 1A / E-048](e-048-issue83-task-1a-objective-identifiability-transfer.md) ·
[E-039 PPO control](e-039-issue94-task-f-effective-update-region.md)

**日期 / 类型 / 目的**：2026-09-24 / 正式单臂训练 transfer gate / 完成 Issue #83 Task 1B：**不重新搜索
reward/PPO**，在 ADR 0039 canonical cadence（k=5 × 0.1 s = 0.5 s decision interval）下验证 #94 冻结的
E-039 candidate PPO control（calibrated R0 + `lr=1.5e-4 / epochs=1 / max_gradient_norm=0.5 /
target_kl=0.006`）是否仍产生有效且稳定的 per-update 更新。协议：**一次** matched transfer run、
`update_count=50`、无 lr/epochs/mgn 网格；复用 #94 Task F 的 **Gate F** 七条判据作为 **Gate T2**。

**裁定：Gate T2 FAILED。** 仅 **c1**（median seeded-MC post-update KL ≥ 1e-6）失败：实测
median `3.258e-7`（< 1e-6）。c2–c6 全部通过；c7 因 c1 失败、runner 不进入候选 held-out 评测而
**未评估**。E-039 candidate 在 canonical cadence 下未通过冻结的 effective-update 判据。

**代码 / 环境**：Git commit `996d0e8bf9f4aa29ae33678cae3d979b8a216bb0`（branch `main`，
`git_status_short=[]` clean，由 `runtime_metadata.json` 记录）；上游源码
`a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`；EMA 276 tensors / 6,042,628 parameters。
Windows-10-10.0.26200-SP0、Python 3.10.20、PyTorch 2.12.1+cu126、CUDA:0（RTX A4000）、Lightning
2.6.6、MetaDrive 0.4.3；rollout `bf16-mixed`。

**进入条件**：Task 1A（E-048）Gate T1 PASSED——冻结 reward representation 在 k=5 下可辨识（endpoint
z head cosine 0.910851、sign-flip 18.75%、λ64 分离 85.29%）。Task 1B 检验 E-039 PPO control 是否同样
transfer。

## 协议与冻结契约

单臂 study manifest `configs/experiments/training/e-049-effective-update-transfer.yaml`
（`TrainingGridConfig`），复用 `experiments/training/grid.py::run` 的 Gate F 实现，grid 为单元素
`(lr=1.5e-4, epochs=1, mgn=0.5)`，不构成搜索：

- protocol `experiments/comparison/calibrated.yaml` arm `r0` =
  `plannerrft_no_energy_calibrated_v1`（E-034 冻结 Progress/Comfort 校准）；
- base job `jobs/training/ppo_conservative`；canonical cadence `0.1 s × 5 = 0.5 s`
  （`ClosedLoopCadenceConfig` typed boundary pin）；
- training seed 0、replay 0；16 scenarios（S/SC × map seeds 0–7）× 8 transitions/env =
  **128 transitions / update**；DDIM5；
- Gate T2 阈值逐值复制自 `configs/experiments/training/grid.yaml`（E-039 冻结 Gate F），
  `mc_draws=4096`、`mc_seed=1000003`。

**命令**：

```powershell
just exp training grid `
  --config configs/experiments/training/e-049-effective-update-transfer.yaml `
  --output-dir outputs/studies/scalar-reward/e-049-issue83-task-1b-effective-update-transfer
```

resolved config 核对：cadence `0.1/5/0.5`、reward `plannerrft_no_energy_calibrated_v1`、
lr=1.5e-4 / epochs=1 / mgn=0.5 / target_kl=0.006、batch=minibatch=128、transitions=8、
update_count=50、training/runtime seed 0、replay 0。initial policy hash
`049697739ab5d6e7bd212938bbac82c35215eb9ed14c02557952098a9718d05d` 与 E-038/E-040/E-048 一致；
frozen planner hash before==after（planner 冻结未被破坏）；final policy hash
`7a7809264b2fd18d3bc950acdd53ed88a5fdb93c4029341c2345be6d105db138`（policy 确有更新，非 no-op）。

## Gate T2 判定结果

```text
Verdict: FAILED（update_gate_passed=false，selected_config=null）
failure_reasons = ["c1_median_kl_above_floor"]
```

| 条件 | 判据 | 实测 | 通过 |
| --- | --- | ---: | :---: |
| c1 median post-update KL | ≥ 1e-6 | **3.258e-7** | ✗ |
| c2 KL within target / 无 runaway | ≥90% 且 tail-10 ≤10×median | within=1.00；tail median −1.6e-8 | ✓ |
| c3 policy ratio 变化 | ≥ 1e-4 | 1.097e-3 | ✓ |
| c4 deterministic Beta mean RMS shift | ≥ 0.01 | 0.02205 | ✓ |
| c5 无 Beta boundary collapse | min α,β ≥0.1 且 boundary mass ≤0.2 | min α/β 1.930/1.986；mass 0.0220 | ✓ |
| c6 无 behavioral collapse | collision+OOR ≤2 且 ep-len 保留 ≥50% | 0/0；8.0 → 8.0 | ✓ |
| c7 held-out 超噪声 | ≥1 指标 ≥0.1% | **未评估**（c1 失败，runner 不进入候选 held-out） | — |

其他训练诊断（50 updates）：`under_update=false`（KL 低于门限但 ratio/RMS 未同时低于门限）；
single-draw k3 median `6.040e-7`（与 MC KL 同量级，交叉验证）；`kl_early_stop_fraction=0`；
`clip_fraction_median=0`；pre-clip gradient norm median `127.53`（post-clip 恒为 mgn=0.5）；
无 NaN/Inf；value loss `141.99 → 135.18`，explained variance `-1.3e-5 → 0.024`；entropy 基本不变。

## E-039(k=1, plain R0) vs E-049(k=5, calibrated R0) 对照

两者为同一 E-039 candidate、同一 initial policy hash、同 batch 尺寸（128 = 16×8）、同 Gate F 阈值；
差异是 cadence（k=1 vs k=5）与 reward profile（E-039 用 plain R0，Task 1B 按 Issue 要求用
calibrated R0）。

| 指标 | E-039（k=1，plain R0） | E-049（k=5，calibrated R0） |
| --- | ---: | ---: |
| MC post-update KL median | **1.160e-6** | **3.258e-7** |
| MC post-update KL max | 2.673e-5 | 1.786e-5 |
| single-draw k3 median | 9.989e-7 | 6.040e-7 |
| policy ratio change | 1.409e-3 | 1.097e-3 |
| policy ratio mean median | 1.0000039 | 1.0000006 |
| probe guidance RMS shift | 0.02894 | 0.02205 |
| min α / β | 1.912 / 1.990 | 1.930 / 1.986 |
| pre-clip grad norm median | 28.63 | 127.53 |
| actor_head param delta | 0.02527 | 0.01771 |
| shared_trunk param delta | 1.7974 | 1.7936 |
| value_head param delta | 0.05085 | 0.05087 |
| collision / OOR | 0 / 0 | 0 / 0 |
| Gate F 结果 | 7/7 PASS | c1 FAIL（6/7） |

k=5 下 actor 侧更新幅度整体略小（ratio 变化 −22%、probe RMS −24%、actor_head param delta −30%），
KL 下降更显著（1.16e-6 → 3.26e-7，factor ~3.6）至冻结 1e-6 门限以下；同时 critic 侧 value loss 与
pre-clip grad norm 大幅增大（28.6 → 127.5）。方向与 ADR 0039 的时间语义后果一致：canonical
transition 覆盖 0.5 s、per-transition reward 为 multi-substep 求和（≈5× scale）、`gamma=0.99`
对应有效折扣视界从 ~10 s 变为 ~50 s，value 目标与 critic 损失量级随之增大，而在同一 lr 下 actor
的 policy KL 反而低于 k=1。

## 实现修复（本次任务暴露）

首次运行在 update 0 即崩溃：`TrainingUpdateSummary.reward_component_means.*` 被约束为
`[0,1]`，但 canonical k=5 下 audit 的 `reward_component_*` 是 `CLOSED_LOOP_EXECUTION_STEPS`
个子步分量分（每子步 ∈[0,1]）的求和，均值可达 ~5（实测 max ttc=5.0、speed=4.999、progress=3.156）。
修复（独立 commit `996d0e8`）：`RewardComponentMeans` 上界改为 `CLOSED_LOOP_EXECUTION_STEPS`，并加
回归测试 `tests/training/test_tracking.py::test_update_summary_persists_multi_substep_component_sums`。
该约束此前只在 k=5 训练路径被触发（E-039/E-040 为 k=1），是 ADR 0039 统一 cadence 后遗留的
schema 未同步。Gate T2 裁定所依据的 KL/ratio/RMS 统计不受此修复影响。

## 验证与产物

- 代码级测试：`just test-target tests/training/test_training_workflows.py`（9 passed，含 e-049
  manifest 组合、Gate 阈值与 canonical cadence 断言）、
  `just test-target tests/training/test_tracking.py tests/training/test_ppo.py`（42 passed，
  含 multi-substep component 回归）；`just lint` 通过、`just typecheck` 0 errors。
- 正式产物经只读核验：50 updates 全部完成、runtime metadata `git_status_short=[]`、
  `git_head=996d0e8`、frozen planner hash 前后一致、initial policy hash 与 E-038/E-040/E-048 一致、
  50 个 update summary 无非有限值。

产物目录（git ignored）：
`outputs/studies/scalar-reward/e-049-issue83-task-1b-effective-update-transfer/`

- `study_manifest.yaml`：resolved 研究清单（单臂 grid + Gate T2 阈值）。
- `lr1.5000e-04-epochs1-mgn0.5/`：`summary.json`（逐 update 训练诊断）、
  `policy-initial/final/update-NNN.pt`、`updates/update-NNN/*.npz`（完整 rollout audit）、
  `resolved_config.yaml`、`runtime_metadata.json`。
- `summary.json`：逐 arm metrics、Gate T2 判定与（空的）选择结果；无 `heldout/`（c7 未评估）。

## Issue Task 5 provenance 清单

- commit `996d0e8`、branch `main`、`git_status_short=[]`（clean 运行）；上游 `a3a621f0…`；
- 环境：Windows-10-10.0.26200 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 /
  Lightning 2.6.6 / MetaDrive 0.4.3；
- resolved config：arm 目录 `resolved_config.yaml`（cadence/reward/PPO/scenarios 已核对）；
- checkpoint identity：initial policy `049697739a…`（与 E-038/E-040/E-048 一致）、frozen planner
  hash 前后不变；final policy `7a780926…`；
- seeds：training/runtime seed 0、replay 0、16 scenarios S/SC seeds 0–7；DDIM5；
- cadence：0.1 s × 5 = 0.5 s（`ClosedLoopCadenceConfig` typed boundary 校验）；
- commands / artifact path：见上；
- 测试结果：9 + 42 passed、Ruff/Pyright clean；failure 状态：Gate T2 FAILED（仅
  `c1_median_kl_above_floor`），无训练/数值崩溃。

## 结论边界

**支持**：在 canonical k=5 cadence、该 initial policy、该 no-traffic 训练池、E-034 冻结校准 R0 下，
E-039 candidate PPO control 的 per-update policy KL median 为 `3.258e-7 < 1e-6`，**未通过冻结的
effective-update 门限**（Gate T2 FAILED）。这是相对 E-039/k=1（1.16e-6 通过）的真实 cadence
transfer 失败，而非机械故障：ratio 变化、probe RMS shift、Beta 边界、行为与数值稳定性均健康
（c2–c6 全通过），policy 参数确有更新。

**不支持 / 限制**：

- 不把 c1 失败解读为「完全没有更新」：KL 已接近 MC 估计量 ~3e-7 的分辨率下限，且 ratio/RMS 仍高于
  各自门限；单点 KL 值不应读作精确量，但 `3.258e-7 < 1e-6` 的冻结阈值比较是明确的。
- c7（held-out 行为超噪声）**未评估**：runner 在 c1–c6 未全通过时不进入候选 held-out；因此本记录
  不提供 held-out learned-effect 证据（E-039/k=1 的 c7 仅由 mean_speed 边际支撑）。
- E-049 使用 calibrated R0（Issue 要求）而 E-039 candidate 的 effective region 原是在 plain R0 上
  测定的；两者 reward profile 差异（仅 Progress/Comfort 校准常数）与 cadence 变化共同构成对照差异，
  本实验不分离二者贡献。
- 不重新搜索 PPO 超参（按 Issue：#83 内不做 optimizer hyperparameter search）；不启动 Task 1C 或
  λ={1,2,4,8} 正式 sweep；不构成节能或 behavior improvement 结论。

**后续（Issue #83 规定）**：Gate T2 失败 → 停止 #83 正式 sweep；E-039 control 在 canonical cadence
下的有效更新问题应由 #80 新建/指定的**独立 cadence-PPO 诊断任务**处理（例如按 canonical 时间语义
重新定位 effective-update region 或调整 `target_kl`/lr 时间校准），并显式记录这是 cadence transfer
失败而非 #83 内重搜。
