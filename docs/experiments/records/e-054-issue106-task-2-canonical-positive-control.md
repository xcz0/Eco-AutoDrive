# E-054 Issue #106 Task 2：canonical positive-control bridge / Gate T3

[返回实验索引](../README.md) · Issue #106 Task 2 · [E-053 Gate T2](e-053-issue106-task-1-gate-t2-reentry.md) ·
[E-052 冻结 control](e-052-issue105-phase-c-canonical-ppo-refreeze.md) ·
[E-048 canonical identifiability](e-048-issue83-task-1a-objective-identifiability-transfer.md) ·
[E-040 历史 k=1 positive control](e-040-issue94-task-g-objective-positive-control.md) ·
[E-047 execution-contract bridge](e-047-issue98-frozen-policy-execution-bridge.md) ·
[ADR 0039](../../adr/0039-unify-closed-loop-cadence.md)

**日期 / 类型 / 目的**：2026-09-29 / 正式配对闭环训练 + 冻结策略 execution bridge / Issue #106 Task 2：
在 canonical k=5（`0.1 s × 5 = 0.5 s`）下，用 E-052/#105 冻结的 canonical PPO control，比较 calibrated R0 与
calibrated-band Rstress（λ=64）两个 matched training seeds（{0,1}），判定 objective→behavior 方向是否建立
且与 energy minimization 兼容（Gate T3）。复用 `compare train` / `compare eval` / `compare analyze` 机制
（ADR 0038），不新增专用 positive-control 目录；Gate T3 裁定在实验记录中给出（与 E-050/E-051 一致，
analysis 层不做 gate 裁定）。executed behavior 用 E-047 frozen-policy execution bridge 在 E-054
checkpoints 上诊断，以解释 canonical temporal mapping。

**裁定：Gate T3 PASSED（6/6）。** 在 canonical k=5 训练 + canonical k=5 主评测下，Rstress 相对 R0
**更慢且单位里程能耗更低**（双 seed mean speed `-1.75%/-1.71%`、energy intensity `-0.637%/-0.613%`，
16/16 场景同向，bootstrap CI 不含 0），方向与 energy minimization 兼容；guidance 分离双 seed 可重复
（probe β-mean RMS `0.10084/0.10001`，纵向维主导）；无 collision/OOR/wrong-direction/stopped/distance
collapse；executed behavior 可由 canonical k=5 planner/execution mapping 解释。该方向**与 E-040（k=1）
及 E-047 对 k=1-trained checkpoints 的 k=5 重放相反**，说明方向转移只在训练与评测同为 canonical k=5 时成立。

## 冻结契约与协议

Task 2 在 #106 冻结研究 contract 下执行：frozen Diffusion Planner / DDIM5；Exploration Policy 架构与同一
initial policy；training scenario family S/SC map seeds 0–7；held-out S/SC seeds 16–23；deterministic Beta
mean 正式主评测；E-034 校准 Progress/Comfort；E-038 calibrated efficiency-band energy；canonical execution
`0.1 s × 5 = 0.5 s`；execution-trace fuel proxy；PPO control 取 #105/E-052 独立冻结结果，不在本 Issue 重搜。
λ 主效应阶段只允许改变 energy weight λ；Task 2 固定双臂（λ=0 / λ=64）。

新增 protocol `configs/experiments/comparison/calibrated-canonical.yaml`（提交 `f2cfaff`）：在
`calibrated.yaml` 的 r0/rstress 双 calibrated 双臂基础上，把 E-052 冻结 control 写入 `training.overrides`：

- `ppo.value_coefficient=0.1`
- `ppo.gamma=0.9509900499000001`（=0.99⁵）
- `ppo.gae_lambda=0.7737809375`（=0.95⁵）

其余保持 canonical：`learning_rate=1.5e-4`、`epochs=1`、`max_gradient_norm=0.5`、`clip_epsilon=0.2`、
`target_kl=0.006`、batch=minibatch=128、`transitions_per_environment=8`、`update_count=50`、
`scheduler_total_optimizer_steps=50`；cadence `0.1 s × 5 = 0.5 s`。训练 seeds `{0,1}`。**不修改**
`calibrated.yaml`，以保护 E-053 已登记实验的声明训练条件。

## 命令

```powershell
# 4 个 matched training runs（r0/rstress × seeds {0,1}），canonical k=5
just exp compare train --config configs/experiments/comparison/calibrated-canonical.yaml `
  --arm r0 --training-seed 0 `
  --output-dir outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control/r0-seed-0
# ... r0-seed-1 / rstress-seed-0 / rstress-seed-1 同理

# 4 个 final checkpoint 的 matched held-out 评测（deterministic Beta mean）
just exp compare eval --config configs/experiments/comparison/calibrated-canonical.yaml `
  --arm r0 --checkpoint final --checkpoint-path .../r0-seed-0/policy-final.pt `
  --output-dir outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control/heldout/r0-seed-0-final
# ... 其余 3 个同理

# 离线配对比较 + guidance 分离证据
just exp compare analyze `
  --config outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control/comparison.yaml `
  --source-dir outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control `
  --output-dir outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control-analysis

# E-047 frozen-policy execution bridge（解释 canonical temporal mapping）
just exp guidance execution-bridge run `
  --source-dir outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control `
  --config configs/experiments/guidance/e-054-issue106-task-2-canonical-bridge.yaml `
  --output-dir outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control-execution-bridge
```

resolved 训练条件（`r0-seed-0/r0-seed-1/rstress-seed-0/rstress-seed-1/resolved_config.yaml`）与协议逐一核对：
cadence `0.1/5/0.5`、control 三 override、reward profile 各自正确、src 训练场景池 S/SC 0–7、
update_count=50、runtime seed = training seed、replay 0、`execution_steps=5`。双臂同 seed 除 reward/tracking
外 resolved 训练条件一致（`compare analyze` 的 matched-condition 校验通过）。

## 训练机械健康（4 runs × 50 updates，全部 completed）

| run | min α / β | collision / OOR | episode length | ratio mean 范围 | clip frac max |
| --- | ---: | ---: | ---: | ---: | ---: |
| r0-seed-0 | 1.898 / 1.985 | 0 / 0 | 8.0 → 8.0 | [0.99968, 1.00081] | 0.0 |
| rstress-seed-0 | 1.728 / 2.000 | 0 / 0 | 8.0 → 8.0 | [0.99819, 1.00344] | 0.0 |
| r0-seed-1 | 1.976 / 1.902 | 0 / 0 | 8.0 → 8.0 | [0.99952, 1.00067] | 0.0 |
| rstress-seed-1 | 1.687 / 1.938 | 0 / 0 | 8.0 → 8.0 | [0.99781, 1.00217] | 0.0 |

50 updates 全程有限（无 NaN/Inf），无 Beta boundary collapse（min α/β ≥1.686 ≫ 0.1），无 collision/OOR，
episode length 保持 8。训练内 total reward：R0 基本持平（~499 → ~498），Rstress 上升（seed0 263→299、
seed1 296→321），energy 分量随之上升（训练内），与 held-out 行为方向不矛盾。

paired initial policy（runner 硬校验）：seed0 `049697739ab5…`、seed1 `feee6d85b681…`；两臂同 seed 相同。
**r0-seed-0 final `39aeb2571cad…` 与 E-052 selected control / E-053 final hash 逐值一致**，确认这是同一
冻结 control 的独立 matched 复跑。

## Gate T3 判定结果

```text
Verdict: PASSED（6/6）
```

| 条件 | 判据 | 实测 | 通过 |
| --- | --- | ---: | :---: |
| c1 guidance separation 双 seed 可重复 | paired probe β-mean RMS ≥0.01 且跨 seed 同构 | seed0 0.10084 / seed1 0.10001（均纵向维主导；before RMS=0） | ✓ |
| c2 至少一个闭环行为指标超预注册 noise | ≥1 指标相对变化 ≥0.1% | mean_speed −1.75%/−1.71%；energy intensity −0.637%/−0.613% | ✓ |
| c3 energy-intensity 方向与 energy minimization 兼容 | energy-heavy arm intensity 向更低 | 双 seed intensity 均 **下降**，无停车 | ✓ |
| c4 方向双 training seed 一致 | 两 seed 同向 | 速度与 intensity 双 seed 均负，16/16 场景同向 | ✓ |
| c5 非 collapse 解释 | 无 collision/OOR/wrong-direction/stopped/distance collapse | 0/0/0；stopped Δ=0；distance −0.37%/−0.38%；route −0.23%/−0.27% | ✓ |
| c6 executed behavior 可由 canonical k=5 映射解释 | bridge Part B 局部 policy→planner 响应一致 | Δg_lon 多数为负、middle forward response 非正；Part A k=5 负向与主评测同向 | ✓ |

## Guidance distribution 分离（rstress − r0）

固定 16 probe contexts、2 维（native `(lateral, longitudinal)`）的 deterministic Beta：

| seed | probe | β-mean RMS | Δmean dim0(lat) | Δmean dim1(lon) | concentration RMS | variance RMS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | after | 0.100836 | +0.005538 | **−0.142495** | 0.021993 | 0.003148 |
| 1 | after | 0.100014 | +0.004494 | **−0.141370** | 0.078361 | 0.003272 |
| 0/1 | before | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

分离双 seed 高度一致、由**纵向 guidance** 主导（Rstress 下移 β-mean），lateral 几乎不变；matched initial
probe 完全一致（RMS 0）。post-training probe concentration ≈4.0、variance ≈0.20，与 E-040/E-041 的
concentration/variance 基本不变一致——分离是 mean-policy 性质，非方差塌缩。

## Held-out 主评测（deterministic Beta mean，S/SC seeds 16–23，16 场景，horizon 300，DDIM5）

每格为该 arm 16 场景 completed-episode 均值；Δ = rstress − r0。

| seed | arm | mean speed (m/s) | energy intensity (mL/km) | total energy (mL) | distance (m) | route completion | arrive_dest | collision / OOR / stopped |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | r0 | 10.008297 | 47.062977 | 7.852907 | 169.622029 | 0.947283 | 13/16 | 0 / 0 / 0 |
| 0 | rstress | 9.832851 | 46.763126 | 7.781290 | 168.998978 | 0.945117 | 13/16 | 0 / 0 / 0 |
| 1 | r0 | 9.975698 | 47.007360 | 7.835490 | 169.433550 | 0.946545 | 13/16 | 0 / 0 / 0 |
| 1 | rstress | 9.805014 | 46.719400 | 7.764673 | 168.791400 | 0.943964 | 13/16 | 0 / 0 / 0 |

Paired delta（rstress − r0，completed 交集 16/16）：

| seed | Δ speed（rel） | Δ energy intensity（rel） | Δ total energy（rel） | Δ distance（rel） | Δ route（rel） | Δ stopped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | −0.175446（**−1.75%**） | −0.299851（**−0.637%**） | −0.071618（−0.912%） | −0.623051（−0.367%） | −0.002166（−0.229%） | 0 |
| 1 | −0.170684（**−1.71%**） | −0.287960（**−0.613%**） | −0.070817（−0.904%） | −0.642150（−0.379%） | −0.002581（−0.273%） | 0 |

速度与 energy intensity 双 seed 全部 16/16 场景同向（均负）。Contrast（total energy）scenario bootstrap：
seed0 estimate −0.071618，95% CI [−0.099697, −0.047841]；seed1 −0.070817，CI [−0.100143, −0.046176]；
均不含 0，`direction_counts lower=2/2`。安全计数双 seed 全 0，arrive_dest 保持 0.8125。

## Executed behavior 与 canonical temporal mapping（E-047 bridge on E-054）

E-047 workflow 在 E-054 四个 final checkpoint 上重放（frozen policy，无训练，k∈{1,2,5}）：

- **Part A（闭环 prefix 方向，rstress − r0）**：speed 与 energy intensity 均为 **k=1 positive、k=2 negative、
  k=5 negative**（双 seed 一致）。canonical k=5 的负向与主 held-out 方向一致（不同 collector/engine，仅方向对照）。
- **Part A k=5 安全**：两臂计数逐值相同（arrive_dest 13、collision 0、OOR 0、terminated 13、truncated 3、max_step 3），
  canonical k=5 行为分离不由安全差异解释。
- **Part B（same-state 局部 policy→planner）**：双 seed `Δg_lon` 多数为负（`delta_g_lon_positive_majority=false`），
  `middle_effect_positive=false`，first-waypoint forward effect median `+0.0136/+0.0135 m`；first-plan response 跨
  prefix 匹配。即 Rstress 将纵向 guidance 下移，planner 中段前向响应非正 → 更慢、单位里程能耗更低。
- 预声明 verdict：`no_material_execution_contract_effect_proceed_to_training_budget`（因 E-047 对 k=1-trained
  checkpoints 预期的 k=1 负→k=5 正 crossover 未出现，此处为 k=1 正→k=2/5 负）。该 verdict 指的是 execution-horizon
  诊断轴本身，不否定 canonical k=5 行为方向可由局部映射解释。

## 与历史 gate 的关系

- **E-040（k=1）**：Rstress 比 R0 **更快且单位里程能耗更高**（speed +1.06%/+1.41%、energy +0.41%/+0.53%），
  Gate G FAILED（仅 c4 方向）。E-047 进一步显示其 k=1-trained checkpoints 在 k=5 重放仍为正。
- **E-054（canonical k=5 训练 + 评测）**：方向**反转**为更慢、更低 intensity，与 energy minimization 兼容。
  因此 objective→behavior 方向转移只在训练与评测同为 canonical k=5 时成立；不能把 E-040/E-047 的 k=5 重放
  方向直接外推为 canonical k=5 训练结果。

## 结论边界

**支持**：在 canonical k=5、E-052 冻结 control、matched 2 seeds、deterministic Beta mean 主评测下，
calibrated-band λ=64（Rstress）相对 calibrated R0 建立了**可复现、双 seed 一致、方向与 energy minimization
兼容**的 objective→behavior 转移：更慢（−1.7%）且单位里程能耗更低（−0.61%），无安全/停车/距离塌缩，
guidance 分离（纵向主导）双 seed 可重复，executed behavior 可由 canonical k=5 planner/execution mapping 解释。
这是 Task 2 的正结果，Gate T3 PASSED，可进入 Task 3（λ={0,1,2,4,8} dose-response / Gate D）。

**不支持 / 限制**：

- 仅 2 seeds、单一 no-traffic 训练池、单一 initial policy、50 updates；不构成跨 seed 总体或长期稳定性结论。
- speed −1.7% 与 intensity −0.61% 是**同一 co-movement**：energy 改善伴随降速，属 Gate T3 允许的
  energy-minimization 兼容方向；本记录**不**主张 non-inferiority（progress/speed trade-off 的正式判据属 #83 Gate F）。
- distance −0.37%、route −0.23% 幅度小且 intensity 为 per-km，不由少走解释；但不排除更细粒度 progress 代价。
- Rstress 训练内 reward 上升而 held-out 能耗下降，训练 reward 不作为候选选择依据，也不解释为 held-out 改善证据。
- 本记录**不**执行 Task 3（λ dose-response）与 #83 候选确认；不主张 λ=64 之外任何 λ 的方向，也不主张
  R0/Rstress 的绝对节能结论。
- c1/c2 阈值沿用 E-040 Gate G（guidance RMS floor 0.01）与 E-031 matched noise bound（0.1%）；本记录不重裁
  E-040/E-047 的历史结论。

## Provenance 与产物

- 实现提交 `f2cfaff`（protocol `calibrated-canonical.yaml` + `analysis/evaluation.py` guidance 配对分离 +
  回归测试）；bridge 配置提交 `d5d5a4e`；4 个训练 run 与 4 个 held-out eval 均在 commit `f2cfaff` 下、
  `git_status_short=[]`（clean）；bridge run 在 commit `d5d5a4e`、clean。
- 环境：Windows-10-10.0.26200 / Python 3.10.20 / torch 2.12.1+cu126 / CUDA:0 RTX A4000 / Lightning 2.6.6 /
  MetaDrive 0.4.3；bf16-mixed；upstream `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`、revision
  `ae5baf1c57229c53f6309332df960ae27d35333f`；EMA 276 tensors / 6,042,628 parameters。
- checkpoint identity：initial seed0 `049697739ab5…`（与 E-038/E-040/E-048–E-053 一致）、seed1 `feee6d85b681…`；
  final r0-seed0 `39aeb2571cad…`（=E-052/E-053 control）、r0-seed1 `7013b6fbabcc…`、
  rstress-seed0 `9d990c9831e5…`、rstress-seed1 `cf106bbe824b…`；bridge runtime `policy_hashes` 逐一一致，
  `optimizer_steps=0`。
- 代码级验证：`just test-target tests/analysis/test_scalar_effects.py`（8 passed，新增
  `test_guidance_contrast_reports_paired_probe_separation`）；`just test-target tests/training/test_training_workflows.py`
  （11 passed，新增 `test_canonical_positive_control_protocol_pins_frozen_control_and_two_arms`）；`just lint`、
  `just format-check`、`just typecheck`（0 errors）。
- 产物（git ignored）：
  - `outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control/`
    - `r0-seed-{0,1}/`、`rstress-seed-{0,1}/`：`summary.json`、`resolved_config.yaml`、`runtime_metadata.json`、
      `policy-initial/final.pt`、`updates/update-NNN/*.npz`（完整 rollout audit）。
    - `heldout/{r0,rstress}-seed-{0,1}-final/`：matched held-out 评测产物。
    - `comparison.yaml`：`compare analyze` 的显式 run/checkpoint/evaluation 分组。
  - `outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control-analysis/`：`analysis.json`（含
    `runs`、matched `contrasts`、`guidance` 配对分离）、`report.md`、`figures/`。
  - `outputs/studies/scalar-reward/e-054-issue106-task-2-positive-control-execution-bridge/`：`analysis.json`、
    `report.md`、`decisions.json`、`episodes.json`、`same_state.json`、`intervention_config.json`、
    `scenarios.json`、`figures/`。

## Issue #106 Task 2 结论

Gate T3 **PASSED**：canonical k=5 下 calibrated R0 vs calibrated-band λ=64 建立了可复现、双 seed 一致、
方向与 energy minimization 兼容的 objective→behavior 转移。下一步为 Task 3（λ={0,1,2,4,8} dose-response /
Gate D），本记录未执行。
