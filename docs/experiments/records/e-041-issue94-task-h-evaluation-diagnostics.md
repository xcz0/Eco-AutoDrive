# E-041 Issue #94 Task H：evaluation diagnostics（deterministic + stochastic 行为比较）

[返回实验索引](../README.md) · [前置 E-040 Task G](e-040-issue94-task-g-objective-positive-control.md)

**日期 / 类型 / 目的**：2026-09-11 / 正式诊断评测 / 按 Issue #94 Task H 要求，在 E-040 冻结的
matched deterministic held-out 主评测之外，增加固定 policy-action seed 的 diagnostic
stochastic evaluation，区分三层：(1) Beta mean 是否变化；(2) Beta
concentration/variance 是否变化；(3) deterministic mean policy 未覆盖的
stochastic behavior shift。本 Task 无 gate 判定；stochastic diagnostic 不替代
matched deterministic 主评测，也不单独作为 Gate G 通过依据。执行前用户裁定
"Gate G（E-040，仅 c4 方向条件失败）暂视为通过"以进入 Task H。

**裁定（descriptive）**：三层区分全部得到明确回答——

1. **Beta mean 变化：是。** matched 固定 probe contexts 上，两 arm 训练后 guidance
   mean 均发生迁移（rstress−r0 paired mean delta RMS 0.0619 / 0.0828，复现
   E-040 c1）；
2. **Beta concentration / variance 变化：基本没有。** concentration 维持
   ≈4.0（final−initial 与 paired delta 均 ≤0.087，约 2% 量级），variance 维持
   ≈0.200（delta ≤0.004）。训练效果几乎完全落在分布均值上，未改变分布形状；
3. **Stochastic behavior shift：存在且系统性。** 对包括 initial 在内的所有
   policy，Beta 采样使闭环 mean speed 系统性 +0.116~+0.122 m/s（+1.16%~1.21%）、
   energy intensity +0.35~+0.38 mL/km（+0.75%~0.81%），远超 0.1% matched 噪声界
   ——deterministic mean 评测不是该 stochastic policy 行为的无偏估计。但
   rstress−r0 的 paired 分离在 stochastic 评测下方向完整复现（2 seeds × 3
   action seeds 共 6/6 同向），说明 E-040 的分离是稳定的 mean-policy 性质，
   不是 deterministic-only 伪影。

## 运行来源与固定协议

- 代码基线为 `scalar-energy-reward` 分支 commit `b0a30c8`（Task G）之上的未提交
  Task H 改动：policy-checkpoint 评测新增 `action_mode: mean|sample`
  （`evaluation.policy_checkpoint.action_mode` + `policy_action_seed`，sample 模式
  要求 serial topology、diffusion 流仍为 ddim_stochasticity=0），以及
  `evaluation-diagnostics` study；每个 run 目录的 `runtime_metadata.json` 与
  `resolved_config.yaml` 保存正式运行来源。
- Windows、Python 3.10、PyTorch 2.12.1+cu126、MetaDrive 0.4.3，GPU NVIDIA RTX
  A4000（与 E-039/E-040 同机）。
- **输入完全复用 E-040 冻结产物**：4 个 run 的 `policy-initial.pt` /
  `policy-final.pt`、`heldout/seed-{0,1}/{initial,r0,rstress}/`（matched
  deterministic 主评测，不重跑）、训练 summary 中的固定 probe（`probe_before` /
  `probe_after` 的 alpha/beta）。
- **Stochastic 评测协议**：与 matched deterministic job 逐字段一致（同一
  held-out 场景池 S/SC seeds 16–23、horizon 300、ddim5 stochasticity=0、
  diffusion noise seed 760025），唯一差异是 policy action 从 Beta 采样：每
  episode 一个 policy RNG stream，由 job 级 `policy_action_seed` 播种，同一
  action seed 下各 arm/场景 matched。固定 policy-action seeds
  `{810001, 810002, 810003}`。
- 评测矩阵：2 training seeds × {initial, r0-final, rstress-final} × 3
  policy-action seeds = 18 个 stochastic 评测 job（每 job 16 episodes，单 job
  评测耗时 ≈89 s）。

```powershell
just experiment training evaluation-diagnostics run `
  --source-dir outputs/studies/scalar-reward/e-040-issue94-task-g-objective-positive-control `
  --output-dir outputs/studies/scalar-reward/e-041-issue94-task-h-evaluation-diagnostics
```

配置清单：`configs/experiments/training/evaluation-diagnostics.yaml`；实现位于
`src/eco_planner/experiments/training/evaluation_diagnostics/`。

## 结果

### Deterministic vs stochastic 闭环行为（held-out 聚合）

| seed | policy | det speed (m/s) | sto speed (m/s)，3 action seeds 均值 [min,max] | det energy (mL/km) | sto energy (mL/km) [min,max] | arrive (det/sto) | collision / OOR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | initial | 9.998465 | 10.114024 [10.0024, 10.1885] | 47.047213 | 47.404048 [47.1948, 47.6027] | 0.8125 / 0.8125 | 0 / 0 |
| 0 | r0 final | 9.964411 | 10.082072 [9.9722, 10.1603] | 46.983970 | 47.342835 [47.1385, 47.5466] | 0.8125 / 0.8125 | 0 / 0 |
| 0 | rstress final | 10.070494 | 10.192778 [10.0838, 10.2640] | 47.174267 | 47.556909 [47.3543, 47.7545] | 0.8125 / 0.8125 | 0 / 0 |
| 1 | initial | 9.998465 | 10.114024 [10.0024, 10.1885] | 47.047213 | 47.404048 [47.1948, 47.6027] | 0.8125 / 0.8125 | 0 / 0 |
| 1 | r0 final | 9.935536 | 10.052340 [9.9374, 10.1229] | 46.936153 | 47.288579 [47.0773, 47.4724] | 0.8125 / 0.8125 | 0 / 0 |
| 1 | rstress final | 10.075349 | 10.196915 [10.0899, 10.2673] | 47.185713 | 47.566141 [47.3688, 47.7608] | 0.8125 / 0.8125 | 0 / 0 |

Initial 的 stochastic 结果双 training seed 逐位一致（10.114024 / 47.404048），
与 E-040 "initial actor 跨 seed 行为相同" 的结构性事实自洽。全部 18 个 job 无
collision、无 OOR、无 stopped，arrive_dest 全保持 0.8125。

### Stochastic sampling 的系统性行为偏移（sto − det，同 checkpoint 配对）

| seed | policy | Δ speed (m/s) | Δ speed (rel) | Δ energy (mL/km) | Δ energy (rel) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | initial | +0.115558 | +1.16% | +0.356835 | +0.76% |
| 0 | r0 | +0.117662 | +1.18% | +0.358865 | +0.76% |
| 0 | rstress | +0.122284 | +1.21% | +0.382641 | +0.81% |
| 1 | r0 | +0.116804 | +1.18% | +0.352426 | +0.75% |
| 1 | rstress | +0.121565 | +1.21% | +0.380428 | +0.81% |

采样偏移对三个 policy（含 initial）几乎同大且同向（更快、单位里程能耗更高），
说明这是 Beta(≈2,2) 采样经 frozen planner / 执行通道非线性传播的通道性质，
不是训练产生的。偏移幅度约为 E-040 paired arm 分离（speed +1.06%/+1.41%）的
同量级、远超 0.1% matched 噪声界。

### Paired rstress−r0 分离在 stochastic 评测下的复现

| seed | 评测 | Δ speed (m/s) | Δ energy (mL/km) | Δ route completion |
| --- | --- | ---: | ---: | ---: |
| 0 | deterministic | +0.106083 | +0.190298 | +0.002382 |
| 0 | stochastic mean | +0.110706 | +0.214074 | +0.000918 |
| 1 | deterministic | +0.139814 | +0.249560 | +0.000932 |
| 1 | stochastic mean | +0.144575 | +0.277561 | +0.001686 |

逐 action seed 的 paired delta（speed / energy，mL/km 与 m/s）：

| seed | action seed | Δ speed | Δ energy |
| --- | --- | ---: | ---: |
| 0 | 810001 | +0.103732 | +0.207923 |
| 0 | 810002 | +0.111596 | +0.215869 |
| 0 | 810003 | +0.116789 | +0.218430 |
| 1 | 810001 | +0.144412 | +0.288400 |
| 1 | 810002 | +0.152507 | +0.291542 |
| 1 | 810003 | +0.136806 | +0.252743 |

6/6（2 training seeds × 3 action seeds）与 deterministic 方向一致（speed、
energy 均为正）；action-seed 间散布（同一 policy 的 sto speed min-max ≈
±0.1 m/s）与 paired delta 同量级，但方向不翻转。E-040 的 c4 失败裁定
（分离方向与 energy 目标相反）在 stochastic 评测下同样成立。

### Beta 分布三层归因（matched 固定 probe contexts，16 contexts × 2 维）

| seed | arm | init conc / var | final conc / var | paired rstress−r0 conc Δ | paired var Δ | paired mean Δ (lon, lat) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | r0 | 4.000 / 0.2000 | 3.979 / 0.2005 | — | — | — |
| 0 | rstress | 4.000 / 0.2000 | 4.002 / 0.1999 | +0.023 / +0.047（RMS 0.037） | −0.0006 / −0.0025（RMS 0.0018） | +0.0253 / +0.0838（RMS 0.0619） |
| 1 | r0 | 4.000 / 0.2000 | 3.956 / 0.2015 | — | — | — |
| 1 | rstress | 4.000 / 0.2000 | 3.961 / 0.1958 | +0.005 / +0.068（RMS 0.048） | −0.0007 / −0.0029（RMS 0.0021） | +0.0239 / +0.1146（RMS 0.0828） |

（conc / var 为两维均值；mean Δ 为 rstress−r0 final 的 per-dimension 均值差。）

> **Errata（2026-09-17，E-044）**：上表 `paired mean Δ (lon, lat)` 的标签与数值顺序互换。
> 权威 guidance 顺序为 `[lateral, longitudinal]`（`src/eco_planner/planning/diffusion/guidance.py:135,153`、
> `src/eco_planner/rl/optimization/gradients.py:43-44`），因此 seed0 `(+0.0253, +0.0838)`、
> seed1 `(+0.0239, +0.1146)` 实为 `(lateral, longitudinal)`——**较大迁移是纵向**。
> 上表数字未改，仅更正标签；E-044 的臂构造已按正确顺序执行。

- Beta mean：两 arm final 均相对 initial 迁移，且 rstress−r0 出现 seed 间方向
  一致的 paired 差异——行为分离的第一层来源；
- concentration / variance：所有 arm 的 final−initial 与 paired 差异都在
  initial 值（4.0 / 0.200）的 ~2% 以内——训练没有改变分布形状；
- stochastic shift：闭环层面的采样偏移（上表二）独立于 arm，是第三层。

## 结论边界

- 本实验是 **diagnostic**，无 gate 判定；不改变 matched deterministic 主评测
  协议（`action_mode: mean` 语义与 E-029/E-031/E-040 完全一致）。
- 进入 Task H 依据的是用户裁定 "Gate G 暂视为通过"；E-040 的 c4 失败裁定
  （分离方向与 energy 目标相反）未被本实验推翻——stochastic 评测下方向依旧
  相反。本实验不支持任何节能结论。
- stochastic-vs-deterministic 的系统性偏移（+1.2% speed / +0.8% energy）是在
  当前 near-Beta(2,2)、concentration≈4 的分布上测得的；它是对
  "deterministic mean 不是 stochastic policy 行为的无偏估计"的量化，不能外推
  到其他 concentration 或其他 planner 通道。其机制（为何对称采样经 planner
  通道偏向更快/更耗能）未被本实验建立。
- Beta concentration/variance 的"基本未变"是 50 updates、E-039 冻结 lr 区域内
  的结论；更长的训练预算或 PPO entropy 项调整可能改变分布形状，本实验不覆盖。
- probe-context Beta 统计来自 training 场景分布（16 个首 transition contexts），
  与 held-out 行为分布不同；本实验用其做 matched-context 归因，不声称覆盖
  held-out 状态分布上的全部 Beta 参数变化。
- Task H 完成后，Issue #94 的下一步是 #83 的 λ 候选筛选与多 seed confirmation
  （以 "Gate G 通过" 为前提的 energy-minimization 结论仍受 E-040 c4 约束）。

## 验证与产物

- 新增测试：`tests/evaluation/test_policy_agent.py`（agent mean/sample 路由与
  构造校验 3 项）、`tests/training/test_evaluation_diagnostics.py`（manifest、
  composition、Beta/probe/stochastic 诊断、summary 组装、source 布局校验 10 项）、
  `tests/configuration/test_scalar_reward.py` 新增 action_mode 校验 2 项；
  `just test-target tests/evaluation`（15 passed）、相关 training/configuration
  测试（32+13 passed）、`just test`（9 passed）、`just test-sim`（8 passed）、
  `ruff check` / `ruff format --check` / `just typecheck`（0 errors）通过。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-041-issue94-task-h-evaluation-diagnostics/`
  - `study_manifest.yaml`：resolved 研究清单（training seeds + policy-action
    seeds 预登记）。
  - `stochastic/seed-{0,1}/{initial,r0,rstress}/action-{810001,810002,810003}/`：
    18 个 stochastic held-out 评测 job（每 job `resolved_config.yaml`、
    `runtime_metadata.json`、`summary.json` + 16 个场景的 `summary.json` /
    `trace.npz`）。
  - `summary.json`：deterministic（复用 E-040）+ stochastic 全量聚合、
    paired rstress−r0（deterministic / stochastic mean / 逐 action seed）、
    direction 复现矩阵、Beta 分布三层统计。
  - `run.out.log` / `run.err.log`：正式运行日志。

Task H 已完成（descriptive diagnostics，无 gate）；本次改动未提交、未推送，
GitHub Issue #94 的状态更新另行执行。
