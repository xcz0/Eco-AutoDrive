# E-043 Issue #98 Task A：Execution-horizon causal intervention

[返回实验索引](../README.md) · [相关 #98 Task D 前置](../records/e-037-issue94-task-d-guidance-control-authority.md)

**日期 / 类型 / 目的**：2026-09-16 / 正式配对闭环人工干预 / 不训练 policy，不依赖 reward
组合、critic 或 PPO。冻结与 E-037 相同的 planner / checkpoint / diffusion sampler /
observation / noise 与 matched 人工 longitudinal guidance，只改变“每次规划后连续执行多少个
0.1 s waypoint 再重规划”，完成 Issue #98 Task A。未执行 Task B–E，未修改 reward、guidance
系数、PPO 主配置或 baseline execution contract。

**裁定：Task A 支持归因 A（guidance temporal response × receding-horizon execution
mismatch）。** 机器 Gate A 状态为 `safety_or_proxy_failure`：预声明判据要求
`safety_clean`，而 k=2（0.2 s prefix）在负 guidance 臂出现 73/1200 个提前 off-route 终止。
该安全项本身是 horizon 干预的直接结果，不改变符号翻转证据；详见下文。

## 运行来源与固定协议

- 代码基线 `23f036bfd63a3c04a984662136ce11457a628d2e`（main）加 Task A 未提交改动。
  `runtime_metadata.json` 保存 `git_head`、`git_branch`、`git_status_short`；工作区 dirty
  （11 个修改文件、5 个新增路径，见 metadata），因此这是诊断干预而非 clean-commit 正式基线。
- Windows、Python 3.10.20、PyTorch 2.12.1+cu126、MetaDrive 0.4.3。实际 GPU 为
  **NVIDIA RTX A4000**（driver 556.18，`cuda:0`），resource profile `rtx_a4000` 与实际一致：
  4 个 rollout workers、每 worker 12 threads、evaluation slots 4。
- 官方 EMA checkpoint：`checkpoints/DP-Origin/model.pth`，276 个 EMA tensors、
  6,042,628 parameters；冻结、eval 模式；不构造 Exploration Policy、critic 或 optimizer。
- 标准高斯 DDIM-5、stochasticity=0、BF16 mixed precision、确定性与 runtime seed=0；
  orthogonal-policy guidance 系数与当前训练配置相同，longitudinal fraction=0.25。
- 场景：no-traffic，S / SC × map seeds 0–7（16 场景）。每场景 noise seeds `[0,1,2]`；每个
  repeat 比较 `g_lon=[-1,-0.5,0,0.5,1]`，`g_lat=0`。simulator reset seed 为该场景 map seed。
- 总窗口固定为 2.0 s = 20 个 0.1 s 决策；execution horizons `k=[1,2,5,10,20]` waypoints，
  `replan 次数 = 20/k`（20/10/4/2/1）。env horizon=21。所有 horizon 在同一 2 s 内可比。
- 同一 matched group 保持物理 slot、batch shape、初始 observation/state 与逐周期 diffusion
  noise 一致；各 arm 与各 horizon 从 reset 开始，只有 guidance 与 execution prefix 改变；
  不从 policy 采样动作。执行 prefix 通过新增的显式 `execution_steps` 覆盖（默认仍为
  `ExecutionMode.ROLLOUT=1` / `EVALUATION=5`）。
- 每个 horizon 的 first-plan forward-displacement 响应在跨 horizon 间逐元素核对匹配，
  不一致即抛错；执行前 5 点仍由 baseline 语义之外、仅供本诊断使用。

正式运行一次，完成 **1,200 episodes / 23,332 transitions**，耗时 **537.15 s**。运行开始后仅对新增
模块做非语义 Ruff 格式化，并更新文档与 Issue；未改变该次采集已加载的代码、统计公式、参数或
Gate。独立离线复算使用重算后的 `episodes.json`，得到与运行目录逐值相同的 `summary.json`。

```powershell
just exp guidance horizon run --output-dir outputs/studies/scalar-reward/e-043-issue98-task-a-execution-horizon

just exp guidance horizon analyze --source-dir outputs/studies/scalar-reward/e-043-issue98-task-a-execution-horizon --output-dir outputs/studies/scalar-reward/e-043-issue98-task-a-execution-horizon-analysis
```

配置：`configs/experiments/guidance/horizon.yaml`。实际 composed 参数保存在
`resolved_config.yaml`；其中训练 job 的 policy/PPO 字段仅因复用配置而存在，不代表创建或运行
了这些对象。

## 预定判据与结果

沿用 authority 的逐场景条件：`|Spearman| >= 0.8` AND `|endpoint effect| >= 2 * noise
scale` AND `endpoint effect != 0`；同方向场景数 >= 9/16 即该指标方向成立。executed-speed
方向定义为该 horizon 下多数通过场景的方向。

下表 effect 为 **16 个场景统计量的中位数**；通过数来自逐场景判定。effect 均为 `+1 − (-1)`。

| Horizon (s) | replans | speed 方向 | speed 通过 | speed effect 中位 (m/s) | speed ρ 中位 | endpoint effect 中位 | energy effect 中位 (mL/km) |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 (k=1) | 20 | **negative** | 10/16 | -0.5933 | -0.90 | -1.8262 | -0.8137 |
| 0.2 (k=2) | 10 | **positive** | 16/16 | +2.8915 | 1.00 | +7.0219 | +5.7348 |
| 0.5 (k=5) | 4 | **positive** | 16/16 | +3.2813 | 1.00 | +3.0516 | +6.0648 |
| 1.0 (k=10) | 2 | **positive** | 16/16 | +3.3114 | 1.00 | +6.0492 | +6.4313 |
| 2.0 (k=20) | 1 | **positive** | 16/16 | +3.3446 | 1.00 | +4.7845 | +6.2703 |

首次规划的完整 8 s planner trajectory 前向位移差（跨 48 场景/repeat 中位，m）：

| 预测时间 | 0.1 s | 0.2 s | 0.5 s | 1.0 s | 2.0 s | 4.0 s | 8.0 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| effect 中位 | -0.125 | +0.492 | +1.696 | +3.186 | +6.668 | +13.451 | +27.919 |
| 方向 | negative | positive | positive | positive | positive | positive | positive |

因此 `g_lon -> executed speed` 的响应在 execution prefix 从 0.1 s 增至 0.2/0.5 s 时由负转正，
并与 planner full-trajectory 的正响应（0.2 s 起）方向一致；k=5/10/20（全部 240 episodes
完整、无失效）给出 16/16 同向正响应，是该结论的 clean-horizon 主证据。

## 安全性与响应链

- 全部 1,200 episodes 中 **73 个提前终止，全部位于 k=2**，全部为 `g_lon ∈ {-1.0,-0.5}`
  的 `out_of_road`（无 collision、无其它终止原因）；最短 6 步，其余完整性正常。k=1/5/10/20
  均为 240/240 `window_complete`。
- 逐 episode 的 planner-to-execution `position_error_m` 约为 `1e-7 m`，heading error 为 0；
  提前终止 episode 的执行位置严格跟随 planner 输出点。因此 73 个 off-route 是 planner 在
  该执行前缀下的规划结果，不是执行器未跟踪目标点。
- Energy proxy：全部 transitions 符合
  `32.5 * exp(0.036 * speed_mps) * distance_m / 1000` mL，容差 rtol=1e-10 / atol=1e-12；
  无 proxy 错误。energy intensity 方向同样由 k=1 负（5/16，未过多数门槛，标记
  `energy_sensitivity_limited`）转为 k≥2 的 16/16 正。
- 机器 Gate A 因 `safety_clean=false` 记为 `safety_or_proxy_failure`。该标签只表示预声明安全
  守卫触发；符号翻转与 planner 一致性证据不受影响。k=2 点因此不作为主证据，只作支持性观察。

## 结论边界

- **归因**：Task A 直接支持解释 **A. guidance temporal response × receding-horizon
  execution mismatch**。E-037 的“full trajectory 正响应 / first waypoint 负响应”并非训练不足
  所致，而是 0.1 s 只执行首点、正向响应落在未执行后续点造成的执行语义错位；执行 0.2–0.5 s
  前缀即可反转符号。
- 该结论未排除 B（learned lon/lat coupling）、C（training budget/critic）或 D（reward
  representation）。Task B–E 仍未执行；E-040 的 c4 failure 归因需要后续 Task 共同完成，
  因此本记录不关闭 Issue #98。
- baseline execution 仍为 MetaDrive kinematic receding-horizon（ROLLOUT 每周期 1 点）；
  execution-horizon 变化只是诊断 intervention，不升级为推荐执行方案、不改写 baseline contract。
- 结论限于本组 no-traffic 场景、2 s 窗口、当前 BF16 冻结 planner 与 MetaDrive fuel proxy；
  不建立真实车辆动力学可跟踪性、真实能耗改善或长程行为。
- k=2 的 off-route 及其对负 guidance 臂的偏置单独保留，未删除、未填补；该现象本身需要后续
  单独解释（可能属于 planner 在该状态/prefix 下的输出特性）。

## 验证与产物

- 新增/修改测试：`tests/training/test_guidance_horizon.py`（config 校验、方向判据、安全/完整性、
  offline recompute 与 live 一致、CLI 路由、跨 horizon planner 匹配）、
  `tests/simulation/test_closed_loop.py::test_environment_slot_executes_explicit_prefix_beyond_fixed_modes`，
  并更新 authority 测试的 `InterventionExecution` 调用。
- 全量 CPU 测试 **288 passed**（`just test-all-cpu`）；定向 simulator 测试 **3 passed**
  （显式 `execution_steps` + authority rollout 窗口/噪声配对）；`just lint`、`just typecheck`
  0 error。独立离线 analyze 的 `summary.json` 与运行目录逐值相同，源文件不变。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-043-issue98-task-a-execution-horizon/`
  - `raw/group-*-arm-*/`：初始 observation、每周期 planner audit NPZ、逐 arm episode 证据；
  - `episodes.json`、`scenarios.json`、`intervention_config.json`、`resolved_config.yaml`；
  - `runtime_metadata.json`、`run.json`、`decisions.json`；
  - `summary.json`、`analysis.json`、`report.md`、三张响应图。

独立复算目录：
`outputs/studies/scalar-reward/e-043-issue98-task-a-execution-horizon-analysis/`。

Task A 已完成；本次改动未提交、未推送，已更新 Issue #98 的 Task A 状态。
