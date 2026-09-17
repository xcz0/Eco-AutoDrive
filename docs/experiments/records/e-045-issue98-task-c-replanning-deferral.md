# E-045 Issue #98 Task C：Replanning-deferral / temporal-consistency trace

[返回实验索引](../README.md) · [E-043 Task A](e-043-issue98-task-a-execution-horizon.md) · [E-044 Task B](e-044-issue98-task-b-guidance-decomposition.md)

**日期 / 类型 / 目的**：2026-09-17 / 正式配对闭环人工干预 / 不训练 policy，不依赖 reward
组合、critic 或 PPO。冻结与 E-043 相同的 planner / checkpoint / diffusion sampler /
observation / noise 与 matched 人工 longitudinal guidance，在 **baseline 0.1 s
receding-horizon execution（每周期只执行首点再重规划）** 下，对连续 20 个 replanning cycle
保存 full-horizon planner audit，直接检查正向 guidance effect 是否被持续推迟到永不执行
的未来，完成 Issue #98 Task C。未执行 Task D/E，未修改 reward、guidance 系数、PPO 主配置
或 baseline execution contract。

**裁定：Task C 观察到稳定的 repeated deferral，支持机制
`positive guidance effect is repeatedly deferred beyond the executed prefix`。** 机器
Gate C 状态为 `repeated_deferral`：13/16 场景同时满足“多数 cycle 首点效应 ≤ 0”“多数
cycle 8 s full-horizon 效应 > 0”“多数 cycle 正 response 越过执行 prefix”且
zero-crossing step 稳定。安全/完整性 clean，proxy 无错误。

## 运行来源与固定协议

- 代码基线 `3f18d5a2a9bf784553671a65476418c248a6a935`（main）加 Task C 未提交改动。
  `runtime_metadata.json` 保存 `git_head`、`git_branch`、`git_status_short`；工作区 dirty
  （4 个修改文件、5 个新增路径），因此这是诊断干预而非 clean-commit 正式基线。
- Windows、Python 3.10.20、PyTorch 2.12.1+cu126、MetaDrive 0.4.3。实际 GPU 为
  **NVIDIA RTX A4000**，resource profile `rtx_a4000` 与实际一致：4 个 rollout workers、
  每 worker 12 threads。
- 官方 EMA checkpoint：`checkpoints/DP-Origin/model.pth`，276 个 EMA tensors、
  6,042,628 parameters；冻结、eval 模式；不构造 Exploration Policy、critic 或 optimizer。
- 标准高斯 DDIM-5、stochasticity=0、BF16 mixed precision、确定性与 runtime seed=0；
  orthogonal-policy guidance 系数与当前训练配置相同，longitudinal fraction=0.25。
- 场景：no-traffic，S（`straight_s0–s7`）/ SC（`gentle_curve_s0–s7`）× map seeds 0–7
  （16 场景），与 E-043 Task A 相同的训练池协议。每场景 noise seeds `[0,1,2]`；每个 repeat
  比较 `g_lon=[-1,-0.5,0,0.5,1]`，`g_lat=0`。simulator reset seed 为该场景 map seed。
- 总窗口固定为 2.0 s = 20 个 0.1 s 决策；`execution_steps=1`（baseline ROLLOUT），每周期
  重规划一次，共 20 cycles；env horizon=21。
- 同一 matched group 保持物理 slot、batch shape、初始 observation/state 与逐周期 diffusion
  noise 一致；各 arm 从 reset 开始，只有 guidance 改变；不从 policy 采样动作。
- 每周期保存完整 8 s（80 waypoint）ego forward displacement，以及既有的 checkpoint 摘要；
  这是 Task A 之后的增量：`evaluation.intervention` 的 per-waypoint 记录为 opt-in，其它
  workflow 与 E-043/E-044 产物不受影响。

正式运行一次，完成 **240 episodes / 4,800 transitions**，耗时 **234.83 s**。独立离线
复算使用 `episodes.json` 重新生成 `summary.json`，与运行目录逐值相同；源文件不变。

```powershell
just exp guidance deferral run --output-dir outputs/studies/scalar-reward/e-045-issue98-task-c-replanning-deferral

just exp guidance deferral analyze --source-dir outputs/studies/scalar-reward/e-045-issue98-task-c-replanning-deferral --output-dir outputs/studies/scalar-reward/e-045-issue98-task-c-replanning-deferral-analysis
```

配置：`configs/experiments/guidance/deferral.yaml`。实际 composed 参数保存在
`resolved_config.yaml`；其中训练 job 的 policy/PPO 字段仅因复用配置而存在，不代表创建或
运行了这些对象。

## 预定判据与结果

matched 效应定义为跨 arm 端点差 `Δ = D(g=+1) − D(g=-1)`（`D` 为沿预测轨迹的累计前向
位移，m）。每个 (scenario, cycle) 先把逐 noise repeat 的 `Δ` 取中位数，再在场景内按 cycle
统计。预声明阈值：`majority_fraction=0.5`、`crossing_tolerance_steps=1`、
`required_scenarios=9`。场景判定 `repeated_deferral` 需同时满足：

1. 多数 cycle `Δ(first waypoint) <= 0`；
2. 多数 cycle `Δ(8 s full horizon) > 0`；
3. 多数 cycle 首个正 response 的 waypoint step 越过执行的 0.1 s prefix（`crossing_step >= 1`）；
4. zero-crossing step 在 defined cycle 上稳定（spread ≤ 1）。

结果：**13/16 场景满足全部条件**，Gate C = `repeated_deferral`。

| Scenario | cycles | median first (m) | median full (m) | first<0 | full>0 | deferred | crossing 中位 step | spread | repeated |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| straight_s0 | 20 | -0.0625 | 27.03 | 14 | 20 | 14 | 1 | 1 | True |
| straight_s1 | 20 | -0.0624 | 26.26 | 16 | 20 | 16 | 1 | 1 | True |
| straight_s2 | 20 | +0.0624 | 29.02 | 8 | 20 | 8 | 0 | 1 | False |
| straight_s3 | 20 | +0.0622 | 29.75 | 7 | 20 | 7 | 0 | 1 | False |
| straight_s4 | 20 | -0.0627 | 26.78 | 14 | 20 | 14 | 1 | 1 | True |
| straight_s5 | 20 | -0.00004 | 28.50 | 10 | 20 | 10 | 0.5 | 1 | True |
| straight_s6 | 20 | -0.0624 | 27.03 | 15 | 20 | 15 | 1 | 1 | True |
| straight_s7 | 20 | -0.0625 | 26.25 | 15 | 20 | 15 | 1 | 1 | True |
| gentle_curve_s0 | 20 | -0.0625 | 27.03 | 14 | 20 | 14 | 1 | 1 | True |
| gentle_curve_s1 | 20 | -0.0624 | 26.26 | 16 | 20 | 16 | 1 | 1 | True |
| gentle_curve_s2 | 20 | -0.0001 | 24.71 | 11 | 20 | 11 | 1 | 1 | True |
| gentle_curve_s3 | 20 | -0.1893 | 18.68 | 16 | 20 | 16 | 1 | 1 | True |
| gentle_curve_s4 | 20 | -0.0627 | 26.78 | 14 | 20 | 14 | 1 | 1 | True |
| gentle_curve_s5 | 20 | +0.00006 | 28.50 | 9 | 20 | 9 | 0 | 1 | False |
| gentle_curve_s6 | 20 | -0.0624 | 27.03 | 15 | 20 | 15 | 1 | 1 | True |
| gentle_curve_s7 | 20 | -0.0625 | 26.25 | 15 | 20 | 15 | 1 | 1 | True |

跨 16 场景中位数（`median_effect_by_cycle`，0.1/0.2/0.5/1/2/4/8 s，m）：

| Cycle | 0.1 s | 0.2 s | 0.5 s | 1.0 s | 2.0 s | 4.0 s | 8.0 s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | **-0.125** | +0.529 | +1.709 | +3.187 | +6.629 | +13.49 | +27.996 |
| 1 | -0.186 | +0.315 | +1.124 | +2.244 | +5.009 | +9.618 | +22.508 |
| 5 | -0.0002 | +0.505 | +1.624 | +2.999 | +6.628 | +12.87 | +24.482 |
| 10 | +0.062 | +0.691 | +1.874 | +3.498 | +7.131 | +14.25 | +28.029 |
| 15 | -0.188 | +0.212 | +1.212 | +2.055 | +4.804 | +9.467 | +23.963 |
| 19 | -0.189 | +0.626 | +1.698 | +2.998 | +6.130 | +13.87 | +30.505 |

机制：每个 cycle 的 matched 效应在首点（0.1 s）为负或 ~0，在 0.2 s 已转正，并在 8 s 累积
到 ~+19–30 m；zero-crossing step 几乎恒为 1（0.2 s），跨 cycle spread = 1 step。因此正
response 的绝对到达时间随 cycle 前移（`c + 1`），在每周期只执行首点（0.1 s）的 baseline
下**永远不会被到达**。

### 与 E-043 Task A 一致性

Task C 的 cycle-0 first-plan 跨场景中位（-0.125 / +0.529 / +1.709 / +3.187 / +6.629 /
+13.49 / +27.996）与 E-043 记录的 first-plan 响应（-0.125 / +0.492 / +1.696 / +3.186 /
+6.668 / +13.451 / +27.919）在 0.1 s 处精确一致，后续 checkpoint 差异仅来自聚合口径
（E-043 为 48 场景/repeat 中位，Task C 为逐场景 noise 中位后再跨场景中位）。这确认两次
运行共享同一冻结 planner / noise / 初始状态，Task C 是 E-043 k=1 的直接时间维展开。

## 安全性与响应链

- 全部 240 episodes 均为 `window_complete`；无 collision、无 out-of-road、无 terminated/
  truncated，`unsafe_or_incomplete_episodes` 为空。
- 逐 episode 的 planner-to-execution `position_error_m` 最大约 `9.7e-7 m`，heading error
  最大约 `6.7e-8 rad`；执行严格跟随 planner 输出点。
- Energy proxy：全部 transitions 符合 `32.5 * exp(0.036 * speed_mps) * distance_m / 1000`
  mL，容差 rtol=1e-10 / atol=1e-12；`proxy_errors` 为空。
- 3 个未触发 repeated deferral 的场景（`straight_s2`、`straight_s3`、`gentle_curve_s5`）
  first-waypoint 中位为正或 ~0（`first_positive_count` 11–13/20），原样保留，未删除、未
  按 seed 选择。

## 结论边界

- **机制**：Task C 直接支持 `positive guidance effect is repeatedly deferred beyond the
  executed prefix`。核对每一 replanning cycle 后，正 response 的 zero-crossing 稳定落在
  0.2 s，而 baseline 只执行 0.1 s 首点，因此 planner full-horizon 正响应无法转化为
  executed response；这与 E-043 Task A 的 k=1 负 executed-speed、k≥2 转正完全一致。
- 该结论是 Task A 归因 A（guidance temporal response × receding-horizon execution
  mismatch）的直接时间维证据，但**不单独关闭 #98**：Task D（training adequacy / critic）与
  Task E（training-budget）仍未执行，E-040 的 c4 失败归因仍需后续 Task 共同完成。
- baseline execution 仍为 MetaDrive kinematic receding-horizon（每周期 1 点）；本 trace 是
  诊断 intervention，不升级为推荐执行方案、不改写 baseline contract。
- 结论限于本组 no-traffic S/SC seeds 0–7、2 s 窗口、当前 BF16 冻结 planner 与 MetaDrive
  fuel proxy；不建立真实车辆动力学可跟踪性、真实能耗改善或长程行为。
- 3 个未复现场景的 first-waypoint 正响应单独保留，提示存在 state-dependent 的残余异质性，
  需要后续单独解释，不作为反例删除。

## 验证与产物

- 新增/修改测试：`tests/training/test_guidance_deferral.py`（config 校验、zero-crossing
  单元、repeated/absent/mixed 裁定、安全/完整性/proxy、per-waypoint horizon 校验、提前终止
  的 matched-common-cycle 处理、offline recompute 与 live 一致、CLI 路由）；受影响
  workflow 的 `tests/training/test_guidance_horizon.py`、`test_guidance_decomposition.py`、
  `test_guidance_control_authority.py` 全部通过（33 passed）。
- `just lint`、`just typecheck` 0 error。独立离线 analyze 的 `summary.json` 与运行目录
  逐值相同，源文件不变。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-045-issue98-task-c-replanning-deferral/`
  - `raw/group-*-arm-*/`：初始 observation、每周期 full planner audit NPZ、逐 arm episode 证据；
  - `episodes.json`（含 opt-in `waypoint_forward_displacement_m`）、`scenarios.json`、
    `intervention_config.json`、`resolved_config.yaml`；
  - `runtime_metadata.json`、`run.json`、`decisions.json`；
  - `summary.json`、`analysis.json`、`report.md`、三张图（effect heatmap、first-vs-full、
    crossing-step）。

独立复算目录：
`outputs/studies/scalar-reward/e-045-issue98-task-c-replanning-deferral-analysis/`。

Task C 已完成；本次改动未提交、未推送，已更新 Issue #98 的 Task C 状态。
