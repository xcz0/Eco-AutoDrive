# E-037 Issue #94 Task D：Guidance control-authority intervention

[返回实验索引](../README.md) · [前置 C4 归因](e-036-issue94-task-c4-critic-gae-common-term-ablation.md)

**日期 / 类型 / 目的**：2026-09-10 / 正式配对闭环人工干预 / 不训练 policy，不依赖 reward
组合、critic 或 PPO，直接验证 longitudinal guidance 经冻结 planner 到实际执行与 Energy
的控制能力。完成 Task D；未执行 Task E–H，未修改 reward、guidance 系数或执行方式。

**裁定：Gate D PASSED，经 executed-speed 主指标通过；Energy intensity 单独未通过。**
该裁定不表示正 guidance 会提高实际执行速度。多数场景的响应方向为负，完整预测与实际执行
的首点存在显著时间分布差异，见下文。

## 运行来源与固定协议

- 代码基线 `b5ea12557031dae74b961c609f524362438e75dd` 加 Task D 未提交改动；
  `runtime_metadata.json`、`tracked_diff.patch` 和 `source/` 保存正式采集时的来源。
- Windows、Python 3.10.20、PyTorch 2.12.1+cu126、Lightning 2.6.5、MetaDrive 0.4.3。
  实际 GPU 为 **NVIDIA GeForce RTX 3050 Laptop GPU**（driver 596.49），设备 `cuda:0`。
  配置名 **`rtx_a4000`** 是预定资源配置，实际使用 4 个 rollout workers、每 worker 12
  PyTorch threads；配置名不代表本机 GPU 型号。
- 官方 EMA checkpoint：`checkpoints/DP-Origin/model.pth`，架构
  `checkpoints/DP-Origin/args.json`；严格加载 276 个 EMA tensors，加载报告参数量
  6,042,628。保持冻结、eval 模式；不构造 Exploration Policy、critic 或 optimizer。
- 标准高斯 DDIM-5、stochasticity=0、BF16 mixed precision、确定性设置，runtime seed=0；
  orthogonal-policy guidance 系数与当前训练配置相同，longitudinal fraction=0.25。
- 场景：no-traffic，S / SC × map seeds 0–7。每场景 noise seeds `[0,1,2]`；每个 repeat
  比较 `g_lon=[-1,-0.5,0,0.5,1]`，`g_lat=0`。simulator reset seed 为该场景 map seed。
- 同一 matched group 保持物理 slot、batch shape、初始 observation/state 和逐周期 diffusion
  noise 一致。各 arm 从 reset 开始，只有 guidance 改变；不从 policy 采样动作。
- 使用 rollout 执行语义：每 0.1 s 重规划，只执行首点。首步为 immediate，固定 20 步为
  2 s short horizon。env horizon=21，窗口结束由 runner 记录，不用 simulator 截断代替。

正式运行一次，完成 **240 episodes / 4,800 transitions**，耗时 **615.42 s**。后续只读
复算与原始数组核验不重新 rollout，不选择 seed。

```powershell
just experiment guidance-control-authority run --output-dir outputs/studies/scalar-reward/e-037-issue94-task-d-guidance-control-authority

just experiment guidance-control-authority analyze --source-dir outputs/studies/scalar-reward/e-037-issue94-task-d-guidance-control-authority --output-dir outputs/studies/scalar-reward/e-037-issue94-task-d-guidance-control-authority-analysis
```

配置：`configs/experiments/guidance-control-authority/intervention.yaml`。实际 composed 参数
保存在 `resolved_config.yaml`；其中训练 job 的 policy/PPO 字段仅因复用配置而存在，不代表
创建或运行了这些对象。

## 预定 Gate 与结果

对每场景，将五个 arm 各自三个 repeats 的均值与 `g_lon` 求 Spearman。endpoint effect
为三个配对 `metric(+1)-metric(-1)` 的平均。noise scale 为五个 arm 内 repeat 样本方差
（ddof=1）的均值的平方根，不能跨地图估计。场景通过条件为：

```text
|Spearman| >= 0.8
AND |endpoint effect| >= 2 * noise scale
AND endpoint effect != 0
```

常量曲线的相关性未定义并失败。执行速度与 Energy intensity 分别判定，任一指标在至少
9/16 个场景通过且 effect 方向一致即可满足数值 Gate；另要求安全/窗口有效、时间响应无
未解释的稳定反转、planner 输出与执行形成可解释链路、Energy 与 proxy 公式一致。

下表的 effect 和 noise 为 **16 个场景统计量的中位数**；通过数来自逐场景判定，不能由
表中的中位数代替。effect 均为 `+1 − (-1)`。

| 窗口 | 指标 | 同方向通过场景 | Spearman 中位数 | Endpoint effect 中位数 | Noise 中位数 |
| --- | --- | ---: | ---: | ---: | ---: |
| immediate 0.1 s | 执行速度 (m/s) | 10/16，负方向 | -0.90 | -1.251129 | 0.510928 |
| immediate 0.1 s | Energy intensity (mL/km) | 10/16，负方向 | -0.90 | -2.146573 | 0.873877 |
| short horizon 2 s | 平均执行速度 (m/s) | **12/16，负方向** | -1.00 | -0.546702 | 0.139360 |
| short horizon 2 s | 累计 fuel / 累计距离 (mL/km) | **8/16，负方向** | -0.80 | -0.716470 | 0.231213 |

short-horizon speed 未通过的场景为 `straight_s2`、`straight_s3`、`straight_s5`、
`gentle_curve_s5`，均完整保留。speed endpoint effect 跨场景范围为
`[-1.580086, +0.355621] m/s`。通过的 12 个场景同时有同方向的 planner 首点速度响应。
immediate 与 short horizon 没有同一指标两窗口均通过但方向相反的场景。

整体跨场景/repeat 的描述性 arm 均值如下；这些均值不用于代替 matched Gate：

| g_lon | 2 s 平均速度 (m/s) | 末端速度 (m/s) | Route progress (m) | Intensity (mL/km) | 平均 Energy score | 累计 fuel (mL) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| -1.0 | 11.339417 | 12.065962 | 22.668000 | 48.932075 | 0.376178 | 1.109924 |
| -0.5 | 11.031651 | 11.553390 | 22.057517 | 48.385026 | 0.380252 | 1.067689 |
| 0.0 | 10.911482 | 11.303902 | 21.820144 | 48.215297 | 0.381841 | 1.052337 |
| +0.5 | 10.913315 | 10.464073 | 21.806378 | 48.298636 | 0.381813 | 1.054460 |
| +1.0 | 10.827846 | 9.995797 | 21.601056 | 48.276632 | 0.382935 | 1.045953 |

## 响应链与结论边界

**guidance target → planner 输出**：target 首点 speed delta 随 `g_lon` 在 16/16 场景
单调增加。完整 8 s planner trajectory 的平均纵向速度也在 16/16 场景正方向通过；
short-horizon 内各次规划的该均值 endpoint effect 中位数为 **+3.123589 m/s**。
但被执行的首点速度随 `g_lon` 在 12/16 场景负方向通过，中位 effect 为
**-0.569183 m/s**，与实际执行速度的 -0.546702 m/s 同方向。

为了区分完整预测与被执行的首点，直接核验每个 matched repeat 的初次规划（相同 observation
与 noise）。`+1` 相对 `-1` 的预测局部前向位移差，跨 48 个场景/repeat 的均值为：

| 预测时间 | 前向位移差 (m) |
| ---: | ---: |
| 0.1 s | -0.088542 |
| 0.2 s | +0.520833 |
| 0.5 s | +1.736979 |
| 1.0 s | +3.229167 |
| 2.0 s | +6.786458 |
| 4.0 s | +13.750000 |
| 8.0 s | +28.208334 |

因此全轨迹平均正响应与首点负响应并不矛盾：planner 将正向响应主要放在未执行的后续点，
10 Hz 闭环却每次只执行首点后重新规划。该证据解释了此次响应方向的时间分布；不进一步
声称已定位 denoiser 内部成因或排除了 BF16 数值表征影响。S 与 SC 的部分初始/短窗口响应
完全相同，不能把这 16 个预定配置视为 16 个统计独立的初始状态。

**planner → execution**：实际 center 对目标点的最大 position error 为
`9.7816e-7 m`。执行结果跟随 planner 首点；这里观察到的反向效应不能归因于执行器没有
执行目标点。Gate D 通过只建立“此通道有可测因果控制能力”，不建立正向 guidance 的
直观加速语义，也不证明所有场景的响应单调。

**execution → Energy**：所有 4,800 步都符合当前 provider 的
`32.5 * exp(0.036 * speed_mps) * distance_m / 1000` mL 公式，容差
rtol=1e-10 / atol=1e-12；Energy distance-valid 全部为真。累计 intensity 的变化还依赖
每步速度与距离的权重，不能仅由窗口平均速度预测其单调性。Energy 的短窗口 Gate
为 8/16，故明确标记 `energy_sensitivity_limited=true`，不宣称 Energy 已有稳健多数分离。

**安全/有效性**：全部 240 episodes 完成 20 步；collision、OOR、提前 termination/truncation、
invalid action/trajectory 均无发生，未填补、跳过或替换 episode。

**研究边界**：结论限于本组 no-traffic 场景、2 s 窗口、当前 BF16 冻结 planner、运动学
首点执行和 MetaDrive fuel proxy。未建立真实车辆动力学可跟踪性、真实能耗改善、长程行为
或训练后效应。Gate C 仍 FAILED；按 Issue 条件现在可以考虑 Task E，但本次未执行 Task E，
更未进入 PPO tuning。

## 验证与产物

- Task D：12 个 CPU 测试与 2 个 simulator 测试通过，覆盖手动 action/端点、RNG、统计、
  安全性、时间反转、离线复算、完整与提前终止窗口。直接相关 guidance/reward 既有测试通过。
- 运行中发现并按用户要求修复 `analysis → rl.__init__ → artifact I/O/rollout` 的提前导入。
  保持现有导出符号，改为按需加载；最小导入回归、完整离线导入检查、导出身份与
  PPO/rollout 定向验证共 14 项通过。离线复算与既有报告的联合验证有 26 项在修复前已通过；
  其中唯一的导入失败项经修复后单独通过。
- 定向 Ruff、Pyright 与 diff 检查通过。相关 CLI suite 仍有一个既有失败：
  `fixed_batch.collection` 从自身导入 `CollectionConfig` 导致循环导入；Task D 不使用该入口，
  本次没有修复它。
- 原始采集与修复后离线复算的 `summary.json` 完全相同。只读核验所有 **1,200 个 cycle
  NPZ**：全部数组有限，初始 observation 与跨 arm noise 匹配，记录动作正确，zero arm
  prediction 与 reference 逐元素一致。响应图与速度时间曲线已检查。
- 正式采集源码快照保留运行当时状态；运行开始后仅修改离线报告 writer 的导入与 RL 根包
  按需加载、测试和文档，未改变该次已加载的采集代码、统计公式、参数或 Gate。独立离线复算
  使用修复后的代码并得到完全相同结果。

正式产物（git ignored）：
`outputs/studies/scalar-reward/e-037-issue94-task-d-guidance-control-authority/`

- `raw/group-*-arm-*/`：初始 observation、每周期全部 planner audit NPZ、对应 episode
  原始执行指标 JSON。
- `episodes.json`、`scenarios.json`、`intervention_config.json`、`resolved_config.yaml`：
  可离线复算的完整原始指标、矩阵与配置。
- `runtime_metadata.json`、`tracked_diff.patch`、`source/`、`run.json`：运行来源与完成状态。
- `summary.json`、`analysis.json`、`report.md`、两张 response 图与 `speed-trajectories.png`：
  逐场景/arm/repeat 统计、Gate、归因与静态图。

独立复算目录：
`outputs/studies/scalar-reward/e-037-issue94-task-d-guidance-control-authority-analysis/`。

Task D 已完成；本次改动未提交、未推送、未写入或关闭 GitHub Issue #94。
