# E-044 Issue #98 Task B：Learned guidance lon/lat 分量归因干预

[返回实验索引](../README.md) · [前置 E-043 Task A](e-043-issue98-task-a-execution-horizon.md) · [臂常量来源 E-041 Task H](e-041-issue94-task-h-evaluation-diagnostics.md)

**日期 / 类型 / 目的**：2026-09-17 / 正式配对闭环人工干预 / 不训练 policy，不依赖 reward
组合、critic 或 PPO。在 E-040 held-out 协议上冻结同一 planner，按 E-041 冻结的
R0 / Rstress final-policy Beta mean 构造四个 matched **常量** guidance 臂
（`r0` / `lon` / `lat` / `joint`），只改变 guidance 的横向或纵向分量，判定 E-040
"Rstress 更快且单位里程更耗能" 的分离主要由哪个分量造成。完成 Issue #98 Task B；
未执行 Task C–E，未修改 reward、guidance 主配置、PPO 或 baseline execution contract。

**裁定：energy 维度双 seed 均为 `longitudinal-dominated`；speed 维度 seed 0 为
`nonlinear-interaction`（lon/joint = 0.72，略低于预声明 0.75 dominance 阈值）、seed 1
为 `longitudinal-dominated`，overall `mixed-across-seeds`。** 与 E-041 记录/汇报中
原始标签相反，**较大迁移来自纵向分量**；`joint` 常量臂的 speed/energy 方向与 E-040
一致（均 positive，多数场景通过），说明常量均值注入可复现 E-040 的分离方向。SC 地图
49/64 episodes off-route，全部保留，未删除、未填补。

## 运行来源与固定协议

- 代码基线 `12afa2c`（main）加 Task B 未提交改动；`runtime_metadata.json` 保存
  `git_head`、`git_branch`、`git_status_short`。工作区 dirty（7 个修改、6 个新增路径），
  因此这是诊断干预而非 clean-commit 正式基线。
- Windows、Python 3.10.20、PyTorch 2.12.1+cu126、MetaDrive 0.4.3；GPU 为
  **NVIDIA RTX A4000**（driver 556.18，`cuda:0`），resource profile `rtx_a4000` 与实际
  一致：4 个 rollout workers、每 worker 12 threads。
- 官方 EMA checkpoint：`checkpoints/DP-Origin/model.pth`，276 个 EMA tensors、
  6,042,628 parameters；冻结、eval 模式；不构造 Exploration Policy、critic 或 optimizer。
- 标准高斯 DDIM-5、stochasticity=0、BF16 mixed precision、runtime seed=760025；
  orthogonal-policy guidance 系数与训练配置相同（lateral_max_offset_m=2.5、
  longitudinal_max_speed_fraction=0.25）。
- 场景：no-traffic，S / SC × map seeds 16–23（held-out 16 场景），`num_scenarios=24`；
  env.horizon=300，`cycles=300`、`execution_steps=1`（ROLLOUT 每周期执行首 waypoint）。
  每个 (worker batch, training seed) 组成一个 matched group，4 个常量臂顺序为
  `r0/lon/lat/joint`；同一 group 保持物理 slot、batch shape、初始 observation/state 与
  逐周期 diffusion noise 一致，只有 guidance 常量改变。noise seed 固定 `760025`
  （与 E-040 runtime seed 相同），故 diffusion 噪声流与 E-040 matched。

### 臂常量（原生顺序 `[lateral, longitudinal]`，index 0=横向、1=纵向）

| seed | g_R0 (r0) | Δg | joint (Rstress mean) |
| --- | --- | --- | --- |
| 0 | (-0.04068838, -0.02526603) | (+0.02525402, +0.08377668) | (-0.01543436, +0.05851064) |
| 1 | (+0.03780416, -0.05099847) | (+0.02390295, +0.11464473) | (+0.06170710, +0.06364626) |

- seed0 `lon` arm = (-0.04068838, +0.05851064)；`lat` arm = (-0.01543436, -0.02526603)
- seed1 `lon` arm = (+0.03780416, +0.06364626)；`lat` arm = (+0.06170710, -0.05099847)
- 来源：E-041 `beta_distribution.{0,1}.{r0,rstress}.final.beta_mean.mean` 与 paired
  delta；`joint` = R0 + Δ = Rstress mean。config 显式给出四个 arm，并由 validator 断言
  `lon`/`lat` 只在对应维不同且 `joint == r0 + (lon-r0) + (lat-r0)`。

```powershell
just exp guidance decomposition run --output-dir outputs/studies/scalar-reward/e-044-issue98-task-b-guidance-decomposition

just exp guidance decomposition analyze --source-dir outputs/studies/scalar-reward/e-044-issue98-task-b-guidance-decomposition --output-dir outputs/studies/scalar-reward/e-044-issue98-task-b-guidance-decomposition-analysis
```

配置：`configs/experiments/guidance/decomposition.yaml`；held-out job：
`configs/jobs/evaluation/no_traffic_heldout_manual.yaml`（E-040 matched 协议 + 手动注入
orthogonal_policy）。实际 composed 参数保存在 `resolved_config.yaml`。

正式运行一次，完成 **128 episodes / 17,872 transitions**，耗时 **1,996.29 s**。运行开始后
未改变该次采集已加载的代码、统计公式、参数或阈值；记录与文档在采集完成后撰写。独立离线
复算使用重算后的 `episodes.json`，得到与运行目录逐值相同的 `summary.json`。

## 预定判据与结果

逐场景配对：对每个 (training seed, scenario) 计算 arm 相对 `r0` 的差，
`interaction = joint - lon - lat`；同一方向场景数 ≥ 9/16 即该方向成立。attribution 阈值
`dominance_share=0.75`、`expected_joint_direction=positive`：joint 方向不为 positive 记为
`not-reproduced-state-dependent`；否则 |lon| ≥ 0.75·|joint| 记 `longitudinal-dominated`，
|lat| ≥ 0.75·|joint|（且 |lat|>|lon|）记 `lateral-dominated`，其余记
`nonlinear-interaction`。

下表 effect 为 **16 个场景配对差的中位数**；括号为同号场景数 `(正/负)`。

| seed | metric | r0 | lon | lat | joint | effect lon | effect lat | effect joint | interaction | verdict |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | speed (m/s) | 10.204 | 10.247 | 10.356 | 10.452 | +0.0598 (12/4) | -0.0001 (8/8) | +0.0833 (13/3) | +0.0002 | `nonlinear-interaction`* |
| 0 | energy (mL/km) | 47.637 | 47.780 | 47.848 | 48.023 | +0.1239 (14/2) | +0.0033 (9/7) | +0.1482 (14/2) | +0.0089 | `longitudinal-dominated` |
| 1 | speed (m/s) | 10.452 | 10.500 | 10.477 | 10.515 | +0.0673 (8/8) | +0.0132 (10/6) | +0.0820 (10/6) | +0.0018 | `longitudinal-dominated` |
| 1 | energy (mL/km) | 47.920 | 48.061 | 47.962 | 48.053 | +0.1946 (14/2) | +0.0120 (10/6) | +0.2159 (12/4) | -0.0011 | `longitudinal-dominated` |

\* seed0 speed 的 lon/joint = 0.0598/0.0833 = **0.72**，仅因低于 0.75 预声明阈值被标为
`nonlinear-interaction`；其 interaction 中位数仅 +0.0002 m/s，且 |lon| ≫ |lat|。因此
"纵向主导" 是跨 seed、跨 metric 的一致结论，seed0 speed 只是阈值边界样本。

补充描述（中位数）：

| seed | 2 s-prefix speed effect lon / lat / joint | 首个执行 waypoint speed effect lon / lat / joint | route completion effect joint |
| ---: | --- | --- | ---: |
| 0 | +0.188 (16/0) / +0.016 (10/6) / +0.219 (16/0) | +0.00125 (16/0) / +0.00016 (12/4) / +0.00119 (16/0) | -0.00031 |
| 1 | +0.312 (16/0) / +0.00002 (10/6) / +0.344 (16/0) | +0.00148 (12/4) / +0.00007 (10/6) / +0.00136 (14/2) | +0.00198 |

- 2 s-prefix 与首个执行 waypoint 上，`lon` 在 16/16 场景正向通过，`lat` 仅弱正向；与
  E-043 的 short-horizon 正响应一致。
- 首个规划 full-trajectory 前向位移 effect 中位（m）：`lon` 与 `joint` 随预测时间增长
  （0.5/2/8 s：seed0 lon ≈ +0.062/+0.313/+1.999，seed1 ≈ +0.094/+0.375/+2.500），
  `lat` 近 0；进一步支持纵向主导。

## 安全性与响应链

- 128 episodes **全部提前终止**，无 `window_complete`：`arrive_dest` 74、`max_step` 5、
  `out_of_road` 49、`collision` 0。49 个 off-route **全部位于 SC 地图**；S 地图
  64/64 无 off-route。
- off-route 在四个臂上均出现（含 `r0` 臂），说明在 SC 地图上长程注入任意常量 guidance
  （即使接近 R0 均值）也可能偏离路线；它不是某个单臂的伪影。因 SC 大量提前终止，SC 的
  route completion / distance / progress 在全 300 步意义上被截断，但配对差分仍在同
  seed/scenario/arm 内成立，结论限定于该干预下实际可达的行为。
- Energy proxy：全部 17,872 transitions 符合
  `32.5 * exp(0.036 * speed_mps) * distance_m / 1000` mL，容差 rtol=1e-10 / atol=1e-12；
  无 proxy 错误。
- `joint` 臂方向与 E-040 一致（speed/energy 均 positive），因此预声明
  `not-reproduced-state-dependent` 未触发；常量均值注入足以复现方向（但幅度不同，
  且未检验与 Rstress policy 逐值一致）。

## 结论边界

- **归因**：Task B 判定 E-040 的分离主要由 **longitudinal 分量**承载；横向分量效应很小，
  不支持 "learned lon/lat coupling 是主因"（Issue #98 归因 B）。结合 Task A（时间响应 ×
  执行错位，支持归因 A），方向误差的解释重心不在横纵耦合。
- seed0 speed 处于 0.75 dominance 阈值边界（0.72），故 speed 维度的 overall 记为
  `mixed-across-seeds`；energy 维度双 seed 均 clean `longitudinal-dominated`，是更稳的主证据。
- `joint` 注入的是 Rstress **常量 Beta mean**，不是 Rstress policy；"复现" 仅指方向与
  多数场景通过，不代表逐值一致。常量注入无法复现的部分（若后续发现）本身属于
  状态依赖性证据，本实验未据此下结论。
- SC 地图 49/64 off-route 是本干预的直接结果，单独保留；它限制了 SC 上全里程指标的
  解释范围，也是后续需要单独解释的现象。
- 结论限于本组 no-traffic held-out 场景、300 步、当前 BF16 冻结 planner 与 MetaDrive
  fuel proxy；不建立真实车辆动力学、真实能耗改善或推荐 guidance 方案。
- 未修改 reward、PPO 主配置或 baseline execution contract；Task C–E 未执行，本记录不关闭
  Issue #98。

## 数据标注更正（实践记录）

- **E-041 记录与 `docs/能耗优化_0914汇报.md` 存在 lon/lat 标签互换**。权威顺序为
  `[lateral, longitudinal]`（代码 `src/eco_planner/planning/diffusion/guidance.py:135,153`、
  `src/eco_planner/rl/optimization/gradients.py:43-44`）。E-041
  `paired_rstress_minus_r0.beta_mean.mean_delta_per_dimension` 为原生
  `[lateral, longitudinal]`，即 seed0 `(+0.0253, +0.0838)`、seed1 `(+0.0239, +0.1146)`：
  **较大迁移是纵向**。已在 E-041 记录加 marked errata，并更正汇报文档的表头标签；
  数字未改。

## 验证与产物

- 新增 `tests/training/test_guidance_decomposition.py`（config/arm 不变量、原生维度顺序、
  attribution 四种裁定、effect/interaction 数学、变长与安全/proxy 统计、`_annotate` 映射、
  offline recompute 与 live 一致、CLI 路由）：**13 passed**。
- 更新 `tests/training/test_guidance_control_authority.py` 的 `InterventionExecution`
  调用（改为 per-arm 2D actions）；authority + horizon 定向测试 **20 passed**；
  `just test-sim` **9 passed**；`just test-all-cpu` **303 passed**；`just lint` /
  `just typecheck` 0 error。独立离线 analyze 的 `summary.json` 与运行目录逐值相同，
  源文件不变。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-044-issue98-task-b-guidance-decomposition/`
  - `raw/group-*-arm-*/`：初始 observation、每周期 planner audit NPZ、逐 arm episode 证据；
  - `episodes.json`、`scenarios.json`、`intervention_config.json`、`resolved_config.yaml`；
  - `runtime_metadata.json`、`run.json`、`decisions.json`、`summary.json`、
    `analysis.json`、`report.md`、三张图。

独立复算目录：
`outputs/studies/scalar-reward/e-044-issue98-task-b-guidance-decomposition-analysis/`。

Task B 已完成；本次改动未提交、未推送。
