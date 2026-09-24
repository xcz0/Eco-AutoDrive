---
kind: protocol
status: proposed
scope: [baseline, sampling, guidance, evaluation]
read_when:
  - changing baseline sampling guidance or evaluation cadence
  - designing a matched evaluation or interpreting its metrics
---

# Planning and evaluation protocol

本篇是 [Issue #102](https://github.com/xcz0/Eco-AutoDrive/issues/102) 的迁移草案，旧入口仍有效，
尚不成为 active owner。MUST／MUST NOT 表示拟保留的强制要求，SHOULD 表示建议，MAY 表示允许；
接受依据与待确认边界列在末尾。概念见 [Semantics](../semantics.md)，执行保证见
[execution](../../contracts/execution.md)，输入与坐标见 [data/model](../../contracts/data-and-model.md)。

## Baseline 与采样

MUST 保持官方预训练 EMA 的模型结构、参数层级、normalization、checkpoint 与 baseline inference
语义；新增研究变量必须显式隔离。Planner MUST 冻结并处于 eval mode，默认不施加 guidance。

| 采样条件 | 初始未来噪声 | 研究定义 |
| --- | --- | --- |
| 官方 `dpm10` baseline | `0.5 × N(0,I)` | 连续线性 VP-SDE、预测 `x_start`；10 步二阶 multistep DPM-Solver++、均匀 lambda spacing，至 `t=0.001` 后额外预测一次 `x_start`。公式及初始尺度不得以任意配置覆盖。 |
| 项目 `ddim5` | `N(0,I)` | 模型时间 `[1.0,0.8,0.6,0.4,0.2]`，转移目标 `[0.8,0.6,0.4,0.2,0.0]`；评测 stochasticity 为 0。 |
| 隔离的 DDIM project-noise 变体 | `0.5 × N(0,I)` | MUST 有独立 parity 标签；不得解释为 PlannerRFT parity，不启用 active guidance。 |

DDIM 的连续时间子序列是本项目复现选择，不是论文公开事实。MUST 显式选择 sampler 并记录
步数、初始尺度、stochasticity、时间序列和 parity。标准高斯 DDIM5 是 guidance 和 policy checkpoint
评测条件；正式 policy evaluation MUST 使用 stochasticity=0。
非零 stochasticity 是显式研究条件，随机流保证见 execution contract。

## Reference-centered guidance

每周期 MUST 刷新 reference；reference 与 guided pass MUST 使用同一冻结 EMA、当前观测、
scene/route encoding、initial noise 和 DDIM transition draws。
切向由有限、非退化的 reference heading 归一化，左法向为 `[-sin(h),cos(h)]`。
速度用当前物理点及未来点按预测采样间隔后向差分；重复位置是零速，不触发 fallback。

固定 action 顺序为 `(lateral,longitudinal)`，值域与软件编码见
[固定干预动作](../../contracts/data-and-model.md#固定干预动作)。正 lateral 表示左移，正 longitudinal
表示沿 reference heading 加速。横向目标为 `2.5 m × lateral`，纵向目标速度变化为
`0.25 × longitudinal × reference_along_track_speed`。

MUST 使用 reference-centered energy-gradient delta，使 `(0,0)` 精确退化为同次 unguided reference。
每个 denoise step 对 physical ego objective 经 normalization 与冻结 denoiser 求 normalized noisy
joint sample 梯度；DDIM transition 后以单位系数施加负梯度，恢复当前点约束并 detach。
MUST 只更新 ego future channels，当前点和邻车梯度不应用；未应用的邻车梯度仍供审计。
该 centered correction、系数、差分速度与 ego-only scope 都是本项目选择，不宣称作者实现 parity。

## 正式闭环 cadence 与执行方法

正式 training/evaluation MUST 采用同一 cadence：simulator 子步 `0.1 s`，每个 planning cycle
执行前 `5` 个子步，共 `0.5 s`，规划频率 `2 Hz`。一个 PPO transition 是一次 decision 加完整执行前缀。
Terminal/truncation 可使该前缀提前结束为 1–5 个子步；这不是另一 baseline。
预测 ABI 与物理推进约束分别归 data/model、execution contract。

MUST 原样执行并保存 planner 轨迹，再从最后实际状态重新规划。MUST NOT 平滑、clip、限幅、旋转
修复、投影到中心线、选择最佳噪声 seed、切换 fallback controller，或在异常时返回零轨迹。
此要求针对轨迹修复，不禁止坐标系的必要刚体转换。

[诊断协议](diagnostic-studies.md) MAY 显式干预 execution prefix；必须标注干预及 matched conditions，
不得将其提升为正式 baseline。旧 k=1 training / k=5 evaluation evidence 保持原条件；统一 cadence
不代表旧 E-039/E-040/E-047 已在新训练条件下重验。
[Issue #83](https://github.com/xcz0/Eco-AutoDrive/issues/83) 的 transfer validation 不由本次迁移宣称完成。

## 场景、窗口与结束

主要能耗比较 SHOULD 使用可重复表达能耗因素的短／中程低交通场景；长程高密度交通可用于鲁棒性
诊断。每个 episode MUST 按场景特征、交通条件、运行阶段与终止类型分层解释，保留所有失败。
交通 history warmup 不进入正式指标窗口，但初始化证据必须单独保存。

Arrival、collision、out-of-road、time-limit 和运行错误 MUST 分别保留。
仍位于道路 lane、但已离开 navigation route edge 集合属于 out-of-road terminal。
Wrong-direction 是记录指标，不自行改变 termination；训练 safety gate 对它的使用见 training protocol。
Completed 不代表到达，失败或截断也不得被合并解释为成功。

## 指标及累计窗口

下列公式定义研究量；单步事实唯一推导及 summary 消费边界见 execution/artifacts contract。

| 指标 | 定义 |
| --- | --- |
| `speed_mps` / `step_distance_m` | 实际 velocity 的欧氏范数／相邻实际 vehicle-center 的几何距离。 |
| `stopped` | `speed_mps < 0.1 m/s`。 |
| `route_heading_error_rad` / `wrong_direction` | 实际 heading 相对 route forward tangent 的 wrapped absolute error（非负）／该误差 `> π/2`。 |
| `collision` | `any(crash_*)`；reward profile 选择 crash 类型的 gate 不重定义该客观事实。 |
| 执行误差 | 实际位置／heading 相对目标的误差；heading 差取最短有向角后形成误差量。 |
| Episode distance | 从 initial state 逐点累计到所有 actual executed states 的几何路径长度。 |
| Mean speed / stopped fraction | 正式执行子步 speed 的均值／stopped 子步占比。 |
| Route completion、arrival、collision、out-of-road | 最终 execution record 的对应事实。 |
| Wrong direction / fraction | 任一执行子步 wrong-direction 为真／该 bool 的子步比例。 |
| Proxy energy | 只聚合 execution-recomputed fuel-proxy 的逐子步 mL 及配套实际 distance；`ml_per_km = 1000 × sum(mL) / sum(m)`。 |

Native MetaDrive `step_energy/episode_energy`（mL）只作 phase-boundary audit，不进入 reward 或
evaluation energy summary。Partial failed episode 有执行轨迹时保留已产生能耗；empty trace 无能耗值。
总距离零时 per-km 为 undefined/null，固定 matrix 不对该未定义指标 bootstrap，必须显式失败。
Reward 子步的无效距离评分规则是另一估计量，见 [training protocol](training.md#reward-定义)。

FASTSim `fastsim_fuel_energy` MUST 在完整实际执行轨迹结束后离线求值，保持动力系统连续状态；
车辆选择、grade、环境温度和初始海拔显式给出。既有 adapter 的车辆条件为 conventional
`2012_Ford_Fusion.yaml`；输出 J/Wh，不推导 fuel mL，不进入在线 reward 或默认 evaluation artifact。
不得把它与 proxy 相加／替换；没有原生坡度不得推断高程。

## Matched comparison 与随机条件

训练与 held-out 场景池 MUST 不相交。Arm、reward profile、方向性 contrasts、训练 seed 与
checkpoint label MUST 显式声明；有 frozen arm 时提供 baseline，无 frozen arm 时 baseline 为 null。
同 training seed 的各臂除声明的 reward 差异与 tracking 元数据外，训练条件必须相同，包含
initial policy、planner、probe 和随机流。完整场景池与 scenario 数必须匹配。

评测 MUST 固定 workload、场景／地图／map seed、traffic、horizon、sampler、runtime diffusion seed
与 evaluation mode；eval seed 与 training seed 分别声明。不同 checkpoint/arm/candidate 使用相同
配对条件。Mean action 不引入 action 抽样；sample action 比较另显式配对 action seeds。
不得按目录名推断身份、挑选最佳 seed 或丢弃失败。

默认 comparison 配置的 S/SC train seeds 0–7、held-out seeds 16–23、no-traffic、300 子步、DDIM5、
runtime seed 760025 是一项具体设计，不是所有实验的普适常量。
数值组合由 [comparison configs](../../../configs/experiments/comparison/) 拥有；新设计必须显式声明。

报告顺序 MUST 为 completion/availability → safety → paired energy：

- Completed rate 的分母是全部回合；arrival、route completion 与 safety rate 的分母是本 arm
  completed 回合。未知结果不计作安全，缺项及失败原因单列。
- 每个 contrast 使用双方 completed 的 matched 交集，差值为 `comparison − reference`。
  每 training seed effect 是场景配对的总 proxy mL 差的均值；负值结合安全、距离与提前终止解释。
- 每 training seed 单独报告点估计、scenario bootstrap CI 和缺项。使用显式 confidence level、
  resample 数、bootstrap seed、percentile 方法和独立 RNG；不重采样 training seeds。
- 无 pair 时 estimate/CI 不可用；单 pair 保留 estimate，CI 不可用；常量双样本允许零宽 CI。
  跨 seed 方向计数只用 final，initial checkpoint 是诊断。CI 只描述固定策略及可用场景的不确定性，
  不是 training-seed 总体区间。不得增加失败能耗 penalty 或混合分数来代替这些结果。

## 规范变更与依据

改变 baseline、采样分布／schedule、guidance objective、正式 cadence、场景分布、配对随机变量、
metric estimator、窗口或终止含义，均是 protocol change，不能作为普通重构静默发生。
实际行为查 code/tests/config/observation；正式切换后与 active 规范不一致应调查为 conformance mismatch，
不能仅凭“代码如此”认定规范过期。本草案阶段不宣称完成 conformance 验证。

接受依据：[ADR 0001](../../adr/0001-preserve-official-baseline.md)、
[0007](../../adr/0007-use-stable-energy-scenarios.md)、
[0008](../../adr/0008-stratify-every-evaluation-episode.md)、
[0011](../../adr/0011-add-explicit-five-step-ddim-sampler.md)、
[0013](../../adr/0013-add-reference-centered-orthogonal-guidance.md)、
[0032](../../adr/0032-separate-online-proxy-and-offline-fastsim-energy.md)、
[0037](../../adr/0037-simplify-experiments-and-report-seed-effects.md)、
[0039](../../adr/0039-unify-closed-loop-cadence.md)。迁移细节来自
[旧 planner](../../agents/contracts/planner.md)、[system contract](../../agents/system-contract.md)、
[旧 experiments](../../agents/contracts/experiments.md)。ADR 保留 rationale，本阶段不改历史正文。

## 代码与测试导航

以下是定位点，不是已运行验证或完整覆盖声明：

| 修改面 | 实现／配置 | 相关测试 |
| --- | --- | --- |
| Sampler / guidance | [diffusion](../../../src/eco_planner/planning/diffusion/) | [sampling](../../../tests/planning/test_sampling.py)、[guidance](../../../tests/planning/test_guidance.py) |
| Cadence | [shared ABI](../../../src/eco_planner/contracts.py)、[jobs](../../../configs/jobs/) | [execution consistency](../../../tests/simulation/test_execution_consistency.py)、[job config](../../../tests/configuration/test_jobs.py) |
| Episode 指标与配对统计 | [summary](../../../src/eco_planner/evaluation/artifacts/summary.py)、[analysis](../../../src/eco_planner/analysis/evaluation.py)、[comparison](../../../src/eco_planner/experiments/comparison/) | [artifacts](../../../tests/evaluation/test_artifacts.py)、[scalar effects](../../../tests/analysis/test_scalar_effects.py) |
