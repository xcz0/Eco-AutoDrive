# E-009 阶段 2 reference guidance 快速验收

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-11，阶段 2 快速验收。验证 reference-centered
orthogonal guidance 的严格配置、几何和量纲、neutral 退化、随机流重放、冻结边界、真实
checkpoint 以及短时 MetaDrive 闭环。按用户决定，预注册的 60 回合正式配对矩阵本次延期，
GitHub Issue #6 保持开启。

**编号说明**：阶段 2 候选计划曾预留 `E-008`，但当前权威实验索引中的 `E-008` 已登记为
Joblib 并行可行性诊断。为保留既有证据，本记录使用下一个可用编号 `E-009`。

**代码与环境**：运行 HEAD `0ef2c2eb25b7ea110a678083210abea4840e73f2`；阶段 2 实现为
未提交 tracked/untracked diff，运行时状态和 tracked diff 分别保存在
`runtime_metadata.json` 与 `tracked_diff.patch`。工作区另有用户预先存在的
`pyproject.toml` 修改，本任务未覆盖。Windows 10、Python `3.10.20`、PyTorch
`2.6.0+cpu`、Lightning `2.6.5`、MetaDrive/assets `0.4.3`；CPU `32-true`。checkpoint、
EMA tensor 数和参数量使用共同资产表中的固定值。

**配置**：无交通 `S`、地图 seed 0、diffusion noise seed 0、2 s / 20 个 `0.1 s` 子步；
标准高斯 DDIM-5，`orthogonal_reference` guidance，action `(lateral=+1,
longitudinal=0)`。最大横向目标为 `2.5 m`，最大纵向速度变化比例为 `25%`，逐规划周期刷新
reference，reference/guided pass 共享 scene encoding、initial noise 和 transition draws，
梯度步系数为 1。视频关闭；本次 smoke 只验证接口和短闭环，不替代正式视频矩阵。

**运行命令**：

```powershell
uv run python scripts/evaluate.py sampler=ddim5 guidance=orthogonal_reference `
  guidance.lateral_scale=1.0 guidance.longitudinal_scale=0.0 `
  runtime.accelerator=cpu runtime.precision=32-true runtime.seed=0 `
  video.enabled=false evaluation.evaluated_horizon_steps=20 env.horizon=20 `
  'scenarios=[{name:straight,map:S,seed:0}]' `
  'hydra.run.dir=outputs/no_traffic/2026-08-11/stage2-smoke/lateral-positive'
```

**结果**：真实 checkpoint 完成 4 个规划周期和 20 个 simulator 子步，无碰撞、出界或运行
错误，因 2 s 上限正常 `truncated`。行驶 `21.3317 m`，最大轨迹位置误差
`4.598883840232763e-4 m`，最大 heading 误差 `8.453547613029855e-7 rad`，低于
`1e-3 m` / `1e-4 rad` 门槛。trace 的全部数值数组有限；首个规划周期相对 reference 的
平均左法向位移为 `+0.0873407 m`，平均沿轨速度差为 `+0.1371361 m/s`。后者不是纵向目标
效果，因为 longitudinal scale 为零，只作为完整输出的观测值，不据此得出加速结论。五个
DDIM 步的 applied-gradient L2 均有限，raw neighbor gradient L2 最大值为 0。

数学与契约测试覆盖手算横纵向目标、10 Hz 后向差分、左右符号、rigid transform、重复点与
零速审计、退化 heading、非有限/非法 action、sampler 不兼容、neutral 逐值退化、单次 scene
encoding、共享 stochastic transition draws、主 generator 单次随机流消耗、当前点/邻车梯度
屏蔽、冻结参数无梯度，以及 guided trace/summary/config schema。

**验证命令**：

- `uv run pytest -m "not gpu and not simulator and not slow"`：136 passed；
- 三个阶段 0/1/2 真实 checkpoint smoke：3 passed；
- `uv run pytest -m simulator`：11 passed；
- `uv run ruff check .`：通过；
- `uv run ruff format --check .`：121 files already formatted；
- guidance PowerShell 运行脚本 dry-run：生成预期的显式 sampler/guidance overrides。

安装的是 CPU-only PyTorch，因此未运行 GPU marker。

**产物**：`outputs/no_traffic/2026-08-11/stage2-smoke/lateral-positive/`，包含 resolved
config、Hydra overrides、runtime metadata、tracked diff、作业/回合 summary 和 `trace.npz`。
此前启动后中止的 `outputs/no_traffic/2026-08-11/stage2-paired/unguided/` 仅保留为诊断残留，
不纳入 E-009，也不覆盖 E-007。

**结论边界**：支持固定 reference guidance 的实现边界和快速退出条件，也支持其在单个 CPU
FP32、无交通直道、2 s MetaDrive 闭环中没有接口或执行失败。不能支持完整 60 回合的
unguided/neutral/正负横纵向配对结论，不能宣布阶段 2 正式矩阵验收完成，也不能推出到达率、
交通、CUDA/mixed precision、policy、reward、rollout、PPO、能耗收益或 PlannerRFT 官方实现
parity。剩余正式矩阵由 GitHub Issue #6 跟踪，Issue 不在本次执行后关闭。
