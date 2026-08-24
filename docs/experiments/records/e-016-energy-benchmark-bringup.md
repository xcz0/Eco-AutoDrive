# E-016 固定能耗 benchmark bring-up

**日期 / 类型 / 目的**：2026-08-24 / 本机验证与诊断 / 验证 Issue #1 的执行轨迹能耗口径、分段限速场景和 paired guidance 运行入口。

**代码**：初始 bring-up 基于 `96bcb15`，运行时工作区包含本 Issue 的未提交实现；CUDA 失败诊断与启动器修复基于 `bde4c67` 后续工作区。每个成功 job 的 `tracked_diff.patch` 已随产物保存。

**环境**：Windows；MetaDrive 0.4.3；CUDA 单 GPU 的 `bf16-mixed` 用于 baseline 和后续 guidance 验证；CPU `32-true` 用于初始 guidance smoke。checkpoint 为 `checkpoints/DP-Origin/model.pth`，EMA 276 tensors / 6,042,628 parameters。

**数据与配置**：程序化 PGMap；scenario/map seeds 分别为 structures 的 `0..3` 和 speed-profile 的 `4`；noise seed `0`；50→30→50 km/h profile；执行 10 Hz、重规划 2 Hz。paired sampler 为 deterministic DDIM5。

**命令**：

```powershell
just energy-matrix outputs/energy_matrix/issue-1-20260824-ddim5
.venv/Scripts/python.exe scripts/evaluate.py --config-name experiment/evaluate_energy_speed_profile planner/guidance=energy_longitudinal_negative runtime.accelerator=cpu runtime.precision=32-true evaluation.evaluated_horizon_steps=5 env.horizon=5 hydra.run.dir=outputs/energy_matrix/issue-1-20260824-guidance-cpu-smoke
```

**结果**：DDIM5 无 guidance 的 structures baseline 完成 4 个回合。`cruise`、`curve`、`intersection`、`merge_lane_change` 的 fuel-proxy 分别为 12.933、11.153、7.950、9.748 mL（43.617–49.265 mL/km）；4 个回合均以 arrive 或 horizon 终止，未发生碰撞或 off-road。5-step CPU negative-longitudinal smoke 完成，50→30→50 profile audit 记录 18 条实际应用 lane、30/50 km/h 分布，且能耗为 0.225 mL / 46.296 mL/km。

**CUDA 失败诊断与修复**：失败产物 `outputs/energy_matrix/issue-1-20260824-rerun/structures/longitudinal_negative/resolved_config.yaml` 显示实际 sampler 为 DPM10，而 matrix manifest 要求 deterministic DDIM5。缩减到一个 planning cycle 后，CUDA BF16 + DPM10 + gradient guidance 稳定报错 `AttributeError: '_DpmSampler' object has no attribute 'sample_guided'`；只把 sampler 改为显式 DDIM5 后，同一 CUDA BF16 场景成功。根因不是 CUDA、BF16 或显存，而是 matrix 启动器只校验 manifest，没有把声明的 sampler 传给 evaluation 子进程，也没有核对 resolved sampler。

启动器现显式传入 `planner/sampler=ddim5`，并在收集产物时校验 resolved sampler name 与 `ddim_stochasticity`。回归测试覆盖启动命令和 sampler 漂移拒绝。修复后的既有 `issue-1-20260824-ddim5` 产物证明 structures 与 speed-limit 的 4 组 profile 共 20 个 CUDA BF16 回合全部完成。traffic-follow 的 4 组运行仍因约 240 m 路线不满足 traffic profile 的 `[2000, 5000] m` 契约而失败；这是独立的场景配置问题，不是 CUDA guidance 失败。

此前一次尝试因无效的 MetaDrive `vehicle_config` 字段在 env 初始化前失败，产物保留于 `outputs/energy_matrix/issue-1-20260824-rerun/`。

**产物**：

- `outputs/energy_matrix/issue-1-20260824-ddim5/structures/baseline/`
- `outputs/energy_matrix/issue-1-20260824-guidance-cpu-smoke/`
- `outputs/energy_matrix/issue-1-20260824-rerun/`（失败 bring-up）

**结论边界**：CUDA BF16 的 DDIM5 guidance 路径及 sampler 启动契约已验证；能耗计算和限速 profile audit 在 structures 与 speed-limit 条件下有效。traffic-follow 场景尚未满足 traffic evaluation 路线长度契约，因此这仍不是完整 6 场景 guidance matrix，也不能据此关闭 Issue #1 的“可观测 guidance 差异”验收项。
