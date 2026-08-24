# E-016 固定能耗 benchmark bring-up

**日期 / 类型 / 目的**：2026-08-24 / 本机验证与诊断 / 验证 Issue #1 的执行轨迹能耗口径、分段限速场景和 paired guidance 运行入口。

**代码**：`96bcb15`，运行时工作区包含本 Issue 的未提交实现；每个成功 job 的 `tracked_diff.patch` 已随产物保存。

**环境**：Windows；MetaDrive 0.4.3；CUDA 单 GPU 的 `bf16-mixed` 用于无 guidance baseline；CPU `32-true` 用于 guidance smoke。checkpoint 为 `checkpoints/DP-Origin/model.pth`，EMA 276 tensors / 6,042,628 parameters。

**数据与配置**：程序化 PGMap；scenario/map seeds 分别为 structures 的 `0..3` 和 speed-profile 的 `4`；noise seed `0`；50→30→50 km/h profile；执行 10 Hz、重规划 2 Hz。paired sampler 为 deterministic DDIM5。

**命令**：

```powershell
just energy-matrix outputs/energy_matrix/issue-1-20260824-ddim5
.venv/Scripts/python.exe scripts/evaluate.py --config-name experiment/evaluate_energy_speed_profile planner/guidance=energy_longitudinal_negative runtime.accelerator=cpu runtime.precision=32-true evaluation.evaluated_horizon_steps=5 env.horizon=5 hydra.run.dir=outputs/energy_matrix/issue-1-20260824-guidance-cpu-smoke
```

**结果**：DDIM5 无 guidance 的 structures baseline 完成 4 个回合。`cruise`、`curve`、`intersection`、`merge_lane_change` 的 fuel-proxy 分别为 12.933、11.153、7.950、9.748 mL（43.617–49.265 mL/km）；前三个以 arrive 或 horizon 终止，未发生碰撞或 off-road。5-step CPU negative-longitudinal smoke 完成，50→30→50 profile audit 记录 18 条实际应用 lane、30/50 km/h 分布，且能耗为 0.225 mL / 46.296 mL/km。

**失败状态**：完整 CUDA BF16 matrix 在 baseline 后进入第一组 gradient guidance 时进程异常退出，未产生 Python traceback 或 failed episode summary；CPU smoke 成功，因此该问题目前只确认在本机 GPU full guidance 路径出现。此前一次尝试因无效的 MetaDrive `vehicle_config` 字段在 env 初始化前失败，产物保留于 `outputs/energy_matrix/issue-1-20260824-rerun/`。

**产物**：

- `outputs/energy_matrix/issue-1-20260824-ddim5/structures/baseline/`
- `outputs/energy_matrix/issue-1-20260824-guidance-cpu-smoke/`
- `outputs/energy_matrix/issue-1-20260824-rerun/`（失败 bring-up）

**结论边界**：能耗计算、限速 profile audit 和 CPU guidance 闭环接口有效；这不是完整 guidance matrix，也不能据此比较四组 guidance 的能耗或关闭 Issue #1 的“可观测 guidance 差异”验收项。
