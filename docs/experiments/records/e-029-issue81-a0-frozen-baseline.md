# E-029 Issue #81 matched 协议下的 A0 frozen baseline

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-09-07 / 正式基线 / 完成 Issue #81 Task C：在当前 commit 与
最终 held-out evaluation harness 下重新生成 A0 frozen Diffusion Planner baseline，作为后续
A1/A2 matched 对照的基准 artifact，避免与 E-017 跨版本直接比较。

**代码**：`9075a16355490255893898e53df05c81d6b5d1d7`（Issue #81 Task B 提交），运行时
工作区干净（`runtime_metadata.json` 的 `git_status_short` 为空，`tracked_diff.patch` 为
空文件）。上游源码与模型 checkpoint 见[共同资产](../README.md#共同资产)。

**环境 / 模型**：Windows 10（10.0.26200）；Python 3.10.20；PyTorch 2.12.1+cu126；
Lightning 2.6.5；MetaDrive 0.4.3；resource profile `rtx_a4000`，单卡 `cuda:0`、
`bf16-mixed`，serial topology（1 worker）。checkpoint 为 `checkpoints/DP-Origin/model.pth`，
EMA 276 tensors / 6,042,628 parameters，与 E-016/E-017/E-028 相同。

**配置**：protocol manifest `configs/experiments/scalar_reward/protocol.yaml` 组合
`jobs/evaluation/no_traffic_heldout`（`matched_no_traffic_heldout`）。held-out 场景为
S/SC map seeds 16–23（16 episodes，与训练池 seeds 0–7 不相交）；no-traffic；horizon 300
（0 warmup + 300 evaluated steps）；deterministic DDIM5（`ddim_stochasticity=0`，parity
`plannerrft_paper_text`）；guidance `none`；runtime seed 760025；`env.num_scenarios=24`；
programmatic lane speed limit 50 km/h；video 关闭。Seed provenance 无独立 manifest 文件：
runtime seed 与 scenario seeds 记录于 `resolved_config.yaml`，planner noise seed（760025，
全 job 单一 generator 流）记录于每个 episode summary 的 `noise_seed`。

**命令**：

```powershell
just scalar-reward evaluate-a0 --output-dir outputs/studies/scalar-reward/e-029-a0-baseline
```

运行 60.574 s，`status: completed`，16/16 episodes 完成，无 launcher 或 runtime failure。
runner 在仿真前通过全部协议校验（runtime seed、horizon、sampler、scenario 集合、
`env.num_scenarios` 覆盖）。

**逐回合结果**（energy 为 MetaDrive fuel proxy，mL；单位距离指标为 mL/km）：

| 场景 | 终止 | Distance (m) | Energy (mL) | mL/km | Progress | Mean speed (m/s) | Steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s16 | arrive_dest | 96.0 | 4.69 | 48.89 | 0.9583 | 11.291 | 85 |
| s17 | arrive_dest | 114.4 | 5.62 | 49.15 | 0.9671 | 11.438 | 100 |
| s18 | arrive_dest | 110.7 | 5.44 | 49.12 | 0.9658 | 11.409 | 97 |
| s19 | arrive_dest | 114.5 | 5.63 | 49.17 | 0.9614 | 11.448 | 100 |
| s20 | arrive_dest | 104.5 | 5.12 | 49.03 | 0.9641 | 11.360 | 92 |
| s21 | arrive_dest | 88.8 | 4.33 | 48.80 | 0.9507 | 11.239 | 79 |
| s22 | arrive_dest | 103.5 | 5.08 | 49.05 | 0.9652 | 11.371 | 91 |
| s23 | arrive_dest | 101.7 | 4.97 | 48.90 | 0.9598 | 11.303 | 90 |
| sc16 | time_truncation | 259.8 | 11.66 | 44.87 | 0.8688 | 8.659 | 300 |
| sc17 | arrive_dest | 183.6 | 8.37 | 45.61 | 0.9766 | 8.783 | 209 |
| sc18 | time_truncation | 262.2 | 11.81 | 45.05 | 0.8810 | 8.741 | 300 |
| sc19 | arrive_dest | 207.6 | 9.37 | 45.13 | 0.9809 | 8.509 | 244 |
| sc20 | arrive_dest | 253.4 | 11.46 | 45.23 | 0.9818 | 8.680 | 292 |
| sc21 | arrive_dest | 229.7 | 10.36 | 45.08 | 0.9825 | 8.508 | 270 |
| sc22 | arrive_dest | 219.6 | 9.84 | 44.80 | 0.9810 | 8.416 | 261 |
| sc23 | time_truncation | 260.1 | 11.66 | 44.85 | 0.7915 | 8.669 | 300 |

**汇总**：arrive_dest 13/16（81.25%）；time_truncation 3/16（sc16/sc18/sc23 至 max_step，
progress 0.79–0.88，无失败语义）；collision / out_of_road / wrong_direction 均为 0/16
（`wrong_direction_fraction` 与 `stopped_fraction` 全部为 0）。S 图 mean speed
11.24–11.45 m/s、mL/km 48.80–49.17；SC 图 mean speed 8.42–8.78 m/s、mL/km 44.80–45.61。
S 图全部 8 回合 arrive；SC 图 3/8 回合因路线较长在 300-step horizon 内未完成。

**验证**：运行前 `just test-target tests/configuration/test_scalar_reward.py` 10 passed
（A0 协议组合与校验）；评测中 runner 协议校验全部通过。本记录为运行登记，无代码改动。

**产物**：`outputs/studies/scalar-reward/e-029-a0-baseline/`（git 忽略）。根目录含
`resolved_config.yaml`、`summary.json`、`runtime_metadata.json`、`tracked_diff.patch`（空）；
16 个场景目录（`s16`–`sc23`）各含 `trace.npz`（含 `executed_route_heading_errors_rad`
等完整 trace 字段）与 `summary.json`。

**结论边界**：本记录证明 matched 协议（held-out S/SC seeds 16–23、horizon 300、DDIM5、
seed 760025、guidance none）下的 A0 frozen baseline 在当前 commit 可复现，并提供
Issue #80 下一阶段 reward ablation 所需的 energy / distance / progress / completion /
speed / collision / out-of-road / wrong-direction / termination 基准。不支持：任何
energy improvement 结论（A1/A2 尚未运行）；与 E-017 的跨版本直接数值比较（不同 commit、
场景集、horizon 与 seed 协议）；真实车辆能耗结论（energy 仍为 kinematic execution trace
的 fuel proxy）；重复性结论（单次运行，同 harness 的逐值可重复性已由 E-017 在其协议下
证明，本记录不重复）。
