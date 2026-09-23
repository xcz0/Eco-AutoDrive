# 0039 — 统一 closed-loop cadence 与 transition 时间语义

- 状态：接受
- 日期：2026-09-21
- 来源：Issue #96 Task E
- 取代：ADR 0017；ADR 0038 中“人工 intervention/rollout 的 0.1 s 与普通 evaluation 的 0.5 s 边界保持不变”的表述

## 背景

ADR 0017 让 PPO 采集使用 10 Hz contract：一个 MDP transition 只执行预测轨迹第一个 0.1 s 点，
而 evaluation 使用 2 Hz receding-horizon contract，每个 planning cycle 执行 5 点、0.5 s。
两者被实现为不同 cadence：`ROLLOUT_EXECUTION_STEPS=1` 与 `EVALUATION_EXECUTION_STEPS=5`，
以及 `ExecutionMode` 与其在各层重复校验的 plumbing。结果 training 与 evaluation 实际运行两个
不同的 closed-loop system，`gamma=0.99` 被隐含解释为 per 0.1 s transition。

## 决定

training rollout 与 evaluation 使用唯一 canonical closed-loop cadence：

- `CLOSED_LOOP_EXECUTION_STEPS=5`，`DECISION_INTERVAL_S=0.5 s`，规划频率 2 Hz。
- 一个 PPO transition 等于一个 planner/policy decision 加上其完整 execution prefix；
  在 terminal/truncation 时 prefix 提前截断为 1–5 个子步。
- 删除 `ExecutionMode`、`ROLLOUT_EXECUTION_STEPS`、`EVALUATION_EXECUTION_STEPS` 与相关重复校验。
  环境执行 API 只接受显式 `execution_steps`，默认 `None` 解析为 canonical。
- matched causal intervention 诊断可显式传入其他 `execution_steps`；该覆盖不改变 baseline 契约或
  正式 transition 语义。

多 substep transition 的 reward 聚合由 `reward.aggregate_substep_rewards` 唯一拥有，在线 transition 边界
`aggregate_transition_reward` 与离线 reweight/rescore 共用同一归约：`total`、`base_total` 与 components 求和，
`safety_gate` 取最小值。因此多 substep 时 `total != base_total * safety_gate`；`total` 是唯一权威 PPO 标量。
`aggregate_transition_reward` 的 diagnostics 规则显式：`route_progress_delta_m`、`step_distance_m`、
`native_step_energy_ml`、`executed_fuel_proxy_step_energy_ml` 求和；`native_episode_energy_ml` 是 MetaDrive
的 episode 累计值，取最后一个 substep 而非求和；`executed_fuel_proxy_ml_per_km` 是 `ml/km` 比率，按
`sum(fuel) / sum(distance) * 1000` 距离加权，而非算术平均；其余 intensive diagnostics 取均值；
`has_ttc_candidate` 取 any、`energy_distance_valid` 取 all，gate-like score（`collision_score`、
`drivable_score`、`wrong_direction_score`）取最小值。transition 级 domain 聚合同样显式：
`distance_m` 求和，`speed_mps`/`position_error_m`/`heading_error_rad` 取均值，
`stopped`/`collision`/`wrong_direction` 取 any。

在线求值把每个 substep 的 component score、safety gate 与重校准输入（progress/comfort 运动量、
fuel-proxy step energy、step distance、energy distance validity、substep count）显式持久化到 rollout audit。
`eco_planner.rl.reward` 的离线 reweight/rescore、`verify_original_components`、energy-band 阈值推导与校准
逐 substep 重建 objective 后再用同一归约，避免用 `min(gate) * sum(base)` 或对非线性逐 substep 分量做单次
缩放造成 online/offline reward–credit 分叉。


`gamma` 保持 per transition（当前 0.99），不做 rebase。canonical cadence 下每个 transition 覆盖
0.5 s，因此有效折扣视界由约 10 s 变为约 50 s。这是有意的科学语义变更，显式记录而不补偿。

## 后果与可比性

- evaluation 的 5 substep/0.5 s 数值行为不变，其历史产物保持可比。
- training rollout 由每周期 0.1 s 改为 0.5 s：同模拟时长下 transition 数约为原先 1/5，每 transition
  reward 尺度约 5×，`gamma` 有效视界约 5×。历史 10 Hz rollout 训练产物与本次之后的训练产物不可
  直接比较。
- 历史实验记录保持原样，不回填、不声称重现。
- training 与 evaluation 仍保留各自独立的编排、产物 schema 与索引单位；统一的是执行 cadence 与
  transition 时间尺度，不是 seed、RNG lifecycle、episode orchestration 或 artifact schema。
