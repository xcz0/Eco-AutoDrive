# Issue #96 Task E 实施计划

统一 closed-loop cadence、正式 trajectory execution 与 RL transition 时间语义。

本文件是 Task E 的临时实施计划，按子任务拆分，供多个会话分次执行。每个子任务完成后更新本文件对应状态；全部完成后按 Issue #96 的“文档写回”规则清理本文件（或将其结论迁入权威文档后删除）。

当前实现事实以代码和测试为准；本文件描述目标与拆分，不代替当前实现契约。

## 背景

- Task A–D 已完成：planning 共享决策、reward core、reward adapters、envs/domain facts 已稳定。
- 当前 cadence 分叉：
  - `SIMULATOR_STEP_S = 0.1`；`ROLLOUT_EXECUTION_STEPS = 1`（10 Hz，0.1 s/transition）；`EVALUATION_EXECUTION_STEPS = 5`（2 Hz，0.5 s/decision）。
  - 一个 PPO transition 被硬锁为“1 decision + 恰好 1 substep + 1 metric + 1 reward”（`rl/rollout/collector.py:_execution_transition_audit`）。
  - GAE 每个 transition 递推一次，`gamma=0.99` 隐含按 0.1 s。
  - evaluation trace 校验硬锁 5-step prefix（`evaluation/artifacts/trace.py`，`EXECUTION_PREFIX_STEPS`）。
- trajectory execution 本身没有重复实现：`MetaDriveEnvSlot.step` → `TrajectoryExecutor.execute` 是唯一入口；`to_world_trajectory` 唯一。重复的是 cadence plumbing（`ExecutionMode.steps` vs 显式 `execution_steps` 四层各校验一次）与 `_stationary_trajectory` helper（三处）。

## 已确定决策

1. **canonical prefix = 5 substeps（0.5 s，2 Hz）**。training rollout 与 evaluation 运行同一真实 closed-loop system。evaluation 数值行为不变；training 由 0.1 s 改为 0.5 s。一个 PPO transition = 一个 planner/policy decision + 其完整 execution prefix。
2. **transition reward = 各 substep `RewardResult.total` 之和**。
3. **非标量 RewardResult 聚合使用显式混合规则**（见 E2）。
4. **safety_gate = 各 substep 最小值**；`total != base_total * safety_gate`（多 substep 时），需在文档与代码说明。
5. **gamma 保持 per transition（0.99）**，不 rebase。effective horizon（秒）约放大 5×，作为科学语义变化显式记录，不做补偿。
6. **删除 `ExecutionMode` 与 mode-specific 常量**；保留显式 `execution_steps` 覆盖，供 matched causal intervention 诊断使用。

## 科学语义变化摘要（必须单独记录，不伪装成重构）

- training rollout decision interval：0.1 s → 0.5 s；每 transition reward 尺度约 5×，但同模拟时长下 transition 数约 1/5。
- `gamma=0.99` per transition 对应的有效折扣视界由约 10 s 变为约 50 s。
- 历史 10 Hz rollout 训练产物与本次之后的训练产物不可直接比较；evaluation 侧产物保持可比。
- 这些变化只影响 training 路径；evaluation 的 5 substep/0.5 s 契约数值不变。

## 子任务拆分

### E1 — canonical cadence contract 与 execution API 统一

状态：已完成

目标：建立唯一权威 cadence 常量，删除 `ExecutionMode` 与 mode-specific 常量，env 执行 API 只接受显式 prefix（默认 canonical），所有调用方迁移。本子任务暂不改变 RL 实际 cadence（RL 过渡期显式传 `execution_steps=1`），保证每步代码可运行、测试可过。

范围 / 文件：

- `src/eco_planner/contracts.py`
  - 删除 `ROLLOUT_EXECUTION_STEPS`、`EVALUATION_EXECUTION_STEPS`、`ExecutionMode`、`ExecutionMode.steps`。
  - 新增 `CLOSED_LOOP_EXECUTION_STEPS: Final = 5` 与 `DECISION_INTERVAL_S: Final = CLOSED_LOOP_EXECUTION_STEPS * SIMULATOR_STEP_S`。
  - `evaluation_plan_cycles` 改为除以 `CLOSED_LOOP_EXECUTION_STEPS`。
- `src/eco_planner/envs/metadrive/slot.py`：删除 `execution_mode` 参数与类型校验；`execution_steps: int | None = None` 解析为 `CLOSED_LOOP_EXECUTION_STEPS`；`_execution_steps` 只保留一份。
- `src/eco_planner/runtime/envs/vector.py`：删除 `execution_mode` 参数、`_validate_configuration` 中 `execution_steps` 范围重复校验；只透传 `execution_steps`。
- `src/eco_planner/runtime/envs/worker.py`：`make_torchrl_scenario_env` 删除 `execution_mode`，透传 `execution_steps`。
- `src/eco_planner/envs/metadrive/execution.py`：保留唯一 prefix 范围校验。
- evaluation：`evaluation/episodes/serial.py`、`vector.py` 删除 `ExecutionMode` 并依赖 canonical 默认；`evaluation/artifacts/trace.py` 用 `CLOSED_LOOP_EXECUTION_STEPS` 取代 `EXECUTION_PREFIX_STEPS`/`EVALUATION_EXECUTION_STEPS`；`evaluation/episodes/recorder.py` 容量守卫同步。
- benchmark：`benchmarking/environment.py`、`throughput.py` 删除 `ExecutionMode`，使用 canonical 默认。
- experiment runners（5 个）：删除 `ExecutionMode` kwarg，改为显式 `execution_steps`：
  - execution_bridge / deferral / horizon 已传显式值，直接删 mode。
  - decomposition / authority 当前隐式依赖 `ROLLOUT.steps==1`，**必须显式补 `execution_steps=1`**。
  - 将 provenance 里的 `"execution_mode": "rollout"` 替换为显式 `execution_steps` 字段。
- RL 过渡期：`rl/rollout/collector.py` 删除 `ExecutionMode`，显式传 `execution_steps=1`（带 TODO 指向 E3），本子任务不改 transition 语义。
- `_stationary_trajectory` 三处重复（`envs/metadrive/slot.py`、`benchmarking/environment.py`、`benchmarking/throughput.py`）收敛为一个 envs-owned helper，benchmark 复用。
- 测试迁移：`tests/simulation/test_closed_loop.py`、`tests/training/test_guidance_control_authority.py` 删除 `ExecutionMode` 用法。

验收：

- 仓库内不再出现 `ExecutionMode`、`ROLLOUT_EXECUTION_STEPS`、`EVALUATION_EXECUTION_STEPS`、`EXECUTION_PREFIX_STEPS`（历史 `docs/experiments/records/*` 除外）。
- evaluation 数值行为不变（5 substep/0.5 s）。
- RL rollout 仍为 1 substep（过渡显式值），测试保持通过。

验证：

- `just test-target tests/simulation/test_closed_loop.py`
- `just test-target tests/training/test_guidance_control_authority.py`
- `just lint`、`just typecheck`

依赖：无。

### E2 — reward transition 聚合（纯函数）

状态：已完成

目标：在 `reward` 包内实现多 substep `RewardResult` 的纯聚合，作为唯一权威数学；不接入 RL，先独立可测。

范围 / 文件：

- 新增 `src/eco_planner/reward/aggregation.py`：`aggregate_transition_reward(results: Sequence[RewardResult]) -> RewardResult`。
  - 输入非空校验；`profile_name` 必须一致，否则失败（不静默混合 profile）。
  - 显式逐字段规则：
    - sum：`total`、`base_total`、`RewardComponents.*`；additive diagnostics（`route_progress_delta_m`、`step_distance_m`、`native_step_energy_ml`、`native_episode_energy_ml`、`executed_fuel_proxy_step_energy_ml`）。
    - mean：intensive diagnostics（`speed_mps`、`speed_limit_mps`、`overspeed_mps`、`longitudinal_acceleration_mps2`、`lateral_acceleration_mps2`、`jerk_mps3`、`yaw_rate_radps`、`min_ttc_s`、`executed_fuel_proxy_ml_per_km`）。
    - any：`has_ttc_candidate`；all：`energy_distance_valid`。
    - min：`safety_gate`、`collision_score`、`drivable_score`、`wrong_direction_score`。
  - 文档说明：多 substep 时 `total != base_total * safety_gate`，`total` 是 PPO 权威标量。
- `src/eco_planner/reward/__init__.py` 导出该函数。

验收：

- 单 substep 输入聚合结果等于原结果（parity）。
- 多 substep 的 sum/mean/any/all/min 分类字段有单元测试固定。

验证：

- 新增 `tests/training/test_reward.py`（或独立测试文件）覆盖聚合规则。
- `just test-target tests/training/test_reward.py`

依赖：无（可先于 E1）。

### E3 — RL transition 时间语义接入

状态：已完成

目标：RL rollout 改为 canonical 5-substep prefix；一个 transition 聚合多个 substep；更新 horizon/quota 数学与配置；删除 E1 的过渡显式 `1`。

实现备注：`rl/rollout/contracts.py` 中 `_validate_audit_trajectory` 原有的 `reward_total == reward_base_total * reward_safety_gate` 断言在 multi-substep 下不再成立（E2 规则），已删除；`reward_component_*` 由“单 substep [0,1] 上界”改为“非负”守卫（多 substep 为各 substep 之和）。

范围 / 文件：

- `src/eco_planner/rl/rollout/collector.py`
  - 删除过渡 `execution_steps=1`，使用 canonical 默认。
  - 重写 `_execution_transition_audit`：用 `len(metrics) == execution.substep_states.shape[0]` 且 `1 <= count <= CLOSED_LOOP_EXECUTION_STEPS` 取代“恰好 1 substep/metric”断言；逐 substep 调 `RewardEvaluator` 后 `aggregate_transition_reward`。
  - transition 级 domain fact 聚合（消费者窗口，显式文档化）：`distance_m`=sum，`speed_mps`=mean，`stopped`/`collision`/`wrong_direction`=any，`position_error_m`/`heading_error_rad`=mean；`route_completion_delta` 已覆盖整个 prefix。
  - `rl/rollout/contracts.py`：更新 `ExecutionTransitionAudit` docstring 为“one closed-loop decision + its execution prefix”。
- `src/eco_planner/rl/config.py::_validate_rollout_environment`：`required_horizon = history_warmup_steps + transition_count * CLOSED_LOOP_EXECUTION_STEPS`。
- YAML：更新所有 RL rollout/training 的 `env.horizon`：
  - `configs/components/ppo_training.yaml` 16 → 80
  - `configs/jobs/training/ppo_conservative.yaml` 16 → 80
  - `configs/jobs/training/rollout_smoke.yaml` 4 → 20
  - benchmark rollout 与 reward collection / training grid / comparison 等复用 env.horizon 的 job，确保 `horizon >= warmup + transitions*5`。
- 测试更新：`tests/training/test_rollout.py`（transition_count 仍等于 decision 数；新增多 substep reward=sum、gate=min 断言）。

验收：

- RL rollout 每个 transition 实际执行 5 个 0.1 s substep（terminal 提前停止时可为 1–5）。
- scalar reward = 各 substep total 之和；`next.reward`、GAE/PPO 形状与递推不变（仍 per transition）。
- 配置解析不再因 horizon 不足失败。

验证：

- `just test-target tests/training/test_rollout.py`
- `just test-target tests/training/test_reward.py`
- 配置解析相关测试（training/rollout config）与 `just typecheck`

依赖：E1、E2。

### E4 — 一致性验证与防回归测试（E3 acceptance）

状态：未开始

目标：用测试固定 rollout/evaluation 共享同一 execution 语义，并防止重新引入 mode-specific cadence 或重复 trajectory execution。

范围 / 文件：

- 新增 cross-entry execution 测试（建议 `tests/simulation/`）：
  - 同一 trajectory / 环境初态下，substep 数、累计 simulated time、`TransitionMetrics` domain facts、raw terminal facts 与 horizon/cycle 换算在 training 与 evaluation 路径一致。
  - 早期 terminal 时 prefix 截断位置与 raw terminal facts 一致。
- 防回归 guard 测试：
  - `contracts` 中不存在 `ExecutionMode` / mode-specific prefix 常量。
  - 不存在第二份 `TrajectoryExecutor` / local→world conversion（例如静态扫描或导入唯一性断言）。
  - env slot 默认 prefix 等于 `CLOSED_LOOP_EXECUTION_STEPS`。
- 更新 `tests/simulation/test_closed_loop.py`：默认 rollout 断言由 `(1,7)`/1 metric 改为 `(5,7)`/5 metrics（或显式 prefix 保持原断言）。
- 固定 warmup=20、evaluation 5-step prefix、terminal early-stop 等既有不变量。

验收：

- 上述测试通过，且能在有人重新引入 mode cadence 或复制执行逻辑时失败。

验证：

- `just test-sim`
- `just test-target tests/simulation/test_closed_loop.py`

依赖：E1、E3。

### E5 — 文档写回与 ADR

状态：未开始

目标：把已实现的 cadence 与 transition/reward/GAE 语义写入权威文档，并按“intentional cadence change 单独记录”要求新增 ADR。

范围 / 文件：

- `docs/agents/system-contract.md`：坐标/时间章节（24、35–36）、轨迹执行（42–51）、评测指标（73）、trace prefix 校验（83）改为单一 canonical cadence 与“一个 PPO transition = 一个 decision + 完整 prefix”。
- `docs/agents/contracts/training.md`：29（`trajectory_execution_steps=1` / evaluation 5）、31（单 substep reward）、39（GAE cadence）、55（policy checkpoint eval 0.1 s vs 0.5 s）更新为统一 cadence、多 substep 聚合、gamma per decision 的视界影响。
- `docs/agents/contracts/experiments.md`：47、53、59、65 中“ROLLOUT 每周期执行 0.1 s / 普通 evaluation 0.5 s”的表述改为 canonical 0.5 s baseline + 诊断显式 `execution_steps` 覆盖。
- 新增 ADR（下一编号）：记录 cadence 统一、reward=sum、safety_gate=min、gamma per transition 与视界/可比性影响；明确指出取代原 10 Hz/2 Hz 分叉决定。
- 检查 `CONTEXT.md`、`docs/agents/domain.md` 是否含需要同步的 cadence/transition 术语；仅在有事实变化时更新。
- 历史 `docs/experiments/records/e-043-*.md`、`e-047-*.md` 保持原样，不回填。
- 完成后按规则处理本 `schemes.md`。

验收：

- 权威文档与最终代码一致；科学语义变化与纯 ownership 重构分开记录。
- 无与 Issue #96 重复的第二份 implementation plan。

验证：

- 文档链接与引用检查；不运行代码测试。

依赖：E1、E3、E4。

## 依赖关系

```text
E1 ──┐
     ├── E3 ── E4 ── E5
E2 ──┘
```

- E1 与 E2 可并行/独立开始。
- E3 依赖 E1（canonical 常量与 API）和 E2（聚合函数）。
- E4 依赖 E1、E3。
- E5 最后。

## 验证入口汇总

```powershell
just test-target tests/simulation/test_closed_loop.py
just test-target tests/training/test_rollout.py
just test-target tests/training/test_reward.py
just test-target tests/training/test_guidance_control_authority.py
just test-sim
just lint
just typecheck
```

按沙箱要求，运行 `just test*` 时申请沙箱外执行。

## 风险与备注

- E1 结束后 RL 仍显式 `execution_steps=1`，是过渡状态；E3 必须删除并接入 canonical，否则 cadence 分叉仍存在。
- decomposition / authority 若漏传 `execution_steps=1`，会在 E1 后静默从 1 变为 5，改变诊断语义。
- `env.horizon` 是 simulator substep 计数的 MetaDrive episode 上限；canonical 后台账仍以 substep 计，但 RL quota 以 decision（transition）计，两者换算集中在 `_validate_rollout_environment`。
- evaluation 数值不变，故其历史产物可比；training 历史产物不可比，已在 ADR/文档记录。
- 不为统一 cadence 合并 training/evaluation 的 seed、RNG lifecycle、episode orchestration、artifact schema。
