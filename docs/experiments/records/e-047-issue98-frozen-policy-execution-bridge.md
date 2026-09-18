# E-047 Issue #98：Frozen-policy execution-contract bridge

[返回实验索引](../README.md) · [相关 E-040](../records/e-040-issue94-task-g-objective-positive-control.md) · [E-043](../records/e-043-issue98-task-a-execution-horizon.md) · [E-045](../records/e-045-issue98-task-c-replanning-deferral.md)

**日期 / 类型 / 目的**：2026-09-17 / 正式配对闭环诊断 + 固定来源 offline 归因 / 不训练、不修改
reward/PPO/planner。用 E-040 的四个 frozen final checkpoints（r0/rstress × seeds {0,1}）在
matched held-out 协议上只改变 execution prefix（k=1/2/5），并额外做同状态双 policy 反事实
planner 审计，直接回答“E-040 的 `Rstress-R0` 闭环方向是否仅因训练 rollout 的 k=1（0.1 s）与
标准 evaluation 的 k=5（0.5 s）execution contract 不同而出现或翻转”。本记录完成 Issue #98
新增诊断任务，未执行原 Task E（training-budget sweep）。

**裁定：`execution_contract_causal_crossover_confirmed`。** 同一 frozen learned policy
difference 在 k=1 为负、k=5 为正，两个 training seed 一致；Part B 同状态局部响应同样呈现
0.1 s ≤ 0、0.2/0.5 s > 0 的 `local_temporal_bridge` 结构。

## 运行来源与固定协议

- 代码基线 `b1d9639f53d12f05288a79c5c9d731adce8836f7`（main）加本次未提交改动；
  `runtime_metadata.json` 保存 `git_head`、`git_branch`、`git_status_short`。工作区 dirty
  （2 个修改文件、5 个新增路径），因此这是诊断干预而非 clean-commit 正式基线。
- Windows、Python 3.10.20、PyTorch 2.12.1+cu126、Lightning 2.6.6、MetaDrive 0.4.3；CUDA
  `cuda:0`，resource profile `rtx_a4000`，4 个 rollout workers。
- 官方 EMA planner：`checkpoints/DP-Origin/model.pth`，冻结、eval；标准高斯 DDIM-5、
  `stochasticity=0`、BF16、runtime seed 760025，orthogonal-policy guidance 系数与训练一致
  （lateral 2.5 m、longitudinal fraction 0.25）。deterministic policy action（Beta mean）。
- 冻结 policy checkpoints（`--source-dir` 指向 E-040 study 目录的相对路径）：

  | label | arm | seed | path | policy_state_hash |
  | --- | --- | ---: | --- | --- |
  | r0-seed-0 | r0 | 0 | `r0-seed-0/policy-final.pt` | `6af05510…af44014c` |
  | rstress-seed-0 | rstress | 0 | `rstress-seed-0/policy-final.pt` | `80648a71…518113ae` |
  | r0-seed-1 | r0 | 1 | `r0-seed-1/policy-final.pt` | `d46196c4…068bc7665` |
  | rstress-seed-1 | rstress | 1 | `rstress-seed-1/policy-final.pt` | `ae11c27b…43c96675f` |

  每个 checkpoint 在每次加载后核对 hash 不变；execution prefix 干预不改变 policy 参数。
- 场景：no-traffic，S/SC × map seeds 16–23（16 场景），runtime/eval job
  `jobs/evaluation/no_traffic_heldout_manual`，`env.horizon=300`，`evaluated_horizon_steps=300`，
  noise seed 760025。Part A 执行 prefix `k=[1,2,5]`（`cycles=300/k`），Part B 上下文采集用标准
  `k=5`。同一 group 内两臂 matched 初始 observation/state、物理 slot、batch shape 与逐周期
  diffusion noise；执行使用显式 `execution_steps` 覆盖（`ExecutionMode.ROLLOUT`）。

配置：`configs/experiments/guidance/execution-bridge.yaml`。实现：`evaluation/policy_intervention.py`、
`experiments/guidance/execution_bridge/`；recompute：`analysis/execution_bridge.py`。

正式运行一次，完成 **192 episodes / 32,596 transitions**（Part A）与 **1,177 same-state
contexts**（Part B），耗时 **1,414.35 s**。independent offline recompute 的 `summary.json`
与运行目录 `summary.json` 文件哈希相同（`1795C30E…D2D9522F`），源文件未修改。

```powershell
just exp guidance execution-bridge run --source-dir outputs/studies/scalar-reward/e-040-issue94-task-g-objective-positive-control --output-dir outputs/studies/scalar-reward/e-047-issue98-frozen-policy-execution-bridge

just exp guidance execution-bridge analyze --source-dir outputs/studies/scalar-reward/e-047-issue98-frozen-policy-execution-bridge --output-dir outputs/studies/scalar-reward/e-047-issue98-frozen-policy-execution-bridge-analysis
```

## Part A：闭环 execution-prefix crossover

逐场景配对差 `Rstress - R0`；方向由 `required_scenarios=9/16` 多数门槛判定。effect 为 16 个
场景配对差的中位数（m/s、mL/km）。

| Seed | k | speed 方向 | speed effect 中位 (m/s) | speed (+/-) | energy 方向 | energy effect 中位 (mL/km) | energy (+/-) |
| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 0 | 1 (0.1 s) | **negative** | -0.0209 | 2/14 | **negative** | -0.0313 | 3/13 |
| 0 | 2 (0.2 s) | **positive** | +0.1404 | 16/0 | **positive** | +0.2469 | 16/0 |
| 0 | 5 (0.5 s) | **positive** | +0.0997 | 16/0 | **positive** | +0.1694 | 16/0 |
| 1 | 1 (0.1 s) | **negative** | -0.0374 | 2/14 | **negative** | -0.0566 | 4/12 |
| 1 | 2 (0.2 s) | **positive** | +0.1958 | 15/1 | **positive** | +0.3321 | 15/1 |
| 1 | 5 (0.5 s) | **positive** | +0.1462 | 16/0 | **positive** | +0.2570 | 16/0 |

- 两个 training seed 同时满足 k=1 negative、k=5 positive → Part A `crossover_confirmed`。
- k=5 与 E-040 matched held-out 的 `Rstress-R0` 方向与量级一致（E-040：seed0 +0.1061 m/s /
  +0.1903 mL/km，seed1 +0.1398 / +0.2496），因此 k=5 复现 E-040 的“更快且单位里程能耗更高”。
- first-plan planner forward-displacement effect（跨 prefix 逐元素匹配后，16 场景中位，m）：
  0.1 s -0.004/-0.005、0.2 s +0.019/+0.026、0.5 s +0.079/+0.106、1 s +0.154/+0.207、
  2 s +0.349/+0.469、4 s +0.681/+0.916、8 s +1.561/+2.109（seed0/seed1）。局部 planner 响应
  在 0.1 s 已为负、0.2 s 起为正，与闭环 crossover 同向。

## Part B：Same-state 局部 policy→planner bridge

在同一 held-out 状态分布上采集 1,177 个 matched contexts，对同一 observation、同一 planner、
同一 diffusion noise 分别评估两个 policy。

| Seed | contexts | Δg_lon 中位 | Δg_lon > 0 | forward effect 中位 0.1 s | 0.2 s | 0.5 s | zero-crossing 中位 step |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 587 | +0.0671 | 587/587 | -0.00756 m | +0.00698 | +0.02339 | 1 (0.2 s) |
| 1 | 590 | +0.0953 | 590/590 | -0.01072 m | +0.01005 | +0.03309 | 1 (0.2 s) |

- 多数（此处全部）context `Δg_lon > 0`，0.1 s 中位 ≤ 0、0.2/0.5 s 中位 > 0 → Part B
  `local_temporal_bridge`。该结构与 E-043 的全幅 `g_lon -1→+1` sweep、E-045 的 0.2 s
  zero-crossing 方向一致，说明 learned R0→Rstress 的**局部** operating point 确实携带同一时间
  响应错位，而不是全幅 sweep 的非线性 artifact。
- 0.5 s 后 forward effect 单调增大（seed0：2 s +0.1013、4 s +0.1928、8 s +0.4617；seed1：
  2 s +0.1428、4 s +0.2811、8 s +0.6640），横向 effect 全程很小（|中位| ≤ 0.015 m），与
  E-044 “主要来自 longitudinal”一致。

## 安全性与边界

- Part A 192 episodes 中 **25 个 unsafe（全部 `out_of_road`，0 collision）**，集中在 Part A
  k=1 的 SC offset（groups 4–7，各臂各 seed 5 个）与 k=2 的 SC offset 2（groups 12–13）；
  **同一 (seed, k) 下 r0 与 rstress 的 safety 计数逐项相同**，因此 crossover 不是 arm
  非对称 safety confound。k=5 无 out-of-road。该 off-route 是 execution prefix / planner
  输出特性，未删除、未填补；`gate.safety_clean=false` 只因预声明 collision/OOR 守卫触发。
- 机器 `gate`：`part_a.status=crossover_confirmed`、`part_b.status=local_temporal_bridge`、
  `proxy_errors=[]`；verdict `execution_contract_causal_crossover_confirmed`。
- energy 与 speed 不是独立证据：MetaDrive fuel proxy 随速度指数增长，速度方向确定后 energy
  机械同向。
- 结论限于 no-traffic S/SC seeds 16–23、300 步 horizon、BF16 冻结 planner、deterministic
  Beta mean 与 MetaDrive fuel proxy；不证明训练时间尺度、真实车辆行为或新的推荐 execution
  contract。Part B 的上下文来自标准 `k=5` held-out 状态分布，未覆盖训练 `k=1` 状态分布。

## 结论边界

- **支持** Issue 评论中的最强证据形态：same frozen policies / scenarios / noise，k=1
  `Δspeed ≤ 0`、k=5 `Δspeed > 0`，两 seed 一致，且 k=5 复现 E-040 的 positive speed/energy。
  这直接把 E-040 的反向行为收敛到 `train–evaluation execution-contract mismatch mediated by
  guidance temporal response`，无需诉诸“policy 没有学到 energy objective”。
- 归因优先级：A（temporal response × receding-horizon execution mismatch）为主；E-044 已排除
  B（横纵耦合）为主因；E-046 显示 critic/GAE 至多为有界次要因素。本实验不重跑 Task B/C/D。
- **不执行**原 Task E（training-budget sweep）：crossover 已在 frozen policy 层面成立，扩大
  训练预算不再是解释方向差异的必要下一步；若后续要研究学习动态仍可单独立项。
- 不修改 baseline execution contract（ROLLOUT 每周期 1 点 / EVALUATION 每周期 5 点）；显式
  `execution_steps` 覆盖仅是本诊断入口。

## 验证与产物

- 新增 `tests/training/test_guidance_execution_bridge.py` 10 项（config 校验、Part A
  crossover/amplify、Part B bridge/differs、zero-crossing、`_prediction_response` 几何、
  offline recompute live 一致、CLI 路由、跨 prefix first-plan response 匹配守卫）。
- `just test-target tests/training tests/analysis` **223 passed, 7 deselected**；
  `just lint`、`just typecheck` 0 error。运行前对新增 collector 做了真实 planner/policy 的
  短程 smoke（Part A 8 episodes + Part B 12 contexts）。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-047-issue98-frozen-policy-execution-bridge/`
  - `raw/group-*-arm-*/`：初始 observation、每周期 planner audit NPZ、逐 arm episodes；
  - `episodes.json`、`same_state.json`、`same-state-group-*.json`、`scenarios.json`、
    `intervention_config.json`、`resolved_config.yaml`；
  - `runtime_metadata.json`、`run.json`、`decisions.json`；
  - `summary.json`、`analysis.json`、`report.md`、3 张图（Part A speed/energy、Part B forward）。

独立复算目录：
`outputs/studies/scalar-reward/e-047-issue98-frozen-policy-execution-bridge-analysis/`（`summary.json`
与运行目录逐值相同）。本次改动未提交、未推送；已更新 Issue #98。
