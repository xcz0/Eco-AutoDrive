# E-046 Issue #98 Task D：training adequacy / critic attribution（离线 backward-only 归因）

[返回实验索引](../README.md) · [前置 E-045 Task C](e-045-issue98-task-c-replanning-deferral.md) · [源实验 E-040 Task G](e-040-issue94-task-g-objective-positive-control.md) · [相关 E-039 Task F](e-039-issue94-task-f-effective-update-region.md)

**日期 / 类型 / 目的**：2026-09-17 / 正式离线固定来源诊断（无训练、无仿真、无 optimizer step）/
完成 Issue #98 Task D：在不改 PPO 配置、不重新训练的前提下，直接复用 E-040 保存的 rollout /
checkpoint，沿 50-update 训练轨迹比较 current standard GAE 与 critic-free V=0 reward-only GAE
对 actor gradient direction 的影响，判断 E-039 "critic explained variance ≈ 0" 是否可能
materially 改变 actor update direction，从而决定归因 C（insufficient training / temporal-credit /
critic issue）能否作为 E-040 c4 方向错误的首要解释。

**裁定：按预登记 all-checkpoint 判据为 `critic_material_candidate`（无法认证 critic 无关）。**
在 4 runs × 50 updates = 200 个 checkpoint 上，standard-versus-reward-only 的 actor
gradient cosine 在 **20/200** checkpoint 上低于 0.99（最小 `longitudinal` 0.8849、`lateral`
0.9441、`actor_head` 0.9816），且失败集中在 update 24–49 后期并偏 `rstress`。但同一批
checkpoint 的 **advantage 排序几乎不变**：sign-flip fraction 最大 0.0078（1/128），Spearman
最低 0.9945、Pearson 最低 0.9959。因此本实验的准确结论是：**critic / temporal-credit 会产生
局部、可测但不重排序的 actor 方向旋转，不能作为 E-040 反向行为的主因，但按预登记判据也不能
认证其完全无关**。这与 Task A（execution-horizon mismatch，主因）和 Task B（横纵耦合非主因）
一致；归因 C 至多为有界的次要贡献，而非首要解释。

## 运行来源与固定协议

- 代码基线：`main` 分支 commit `319000e`（Task C）之上的**未提交**本次改动（新增 training
  `critic-attribution` 工作流、离线分析与 `read_rollout_episode` artifact primitive）；运行目录
  `runtime_metadata.json` 保存实际 `git_head` 与 `git_status_short`。
- Windows、Python 3.10.20、PyTorch 2.12.1+cu126；**CPU** 执行 backward（无 CUDA / MetaDrive /
  planner；不加载 reward、环境或执行器）。无仿真、无训练、无数据生成。
- **输入完全复用 E-040 冻结产物**：`r0-seed-{0,1}` / `rstress-seed-{0,1}` 的
  `resolved_config.yaml`、`summary.json`、`policy-initial.pt` 与 `policy-update-000..049.pt`、
  `updates/update-NNN/slot-*-episode-*.npz`。E-040 的 rollout audit schema 与 checkpoint
  格式在当前代码下逐字段可加载（见验证）。
- **Update 采样**：`update_indices = 0..49`（全轨迹），`advantage_form = z`（实际 actor
  preprocessing），`credit_forms = {standard_gae, reward_only_gae}`。
- **Policy 语义**：每个 update `k` 使用该 update rollout 的 **pre-update policy**
  （`k=0` 用 `policy-initial.pt`，否则 `policy-update-(k-1)`），即真正产生该 batch 并计算
  advantage 的策略；每个 checkpoint 只做 `loss_objective` backward，不 clipping、不 optimizer。

```powershell
just exp training critic-attribution run `
  --source-dir outputs/studies/scalar-reward/e-040-issue94-task-g-objective-positive-control `
  --output-dir outputs/studies/scalar-reward/e-046-issue98-task-d-training-adequacy
just exp training critic-attribution analyze `
  --source-dir outputs/studies/scalar-reward/e-046-issue98-task-d-training-adequacy `
  --output-dir outputs/studies/scalar-reward/e-046-issue98-task-d-training-adequacy-analysis
```

配置清单：`configs/experiments/training/critic-attribution.yaml`（runs、credit forms、
`update_indices`、provenance 容差、预登记 materiality 阈值）；实现位于
`src/eco_planner/experiments/training/critic_attribution/`，离线重算位于
`src/eco_planner/analysis/critic_attribution.py`，报告/绘图位于
`src/eco_planner/analysis/reporting/critic_attribution.py`。

## 测量语义

- 每个 update 由 `updates/update-NNN/slot-*-episode-*.npz` 经 `read_rollout_episode` 从 audit
  轨迹 + `tail_kind`/`tail_bootstrap_value` 完整重建 `RolloutEpisode`（不再依赖
  `training-batch.pt`），拼接后 transition 数必须等于 resolved `ppo.batch_size = 128`。
- 对同一 episodes、boundary 与 reward，分别经 `credit_batch` 构造 standard GAE 与 V=0
  reward-only GAE，再用训练同一 `ClipPPOLoss` 做 full-batch `loss_objective` backward，取
  `actor_head` / `lateral` / `longitudinal` / `shared_trunk` / `actor` 分组梯度。
- **Provenance guard**：standard GAE 重建的 raw advantage `mean`/`std(ddof=1)` 必须与 E-040
  `summary["updates"][k]["raw_advantage_mean"/"raw_advantage_std"]` 在 `rtol=1e-4` 内一致，
  否则运行显式失败；200/200 通过（update 0 rstress 均值逐位 `2.0935258865356445`）。
- **策略不变 guard**：每个 checkpoint backward 前后 `policy_state_hash` 不变、
  `completed_optimizer_steps == 0`；product `optimizer_steps = 0`。

## Materiality gate（预登记）

判据（配置预登记）：对每个 (run, update)，要求 `actor_head` / `lateral` / `longitudinal`
cosine ≥ 0.99 且 advantage sign-flip fraction ≤ 0.05；`shared_trunk` 零向量 cosine 为
undefined 时记录但不计入。全部通过 → `critic_not_material_to_actor_direction`；任一失败 →
`critic_material_candidate`。

| Comparison | passed | checkpoints | failures | undefined |
| --- | --- | ---: | ---: | ---: |
| standard_gae vs reward_only_gae | False | 200 | 20 | 0 |

失败类型：`longitudinal_cosine` 9、`lateral_cosine` 7、`actor_head_cosine` 4（部分 update
多组同时失败）。失败 update 分布：{24, 24, 25, 31, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46,
47, 49}，全部落在后期。

## 结果

### Within-arm standard GAE vs reward-only GAE（z，200 checkpoints）

| 指标 | min | median | max |
| --- | ---: | ---: | ---: |
| actor_head cosine | 0.981627 | 0.999781 | 1.000000 |
| lateral cosine | 0.944080 | 0.999975 | 1.000000 |
| longitudinal cosine | 0.884854 | 0.999965 | 1.000000 |
| actor_head norm ratio (j/i) | 0.7726 | — | 1.2490 |
| advantage sign-flip fraction | 0.000000 | — | 0.0078125 |
| advantage Pearson | 0.995905 | — | 1.000000 |
| advantage Spearman | 0.994501 | — | 1.000000 |

`count(cosine < 0.99)`：actor_head 4、lateral 7、longitudinal 9；`count(< 0.95)`：lateral 1、
longitudinal 1。最小 angle（`acos`）约 27.8°（longitudinal，rstress-seed-0 u47），其余低于
0.99 的多在 8–19°。失败 checkpoint 的 actor-head 梯度范数经独立复核为 0.15–1.68（非
ill-conditioned），因此方向偏差不是零梯度噪声。

### Per-run（min cosine / critic 轨迹）

| run | min actor_head | min lateral | min longitudinal | max sign-flip | EV [min,max] | value loss [min,max] |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| r0-seed-0 | 0.994832 | 0.980195 | 0.996183 | 0.000000 | −0.0004 / +0.0064 | 5.129 / 5.987 |
| r0-seed-1 | 0.996611 | 0.997839 | 0.986689 | 0.000000 | −0.0004 / +0.0118 | 5.163 / 5.984 |
| rstress-seed-0 | 0.987914 | 0.966978 | 0.884854 | 0.0078125 | −0.0006 / +0.0078 | 1.900 / 3.296 |
| rstress-seed-1 | 0.981627 | 0.944080 | 0.975460 | 0.0078125 | −0.0004 / +0.0181 | 2.153 / 3.230 |

- **critic explained variance 全程 ≈ 0**（最大 +0.018），复现 E-039 "50 updates 内 critic 尚未
  拟合"；但 V=0 与 standard 的 advantage `std` 存在约 5% 的系统差（如 r0-seed-0 u49 1.364 vs
  1.434），说明 under-fit critic 的 state-dependent value baseline 仍会轻微重塑 advantage。
- arm 层面：`r0` 双 seed 的 min cosine 全部 ≥ 0.9802，sign-flip 恒为 0；更明显的方向旋转只出现
  在 `rstress`。

### Cross-arm R0 vs Rstress（descriptive，unmatched）

不同 arm 的 rollout transitions 与 policy 均不同，故为**非 matched** 描述量，不作因果解释：

| group | min cosine | median cosine |
| --- | ---: | ---: |
| actor_head | −0.7967 | 0.8854 |
| lateral | −0.8683 | 0.9870 |
| longitudinal | −0.9399 | 0.8163 |

即两 arm 的 actor 梯度方向本身差异很大（longitudinal 中位 0.816、可出现反向），与 E-040
两 arm 向相反方向移动的行为分离自洽；这也说明 Task D 的 within-arm critic 对照必须与
cross-arm reward 差异分开看。

## Task D 归因裁定

```text
Pre-registered all-checkpoint materiality: FAILED (20/200 checkpoints below 0.99)
Verdict: critic_material_candidate
Mechanism supported: bounded, non-reordering actor-direction rotation from the under-fit critic
```

- **advantage 排序未被 critic 重排**：sign-flip ≤ 1/128、Spearman ≥ 0.9945，说明 E-040 c4
  的"更快更耗能"不是 critic 把 transitions 排序翻转造成；该方向问题的主因仍在
  Task A/C 的 guidance temporal response × 0.1 s receding-horizon execution mismatch
  （Task B 已排除横纵耦合为主因）。
- **actor 方向在 90% checkpoint 上高度一致**（median cosine ≈ 0.9998），但在后期少数
  checkpoint 上出现 8–28° 的局部旋转，梯度范数健康，因此不能按预登记阈值认证 critic 完全
  无关。若要把这一有界差异关闭，需要单独 critic / temporal-credit 研究（例如 critic 拟合、
  value target、GAE λ 或 advantage normalization 的控制实验），但这**不是** E-040 反向行为的
  首要解释。按 Issue #98 决策顺序，这不改变 Task E 的"仅在 A–D 不足时进入"的定位。
- E-039 的 "EV ≈ 0" 在本诊断中被复现，但"critic 未拟合"不等于"critic 对 actor 方向无影响"：
  under-fit 的 state-dependent baseline 会在少量后期 checkpoint 上轻微旋转归一化后的方向。

## 结论边界

- 无训练、无仿真、无 planner / MetaDrive / 执行器；全部结论限于 E-040 的固定协议（lr=1.5e-4、
  epochs=1、mgn=0.5、batch=128、50 updates、training seeds {0,1}、no-traffic、BF16 rollout）
  及其保存 rollout / checkpoint。
- actor gradient 是离线 backward-only 测量，**不证明**训练后行为改变、不重新训练、不选择 reward
  或 PPO 配置、不修改 baseline 执行契约。
- 预登记 all-checkpoint 阈值 0.99 是判定选择；本结果对阈值敏感：20/200 失败但 median ≈ 0.9998、
  仅 1 个 group-checkpoint < 0.95。`critic_material_candidate` 表示"不能认证无关"，不等于
  "critic 重新定向了优化目标"。
- cross-arm 梯度对照为 unmatched，仅作描述，不作因果；within-arm 对照才是本 Task 的判据。
- 不把 proxy / 运动学结果外推到真实车辆能耗。

## 验证与产物

- 新增 `tests/training/test_critic_attribution.py` 7 项（config 校验、materiality 三分支与
  undefined 记录、`read_rollout_episode` roundtrip、runner + 离线 recompute 一致、
  diagnostics.npz 篡改检测、CLI 路由）；`just test-target tests/training tests/analysis
  tests/configuration` 289 passed；`just test-all-cpu` 324 passed；`just lint`、
  `just typecheck`（0 errors）通过。
- 离线 `analyze` 从 `diagnostics.npz` raw advantage 重算全部 advantage 对照并与记录逐值一致
  （不一致会显式报错）；记录的 gate 不做重裁定。
- 正式产物（git ignored）：
  `outputs/studies/scalar-reward/e-046-issue98-task-d-training-adequacy/`
  - `diagnostic_config.yaml`：resolved 设计（runs / forms / updates / 阈值）。
  - `runtime_metadata.json`：`git_head`、dirty status、CPU、`optimizer_steps=0`、pre-update
    policy 语义。
  - `sample_index.json`：200 条 (run, update, prefix, transition_count, credit_forms)。
  - `diagnostics.npz`：每个 (run, update, credit) 的 raw advantage 与 value_target。
  - `summary.json`：per-run/per-update 的 advantage 对照、梯度分组对照、loss、critic EV /
    value loss、cross-arm 描述量、materiality gate。
  - `report.md` / `analysis.json` / `figures/`：报告与梯度 cosine、advantage sign-flip /
    Spearman、actor gradient cosine、critic EV / value loss、cross-arm 图。
  - `e-046-run.out.log` / `e-046-analysis.out.log`：正式运行日志。

Task D 已完成（`critic_material_candidate`，真实、有界的次要效应）；本次改动未提交、未推送，
GitHub Issue #98 的状态更新另行执行。Task E（training-budget sweep）未执行。
