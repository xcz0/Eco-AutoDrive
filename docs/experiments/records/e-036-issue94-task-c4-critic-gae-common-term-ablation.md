# E-036 Issue #94 Task C4：Critic / GAE common-term ablation

[返回实验索引](../README.md) · [源实验 E-035](e-035-issue94-task-c-objective-decomposition.md)

**日期 / 类型 / 目的**：2026-09-09 / 正式离线固定批次诊断 / 完成 Issue #94 Task C4：在
E-035 的同一 fixed batch 上，对 R0 与 Energy-only 两个 objective endpoint 比较三种
temporal-credit 形式，回答 E-035 保留的归因边界——共线是来自 reward/batch 本身（A），还是被
全部 arm 共享的 critic / GAE common term 主导（B）。本实验不重新 rollout、不执行 optimizer
step、不改变正式 PPO 训练定义。

**代码 / 环境**：基线 `38da0ba` 加本次未提交改动；`runtime_metadata.json` 和
`tracked_diff.patch` 记录运行时仓库状态，`source/` 保存实际执行的模块与 CLI。Windows、
Python 3.10.20、PyTorch 2.12.1+cu126、CUDA:0；actor backward 为 float32，无 rollout
autocast。

**批次来源**：直接复用 E-035 源 batch（`e-035-issue94-task-c-source-batch/`，16 episodes /
128 transitions，initial policy hash `04969773…`）。Task B 冻结校准规则在本 batch 上重新执
行并通过 `expected_calibration`（rtol=0.05）溯源守卫；与 E-035 相同的
`calibration_verification.json` 记录在产物中。

**与 E-035 的交叉核验**：本实验 standard-GAE arm 的全部 40 个数组（r0=arm_0、
Energy-only=arm_4 的 reward / raw / center / z advantage / value_target，以及三种 advantage
形式 × 五个参数组的梯度向量）与 E-035 `diagnostics.npz` **逐位一致**（max abs error
`0.0`，容差 rtol=1e-5 / atol=1e-6）。因此 C4 与 E-035 严格共享同一 batch、同一 policy、
同一校准与同一 standard-GAE 基线。

**配置 / 命令**：

```powershell
just critic-gae-ablation `
  --source-dir outputs/studies/scalar-reward/e-035-issue94-task-c-source-batch `
  --reference-dir outputs/studies/scalar-reward/e-035-issue94-task-c-objective-decomposition `
  --output-dir outputs/studies/scalar-reward/e-036-issue94-task-c4-critic-gae-ablation
```

`configs/experiments/scalar_reward/critic_gae_ablation.yaml` 显式指定量化分位、校准目标分数、
E-034 冻结值、溯源容差、E-035 交叉核验容差与 Gate C endpoint 阈值（复用
`cos <= 0.99 AND (RMSE >= 0.10 OR sign-flip >= 0.05)`）。

**计算语义**：arms 为校准 R0 与 Energy-only endpoint（同 E-035 定义）。三种 temporal-credit
形式共享同一批 component scores、safety gate、policy context、动作、old log-prob、reward
与 episode boundary，只改变 credit 计算：

```text
standard_gae      当前固定 critic V(s), V(s')（tail bootstrap 已并入末步 next value）
reward_only_gae   固定 V(s)=V(s')=0（置零 next value 同时消融 tail bootstrap），
                  保持相同 gamma=0.99 / gae_lambda=0.95 / episode boundary
discounted_return 逐 episode 反向递推 R_t = r_t + gamma * R_{t+1}，不使用 critic 与
                  bootstrap，仅作额外 temporal-return diagnostic
```

每 arm × credit form 在 raw / center-only / z 三种 advantage 形式下各做一次 full-batch
`loss_objective` backward（primary comparison 为 z，与实际 actor preprocessing 可比）。
backward 后 policy hash 不变，optimizer steps 为 0。

## 结果：R0 vs Energy-only（z-normalized，primary）

| Credit form | Pearson | Spearman | Sign flip | Z-adv RMSE | Head cosine | Head norm ratio | Lateral / Longitudinal cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard GAE | 0.997583 | 0.980109 | 0 | 0.069255 | 0.999713 | 0.907619 | 0.999639 / 0.999754 |
| reward-only GAE (V=0) | 0.997577 | 0.980115 | 0 | 0.069343 | 0.999712 | 0.907502 | 0.999650 / 0.999752 |
| discounted return | 0.998051 | 0.980916 | 0 | 0.062184 | 0.999783 | 0.924558 | 0.999960 / 0.999799 |

raw / center-only 辅助行：raw 形式 head cosine 分别为 0.997131 / 0.997127 / 0.997959，
center-only 与 z 的 cosine 相等（恒等式），三者同样全部高于 0.99 阈值。shared-trunk 梯度
仍因 actor-head 零初始化严格为 0，cosine 记 `null`，不作为证据。

各 arm raw advantage（标准 GAE → V=0 → discounted return）：

| Arm | Raw A mean | Raw A std |
| --- | ---: | ---: |
| R0 | 3.108009 → 3.106143 → 3.476937 | 1.437447 → 1.436642 → 1.750349 |
| Energy-only | 1.516289 → 1.514422 → 1.695475 | 0.697276 → 0.696473 → 0.849516 |

## 归因证据

1. **共享 critic 项在 arm 间差分中严格抵消**：在相同 V、相同 boundary 下，
   `A_R0(t) − A_E(t)` 中所有 `V(s)/V(s')` 项在 GAE 递推差分中两两相消，两个 arm 的
   advantage 差等于 (γλ)-discounted reward 差。产物数值证实：standard 与 reward-only 的
   between-arm raw advantage 差分最大仅差 `1.43e-6`（float32 噪声），raw RMSE 完全相等
   （`1.7555494`）。因此 **standard GAE 与 reward-only GAE 的 endpoint 对比在数学上只可能
   通过归一化 σ 产生微小差别**，实测 head cosine 差 `6e-7`。
2. **critic 对本 batch 的 advantage 形状/尺度贡献极小**：把 V 全部置零后，R0 的 raw
   advantage mean 仅从 3.108009 移到 3.106143、std 从 1.437447 到 1.436642（逐点最大位移
   `0.0032`）；E-035 记录中"raw advantage 尺度（std 0.70–1.44）主要来自共享 critic
   value/GAE 结构"的解释被修正：该尺度几乎全部来自 reward 流自身的 (γλ)-discounted
   return。
3. **discounted return（γ 折扣、无 critic）同样共线**：Pearson 0.998051、head cosine
   0.999783，甚至略高于 reward-only GAE——更长的时间混合没有释放额外的排序信息，也未出现
   temporal-credit-structure sensitivity。

## C4 归因裁定

```text
Standard GAE endpoint: FAILED（cos 0.999713 > 0.99，RMSE 0.069 < 0.10，sign flip 0 < 0.05）
Reward-only GAE endpoint: FAILED（cos 0.999712，RMSE 0.069，sign flip 0）
Discounted return endpoint: FAILED（cos 0.999783，RMSE 0.062，sign flip 0）
Attribution: reward_batch_collinearity
```

按 Issue #94 C4 判据：standard GAE 失败且 reward-only GAE 同样近共线，落入"reward/batch
collinearity 的更强证据"分支；discounted return 也未分离，故不标记 temporal-credit
sensitivity。E-035 保留的归因边界（A. reward/batch 缺少独立排序信息 vs B. 共享
critic/GAE common term 主导）现在可以关闭：**答案为 A**。共线不是被 critic / GAE / 
normalization 制造出来的——在完全移除 critic 后，两个 objective 对该 batch transitions 的
排序本身几乎相同（Pearson ≈ 0.998）。E-035 的 Gate C 裁定（FAILED）维持不变，本实验只
细化其失败原因，不改写原始裁定。

**含义**：在该 initial-policy / no-traffic update-0 batch 上，R0 的组合分量（校准后
Progress/Comfort/TTC/Speed 经 safety gate）与 Energy 分量对 transition 的排序高度一致；
Energy-only endpoint 不提供独立的优化方向。reward redesign 若要在 actor-gradient 层面可辨
识，需要改变 reward 对 transitions 的**排序结构**（分量形状/门控结构），而不是 λ 相对尺度
或 credit 计算方式——这与 E-033/E-034/E-035 的结论一致并加以强化。

**结论边界**：以上结论只支持该 initial policy、该 no-traffic update-0 batch、该校准配置与
该 Energy-only endpoint 定义；不证明 optimizer 问题，不证明训练后行为，不选择训练 reward
或 λ，不修改 PPO 训练定义。按 Issue #94 顺序，无论归因落在哪一类都进入 Task D（guidance
control-authority intervention）；Task E（Energy objective representation）的进入条件之一
（Gate C failed）在本 batch 上继续成立。

## 验证与产物

新增测试 `tests/training/test_critic_gae_ablation.py` 6/6 通过，覆盖：V=0 GAE 与
(γλ)-discounted return 的逐元素等价（含 rollout_limit tail bootstrap 置零）、standard GAE
含 bootstrap 的手工递推对照、discounted return 反向递推与 `value_target == advantage`、
`zero_critic_values` 仅改变两个 value 键、analyze 的结构/恒等式（center=raw−mean、
z=center/σ、`g_z = g_center/σ`、单步 episode 下三种 credit 形式的解析值）与 policy/optimizer
不变守卫、归因三分支标签、配置校验。直接相关既有测试（Task C 7、PPO 7）通过；Ruff 通过；
新增/修改模块 Pyright 0 errors（`tracking.py` 的 mlflow import 报错为预先存在，与本次无
关）。

正式产物只读核验：40 个 standard-GAE 数组与 E-035 逐位一致；policy hash 前后一致；
optimizer steps 为 0；输出目录由 runner 独立创建（`exist_ok=False`）。

产物目录：`outputs/studies/scalar-reward/e-036-issue94-task-c4-critic-gae-ablation/`
（git ignored）：

- `report.md` / `summary.json`：arm × credit form 统计、endpoint 对比（raw/center/z）、
  C4 归因裁定与 E-035 交叉核验记录。
- `diagnostics.npz` / `sample_index.json`：逐 transition reward / 三种 credit × 三种
  advantage 形式的 advantage 与 value_target、逐 form 逐参数组梯度向量。
- `calibration_verification.json`：本 batch 校准值与 E-034 冻结值对照。
- `resolved_config.yaml` / `diagnostic_config.yaml` / `runtime_metadata.json` /
  `tracked_diff.patch` / `source/`：运行来源与实际执行的源码快照。

Task C（C1–C4）已完成；未执行 Task D–H，未修改全局 reward 默认配置，未关闭 GitHub
Issue #94。
