# 0040 — Beta interior concentration 参数化

- 状态：接受并已实现
- 日期：2026-10-08
- 来源：Issue #114 Task 2（证据链：E-056 / #83 Task 1，上游 E-055 / #106）
- 取代：ADR 0016 中 `softplus(raw) + minimum_concentration` 的 concentration 参数化条款；0016 其余决定继续有效

## 背景

E-056（#83 Task 1）在 λ=4/8 的 100-update 多 seed 确认协议中把纵向 Beta α 压到 < 1：分布进入
boundary-seeking regime，精确/近边界 draw 使 `old_joint_guidance_log_prob` 非有限（seed 直接崩溃），
或 final probe boundary mass 远超 ceiling。Gate M FAILED，无候选进入 #83 final A2。

原参数化 `softplus(raw) + minimum_concentration` 只保证 α, β > 0，允许 α 或 β < 1；此时 Beta density
在 0/1 边界发散（singular mode），Gamma-based sampler 可产生精确边界样本，其 log density 无限。

## 决定

concentration 参数化为：

```text
concentrations = 1.0 + interior_epsilon + softplus(raw)
```

- `interior_epsilon` 是 Hydra 必需字段（`configs/components/policy.yaml`），冻结为 `1e-4`；该值在实现前
  冻结，不根据训练结果事后调整。
- α, β 恒大于 `1 + interior_epsilon`（softplus 严格为正）；Beta density 在 0/1 边界为零，可达分布族
  不再包含 boundary-seeking singular mode。
- 对称初始化保持 α=β=initial_concentration、初始 guidance 均值为零：actor head bias 初始化为
  `inverse_softplus(initial_concentration - 1.0 - interior_epsilon)`。
- 配置契约：`initial_concentration > 1.0 + interior_epsilon`（严格大于）。

拒绝的替代方案，不得作为主修复：rollout 直接 `validate_args=False`、把 guidance action 粗暴 clamp 到
`[-0.999, 0.999]` 后继续训练、捕获非有限 log-prob 并跳过 transition、仅降低 λ。probe 允许读取 boundary
sample 以测量失败；training probability semantics 保持严格——边界 action、非有限参数仍立即失败，不 clamp。

## 后果与可比性

- 这是 policy probability family 的科学语义变更：可达分布族被限制在 interior（α, β > 1 + ε）。
- 初始 actor bias 与全部 `policy_state_hash` 改变；旧 resolved config（含 `minimum_concentration`）在
  新代码下解析失败，不加兼容层。E-052/E-055/E-056 是旧 policy family 下的历史证据；E-055 的 λ=4/8
  nomination 不能直接作为新 family 的 candidate 证据，candidate provenance revalidation 见 Issue #114 Task 4。
- characterization golden fixture（`learned_guidance_decision.pt`）已显式再生成。
- 历史实验记录保持原样，不回填、不声称重现。
- ε 只消除 singular boundary mode，不约束 boundary mass 或 mean saturation；行为饱和由 Gate S1/S2 的
  boundary mass ceiling 与诊断区分（Issue #114 Task 3）。mean + concentration 参数化（Task 2 第二阶段）
  仅在第一阶段消除数值坍缩后仍出现长期 mean saturation 时另行评估，不与本决定同时引入。
