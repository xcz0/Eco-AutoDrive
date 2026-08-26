# Add PlannerRFT-style energy reward at the MetaDrive execution boundary

**Status:** Accepted and implemented
**Date:** 2026-08-26

Issue #59 需要在现有 closed-loop PPO 上加入可配置、可审计的 PlannerRFT-style reward，
同时保持 `metadrive_builtin_v1` 的既有数值语义。PlannerRFT 使用的 nuPlan scorer、车辆动力学
与 MetaDrive 当前运动学 waypoint execution 不同，因此不能把本项目的阈值和适配描述为作者实现
或 scorer parity。

因此，RL reward 使用以 `name` 判别的严格配置联合。`metadrive_builtin_v1` 继续委托 MetaDrive
原生 reward；`plannerrft_energy_v1` 由 `TrajectoryMetaDriveEnv` 在每个实际 10 Hz execution
子步有状态计算。environment slot、TorchRL vector worker 和 serial path 显式接收同一个 typed
profile。环境边界拥有前一 position、velocity 和 acceleration，并在 MetaDrive done/cost、当前
traffic frame、route/reference lane 与 lane speed limit 都可用后生成不可变 reward audit。trainer、
collector、summary 和 artifact writer 只传播或聚合该结果，不再实现 reward 公式。

MetaDrive native `step_energy`/`episode_energy` 与实际 execution fuel proxy 是两个独立流。native
值保留为 phase-boundary audit；E-019 已证明它在当前 kinematic execution 下恒为零，不能作为
reward。execution boundary 以相邻实际 center position 和执行速度按同一 MetaDrive fuel proxy
公式重算 step mL，并同时保存 step distance、mL/km 和 denominator-valid。evaluation summary、
energy reward 与 training update summary 只聚合这条重算流，不从 summary 或 collector 重算。

`plannerrft_energy_v1` 的 gate 为 collision、drivable-area 和 wrong-direction 三项之积；未 gated
score 为 `(5*TTC + 5*Progress + 2*Comfort + 4*Speed + Energy) / 17`。TTC 使用 ego-forward
corridor constant-velocity closing estimate；progress 使用当前 route/reference lane 的非负纵向
step delta；comfort 使用实际 execution acceleration、jerk magnitude 和 yaw rate；speed 使用当前
lane limit；energy 使用 `exp(-ml_per_km/50)`，位移小于 0.01 m 时为零。阈值、权重、margin 和
归一化尺度全部保存在 resolved reward profile，50 mL/km 仅是 E-019 支持的 smoke normalization。

PPO TensorDict 只保存最终 scalar reward。CPU rollout audit、`TrajectoryExecutionRecord`、TorchRL
remote result、NPZ 和 update summary 保留 profile-specific typed audit。builtin profile 保持既有
dense/terminal artifact schema；energy profile 保存 gate、component、原始量、collision/termination
和双能耗字段。该决定支持一次真实 PPO update 的链路验证，不构成 A/B、PlannerRFT parity、真实
车辆舒适性或节能改善证据；在对应实验完成前不关闭 Issue #59。
