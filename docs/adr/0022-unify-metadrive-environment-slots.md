# Unify planner-facing MetaDrive environment slots

**Status:** Accepted and implemented
**Date:** 2026-08-25

Serial evaluation、single-environment rollout 和 vector worker 曾分别创建 MetaDrive
environment、observation adapter、traffic history，并各自实现 reset、stationary warmup、
observation build、trajectory step、vehicle state 与 route-length 读取。这些路径表达同一仿真
生命周期，但重复实现会使交通历史、换图和失败语义逐渐分叉。

因此，使用 `MetaDriveEnvSlot` 作为单个物理环境的共同所有者。slot 持有
`TrajectoryMetaDriveEnv`、严格 traffic/no-traffic adapter 和地图缓存，提供 reset、warmup、
observe 与 step 边界。Serial evaluation 和 rollout 直接使用 slot；vector worker 在独立进程中
持有同一对象，IPC client 只负责命令调度、transport timing 和错误传播。Evaluation trace、
artifact failure classification、RL reward/GAE/PPO 仍由各自调用方拥有，不下沉到环境层。

环境适配器只接收 `PlannerObservationSpec` 中实际使用的 observation shape 字段，不再依赖或在
worker 中重建完整 `OfficialDiffusionPlannerConfig`。Windows spawn payload 保持为仅含普通 Python
值的映射；torch-free trampoline 在子进程开始执行后才导入 worker runtime。

该选择不改变 trajectory、坐标、交通历史、限速、随机流、reward、termination、artifact 或
timing 字段语义，也不实现 ADR 0009 中尚未完成的 ego-independent traffic prewarm。
