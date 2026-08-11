# Parallelize isolated evaluation jobs

**Status:** Accepted, not implemented  
**Tracking:** [GitHub Issue #5](https://github.com/xcz0/Eco-AutoDrive/issues/5)

MetaDrive 0.4.3 的引擎是进程内单例，因此独立场景不能在线程内并行。同一评测作业内复制 MetaDrive 闭环或 Fabric runtime 还会改变当前随机性、产物所有权和单设备推理边界。

实现完成后，只允许在相互隔离的 Hydra 评测作业之间使用进程并行。每个进程仍只拥有一个 MetaDrive 引擎、一个 `devices=1` 的 Fabric runtime 和一个 artifact writer；场景在作业内保持串行，地图 seed 与噪声 seed 继续独立记录。该决定细化 ADR 0010，不改变单设备推理契约。

Windows CPU 的 traffic matrix 目标入口使用两个 Joblib `loky` worker，并显式限制 PyTorch 线程预算。短 smoke、no-traffic 和普通 full 运行保持串行；并行模式关闭视频。只有在固定 traffic matrix 上证明总墙钟时间至少改善 20%，且串行/并行的 CPU FP32 summary 与 trace 除运行元数据外逐值一致后，才能把该入口描述为可用。

配套重构、Artifact v2、失败回合记录、v1 只读兼容和动态 matrix 网格均属于 Issue #5，在该 Issue 完成前不得写入当前系统能力说明。集中式 GPU batch inference、多 GPU 调度、线程内 MetaDrive 并行和背景交通独立预热不属于本决定。
