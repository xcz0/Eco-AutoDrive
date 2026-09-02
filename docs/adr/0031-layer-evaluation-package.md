# Layer the evaluation package by execution phase

**Status:** Accepted and implemented
**Date:** 2026-09-02

ADR 0029 收口了 evaluation 的包根 API 和当前 artifact 契约，但扁平目录随后同时包含
planner inference、serial/vector episode、在线 recorder 和离线 artifact/report 等十余个模块。
其中 `episode.py` 与 `runtime.py` 也分别混合了多种控制流和边界职责。

因此保留 `evaluation.engine` 与 `evaluation.config` 作为 job 级入口，将其余实现分为三个内部域：

- `evaluation.inference`：agent protocol、Diffusion adapter、Fabric runtime、decision validation
  与 host transfer；
- `evaluation.episodes`：共享 lifecycle、serial/vector 执行、trace recorder 与 rendering；
- `evaluation.artifacts`：typed models、trace schema、reader/writer、summary/metrics 与 matrix report。

`episode.py` 按 lifecycle、serial 和 vector 拆分，`runtime.py` 将 decision/transfer 边界独立出来；
原 `validation.py` 并入 artifact I/O，原 `metrics.py` 并入 summary。声明式 models、trace schema
和单一 recorder 保持完整，不按文件长度继续拆分。

本决定只取代 ADR 0029 的物理目录布局。`eco_planner.evaluation` 仍是仓库级延迟公开入口，
不提供旧内部路径转发；三个子包各自通过 `__init__.py` 导出跨域使用的接口，其中
`evaluation.artifacts` 保持延迟导出。配置、随机流、执行语义以及 JSON/NPZ/report artifact
契约均不改变；离线 reader 与 report 的导入路径仍不得加载 Torch、MetaDrive、Panda3D 或 rendering。
