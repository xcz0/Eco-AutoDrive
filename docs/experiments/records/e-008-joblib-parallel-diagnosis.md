# E-008 Windows CPU 作业级并行可行性诊断

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-11，方案实施前诊断。确认 MetaDrive 独立进程与 Hydra Joblib 作业并行能否共同运行，核对短回合串并行产物，并测量 Windows worker 启动成本。该运行不表示 Issue #5 或 ADR 0012 已实现。

**代码与环境**：运行 HEAD `a770b5bcf3488ff01cc6fe12f2699da20a973ea6`。首次剖析时 tracked diff 只有 `pyproject.toml`；后续基准还包含当时未提交的 `docs/agents/issue-tracker.md`，每个作业的实际 diff 保存在各自 `tracked_diff.patch`。Windows 10、16 个逻辑处理器、Python `3.10.20`、PyTorch `2.6.0+cpu`、Lightning `2.6.5`、MetaDrive/assets `0.4.3`；临时解析`hydra-joblib-launcher 1.2.0` 和 `joblib 1.5.3`。模型、checkpoint 与 DPM-10 使用共同资产。

**配置与命令**：全部使用 CPU `32-true`、关闭视频。Joblib 使用两个 `loky` worker；串行对照使用 Hydra BasicLauncher。以下是四作业完整无交通对照的核心命令，串行命令移除 launcher overrides 并替换输出目录：

```powershell
uv run --with hydra-joblib-launcher python scripts/evaluate.py --multirun `
  hydra/launcher=joblib hydra.launcher.n_jobs=2 hydra.launcher.backend=loky `
  runtime.accelerator=cpu runtime.precision=32-true 'runtime.seed=0,1,2,3' `
  video.enabled=false `
  'hydra.sweep.dir=C:/Users/xcz/AppData/Local/Temp/eco-autodrive-joblib-full-benchmark/parallel2'

uv run python scripts/evaluate.py --multirun `
  runtime.accelerator=cpu runtime.precision=32-true 'runtime.seed=0,1,2,3' `
  video.enabled=false `
  'hydra.sweep.dir=C:/Users/xcz/AppData/Local/Temp/eco-autodrive-joblib-full-benchmark/serial2'
```

另运行两个 2 s、固定地图 seed 0 的短作业，以及一个带 `cProfile` 的 2 s 直道回合。短作业显式覆盖 `evaluation.evaluated_horizon_steps=20`、`env.horizon=20` 和单一 `straight` 场景。

**结果**：

- 2 s 直道进程总耗时 `9.444 s`，其中 `run_evaluation` 为 `1.797 s`、回合为 `1.303 s`；四次推理累计 `0.641 s`，地图适配 `0.070 s`，环境 trajectory step `0.061 s`。短进程主要成本是 Python、Torch、MetaDrive 等模块导入。
- 两个 2 s 作业的 2-worker 总耗时为 `24.988 s`，串行为 `9.379 s`，串行/并行比为 `0.375`。两个 seed 的 job summary 与 `trace.npz` 全部数组逐值一致。
- 四个默认无交通作业（每作业依次运行 `straight` 与 `gentle_curve`，最多 200 个正式子步）的 2-worker 总耗时为 `44.076 s`，串行为 `37.918 s`，串行/并行比为 `0.860`。去除一次性启动后，并行两波作业约 `20.2 s`，串行四作业约 `30.5 s`，说明进程执行有吞吐收益，但本矩阵仍被额外 worker 启动成本抵消。

**产物**：`%TEMP%/eco-autodrive-plan-profile/`、`%TEMP%/eco-autodrive-joblib-benchmark/` 和 `%TEMP%/eco-autodrive-joblib-full-benchmark/`。这些是本机临时诊断目录，可能被系统清理；各 Hydra 作业在生成时包含 resolved config、runtime metadata、tracked diff、summary 和 episode trace。

**结论边界**：支持 Windows 上“每进程一个 MetaDrive 与一个 Fabric runtime”的作业隔离可以运行，并支持已核对的两个短作业在 CPU FP32 下串并行逐值一致。不支持“并行已经提升当前默认矩阵吞吐”的结论，也没有验证 traffic、长回合、DDIM、视频、CUDA 或多 GPU。只有 Issue #5 的 6-job、300-step traffic matrix 达到至少 20% 总墙钟改善并完成回归后，才能把并行入口描述为可用。
