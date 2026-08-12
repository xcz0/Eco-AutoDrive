# E-010 Windows CPU 与单 GPU 并行评测验收

[返回实验索引](../README.md)

**日期 / 类型 / 目的**：2026-08-12，Issue #5 正式本机验收。验证 Windows 上两个
Joblib `loky` worker 的长交通矩阵吞吐、CPU 串并行 artifact 一致性，以及 RTX 3050
Laptop GPU 上两个隔离进程共享单张可见 GPU 的正确性和显存边界。

**代码与环境**：基于 `c0dc1be856a73940d945db9a028c68a6c69d76de`，运行时包含
Issue #5 的未提交 diff；每个 job 的 `runtime_metadata.json` 保存了当时的完整
`git status --short`。Windows 10、8 物理核/16 逻辑处理器、Python 3.10.20、
Lightning 2.6.5、MetaDrive/assets 0.4.3、hydra-joblib-launcher 1.2.0。CPU 使用 PyTorch
`2.6.0+cpu`；CUDA 使用
PyTorch `2.6.0+cu124`、单张 NVIDIA GeForce RTX 3050 Laptop GPU（4 GiB）。模型和
checkpoint 使用共同资产。

**配置**：CPU 正式矩阵使用 FP32、DPM-10 或 DDIM-5、seeds `[0,1,2]`、traffic
densities `[0.05,0.10]`、`long_straight` 与 `long_mixed`、20 个预热步、300 个正式步、
视频关闭。串行使用 BasicLauncher 和一个 worker；并行使用 Joblib 1.2.0、`loky`、
两个 worker、每 worker 8 个 PyTorch 线程。CUDA preflight 和一致性矩阵使用 FP32、
seeds `[0,1]`、density `0.05`、10 个正式步（preflight 为 5 步）、每 worker 1 个
PyTorch 线程、确定性算法、关闭 cuDNN benchmark，并设置
`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

**命令**：使用 `scripts/evaluate.py --config-name evaluation/traffic_matrix --multirun`，
分别覆盖 `hydra/launcher=basic` 或默认 Joblib launcher、sampler、seed、density、设备与
execution 设置；产物通过 `scripts/compare_evaluation_artifacts.py` 逐 summary 字段和逐
NPZ 数组比较，并通过 `scripts/summarize_traffic_matrix.py` 生成矩阵报告。CUDA 运行按本机
环境要求直接调用 `.venv\\Scripts\\python.exe`。

**结果**：

- DPM-10 CPU 正式矩阵串行 `229.023 s`，双 worker `163.703 s`，单次正式对照墙钟改善
  `28.52%`；按用户要求不再重复独立计时。6 个 job、12 个 episode、468 个 NPZ 数组及
  全部非运行时 summary 字段完全一致。
- DDIM-5 CPU 正式矩阵串行 `182.910 s`，双 worker `117.557 s`，墙钟改善 `35.73%`；
  同样为 6 个 job、12 个 episode、468 个数组完全一致。
- 两个正式矩阵的 12 个 episode 均成功写入 Artifact v2；各为 10 个 time truncation、
  2 个 out-of-road termination，无失败 episode，视频关闭时报告未要求 GIF。
- CUDA 双 worker preflight 成功；两个进程 PID 分别为 `4600`、`18324`，每进程 peak
  allocated `62,640,640` bytes、peak reserved `65,011,712` bytes，无 OOM 或不确定算子
  错误。
- DPM-10 与 DDIM-5 CUDA 短矩阵各有 2 个 job、4 个 episode、156 个数组；串行与双
  worker artifact 均完全一致。
- 快速测试 `154 passed, 18 deselected`，显式 simulator 测试 `11 passed, 161
  deselected`；Ruff lint 和 format check 通过。

**产物**：CPU 正式产物在 `outputs/issue5-cpu/`，CUDA preflight 与一致性产物在
`outputs/issue5-cuda/`，短期诊断产物在 `outputs/issue5-smoke/`。这些目录被 Git 忽略，
不提交。

**结论边界**：支持 Windows CPU 两个隔离作业达到 Issue #5 的至少 20% 单次正式墙钟
改善，并支持 CPU 与单张 CUDA GPU 上两个隔离进程的可复现串并行一致性。CUDA 结果不构成
性能承诺；没有实现或验证多 GPU 作业分配，也不支持视频并行写入。
