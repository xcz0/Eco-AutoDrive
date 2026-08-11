# Use a single-device Fabric inference runtime

Use Lightning Fabric, without Trainer, as the sole assembly boundary for closed-loop inference device placement, forward precision and global seeding. Keep one MetaDrive loop and one artifact writer per Hydra job; reject `devices != 1` instead of implicitly launching distributed simulator processes.

The default runtime resolves CPU to `32-true`. CUDA resolves to `bf16-mixed` when the device supports BF16 and otherwise to `16-mixed`; every artifact records the resolved value. This refines ADR 0001: official weights, model hierarchy, normalization and sampling semantics remain the controlled baseline, but automatic CUDA mixed precision is not a strict FP32 numerical reference. Comparisons that require that reference must explicitly select `32-true`.
