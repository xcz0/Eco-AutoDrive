# Separate fixed-batch collection and unify experiment workflows

**Status:** Accepted and implemented
**Date:** 2026-09-10

Experiment runners depended on other runners for batch restoration and calibration guards.
Reusable reward transformations and gradient extraction were owned by individual diagnostics.
Collection was inseparable from lambda analysis, and critic/GAE reference comparisons assumed
a fixed number of decomposition arms. These dependencies followed the order experiments were
introduced rather than the responsibilities of the code.

Place batch collection/storage, reward transformations/calibration and actor backward primitives
in `experiments.fixed_batch`. Keep each diagnostic's configuration, numerical decisions and
orchestration separate. The four diagnostics consume an explicit batch directory; calibration and
ablation additionally consume explicit diagnostic references. Match references by source, policy
and sample order, and find decomposition endpoints by their recorded labels and indices.

Training and diagnostics use the same named PPO batch/normalization operations. Seed derivation
belongs to rollout, not the trainer. Scalar-reward and stability configuration composition are
independent of their execution runners, and stability comparison models do not import execution.
These are repository-internal interfaces, not an external SDK or an experiment framework.

Use `just experiment <experiment> <action>` and `python -m scripts.experiments`, with explicit
subcommand dispatch and lazy execution imports after argument parsing and bootstrap. Config
directories follow experiment command names. This supersedes ADR 0030's experiment-specific
CLI aliases and ADR 0034's standalone analysis command and artifact-compatibility requirement;
ADR 0034's execution/analysis/presentation responsibilities remain in place.

Only new experiment artifacts are supported. Do not provide legacy import/CLI aliases, schema
migrations or automatic fallbacks. Preserve the numerical algorithms and tolerances, and change
storage only where collection/reference separation requires it. Historical experiment records
remain historical evidence with their original commands and paths.

Verification uses the original diagnostic tests, exact comparison of synthetic numerical results,
storage roundtrips, a synthetic complete offline chain, explicit reference failures, CLI tests and
a small real-simulator collection test. These checks are software verification, not new research
evidence or a rerun of historical studies. Current invocation and artifact details are maintained
in README and the system contract rather than duplicated here.
