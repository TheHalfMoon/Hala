# H14/H15 security repair delta

Verdict: **no new must-fix security finding**. Reviewed narrowly after the independent correctness repairs; the original security report remains the baseline, with the source hashes below superseding its repaired files.

- **must-fix:** none.
- **worth-considering:** none for the existing operator-owned experiment boundary.
- **skip-it:** no external intake, execution authority, policy bypass or shell surface was added by either repair.

H14 `evaluation/heldout_runner.py:14,48,74` now publishes result and image-freeze JSON through the existing `training.checkpoints.atomic_json`. The helper imports no model package; it serializes with `allow_nan=False`, creates an exclusive UUID temporary file next to the destination, flushes/fsyncs and atomically replaces the destination, then syncs the directory on Linux. A replacement failure propagates instead of implying publication and preserves the prior JSON. A leftover temporary file does not become the named result or freeze. Destination paths remain operator-controlled, within the fresh output directory. `tools/freeze_heldout.py:10` adds checkpoint/artifact helper bytes to the trusted pre-inference source manifest. No model text becomes a filename or callable through this import.

H15 sibling `training/scripts/qualify_lora.py:104` compares ordered step/loss pairs using the existing finite-value, type/structure and floating-tolerance comparator. `snapshot_state` already validates checkpoint state: metrics length equals step, steps are sequential, values finite/nonnegative and metadata bounded to 4 MiB. The qualification comparison additionally requires final step two; consequently this new list contains two validated records per trajectory. Excluding measured seconds avoids making scheduling noise into a qualification mismatch. Loss mismatch now participates in the existing overall result and cannot escape the real-runtime/unqualified-CPU flag logic. Safe tensor/snapshot readers are unchanged.

Independent bundled-Python probes through RTK: `tests.test_heldout`, `tests.test_heldout_persistence`, `tests.test_training_integrity`: **26 passed**; sibling `tests.test_lora_qualification`: **7 passed**. The persistence regression injects replacement failure for both named JSON files and checks preservation; the loss regression distinguishes changed loss from changed timing. An initial invocation also named nonexistent `tests.test_checkpoints` and failed module discovery; the corrected invocation above covers the actual integrity module and passed. No GPU/tensor runtime, dependency installation, source edit or job was executed by this judge.

Recomputed prospective manifest SHA-256: `0d81cc61ecaafb79b52b6a0368f76d550c92f5e4c5ccb05ef468e92a83c5b8d6`; `verify_freeze` accepted all 64 cases. Dataset SHA-256: `0efd21686fb1fc6255ffe463256f290c8d1cbb2fbcd61ad8d982ae1c5990e6c5`.

Reviewed source hashes:

- H14 runner: `b7797e938c9ecb74e71a1ecbd834cd12bc41a70b61539e9299ec68eb59c5b014`.
- H14 freeze tool: `ae86392c1357011c0a83331d8562f39190c0e1a7367045c504a53aff845351ba`.
- Checkpoint helper: `93e34ea8e20e77e7424a400656e15c814457d608382c8b4db75597a65a891826`.
- Artifact helper: `80742d1ea6bdafc2d16db27d57aec4e653ca53b8ca54776908bd07869d97e2c1`.
- H15 qualifier: `8fee099a627fdafd05d737d188288ae05aa37f69e9b273447de0318bef812523`.

Manifest: **security delta ✓ independent PStack session-model judge; targeted CPU probes ✓; prospective manifest verified ✓; d1 not used; real GPU, package-origin and parent Jev/Graft gates remain separate.** This report supplies no release, deployment or measured-runtime approval.
