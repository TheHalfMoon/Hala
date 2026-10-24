# H82 — Restore orphaned optimizer checkpoints after atomic-pointer interruption

**Parent:** H81 Draft PR #65, exact parent `b18edffb998a888cc3aa741dd8a26e16ca4a1af0`.

This is a CPU-only correctness improvement to the H61 remote trainer's resume path, not a model release or evidence of executing a real GPU step.

## Defect and recovered window

The existing immutable `training.checkpoints.publish_checkpoint` first fully writes and SHA-validates a version's adapter and optimizer state, then atomically renames it to `checkpoints/step-XXXXXXXX-<uuid>`, then updates `latest.json`. If a free Colab T4 job crashes **after the complete version rename but before its pointer update**, a future run of the H61 trainer sees the older pointer. It may re-run the already completed optimizer step, and could publish a second divergent version at the **same step** instead of resuming from the actual saved state. If this was the **first** step, no `latest.json` exists at all, and previously H61 would reject the preexisting training output.

H82's `recover_orphaned_checkpoint(output, expected_identity)` refuses ambiguous state and promotes **at most one** complete, contiguous `previous_step + 1` checkpoint using original `validate_version`, the existing artifact SHA-256 manifest and optimizer file-set checks. It requires exact configuration/dataset/manifest/`max_steps` identity, ordered finite nonzero gradient evidence for each saved step, positive adapter updates, matching version step and immutable path. It rejects missing or extra versions, incomplete or corrupt complete-named artifacts, a fork at the same step, jump-ahead progress, unexpected version entries, and linked output, pointer or checkpoint directories. Harmless incomplete dot-staging files are ignored; **never** adopted as checkpoints.

After validating, it writes `latest.json` atomically and independently calls `load_checkpoint` to reverify the newly published result. It never recreates, overwrites or deletes optimizer versions, and does not import CUDA, torch or weights. H61 now enters the strict resume path even if a `checkpoints` directory exists **without** a latest pointer; the real trainer invokes recovery after verifying source/config identities and before model loading. On an already current pointer with no orphan, it is a no-op.

## CPU-only evidence

Synthetic immutable adapter and optimizer file bundles exercise a successful first-step recovery, a successful second-step recovery preserving the previous pointer until qualification, independent full SHA validation, no duplicate step, no forward jump, no switched TRAIN data, no linked storage, zero changed-tensor / zero-gradient rejection, and safe no-op on an already published checkpoint. The end-to-end test uses the **real H61 step_train → real train_adapter.train → H82 recovery → H61 train finalization** call path against synthetic dataset/model artifacts (not weights). It proves that a crashed first-and-final optimizer checkpoint can be finalized **without any new GPU probe or second optimizer step**.

No GPU session has been allocated, no real Qwen model trained and no private Kaggle checkpoint published/read back. H65–H81 checkpoint egress, user zero-cost compute policy and original H71 data provenance remain unchanged. The first actual real optimizer update and independent Arabic/English GUI quality tests are still outstanding.
