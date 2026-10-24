# H90 — Recover a complete model checkpoint after Kaggle receipt interruption

**Parent:** H89 [Draft PR #73](https://github.com/TheHalfMoon/Hala/pull/73), source SHA `2e43369d80ff3fca1330e5c34d40c47c049ef119`. Research-only; do not merge or publish.

## Practical checkpoint survivability defect

The H87/H88 private Kaggle research runner persists the original H61 **atomic immutable adapter-plus-optimizer checkpoint** first; only afterward does it write `training-result.json`, `hala-h87-training-receipt.json` and `hala-h88-research-attempt.json`. A GPU notebook can lose its session *after* a real optimizer checkpoint is durable but *before* the final receipts. The H88 exception handler then marks its attempt failed, while the strict H89 downloaded-output verifier necessarily refuses incomplete receipts. This can make a real valid checkpoint unrecoverable despite preserved checkpoint bytes.

H90 provides a **separate, deliberately weaker, CPU-only forensic recovery**. It does not fabricate missing H87/H88 successful receipts or pretend independent provider authentication.

## Invocation

After privately downloading **all** notebook output files, preserving the `hala-h88-research-checkpoint/` directory, run:

```bash
python -m h90.recover_interrupted --downloaded-output /private/hala-h88-output
```

This imports no model weights or Torch. It reuses H61's original `load_checkpoint` pointer, full member inventory and SHA256 checks. The candidate must bind the exact H87 noncommercial research configuration **derived from the pinned committed H61 JSON** (including the exact 1-step override), original H71 TRAIN SHA, pinned Qwen model/revision, complete adapter safetensors and `resume.pt`, and positive first-step *reported* CUDA gradient evidence and loss. A present `training-result.json` must match the immutable complete checkpoint (including the **exact artifact hash manifest**). A missing finalization is allowed because this is expressly interruption recovery, not a completed training run.

The output status is **`RECOVERABLE_LOCAL_BYTES_ONLY_NEEDS_PROVIDER_PROOF`**. It always explicitly records `provider_authenticated_readback_verified=false`, `model_training_independently_verified=false`, `model_quality_evaluated=false` and `commercial_use=false`. Its self-consistent records are not cryptographically signed attestations from Kaggle; the operator must separately verify authenticated private-provider readback and then actually evaluate the model before any success or release qualification.

## Evidence

The tests use original H61 checkpoint writer with **synthetic** adapter/optimizer bytes and intentionally missing H87/H88 receipts. They cover valid first-step salvage, legitimate missing finalization, stale failed notebook receipt, checksum tampering, missing optimizer/adapter/pointer, mismatched train source/config, nonfinite gradient, and forged finalization/artifact hash manifest. The tests do **not** execute Hala Qwen3-VL on real Kaggle; do not confuse successful GitHub CI with a trained model.

The real H87 v4 Kaggle result remains unavailable because Windows Remote Desktop Commander monthly usage was exhausted and no authenticated Kaggle connector is available here. Do not retry exhausted connection, do not publish datasets or model, and keep Hala NO-GO.
