# H77 — Opt-in private checkpoint continuation after H65 host egress

**Parent:** H76 PR #60, SHA `191f593534dbfb68238e329f82617d466779706b`. **Status:** CPU-tested integration only; no actual GPU allocation, new trained checkpoint or provider upload.

## Problem solved

H65 already includes:
- `safe_diagnostic`: supervised training on an **already-granted eligible free T4**, with host-side immutable checkpoint egress.
- `provider_publish.publish_latest`: an explicit *separate* Kaggle private dataset upload, privacy inspection, unauthenticated HTTP 403 verification and authenticated **byte-for-byte** downloaded artifact readback.

On a revoked Colab session, even a previously rescued checkpoint was only staged on the host; private durable publication required a later manual command. H77 optionally connects these existing source-first components with a fail-closed continuity step after the H65 supervisor returns, **even when the child failed and the GPU is gone**. This happens on the host CPU; it does not create a new GPU session or train on Kaggle.

## Explicit invocation

The default supervisor is **unchanged**: no publication and no secret read. To opt in for a genuine approved session, configure your *host* `KAGGLE_API_TOKEN` environment variable privately, then add:

```bash
python -m h65.safe_diagnostic \
  --session EXISTING_AUTHORIZED_FREE_T4 \
  --pinned-sha REVIEWED_H77_COMMIT_SHA \
  --destination PRIVATE_HOST_RECEIPTS \
  --steps train \
  --stage-sha256 4a8e602c785c2b90a60e49f78381a7af983401cb99b16c53f3df69e930cc073a \
  --stage-source-sha acd694fbae10752ff4ee3fe6cc3e52142ab7f0fe \
  --max-steps 1 \
  --processor-audit-records 32 \
  --private-readback-after-egress
```

The user must **independently confirm** that Colab is an authorized, zero-paid-compute free session, that the H71 private archive is verified and staged, and that the pinned Qwen weights can be prepared. Nothing in this new flag bypasses GPU admission, privacy or the existing training policy.

## Strict state semantics

- No valid provider token: reject **before** the GPU hardware probe or operator launch.
- No verified H65 host checkpoint: never try to upload; write `SKIPPED_NO_VERIFIED_HOST_CHECKPOINT` and fail closed.
- H65 has verified host checkpoint: re-run local `load_checkpoint`, use the **existing** Kaggle private uploader (which itself verifies privacy, anonymous refusal and full artifact readback), and independently verify the returned checkpoint step and result flags.
- Completed H65 + successful independent private readback: `COMPLETED_WITH_HOST_EGRESS` with `private_provider_readback_verified=true` and `PRIVATE_READBACK_VERIFIED`. **This is storage durability, not model or adapter qualification.**
- Failed H65 (for example, T4 revoked) + successful independent private readback: still **FAILED**, with durable checkpoint receipt preserved, not a successful training receipt.
- Upload error, public provider, missing/incorrect checkpoint, wrong step or absent provider readback: `PRIVATE_READBACK_FAILED`, fail closed; original host evidence untouched.
- Host copy deletion is **never** requested. Secrets are never accepted as CLI arguments, printed or stored in receipts; error type only is retained.

## Evidence and limitation

`tests/test_h77_private_checkpoint_continuation.py` exercises the CLI opt-in vs default, deterministic synthetic positive and negative scenarios, fake failures, no checkpoint, identity mismatch, encrypted-secret redaction and preservation of a failed monitor outcome. Original Kaggle publisher implementation and its existing tests remain unchanged.

All tests use local synthetic checkpoints and mock publisher responses. **No actual Kaggle private publication or authenticated real byte readback** was attempted. Existing H71 private corpus, H73 private CPU admission, H74 source SHA isolation, H75 bounded processor audit and H76 crash-safe egress remain intact.

The remaining gate is a genuinely available, zero-cost eligible T4, one real optimizer step, real H65 egress, **actual** authorized private provider readback and an independently verified held-out Arabic/English GUI assessment.
