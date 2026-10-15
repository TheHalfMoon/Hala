# H15 gradient repair delta review

Reviewed 2026-10-14 as a fresh independent PStack judge. Scope: `git diff HEAD -- training/scripts/train_adapter.py tests/test_training.py` only; surrounding trainer/checkpoint/spec code read to establish numerical and publication contracts. No source edits, commits, model downloads or GPU execution.

HEAD: `bd539bc012e9a2ec04af92136599890bf2a3bc42`.

Reviewed SHA256:
- `training/scripts/train_adapter.py`: `dfb73c9a1e84151357ffc55fbd2bfb47315f00568e57cbd052cddbb13dbf6b9e`
- `tests/test_training.py`: `1772eebe99767847101eafa9f4b7fd44785acb20f5e9dc5490d71754deabdf34`

## Verdict

No must-fix finding in this delta. Suitable for another bounded actual free-Colab smoke, subject to parent gates; no GPU qualification or proven overflow diagnosis from this review.

## Findings

- Worth-considering — `tests/test_training.py`: both added tests establish the exact factory arguments, but the new post-update nonfinite rejection has no executable negative test. Add a bounded control-path test in future work that makes an optimizer update produce NaN/Inf and asserts failure before checkpoint publication, alongside a finite-update case. Current passing mocks do not demonstrate that numerical guard or actual FP16 backward behavior.
- Skip-it — no extra scaler abstraction or general retry/backoff loop is warranted for this repair. The small factory permits a CPU regression check without importing model dependencies.

## Correctness and numerical contract

The FP16 scaler now starts at 1.0, avoiding the default 65536 multiplication as a plausible source of first-backward overflow. This is a hypothesis-directed mitigation, not evidence that the observed gradient was finite before scaling. Dynamic growth remains enabled. If later growth produces nonfinite gradients, the existing unscale/finite-gradient gate still rejects the run; the change does not silently skip a failed qualification update.

BF16 retains disabled scaling. After construction, real resume still loads the checkpoint scaler state, so the initial scale does not overwrite saved scale/growth state. Optimizer and RNG restore are unchanged. The inserted finite-parameter check occurs after scaler step/update and before metrics, paired checkpoint publication or final result; a corrupt update cannot become a newly published success artifact. Existing finite loss, finite/nonzero gradient norm, language-only LoRA trainability, frozen vision/base validation and per-step changed-parameter requirements remain intact. Completion masks and all processor label construction are unchanged.

## Product, evidence and security

The bounded repair matches H15's allowed trainer defect repair and finite-gradient/update acceptance. It adds no training scope, quality claim, paid infrastructure or H14 protocol change. Completed results still set `adapter_qualified: false` and require evaluation. Actual uninterrupted/interrupted/reloaded/resumed numerical comparison remains necessary; CPU tests cannot satisfy it.

No new external input, deserialization, shell, secrets or authorization surface is introduced. Existing checkpoint loading and `weights_only=True` behavior are unchanged. This is a delta security assessment, not a full repository security scan. Jev/d1 advisory output was not used to approve the patch.

## Verification

Ran with bundled Python through `rtk proxy`:

`python -m unittest tests.test_training tests.test_training_integrity tests.test_lora_qualification -v`

Result: 24 tests passed in 3.808 seconds. The narrower `tests.test_training` run also passed all 6 tests. Coverage includes scaler factory arguments, masking, checkpoint integrity/identity, interruption publication, completed resume and qualification comparison contracts. Actual model training, backward, optimizer mutation, saved model reload and GPU equivalence were not executed by this judge.

panel: light independent combined (session model) — correctness ✓ · parsimony ✓ · product/evidence ✓ · security delta inspected, no new trigger · GPU qualification pending
