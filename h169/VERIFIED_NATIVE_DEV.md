# H169 — independently verified real native browser same-family DEV reward

**2026-10-28; one-shot H169 DEV completed, sealed and consumed.**

## Why H169 was necessary

H168 proved that 32 unique true original-browser TRAIN examples updated rank-4
visual-merger LoRA within free Tesla T4 memory. That is **training feasibility**,
not evidence of improved browser behavior. H169 was preregistered before any
new model proposals: a direct comparison of **the same H164 frozen language LoRA
before and after addition of H168's frozen vision adapter** on 32 **new**
original browser scenes (16 English, 16 Arabic; 16 text tasks, 16 drags).

H169 DEV uses new source variants 16–31 rather than H168 TRAIN 0–15;
underlying app families Choices.js and noUiSlider are the same, so this is
**same-app-family DEV**, **NOT new-app OOD**. A separate Chromium session
replayed all **32/32 byte-identical original source PNGs**, and SHA registry
excludes old TRAIN/H140/H162/H163/H164 benchmark media. New DEV owner
reference actions stay under D drive only, never on Kaggle.

## Real private Kaggle paired inference

- Private no-paid T4 kernel:
  `abdulazizshehri/hala-h169-private-paired-new-dev-t4` COMPLETE.
  Pinned code source `2c4d17684086b8dee4d27aa02338606a770c65da`.
- Immutable DEV-only ZIP SHA-256
  `2dca00e6c6a0983529c625f47c863f9e5d19c94d12ac51341b4aa93bc1f2a215`,
  **zero owner gold**, **zero old OOD**.
- BEFORE: frozen H164 language LoRA, unchanged SHA
  `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
- AFTER: identical language LoRA plus H168 vision-only adapter
  `506f571d0ff45bbe8932d9ea1f56dd6e4b5b5dc5f90cfdd1397243285523cd88`.
- **64/64** real model-generated GPU inferences (32/role), **zero optimizer
  steps**, **19/32 schema-valid outputs for each role**, peak CUDA
  allocated **4.306 GiB**. No model updates/weights saved or released.
- Separate independent auditor returned
  `H169_INDEPENDENT_64_PAIRED_NEW_DEV_GPU_PROPOSALS_PASS`.

## Actual original-browser task reward (not coordinate proximity)

The owner Windows native scorer reproduced each frozen original GUI screenshot
before letting the model-generated pixel action click/type/drag **the actual
app**, and used actual application state to check success. One-shot 64/64
replays completed; independent `h169.audit_reward` returned
`H169_INDEPENDENT_NEW_DEV_NATIVE_PAIRED_REWARD_VERIFIED`:

| Metric | BEFORE H164 language-only | AFTER H164 language+H168 vision |
| -- | --: | --: |
| Actual native task success | **5/32 (15.625%)** | **5/32 (15.625%)** |
| Arabic native | 1/16 | 1/16 |
| English native | 4/16 | 4/16 |
| Native type | 3/16 | 3/16 |
| Native drag | 2/16 | 2/16 |
| Schema-valid model actions | 19/32 | 19/32 |
| Paired wins | — | **0** |
| Paired regressions | — | **0** |

Two-sided exact McNemar `p=1.0` (no discordant paired outcomes). This is a
**measured null result**, not evidence that the new vision adapter improved
Hala. The DEV set is now **permanently consumed** and locked in owner-private
`H169_NEW_ORIGINAL_DEV_ONE_SHOT_CONSUMED.lock`. Never tune to, rescore, or
train on these 32 scenes.

**Engineering decision: NO-GO to promote H168 over H164.** Retain frozen H164
language-only baseline; keep H168 visual adapter private as a research
checkpoint for further *fresh* training with broader native app coverage.
A separate preregistered never-before-seen-app OOD is still necessary to
measure out-of-family generalization after any future model selection.

The Hala flagship 35B did not change. No paid inference/GPU, no publication,
no human review, no main merge, no release. Branch and PR remain Draft.
