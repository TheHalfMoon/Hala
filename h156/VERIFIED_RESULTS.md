# H156 — independently verified Qwen3.5-2B typed-action LoRA

**Status (2026-10-27):** Genuine neural improvement on SAME-APP DEV
demonstrated, **not** an independent multi-application holdout or Hala
flagship 35B improvement.

## Frozen source and model provenance

- Original upstream application: unchanged MIT MiniWoB++ at
  `7fd85d71a4b60325c6585396ec4f48377d049838`.
- **48 TRAIN + 12 DEV** original 960x720 screenshots, all native
  source-oracle rewards positive; 60 distinct PNG SHA values, disjoint
  from 81 prior H154/H155 captured images.
- Private source manifest SHA-256:
  `fe274bc48225685ea135137e1607169ef9481903c5848f9bb1e2ede78f509562`.
- Frozen prior **external research** Qwen3.5-2B LoRA SHA-256:
  `992fa18fead6724329dbe7ec7b95c8acb0a56229c1fa5b56719e5f4829130a68`.
- Completed **24 real optimizer steps** on owner-private **FREE**
  Kaggle Nvidia T4, training *only* language-model LoRA projections;
  base and vision tower frozen.
- New **trained external research LoRA** SHA-256:
  `732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6`.
- Actual new adapter and GPU receipts are kept owner-private under
  `private-h156/kaggle-output-v4/`. Model weights/media are not in GitHub.

## True before/after neural outcome

Both frozen adapters generated model-native typed action JSON on the same
12 DEV screenshots, then each action plan was replayed into its own
isolated **original Chromium instance** using only native mouse/keyboard.
Every source screenshot was replayed and SHA-verified before the action;
native reward was read **after** actions. No original DOM labels/points
were passed to the neural controller.

| Original action task | Old frozen LoRA native successes | New LoRA successes |
|---|---:|---:|
| `click-test` | 0/4 | **1/4** |
| `enter-text` | 0/4 | **0/4** |
| `choose-list` | 0/4 | **4/4** |
| **Total true browser reward** | **0/12** | **5/12** |

**Paired difference:** 5 new wins, 0 regressions, 7 both fail.
Exact two-sided paired sign test `p=0.0625`, small DEV cohort; do not
overclaim significance or real-world generalization.

Separate format learning: full valid typed action sequences
**0/12 -> 12/12**; correctly predicted text values among the 8
text-requiring tasks **0/8 -> 8/8**. The 4 click-only examples are NOT
counted as correct text predictions. Therefore 12/12 JSON is **NOT**
12/12 task completion.

## Failure and one-shot integrity

The Kaggle v1 and v3 workers aborted with zero optimizer steps due
to data mount assumptions; owner-only failure receipts retained. Private
v4 completed real optimization and saved a genuinely changed adapter.

First local model-native browser replay wrote a one-shot lock but hit
`KeyError('image_sha')` **before opening its first browser context**.
The original lock and stack trace are retained; no browser score was
produced. Corrected code performs full SHA prediction preflight before
the new lock. The **V2 replay consumed the DEV cohort** for both arms
and returned all 24 original Chromium trials and 12 paired tasks.
Do not rerun or tune on this DEV cohort.

Independent auditors verify GPU output/weight SHA and the final
native before/after paired reward outcomes. Focused local H150-H156
unit tests **42/42 passed** as of source SHA `33d299e`; GitHub CI on
later report commits must be checked independently.

## What is demonstrated, what is not

**Demonstrated:** Training the research 2B LoRA improved valid
image-to-action generation and *actual browser task completion* for
some original MiniWoB++ tasks, particularly native dropdown selection.

**Not demonstrated:** reliable click/text field spatial grounding
or full tasks across different apps, Arabic input transfer, robust
out-of-distribution CUA, and any improvement to Hala flagship 35B.

**Next H157:** collect independently frozen, SHA-disjoint new
source training/validation examples emphasizing correct click points
and text-field focus, plus a second licensed application; train a
separate candidate on free research compute, then compare genuinely
native task success on a new, one-shot evaluation cohort. Do NOT
retest H156 DEV or unseal H140 sealed OOD BLIND.

No paid compute, H140 BLIND use, main merge, publication or release.
