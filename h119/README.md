# H119 — full-512 original TRAIN anchor-robust END supervision (CPU source only)

## Reason and observed evidence
H118 finished real 64-scene public EVAL indices 76..91/domain. Kaggle
and independent owner-Windows original source/prompt/raw-output SHA and
unchanged H61 scorer replay agree: strict complete drag H106 **0/64**,
H114 **0/64**, H117 **0/64**. H117 is **not** a model improvement.
H117 valid actions were 55/64 versus H114 64/64; H117 START hits 7
versus H114 11; END hits each 1. Stop treating lower loss as GUI skill.

## New source-only hypothesis
The prior H117 optimizer saw only 128 authentic TRAIN images and assumed
an oracle/authentic TRAIN START anchor. During inference its own model
START prediction may be wrong. H119 isolates an untouched-source,
more-diverse *candidate curriculum* and explicitly perturbs TRAIN anchors:

- Re-admit **only** the original, SHA-locked H113 1024 START/END TRAIN
  records from 512 genuine unique screenshots, preserving exact image SHA.
- Exactly 512 START point examples and 512 END signed-delta examples,
  source paired, multilingual (original 128 Arabic/384 English) and
  spanning all five qualified original TRAIN families.
- Train END prompt anchor is derived deterministically from authentic
  TRAIN START, at scales 0/24/64/128 normalized units, using image SHA
  and a fixed H119-only nonce. Its signed delta reconstructs authentic
  TRAIN END *exactly*. Neither GT END nor EVAL answers enter the prompt.
- Never read older H115/H118 EVAL targets for TRAIN construction, change
  previous checkpoints, or reuse an H117 adapter. Raw data remains private.

This is only a **CPU-verified training-data plan**, NOT a new trained model
or a demonstrated accuracy improvement. The existing H106/H117 trainer
explicitly restricts joint LoRA to 256 optimizer steps and the original
first 128 source pairs. A **new opt-in isolated trainer path**, new run
identity, source/checkpoint/optimizer SHA attestations, 1024-step T4
resource budget, and regression tests are required *before* H119 may
run real training. Do not remove inherited safety constraints or silently
reuse existing training flags.

## Pre-registered future evaluation
Only after a distinct actual H119 adapter has independent byte replay,
freeze genuinely untouched original public EVAL indices 92..107 per each
four domains (64). Compare H119 vs frozen H114 and H117 using identical
original screenshots and unchanged H61 scorer. Model-produced START
anchors only; strict raw JSON, no clipping, no oracle prompts, genuine
START and END hits, complete task success, per-domain and paired
significance, and inference-call budgets. Only improvement in real strict
complete GUI tasks can justify calling the candidate better.

No model/weights/media publication, no paid compute, no deployment, no
merge, no hidden TEST use or release.
