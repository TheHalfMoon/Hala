# H121 — original TRAIN geometric diagnosis and replacement grounding strategy

## What is verified (2026-10-25)

On the **held-out original DragOn EVAL 92..107/domain** (64 scenes),
H120 owner-independent Windows replay of model screenshots, prompts,
raw neural outputs and unchanged H61 strict scorer is **PASS**.
H114/H117/H119 all achieved **0/64 complete correct GUI drags**.
H119 was nevertheless a valid training event: 1024/1024 actual free-T4
optimizer steps, final adapter/checkpoint/optimizer independently SHA
verified 18/18 checks. TRAIN success/loss is NOT GUI quality.

The owner-only H121 aggregate diagnostics read **only the original
SHA-qualified H113 512-pair TRAIN completions** and H120 **model
prediction strings without opening any EVAL labels or boxes**:

| Aggregate measure | Original TRAIN movement (512) | H114 model output (64) | H117 model output (64) | H119 model output (64) |
|---|---:|---:|---:|---:|
| Unique predicted/true signed displacements | 427 | 42 | 16 | 12 |
| Fraction of single most frequent vector | 1.17% | 26.56% | 59.38% | 73.44% |
| Positive X (rightward) movement | 50.20% | 48.44% | 1.56% | 3.12% |
| Median movement magnitude (normalized) | 200.0 | 11.5 | 141.42 | 181.02 |
| Near-zero movement fraction | 22.46% | 67.19% | 10.94% | 3.12% |

The H119 prediction-only direction bucket **left/up** appeared 58/64
times. Predicted H119 END went outside normalized frame in 14/64 cases;
H117 12/64. H114 has 18/64 zero-displacement composite actions.
These are **descriptive associations, not controlled causal proofs**:
TRAIN and EVAL may differ in task distribution and predictions reflect
both text and visual representation. They do explain *what outputs
failed* more precisely than the model loss, and they invalidate any
claim that H119 improved strict GUI skills.

## What to build next — NOT more identical fine-tuning

1. **Change spatial target representation.** Reuse ONLY H71 ORIGINAL
   TRAIN labels, images and source-defined boxes. Prioritize source-
   verified absolute END element grounding (point/region) and separate
   START grounding, not another weak signed-delta token objective.
   Require original screenshot SHA, image-unique source admission, and
   pixel-to-normalized exact round-trip into each original TRAIN box.
2. **Pretrain/evaluate on TRAIN-only image-disjoint holdouts** without
   touching official EVAL/TEST. Track both full role geometry, direction
   marginal, unique output diversity, repeated-vector frequency,
   quantization, right/left and up/down, and native action validity.
   Stop before costly T4 evaluation when geometry repeatedly collapses.
3. **Test geometry-specific methods** rather than scaling steps on the
   same 512 records: coarse-to-fine crops, spatial ruler-style input
   guidance, and a direction-classification auxiliary objective
   sourced strictly from original TRAIN geometry. Never derive
   target points from H118/H120 model failures.
4. **Preregister H122** with the next unused original public EVAL
   indices 108..123 for each domain, 64 fresh scenes; keep scorer H61,
   model-produced coordinates only, paired exact comparison against
   H114 and frozen H119, separate owner SHA replay. Do not inspect
   these EVAL answers during TRAIN design.
5. **Resources/licensing:** no paid models, no commercial Kaggle run,
   no local heavyweight inference. Recent GroundGUI (Oct 2026)
   published Apache-2.0 method code; its ScreenRef imagery dataset
   is RESTRICTED to academic research and ~330 GB unpacked. Do not
   import, copy, mirror or train on ScreenRef without proving access
   and compatibility with prospective Hala licensing. The paper may
   inform architecture design without using restricted media.

References for methods, not imported training data:
- GroundGUI (SFT + distillation/RL): https://arxiv.org/abs/2610.19100
- GroundGUI code (Apache-2.0 with NOTICE): https://github.com/PrentisAI/GroundGUI
- Restricted ScreenRef dataset: https://huggingface.co/datasets/PrentisAI/ScreenRef
- Explicit position-to-coordinate mapping: https://arxiv.org/abs/2510.03230
- Multi-View Prediction: https://arxiv.org/abs/2512.08529
- GUI-Cursor visual-feedback grounding: https://arxiv.org/abs/2509.21552

## Governance / artifacts

`h121/geometry.py` contains strict original role-format parsing,
TRAIN-only derived movement aggregates, prediction-only movement
aggregates, and **post-hoc diagnostic warnings**. It neither accesses
EVAL labels nor trains. Raw original H113 TRAIN labels, raw H120 model
predictions, screenshots and owner-aggregate report remain on Windows;
never commit them to GitHub. Only aggregate summaries and code may be
discussed publicly. Warnings are NOT an evaluation score and must not
be used to cherry-pick benchmark samples or claim model skill.
No merge, release, publication of model weights, or paid GPU.
