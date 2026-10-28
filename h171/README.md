# H171 — four original-app bilingual real browser SFT candidate

**Draft research. Prior H169 same-family DEV and H164 unseen-app OOD are consumed, never reuse.**

## Motivation

H169 measured a real null result: frozen H164 language LoRA 5/32 original
browser native success and H164 language+H168 rank4 vision also 5/32,
zero paired wins or regressions. H170 proved real H168 rank4 vision
merger activations and final logits do change, though no 32 sampled
greedy action strings did. The next candidate must broaden original
native-action supervision beyond H168's 16 type and 16 drag scenes.

## Fresh original click TRAIN source

- Unmodified original MIT [flatpickr](https://github.com/flatpickr/flatpickr)
  Git `548330e6e18871d6ca1ca78746c0e89cb12a0303`,
  original HTML `index.template.html` SHA
  `5fd7cb0b9a6f3e6a7d24f76e310ebe4e2edf050e82065933df98e4689f3e4cee`.
  Packaged original npm compiled JS @4.6.13 SHA
  `912da5a92aea3447d39ab83d83706c004c999ee9b013a6267fab3d8710e8a7a8`.
  16 date selections checked against actual original input value.
- Unmodified original MIT [Awesomplete](https://github.com/LeaVerou/awesomplete)
  Git `596bc5d5916266437970ecbf65434077d3950a3d`,
  original HTML SHA
  `d3633eb827d8524f843a1f76913b6888e178e712e761d9f116cb362ef4ae98b9`.
  16 real autocomplete suggestions clicked and verified against input value.
- Arabic 16 / English 16, all original screenshots 960x720. Per-example
  reference action succeeds in original browser. Only source SHA metadata
  of prior H164 OOD/H168 TRAIN/H169 consumed DEV is consulted to exclude
  prior screenshot overlap. Never open their sealed OOD pixels.
- Owner gold is stored in separate D-only JSON from screenshot manifest;
  neither original donor source nor owner media/gold may enter GitHub.

## Original-browser exact replay gate

The first source capture `click-train-32` was **QUARANTINED**, because
Awesomplete case 171114's initial PNG was not byte-identical on replay
despite the original click succeeding. Do not train on that original
capture and do not modify its evidence to force acceptance.
The independent `click-train-32-v2` capture was also quarantined after a
single RGB pixel changed in the Awesomplete case 171114. The variant was
replaced with an original app autocomplete prefix and a **new third
capture** `click-train-32-v3` underwent full independent replay.
**All 32/32 scenes passed exact RGBA pixel identity.** The frozen source
PNG itself remains SHA-256 pinned; PNG compression bytes are not confused
with visual pixels. Private 32-case TRAIN ZIP SHA-256:
`8690b8bf40db1de82dec3620cdbb02ca1ea7fca953f253b8e4208dd559c81235`.
Both prior captures remain quarantined and were never used for training.

## Predeclared private TRAIN SFT

If the new capture qualifies, combine 32 novel original click TRAIN images
with H168's previously used **32 original TRAIN** images (16 Choices.js type,
16 noUiSlider drag) for a **64-image TRAIN-only four-app dataset**, balanced
32 Arabic/32 English, actions 32 click/16 type/16 drag.

Train a new rank16 visual-merger LoRA from the frozen H164 language baseline,
**128 optimizer steps: exactly 2 exposures of each source TRAIN image**,
seed 171, LR 2e-5, alpha32, no trainable language/base weights. Strict
single free private Kaggle Tesla T4, CUDA allocated cap 13 GiB, no paid GPU
fallback, no DEV/OOD or owner-gold from any evaluation. Save only visual
LoRA privately, and require independent receipt/weights audit.

H171 TRAIN feasibility does NOT prove improved task success. H172 must
capture entirely NEW DEV then later preregister unrelated original-app OOD,
with owner-only native real browser reward, one-shot results and frozen
comparison. Never use H169 or H164 consumed evaluations for selection.

No public weights, no main merge, no Hala 35B change. Draft only.
