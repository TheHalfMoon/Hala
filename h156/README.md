# H156 — executable GUI action labels and first true LoRA training pilot

Status: source/action QA COMPLETE, GPU training prepared; no model training
result should be claimed until Kaggle records real optimizer steps.

## Why this follows H155

H155 produced 48 TRAIN + 12 DEV original MiniWoB++ reference labels
but its `select(text)` role had no visible coordinate for a native
dropdown action. Selecting by CSS/DOM selector would not be valid
computer use. H156 fixes this as `select(point:[x,y],text)`. The actuator
uses **mouse click + keyboard Home + typing + Enter**, no DOM access.
Five exploratory owner-only native list tasks succeeded 5/5 using this
path; these are NOT model-generated actions or unseen-app holdout.

## Fresh source and privacy

- Untouched original Farama MiniWoB++ pinned MIT source
  `7fd85d71a4b60325c6585396ec4f48377d049838`.
- New 960x720 screenshots: **48 TRAIN / 12 DEV** (balanced across
  click-test, enter-text, choose-list); every original app reward positive.
- **60/60** reference action plans actually completed with the
  DOM-free native mouse/keyboard replay of source-oracle-generated labels.
- **81 previously captured H154/H155 image SHA values excluded**,
  plus 60 current image SHA values unique between TRAIN and DEV.
- Owner-independent audit: `H156_OWNER_INDEPENDENT_ACTION_SOURCE_AUDIT_PASS`.
  Owner manifest SHA-256:
  `fe274bc48225685ea135137e1607169ef9481903c5848f9bb1e2ede78f509562`.
- Original images, labels and model adapter kept private under
  `private-h156` on Windows and in **private Kaggle dataset**, not Git.
- Dataset slug: `abdulazizshehri/hala-h156-private-typed-action-pilot-v1`.
  Source media ZIP SHA-256:
  `6bb182e432e986d1c89554e8887a3ef1c64b297119b25c9e4278685da4c0f423`.
- Existing frozen H142 Qwen3.5-2B research LoRA model SHA:
  `992fa18fead6724329dbe7ec7b95c8acb0a56229c1fa5b56719e5f4829130a68`.

## New experimental action model

`h156.train_gpu` defines a bounded pilot: **24 actual optimizer steps**,
two genuine examples per gradient update, frozen base/vision tower,
trainable research language LoRA only, and **48 TRAIN examples**.
Twelve distinct DEV examples are for actual neural baseline/after
structured-action comparison only, not the optimizer. Exact 2B
base revision, pinned torch/transformers/peft and **free private
Nvidia T4** are admission requirements, no paid fallback.

Success criteria are strict valid full-JSON action sequences and correct
text values on DEV. This DEV output is **not** true browser reward or a
cross-app benchmark. A valid next gate will replay **new model predictions**
with DOM-free mouse/keyboard in actual MiniWoB, never supplying model
the reference points/texts, and only then measure true task completion.
Unseen apps and Arabic remain unqualified pending separate research.

No H140 sealed blind access, flagship Hala 35B update, model publishing,
main branch merge, or release.
