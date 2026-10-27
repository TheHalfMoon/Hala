# H157 — original variable-layout spatial grounding source

**Verified on owner Windows (2026-10-27):** **352/352** original native
source-reference tasks passed; **280 new TRAIN / 72 new DEV** screenshots.
Independent owner-source audit PASS. Manifest SHA-256:
`a28b845b212ba7b745f1b155944174278b8aac69e69ad0f0eee3ca3acd8b7f78`.
All 352 screenshots unique and disjoint from 146 prior source/smoke images.
The source+frozen H156 research LoRA dataset is READY and PRIVATE on
Kaggle at `abdulazizshehri/hala-h157-private-spatial-action-v1`.
This is **source/reference success, not H157 model accuracy**.

**Planned H157 SFT:** 152 bounded real optimizer steps at 2 examples per
update on free non-commercial T4, using 280 unique original TRAIN images
and repeating the 24 dropdown TRAIN examples once (304 action examples)
to retain H156's previously improved list-selection ability. Exactly
72 new DEV images remain optimizer-excluded. No new H157 model checkpoint
has been verified yet.

**Purpose:** close the demonstrated H156 gaps in button coordinates and
text-field/submit targeting, while retaining dropdown selection.

## Why H157 exists

H156 **actually trained** the external 2B research LoRA, with verified
24 real free-T4 optimizer steps and a SHA-distinct checkpoint. In paired,
same-app original MiniWoB++ DEV (12 tasks) native browser completion
changed **0/12 → 5/12**: dropdown **4/4**, click-test **1/4**, entering
text **0/4**. Exact paired sign p=0.0625. The real model improved on this
small DEV cohort, but did not establish broad computer-use generalization.

Separate diagnostic measurement on the **already consumed H156 DEV**
showed mean point-center distances of approximately: click-test 81.5 px,
text-field focus 22.4 px, text submit 53.9 px, dropdown select 0.6 px.
These observations inform new training, **not** a repeat test or fresh
holdout. Do not feed the old DEV into H157 training or score it again.

## Independent original source sampling

Reuse the unmodified, MIT-licensed Farama MiniWoB++ HTML/JS pinned at
`7fd85d71a4b60325c6585396ec4f48377d049838`, with new SHA-disjoint
native Chromium captures at 960×720.

| Original task | Action | TRAIN | DEV |
|---|---|---:|---:|
| `click-test` | Spatial button click | 64 | 16 |
| `click-button` | Select the named button among distractors | 64 | 16 |
| `enter-text` | Focus, type, submit | 64 | 16 |
| `enter-text-dynamic` | Moving input and submit button | 64 | 16 |
| `choose-list` | Retain dropdown skill | 24 | 8 |
| **Total** | Five original action-layout families | **280** | **72** |

Every screenshot and instruction is collected **before** reference
execution. Source-only oracle targets are generated from the task's
visible UI instruction and the original HTML solely for supervision.
The original app's own positive reward must then be produced using
strict mouse and keyboard operations without DOM lookups.

The prior 141 H154–H156 source images and five separate H157 smoke
samples (146 SHA values) are excluded. TRAIN and DEV are newly drawn
with distinct source seeds, file/image SHA checks, fixed task quotas
and create-only owner-private output.

New source root: `private-h157/source-spatial-train-dev-v1/`.
Only source code, tests and aggregate evidence go on GitHub.
**Source QA is not a neural performance metric.** All images and labels
remain private; H140 sealed OOD BLIND remains untouched.

## Future real-model gate

Use exact approved free Nvidia T4 only and the frozen H156 adapter SHA
`732f54893172262348f7ef802a3d645626d6721f2214f28944740f0f07c91ac6`
as the research starting checkpoint; freeze source/model receipts first.
New DEV role remains **DEV only**, not an independent BLIND. After
training a new bounded LoRA candidate, report exact parsing/point
accuracy and actual reward in original Chromium using model predictions,
then require a **separate unseen-application holdout** before any
cross-app claim. Never reuse H156 consumed DEV/H140 blind for tuning.

No paid compute, Hala 35B weight update, main merge or release.
