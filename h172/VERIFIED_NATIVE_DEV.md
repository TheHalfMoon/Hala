# H172 — independently verified new original-browser paired native DEV reward

**Owner-private one-shot new DEV evaluation completed 2026-10-28.**

## Predeclared study and exact immutable source

This study was preregistered before any H172 GPU model inferences.
It compares frozen H164 language-only research baseline against the
**same frozen H164 language adapter plus H171 newly trained rank16 visual
adapter**. No optimizer steps, hyperparameter search, or weight updates.

Two original unmodified MIT-licensed application families, **same as H171
TRAIN but entirely new scenes**, were used: flatpickr November 2026
original calendar selections (vs H171 October TRAIN), and Awesomplete
Node.js autocomplete suggestions (vs H171 JavaScript TRAIN). Exactly
16 cases/app, 16 English and 16 Arabic, 32 original source-native
reference success checks. First capture was **quarantined** for a 1-pixel
replay discrepancy; the fresh replacement `new-dev32-v2` passed
**32/32 exact independent RGBA pixel replay**. This is SAME-FAMILY DEV,
**not** previously unseen application OOD.

Private DEV-only model-input ZIP SHA-256:
`0be919e841d737b16a86875e81ab7c640d430a2baceb05596d94e0611a79b231`.
It holds original Chromium screenshots and visible instructions only:
**zero owner reference points/gold**, zero prior consumed DEV/OOD
screenshots, and zero H168/H171 TRAIN images. Owner reference actions
were kept in a separate private file on D.

## Private free GPU and owner browser replay

- Research baseline: exact H164 frozen language LoRA SHA
  `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
- New research candidate: H164 language + exact H171 rank16 visual-merger
  checkpoint SHA
  `8796e964b9f830d353daf8bc21d90543b9b792931fe7733c6f443ba97057622e`.
- Exact pinned code SHA
  `494637a141feb68adf64d3e257a4173054a586ff`.
- Private free Kaggle Tesla T4
  `abdulazizshehri/hala-h172-private-paired-new-dev-t4` COMPLETE.
  **64/64 real frozen paired inferences** (32/role), no optimizer steps,
  GPU CUDA peak allocated **4.307 GiB**. Schema-valid responses:
  baseline **32/32**, candidate **31/32**, emphasizing that valid
  syntax alone cannot measure browser success.
- Separate independent frozen GPU output audit:
  `H172_INDEPENDENT_64_PAIRED_NEW_DEV_GPU_PROPOSALS_PASS`.
- Owner Windows Chromium replayed exact original preaction pixels
  before applying **only** model-generated real screen-coordinate
  clicks. Ground truth was actual selected original datepicker or
  autocomplete input value, **not proximity to a reference click point**.
  It executed **64/64** one-shot paired native actions. The H172 owner
  new DEV was then permanently marked consumed in
  `private-h172/H172_FRESH_NATIVE_DEV_ONE_SHOT_CONSUMED.lock`.
- Independent owner-browser reward audit:
  `H172_INDEPENDENT_ORIGINAL_NATIVE_32_PAIRED_DEV_VERIFIED`.

## Real app-native task success, no benchmark reuse

| New original-browser DEV (n=32) | H164 frozen language only | H164 + H171 vision |
| -- | --: | --: |
| **Total actual native success** | **4/32 = 12.5%** | **15/32 = 46.875%** |
| Arabic (n=16) | 2/16 | 7/16 |
| English (n=16) | 2/16 | 8/16 |
| flatpickr November date selections (n=16) | **4/16** | **2/16** |
| Awesomplete Node.js suggestions (n=16) | **0/16** | **13/16** |

**Paired winners: 13, regressions: 2**; exact two-sided McNemar
`p=0.00738525390625`. The +11/32 net cases and strong autocomplete
performance provide direct evidence of improvement on this particular
new same-family DEV. But the **datepicker regressions are real**:
do NOT describe the checkpoint as universally better, better on all
visual actions, or ready for production. The sample is small and
limited to two known app families.

**Decision:** H171 is a promising *frozen research candidate* for
unseen original-app OOD. **NO-GO for general promotion/release** until
a fully preregistered pair of unrelated original-app OOD families,
disjoint from all H164/H169/H172 evaluated scenes, is qualified and
native-reward compared without tuning. The H172 32 cases are now
irrevocably consumed; **never train on, rescore, tune, or select on them**.
Do not recycle H164 Pell/Pikaday, H140/H162/H163, or other past
consumed OOD.

Windows focused tests: **211 passed, 1 skipped**; exact GitHub Actions
for final head are a separate independent gate. No paid GPU, no
model publication or public weights, no Hala flagship 35B change,
no main-branch merge; all work remains a stacked Draft PR.
