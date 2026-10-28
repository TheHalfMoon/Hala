# H171 — verified 128 real optimizer steps on four original browser app TRAIN families

**Evidence qualified on 2026-10-28; private free Kaggle Tesla T4. No native quality gain yet proven.**

## New source admission

The new original MIT-licensed flatpickr and Awesomplete apps produced
32 native successful click TRAIN scenes, bilingual 16 Arabic/16 English.
Two earlier drafts were rejected over one-pixel screenshot nondeterminism;
only the third frozen `click-train-32-v3` was admitted. Independent
browser replay verified all **32/32 exact decoded RGBA pixel rasters** and
checked frozen PNG SHA-256 source bytes. Prior H164 consumed OOD,
H168 TRAIN and H169 consumed DEV image SHA registry was excluded.

The private source was packed as TRAIN-only with 32 native click reference
labels, no DEV/OOD and no benchmark scenes. New owner-private click ZIP:
`8690b8bf40db1de82dec3620cdbb02ca1ea7fca953f253b8e4208dd559c81235`.
Combined with immutable H168 original TRAIN ZIP
`d6d23a0b4c0cebd87d5ebd9830ceeca18108037f82bcbbc2c0fb7749aae86fc6`,
we froze 64 distinct original-app TRAIN examples: Choices.js (16 type),
noUiSlider (16 drag), flatpickr (16 click), Awesomplete (16 click), equally
32 Arabic and 32 English.

## Completed PRIVATE real free T4 training

- Research base: `patrickballou/gui-assist-grounder`, revision
  `31632edaa20749c7f532868a42635f7240033106`; not flagship Hala 35B.
- Exact pinned private source commit:
  `d2de3c9cc3cee75e8de55fe73f533fbf5d819c4f`.
- Kernel: `abdulazizshehri/hala-h171-private-four-app-rank16-t4`, COMPLETE.
- Frozen H164 language adapter SHA:
  `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88`.
- Trainable rank16 visual-merger adapter only, alpha32, LR2e-5, deterministic
  shuffle seed171, **128 real AdamW optimizer steps**, exactly two original
  TRAIN-image exposures each of 64 unique source cases. **229,376** trainable
  visual parameters, unchanged frozen H164 language parameters. Finite
  losses and nonzero visual gradients for each of all 128 real steps.
- Tesla T4 peak CUDA allocated **6.696 GiB**, reserved **11.914 GiB**,
  under 13.000 GiB allocated cap. No paid compute.
- Private **vision-only checkpoint SHA-256**:
  `8796e964b9f830d353daf8bc21d90543b9b792931fe7733c6f443ba97057622e`.
  Weights and source app PNGs are **not committed to GitHub, published to
  Hugging Face, or released**.
- The independent `python -m h171.audit_gpu --output
  D:\Hala-H171-Private\gpu-output` result was
  **`H171_INDEPENDENT_FOUR_APP_128_REAL_T4_TRAIN_VERIFIED`**.
  It checked frozen source SHA, exact model ID/revision, complete optimizer
  trace and per-case exposure counts, 32 click/16 type/16 drag proportions,
  bilingual quotas, finite gradients, memory bounds, frozen language
  identity and the actual **four visual-merger-only safetensors entries**.
  No previous consumed DEV/OOD screens or any owner DEV gold were used.

## Quality and release status

**Do not promote H171 yet.** This result proves that broader, stronger
visual-only adaptation can run with zero paid compute and preserved frozen
language LoRA. It DOES NOT establish improved browser task success, Arabic
grounding, unseen-app generalization or real-world safety. H169 old
same-app DEV is permanently consumed. H172 will use entirely new frozen
November flatpickr and Node.js Awesomplete DEV scenes, native owner-browser
success with paired H164 vs H171 proposals, then an unrelated original
app OOD only after freezing any selection. **No model release, no main merge,
Hala 35B unchanged.**

Windows H16/H170/H171 focused tests: **198 passed, 1 skipped**.
GitHub Actions are a separate check and should not be marked successful
until the exact current head jobs finish.
