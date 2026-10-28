# H168 — completed 32-step NEW ORIGINAL browser TRAIN-only visual SFT (FREE T4)

**Owner-private actual GPU experiment — 2026-10-28. Training and independent receipt audit COMPLETE.** This is **not** new DEV or OOD task success.

## Original TRAIN-only source qualification

- Clean MIT-licensed pinned unmodified Choices.js (`e132628ac74dd427a9a454caf042c65d3065d170`) and noUiSlider (`57033672d07e8bac527d7147721a4a2e07e0d699`) upstream application pages, served locally on owner Windows. Neither is a previously consumed H164 OOD app.
- Native Playwright Chromium action/reference result checked **32/32**: 16 text entries, 16 true horizontal or vertical slider drags; **16 English, 16 Arabic** task instructions. Source replay independently returned **32/32 byte-identical PNG** screenshots on fresh Chromium contexts (960×720).
- All 32 image SHA-256 identities are fresh against the registered old H140/H161/H162/H163/H164 source manifest hashes. No previous sealed images read or evaluated.
- Source images and private training reference actions are in separate files *on D drive only*. Immutable owner-private TRAIN ZIP SHA `d6d23a0b4c0cebd87d5ebd9830ceeca18108037f82bcbbc2c0fb7749aae86fc6`; it contains precisely 32 TRAIN cases and **zero DEV/OOD**.
- A third MIT-pinned flatpickr application is a candidate **only**, not captured, trained, evaluated or used for hyperparameter selection.

## GPU training actually executed

- Private FREE Kaggle Tesla T4 kernel `abdulazizshehri/hala-h168-private-real-ui-32-visual-t4`, version 1, completed. Executed pinned code SHA `da7d20367b7bb1db439cf23fbff0700bfb0cab89`.
- External research model `patrickballou/gui-assist-grounder` base revision `31632edaa20749c7f532868a42635f7240033106` (not the planned flagship Hala 35B).
- **Frozen H164 language LoRA** SHA `43e0f6a260a40bda0fb5604aabde897d8aa3fdf1ced2877a32babddabbcf6b88` stayed exactly unchanged, independently verified by the frozen tensor digest. **Trainable rank-4 vision merger LoRA**, only two visual merger linear projection modules; no language layer trainable.
- **32/32 genuine optimizer steps with 32 unique original-browser TRAIN image exposures** (one shuffled exposure per image, seed 168), 16 Arabic/16 English and 16 type/16 drag, with finite supervised losses and **2–4 nonzero visual gradient tensors per step**. First and last example losses were 1.140080 and 1.125555 respectively; they are **different training examples**, so this pair is NOT a learning curve or measured accuracy gain.
- CUDA **peak allocated 6.693 GiB**, peak reserved **7.238 GiB**; passed <=13.000 GiB safety cap. No paid GPU, no additional Kaggle data source, no old OOD/DEV images, no native evaluation.
- Private vision-only adapter `adapter_model.safetensors` was saved to owner-only Kaggle output and downloaded to D; SHA-256 **`506f571d0ff45bbe8932d9ea1f56dd6e4b5b5dc5f90cfdd1397243285523cd88`**. **Do not upload these weights to GitHub, Hugging Face or public model registries.**

### Independent audit

The owner-independent, no-GPU-checkpoint-publication command
`python -m h168.audit_gpu --output D:\Hala-H168-Private\gpu-output`
returned **`H168_INDEPENDENT_REAL_BROWSER_TRAIN_32_FREE_T4_VERIFIED`**.
It checks two separately generated private job/training receipts, pinned source ZIP and frozen language adapter SHA, 32 distinct source tasks with exact language/action/app/index pairing, finite losses and gradients for all 32 real steps, T4 memory bounds, and checks the **actual saved safetensors binary header contains ONLY four visual merger LoRA parameter tensors**, with exact SHA agreement against both receipts.

## Next quality gate (not yet performed)

**The trained adapter's actual task performance is UNKNOWN.** A new, fully preregistered, previously unseen original-app DEV (labels kept owner-side) must be captured and evaluated before deciding whether visual SFT improved Arabic/type/drag behavior. After freezing that choice, evaluate on new, unrelated original OOD applications with one-shot real browser-native state reward; past Pell/Pikaday, H140/H162/H163, and other consumed DEV/OOD remain permanently off limits. A favorable training loss alone is insufficient.

All work remains **Draft**. No `main` merge, no model publish or release, no paid compute, no Hala 35B training or change.
